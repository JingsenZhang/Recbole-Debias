# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import copy
import torch

from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction
from recbole.utils import InputType, FeatureType, FeatureSource


class DebiasDataloader(TrainDataLoader):
    """:class:`DebiasDataLoader` is a dataloader for training Debiased algorithms.

    Args:
        config (Config): The config of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)


class DICEDataloader(TrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=True):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.mask_field = config['MASK_FIELD']

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.dl_format = dl_format
        self.neg_sample_args = neg_sample_args
        self.times = 1
        if (
            self.neg_sample_args["distribution"] == "uniform"
            or "popularity"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            self.neg_sample_num = self.neg_sample_args["sample_num"]

            if self.dl_format == InputType.POINTWISE:
                self.times = 1 + self.neg_sample_num
                self.sampling_func = self._neg_sample_by_point_wise_sampling

                self.label_field = config["LABEL_FIELD"]
                dataset.set_field_property(
                    self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
                )
            elif self.dl_format == InputType.PAIRWISE:
                self.times = self.neg_sample_num
                self.sampling_func = self._neg_sample_by_pair_wise_sampling

                self.neg_prefix = config["NEG_PREFIX"]
                self.neg_item_id = self.neg_prefix + self.iid_field

                columns = (
                    [self.iid_field]
                    if dataset.item_feat is None
                    else dataset.item_feat.columns
                )
                for item_feat_col in columns:
                    neg_item_feat_col = self.neg_prefix + item_feat_col
                    dataset.copy_field_property(neg_item_feat_col, item_feat_col)
            else:
                raise ValueError(
                    f"`neg sampling by` with dl_format [{self.dl_format}] not been implemented."
                )

        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            raise ValueError(
                f'`neg_sample_args` [{self.neg_sample_args["distribution"]}] is not supported!'
            )

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids, mask = self.sampler.sample_by_user_ids(user_ids, item_ids,
                                                                      self.neg_sample_num * candidate_num)
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self._neg_sample_by_pair_wise_sampling(inter_feat, neg_item_ids, mask)
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            # neg_item_ids [user_ids.size()*neg_ratio]   mask [True,False.....]
            neg_item_ids, mask = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            ) # 更改
            return self._neg_sample_by_pair_wise_sampling(inter_feat, neg_item_ids, mask)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids, mask):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_mask = Interaction({self.mask_field: mask})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_mask = self._dataset.join(neg_item_mask)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        inter_feat.update(neg_item_mask)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_model(self, model):
        self.model = model
