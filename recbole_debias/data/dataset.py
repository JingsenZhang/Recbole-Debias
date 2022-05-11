# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import torch

from recbole.data.dataset import Dataset


class DebiasDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.USER_ID = config['USER_ID_FIELD']
        self.n_items = self.num(self.ITEM_ID)
        self.n_users = self.num(self.USER_ID)

        self.pscore_method = config['pscore_method']
        self.eta = config['eta']

    def estimate_pscore(self):
        r"""
            estimate the propensity score
        """
        interaction_data = self.inter_feat  # interaction for training
        pscore_cnt = None
        column = ''

        if self.pscore_method == 'item':  # item_id may not be consecutive
            column = 'item_id'
            pscore = torch.unique(interaction_data[column], return_counts=True)  # (arr(item_id), arr(count))
            pscore_id = pscore[0].tolist()
            pscore_cnt = pscore[1]

            pscore_id_full = torch.arange(self.n_items)
            pscore_cnt_full = torch.zeros(pscore_id_full.shape).long()
            pscore_cnt_full[pscore_id] = pscore_cnt

            pscore_cnt_full = pow(pscore_cnt_full / pscore_cnt_full.max(), self.eta)
            pscore_cnt = pscore_cnt_full

        elif self.pscore_method == 'user':
            column = 'user_id'
            pscore = torch.unique(interaction_data[column], return_counts=True)  # (arr(user_id), arr(count))
            pscore_id = pscore[0].tolist()
            pscore_cnt = pscore[1]

            pscore_id_full = torch.arange(self.n_users)
            pscore_cnt_full = torch.zeros(pscore_id_full.shape).long()
            pscore_cnt_full[pscore_id] = pscore_cnt

            pscore_cnt_full = pow(pscore_cnt_full / pscore_cnt_full.max(), self.eta)
            pscore_cnt = pscore_cnt_full

        elif self.pscore_method == 'nb':  # uniform & explicit feedback
            column = 'rating'
            pscore = torch.unique(interaction_data[column], return_counts=True)
            pscore_id = pscore[0].tolist()
            pscore_cnt = pscore[1]

            pscore_id_full = torch.arange(6)
            pscore_cnt_full = torch.zeros(pscore_id_full.shape).long()
            pscore_cnt_full[pscore_id] = pscore_cnt

            pscore_cnt_full = pow(pscore_cnt_full / pscore_cnt_full.max(), self.eta)
            pscore_cnt = pscore_cnt_full

        return pscore_cnt, column
