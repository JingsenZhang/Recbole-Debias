# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
CausE
################################################
Reference:
    Stephen Bonner et al. "Causal embeddings for recommendation" in RecSys 208
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class CausE(DebiasedRecommender):
    r"""
        CausE model:
        The version we implemented is not ideal and needs further improvement. And We speculate that the problem lies in
    the mask setting.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CausE, self).__init__(config, dataset)

        self.intervene_mask_field = config['INTERVENE_MASK']
        self.LABEL = config['LABEL_FIELD']
        self.dis_pen = config['dis_pen']
        self.lambda_1 = config['lambda_1']
        self.lambda_2 = config['lambda_2']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        self.items_emb_control = nn.Embedding(self.n_items, self.embedding_size)
        self.items_emb_treatment = nn.Embedding(self.n_items, self.embedding_size)

        self.criterion_factual = nn.BCEWithLogitsLoss()
        self.criterion_counterfactual = nn.MSELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_emb(self, user):

        return self.user_emb(user)

    def get_item_emb_control(self, item):

        return self.items_emb_control(item)

    def get_item_emb_treatment(self, item):

        return self.items_emb_treatment(item)

    def forward(self, user, item, factor):
        user_emb = self.get_user_emb(user)
        item_emb = None
        if factor == 'control':
            item_emb = self.get_item_emb_control(item)
        elif factor == 'treatment':
            item_emb = self.get_item_emb_treatment(item)
        return torch.mul(user_emb, item_emb).sum(dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        mask = interaction[self.intervene_mask_field]

        score_control = self.forward(user[~mask], item[~mask], 'control')
        label_control = label[~mask]
        control_loss = self.criterion_factual(score_control, label_control)
        # control_distance = (torch.sigmoid(score_control) - label_control).abs().mean().item()

        score_treatment = self.forward(user[mask], item[mask], 'treatment')
        label_treatment = label[mask]
        treatment_loss = self.criterion_factual(score_treatment, label_treatment)
        # treatment_distance = (torch.sigmoid(score_treatment) - label_treatment).abs().mean().item()

        item_all = torch.unique(item)
        item_emb_cotrol = self.get_item_emb_control(item_all)
        item_emb_treatment = self.get_item_emb_treatment(item_all)
        discrepency_loss = self.criterion_counterfactual(item_emb_cotrol, item_emb_treatment)

        loss = self.lambda_1 * control_loss + self.lambda_2 * treatment_loss + self.dis_pen * discrepency_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item, 'control')
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_emb(user)
        score = torch.matmul(user_e, self.items_emb_control.weight.transpose(0, 1))
        return score.view(-1)
