# -*- coding: utf-8 -*-
# @Time   : 2022/4/18
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
PDA
################################################
Reference:
    Yang Zhang et al, "Causal Intervention for Leveraging Popularity Bias in Recommendation"
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, RegLoss, EmbLoss
from recbole.utils import InputType
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class PDA(DebiasedRecommender):
    r"""
        Since the dataset does not have the condition to be divided by time period，we similarly define the global
    popularity of an item based on its interaction frequency in D.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PDA, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.predict_method = config['predict_method']
        self.eta = config['eta']
        self.reg_weight = config['reg_weight']
        self.device = config['device']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.elu = nn.ELU()
        self.propensity_score, self.column = dataset.estimate_pscore()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_item_weight = self.propensity_score[pos_item].to(self.device)
        neg_item_weight = self.propensity_score[neg_item].to(self.device)

        pos_score = self.elu(torch.mul(user_e, pos_e).sum(dim=1)) + 1
        pos_score = pos_score * pos_item_weight
        neg_score = self.elu(torch.mul(user_e, neg_e).sum(dim=1)) + 1
        neg_score = neg_score * neg_item_weight

        loss = self.loss(pos_score, neg_score)
        reg_loss = self.reg_weight * self.reg_loss(user_e, pos_e, neg_e)
        return loss + reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        score = self.elu(torch.mul(user_e, item_e).sum(dim=1)) + 1  # [batch,dim] -> [batch]
        if self.predict_method == 'PDA':
            item_weight = self.propensity_score[item].to(self.device)
            score = score * item_weight
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = self.elu(torch.matmul(user_e, all_item_e.transpose(0, 1))) + 1  # [user_batch_num,item_tot_num]
        if self.predict_method == 'PDA':
            item_weight = self.propensity_score.to(self.device)
            score = score * item_weight
        return score.view(-1)
