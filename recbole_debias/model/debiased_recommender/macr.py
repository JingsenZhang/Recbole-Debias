# -*- coding: utf-8 -*-
# @Time   : 2022/4/20
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
MACR
################################################
Reference:
    Tianxin Wei et al, "Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System"
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class MACR(DebiasedRecommender):
    r"""
        MACR model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MACR, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.item_loss_weight = config['item_loss_weight']
        self.user_loss_weight = config['user_loss_weight']
        self.c = config['c']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        size_list = [self.embedding_size] + self.mlp_hidden_size
        self.user_module = MLPLayers(size_list, self.dropout_prob)
        self.item_module = MLPLayers(size_list, self.dropout_prob)

        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        yk = torch.mul(user_e, item_e).sum(dim=1)
        yu = self.sigmoid(self.user_module(user_e)).squeeze(-1)
        yi = self.sigmoid(self.item_module(item_e)).squeeze(-1)
        yui = self.sigmoid(yk * yu * yi)

        return yk, yui, yu, yi

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        yk, yui, yu, yi = self.forward(user, item)
        loss_o = self.loss(yui, label)
        loss_i = self.loss(yi, label)
        loss_u = self.loss(yu, label)
        loss = loss_o + self.item_loss_weight * loss_i + self.user_loss_weight * loss_u

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        yk, _, yu, yi = self.forward(user, item)
        score = (yk - self.c) * yu * yi
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight

        yu = self.sigmoid(self.user_module(user_e))  # [user_num,1]
        yi = self.sigmoid(self.item_module(all_item_e)).squeeze(-1)  # [item_num]
        yk = torch.matmul(user_e, all_item_e.transpose(0, 1))  # [user_num,item_num]
        score = (yk - self.c) * yu * yi
        return score.view(-1)
