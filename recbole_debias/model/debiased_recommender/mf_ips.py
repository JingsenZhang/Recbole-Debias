# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
MF-IPS
################################################
Reference:
    Tobias Schnabel et al. "Recommendations as Treatments: Debiasing Learning and Evaluation"
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class MF_IPS(DebiasedRecommender):
    r"""
        Inverse Propensity Score based on MF model.
        We simply implemented three methods (in recbole-debias.data.dataset) to calculate Propensity Score:
            1. User Propensity
            2. Item Propensity
            3. Naive Bayes (uniform) Propensity
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MF_IPS, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

        self.propensity_score, self.column = dataset.estimate_pscore()

        # parameters initializationBCE
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
        return torch.mul(user_e, item_e).sum(dim=1)  # MSELoss 需要加sigmoid吗

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        weight = self.propensity_score[interaction[self.column].long()].to(self.device)
        loss = torch.mean(1 / (weight + 1e-7) * self.loss(output, label))
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
