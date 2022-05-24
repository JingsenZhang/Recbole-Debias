# -*- coding: utf-8 -*-
# @Time   : 2022/4/15
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
REL_MF
################################################
Reference:
    Tuta Saito et al. "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback"
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import RegLoss, EmbLoss
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class REL_MF(DebiasedRecommender):
    r"""
        Two choices for loss function:
            1. nn.BCELoss  (loss_1) （suggest）
            2. Unbiased_BCELoss, referring to Eq.(9) in original paper.  (loss_0) (For this way,
            the learning rate should be set lower, e.g. 0.0001)
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(REL_MF, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.reg_weight = config['reg_weight']
        self.loss_choice = config['loss_choice']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.customized_loss = Unbiased_BCELoss()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.reg_loss = EmbLoss()
        self.sigmoid = nn.Sigmoid()

        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.global_bias = nn.Parameter(torch.zeros(1))

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
        user_bias = self.user_bias[user]
        item_bias = self.item_bias[item]
        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1) + user_bias + item_bias + self.global_bias)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        weight = self.propensity_score[interaction[self.column]].to(self.device)
        if self.loss_choice == 0:
            loss = self.customized_loss(output, label, weight)
        else:
            loss = torch.mean(1 / (weight + 1e-7) * self.bce_loss(output, label))
        reg_loss = self.reg_weight * self.reg_loss(self.user_embedding.weight, self.item_embedding.weight)
        return loss + reg_loss

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


class Unbiased_BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Unbiased_BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, label, weight):
        loss = - (label / (weight + 1e-7)) * torch.log(prediction) - (1 - label / (weight + 1e-7)) * torch.log(
            1 - prediction)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
