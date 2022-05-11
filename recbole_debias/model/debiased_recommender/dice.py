# -*- coding: utf-8 -*-
# @Time   : 2022/4/9
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
DICE
################################################
Reference:
    Yu Zheng et al. "Disentangling User Interest and Conformity for Recommendation with Causal Embedding" in WWW 2021
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class DICE(DebiasedRecommender):
    r"""
        DICE model, which equipped with DICESampler(in recbole-debias.sampler) and DICETrainer(in recbole-debias.trainer)
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DICE, self).__init__(config, dataset)

        self.mask_field = config['MASK_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.dis_loss = config['dis_loss']
        self.dis_pen = config['dis_pen']
        self.int_weight = config['int_weight']
        self.pop_weight = config['pop_weight']
        self.adaptive = config['adaptive']

        # define layers and loss
        self.users_int = nn.Embedding(self.n_users, self.embedding_size)
        self.items_int = nn.Embedding(self.n_items, self.embedding_size)
        self.users_pop = nn.Embedding(self.n_users, self.embedding_size)
        self.items_pop = nn.Embedding(self.n_items, self.embedding_size)

        if self.dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif self.dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif self.dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_emb_total(self, user):

        user_emb = torch.cat((self.users_int.weight, self.users_pop.weight), 1)
        return user_emb[user]

    def get_item_emb_total(self, item):

        item_emb = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        return item_emb[item]

    def dcor(self, x, y):

        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item, factor):
        user_emb = None
        item_emb = None
        if factor == 'int':
            user_emb = self.users_int(user)
            item_emb = self.items_int(item)
        elif factor == 'pop':
            user_emb = self.users_pop(user)
            item_emb = self.items_pop(item)
        elif factor == 'tot':
            user_emb = self.get_user_emb_total(user)
            item_emb = self.get_item_emb_total(item)
        return torch.mul(user_emb, item_emb).sum(dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_p = interaction[self.ITEM_ID]
        item_n = interaction[self.NEG_ITEM_ID]
        mask = interaction[self.mask_field]

        score_p_int = self.forward(user, item_p, 'int')
        score_n_int = self.forward(user, item_n, 'int')
        score_p_pop = self.forward(user, item_p, 'pop')
        score_n_pop = self.forward(user, item_n, 'pop')

        score_p_total = score_p_int + score_p_pop
        score_n_total = score_n_int + score_n_pop

        loss_int = self.mask_bpr_loss(score_p_int, score_n_int, mask)
        loss_pop = self.mask_bpr_loss(score_n_pop, score_p_pop, mask) + self.mask_bpr_loss(score_p_pop, score_n_pop,
                                                                                           ~mask)
        loss_total = self.bpr_loss(score_p_total, score_n_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_emb_int = self.items_int(item_all)
        item_emb_pop = self.items_pop(item_all)
        user_all = torch.unique(user)
        user_emb_int = self.users_int(user_all)
        user_emb_pop = self.users_pop(user_all)
        dis_loss = self.criterion_discrepancy(user_emb_int, user_emb_pop) + self.criterion_discrepancy(item_emb_int,
                                                                                                       item_emb_pop)

        loss = loss_total + self.int_weight * loss_int + self.pop_weight * loss_pop - self.dis_pen * dis_loss
        return loss

    def adapt(self, decay):

        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item, 'tot')
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_emb_total(user)
        all_item_e = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
