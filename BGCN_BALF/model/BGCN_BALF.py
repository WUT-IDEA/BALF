#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import numpy as np
from .model_base import Info, Model
from .BGCN import BGCN_Info, BGCN


class BGCN_BALF_Info(BGCN_Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act)
        self.BL_BALF_lambda = 1.0e-10  # Meal: 1.0e-8   iFashion: 1e-10   NetEase: 1.0e-8  Youshu: 5.0e-8
        self.IL_BALF_lambda = 5.0e-10  # Meal: 5.0e-9   iFashion: 1e-10   NetEase: 1.0e-8  Youshu: 4.0e-8


class BGCN_BALF(BGCN):
    def get_infotype(self):
        return BGCN_BALF_Info

    def __init__(self, info, dataset, raw_graph, device, pretrain=None):
        super().__init__(info, dataset, raw_graph, device, pretrain)
        self.BL_BALF_lambda = info.BL_BALF_lambda
        self.IL_BALF_lambda = info.IL_BALF_lambda

    def propagate(self):
        #  =============================  item level propagation  =============================
        atom_users_feature, atom_items_feature = self.one_propagate(
            self.atom_graph, self.users_feature, self.items_feature, self.dnns_atom)
        atom_bundles_feature = F.normalize(torch.matmul(self.pooling_graph, atom_items_feature))

        #  ============================= bundle level propagation =============================
        non_atom_users_feature, non_atom_bundles_feature = self.one_propagate(
            self.non_atom_graph, self.users_feature, self.bundles_feature, self.dnns_non_atom)

        users_feature = [atom_users_feature, non_atom_users_feature]
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]

        return users_feature, bundles_feature, atom_items_feature


    # item level
    def IL_reg_regularizer(self, U, V, B, bundles):
        """
        U: [batch_size, 1+copy, emb_size]
        V(items_feature): [item_num, emb_size]
        B(bi_graph): [bundle_num, item_num]
        bundles: [batch_size, 1+neg_num]
        """
        # reshape user embedding 成二维
        U = U.reshape(-1, U.shape[-1])  # [batch_size * (1+copy), emb_size]
        batch_size = U.shape[0]

        e = torch.ones(batch_size, 1).to(U.device)  # [batch_size, 1]

        # 分母 ||B V U^T e||^2
        U_transpose_e = torch.mm(U.T, e)  # [64, 2048] @ [2048, 1] = [64, 1]
        VU_e = torch.mm(V, U_transpose_e)  # [item_num, emb_size] @ [emb_size, 1] = [item_num, 1]
        B_VU_e = torch.sparse.mm(B, VU_e)  # [bundle_num, item_num] @ [item_num, 1] = [bundle_num, 1]

        bundle_ids = bundles.view(-1)               # [batch_size * (1+neg_num)]
        B_VU_e_selected = B_VU_e[bundle_ids]        # [batch_size * (1+neg_num), 1]
        denominator = torch.norm(B_VU_e_selected) ** 2  # 标量
        
        # 分子 ||U V^T B^T B V U^T e||^2
        BTB = torch.sparse.mm(B.transpose(0, 1), B)  # [item_num, bundle_num] @ [bundle_num, item_num] = [item_num, item_num]
        U_V = torch.mm(U, V.T)  # [batch_size, emb_size] @ [emb_size, item_num] = [batch_size, item_num]
        BTBVU_e = torch.sparse.mm(BTB, VU_e)  # [item_num, item_num] @ [item_num, 1] = [item_num, 1]
        U_V_BTBVU_e = torch.mm(U_V, BTBVU_e)  # [batch_size, item_num] @ [item_num, 1] = [batch_size, 1]
        numerator = torch.norm(U_V_BTBVU_e) ** 2  # 标量
        reg_regularizer = numerator / (denominator + 1e-8)

        return self.IL_BALF_lambda * reg_regularizer
    


    def calculate_reg_regularizer(self, U, V):
        """
        U: user embedding [batch_size, 1, emb_size] or [batch_size, neg_num, emb_size]
        V: bundle embedding [batch_size, 1, emb_size] or [batch_size, neg_num, emb_size]
        return: 标量
        """
        batch_size = U.shape[0]
        U = U.reshape(-1, U.shape[-1]) # [batch_size * 1 or neg_num, emb_size]
        V = V.reshape(-1, V.shape[-1]) # [batch_size * 1 or neg_num, emb_size]

        e = torch.ones(batch_size, 1).to(U.device)  # [batch_size, 1]
        
        # 计算分母 ||VU^T e||^2
        U_transpose_e = torch.mm(U.T, e)  # [64, 2048] @ [2048, 1] = [64, 1]
        VU_e = torch.mm(V, U_transpose_e)  # [batch_size, emb_size] @ [emb_size, 1] = [batch_size, 1]
        denominator = torch.norm(VU_e) ** 2  # 标量
        
        # 计算分子 ||U V^T V U^T e||^2
        VTV = torch.mm(V.T, V)  # [64, 2048] @ [2048, 64] = [64, 64]
        VTV_U_e = torch.mm(VTV, U_transpose_e)  # [64, 64] @ [64, 1] = [64, 1]
        UVTVU_e = torch.mm(U, VTV_U_e)  # [2048, 64] @ [64, 1] = [2048, 1]
        numerator = torch.norm(UVTVU_e) ** 2  # 标量

        # 计算正则化项
        reg_regularizer = numerator / (denominator + 1e-8)

        return reg_regularizer

    # Bundle level
    def BL_reg_regularizer(self, users_feature, bundles_feature):
        """
        users_feature: [batch_size, 1+copy, emb_size]
        bundles_feature: [batch_size, 1+neg_num, emb_size]
        """
        IL_users_feature, BL_users_feature = users_feature
        IL_bundles_feature, BL_bundles_feature = bundles_feature

        # item-level 
        IL_U_pos = IL_users_feature[:, 0:1, :] # [batch_size, 1, emb_size]
        IL_U_neg = IL_users_feature[:, 1:, :] # [batch_size, neg_num, emb_size]
        IL_B_pos = IL_bundles_feature[:, 0:1, :] # [batch_size, 1, emb_size]
        IL_B_neg = IL_bundles_feature[:, 1:, :] # [batch_size, neg_num, emb_size]

        # bundle-level
        BL_U_pos = BL_users_feature[:, 0:1, :] # [batch_size, 1, emb_size]
        BL_U_neg = BL_users_feature[:, 1:, :] # [batch_size, neg_num, emb_size]
        BL_B_pos = BL_bundles_feature[:, 0:1, :] # [batch_size, 1, emb_size]
        BL_B_neg = BL_bundles_feature[:, 1:, :] # [batch_size, neg_num, emb_size]

        # item level
        IL_reg_pos = self.calculate_reg_regularizer(IL_U_pos, IL_B_pos)
        IL_reg_neg = self.calculate_reg_regularizer(IL_U_neg, IL_B_neg)
        IL_reg = (IL_reg_pos + IL_reg_neg) / 2

        # bundle level
        BL_reg_pos = self.calculate_reg_regularizer(BL_U_pos, BL_B_pos)
        BL_reg_neg = self.calculate_reg_regularizer(BL_U_neg, BL_B_neg)
        BL_reg = (BL_reg_pos + BL_reg_neg) / 2
        
        return self.BL_BALF_lambda * BL_reg + self.IL_BALF_lambda * IL_reg

    
    def forward(self, users, bundles):
        users_feature, bundles_feature, item_features = self.propagate()
        users_embedding = [i[users].expand(- 1, bundles.shape[1], -1) for i in users_feature]  # u_f --> batch_f --> batch_n_f
        bundles_embedding = [i[bundles] for i in bundles_feature] # b_f --> batch_n_f
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_embedding, bundles_embedding)
        BL_reg_loss = self.BL_reg_regularizer(users_embedding, bundles_embedding)
        
        # B = self.pooling_graph
        # IL_reg_loss = self.IL_reg_regularizer(users_embedding[0], item_features, B, bundles)
        IL_reg_loss = torch.tensor(0.0)
        reg_loss = BL_reg_loss + IL_reg_loss

        return pred, loss, reg_loss