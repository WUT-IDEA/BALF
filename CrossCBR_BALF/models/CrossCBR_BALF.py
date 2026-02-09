#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

from models.CrossCBR import CrossCBR


class CrossCBR_BALF(CrossCBR):
    def __init__(self, conf, raw_graph):
        super().__init__(conf, raw_graph)
        self.BL_BALF_lambda = conf["BL_BALF_lambda"]
        self.IL_BALF_lambda = conf["IL_BALF_lambda"]
    
    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature, IL_items_feature


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
        # B_VU_e = torch.mm(B, VU_e)  # [bundle_num, item_num] @ [item_num, 1] = [bundle_num, 1]
        B_VU_e = torch.sparse.mm(B, VU_e)  # 替换 torch.mm

        bundle_ids = bundles.view(-1)               # [batch_size * (1+neg_num)]
        B_VU_e_selected = B_VU_e[bundle_ids]        # [batch_size * (1+neg_num), 1]
        denominator = torch.norm(B_VU_e_selected) ** 2  # 标量
        
        # 分子 ||U V^T B^T B V U^T e||^2
        BTB = torch.sparse.mm(B.transpose(0, 1), B)  # 仍然稀疏
        # BTB = torch.mm(B.transpose(0, 1), B)  # [item_num, bundle_num] @ [bundle_num, item_num] = [item_num, item_num]
        U_V = torch.mm(U, V.T)  # [batch_size, emb_size] @ [emb_size, item_num] = [batch_size, item_num]
        BTBVU_e = torch.sparse.mm(BTB, VU_e)
        # BTBVU_e = torch.mm(BTB, VU_e)  # [item_num, item_num] @ [item_num, 1] = [item_num, 1]
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


    # bundle level
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
        
        return self.BL_BALF_lambda * (IL_reg + BL_reg)
        

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_feature, bundles_feature, items_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        BL_reg = self.BL_reg_regularizer(users_embedding, bundles_embedding)
        # BL_reg = 0

        B = self.bundle_agg_graph
        
        IL_reg = self.IL_reg_regularizer(users_embedding[0], items_feature, B, bundles)
        # IL_reg = 0

        return bpr_loss, c_loss, BL_reg + IL_reg

