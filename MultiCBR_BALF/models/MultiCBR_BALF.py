#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch.sparse
import torch.sparse 

from models.MultiCBR import MultiCBR


class MultiCBR_BALF(MultiCBR):
    def __init__(self, conf, raw_graph):
        super().__init__(conf, raw_graph)
        self.BL_BALF_lambda = conf["BL_BALF_lambda"]
        self.IL_BALF_lambda = conf["IL_BALF_lambda"]


    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep = torch.stack(users_feature, dim=0)
        bundles_rep = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_rep * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_rep * self.modal_coefs, dim=0)

        return users_rep, bundles_rep, users_feature, bundles_feature, UI_items_feature, BI_items_feature


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


    def reg_regularizer(self, users_feature, bundles_feature):
        """
        users_feature: [batch_size, 1+copy, emb_size]
        bundles_feature: [batch_size, 1+neg_num, emb_size]
        """

        U_pos = users_feature[:, 0:1, :]
        U_neg = users_feature[:, 1:, :]
        B_pos = bundles_feature[:, 0:1, :]
        B_neg = bundles_feature[:, 1:, :]

        reg_pos = self.calculate_reg_regularizer(U_pos, B_pos)
        reg_neg = self.calculate_reg_regularizer(U_neg, B_neg)

        reg = (reg_pos + reg_neg) / 2

        return self.BL_BALF_lambda * reg


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_rep, bundles_rep, users_feature, bundles_feature, UI_items_feature, BI_items_feature = self.get_multi_modal_representations()

        # loss
        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)


        # bundle level reg
        BL_users_embedding = users_feature[0][users].expand(-1, bundles.shape[1], -1)
        BL_bundles_embedding = bundles_feature[0][bundles]
        BL_reg = self.reg_regularizer(users_embedding, bundles_embedding)
        #BL_reg = 0

        # item level reg
        B = self.BI_aggregation_graph
        UI_users_embedding = users_feature[1][users].expand(-1, bundles.shape[1], -1)
        UI_bundles_embedding = bundles_feature[1][bundles]
        BI_users_embedding = users_feature[2][users].expand(-1, bundles.shape[1], -1)
        BI_bundles_embedding = bundles_feature[2][bundles]
        # UI_reg = self.IL_reg_regularizer(UI_users_embedding, UI_items_feature, B, bundles)
        # BI_reg = self.IL_reg_regularizer(BI_users_embedding, BI_items_feature, B, bundles)
        # IL_reg = UI_reg + BI_reg
        IL_reg = self.reg_regularizer(UI_users_embedding, UI_bundles_embedding) + self.reg_regularizer(BI_users_embedding, BI_bundles_embedding)
        

        return bpr_loss, c_loss, BL_reg + IL_reg
