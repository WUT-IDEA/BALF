#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .HGT import HGT
import scipy.sparse as sp
import numpy as np
eps = 1e-9 

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


class BundleGT_BALF(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.BL_BALF_lambda = conf["BL_BALF_lambda"]
        self.IL_BALF_lambda = conf["IL_BALF_lambda"]

        # Recommendation Basic <<<
        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.eval_bundles = torch.arange(self.num_bundles).to(
            self.device).long().detach().view(1, -1)
        self.eval_users = torch.arange(self.num_users).to(
            self.device).long().detach().view(1, -1)
        # Recommendation Basic <<<

        self.num_ui_layers = conf["num_ui_layers"]  # [1,2,3]
        self.num_trans_layers = conf["num_trans_layers"]  # [1,2,3]

        # GCN Configuration >>>
        self.gcn_norm = self.conf["gcn_norm"]
        self.layer_alpha = self.conf['layer_alpha']  # None
        # GCN Configuration <<<

        # ML Basic >>>
        self.embed_L2_norm = conf["l2_reg"]
        # ML Basic <<<

        # Attention part >>>
        # 0 means keep the default
        self.num_token = conf["num_token"] if "num_token" in conf else 0

        self.dropout_ratio = conf["dropout_ratio"]
        # <<< Attention part

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        bi_graph = self.bi_graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

        self.user_bundle_cf_count = self.ui_graph.sum(axis=1)

        self.HGT = HGT(conf={
            "n_user": self.num_users,
            "n_item": self.num_items,
            "n_bundle": self.num_bundles,
            "dim": self.embedding_size,
            "n_ui_layer": self.num_ui_layers,
            "n_trans_layer": self.num_trans_layers,
            "gcn_norm": self.gcn_norm,
            "layer_alpha": self.layer_alpha,
            "num_token": self.num_token,
            "device": self.device,
            "data_path": self.conf["data_path"],
            "dataset": self.conf["dataset"],
            "head_token": False,
            "dropout_ratio": self.dropout_ratio,
            "ub_alpha": self.conf["ub_alpha"],
            "bi_alpha": self.conf["bi_alpha"],
        },
            data={
                "graph_ui": self.ui_graph,
                "graph_ub": self.ub_graph,
                "graph_bi": self.bi_graph,
        }
        )

        self.MLConf = {}
        for k in ['lr', 'l2_reg', 'embedding_size', 'batch_size_train', 'batch_size_test', 'early_stopping', 'BL_BALF_lambda', 'IL_BALF_lambda']:
            self.MLConf[k] = self.conf[k]
        print("[ML Configuration]", self.MLConf)
        print("[HGT Configuration]", self.HGT.conf)

    def propagate(self):
        return self.HGT()
    
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
    
    
    def BL_reg_regularizer(self, users_feature, bundles_feature):
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
    

    def forward(self, batch):
        losses = {}
        users, bundles = batch
        
        users_feature, items_feature, bundles_features = self.propagate()
        self.batch_size = users.shape[0]

        i_u = users_feature[users].expand(-1, bundles.shape[1], -1)
        i_b = bundles_features[bundles]

        score = torch.sum(torch.mul(i_u, i_b), dim=-1)

        loss = torch.mean(torch.nn.functional.softplus(
            score[:, 1] - score[:, 0]))

        l2_loss = self.embed_L2_norm * self.HGT.reg_loss()
        
        B = self.bundle_agg_graph

        BL_reg = self.BL_reg_regularizer(i_u, i_b)
        IL_reg = self.IL_reg_regularizer(i_u, items_feature, B, bundles)
        reg = BL_reg + IL_reg
        
        loss = loss + l2_loss + reg
        
        losses["l2"] = l2_loss.detach()
        losses["reg"] = reg
        losses["loss"] = loss

        return losses


    def evaluate(self, propagate_result, users):

        users_feature, _, bundles_feature = propagate_result
        users_embedding = users_feature[users]
        scores = torch.mm(users_embedding, bundles_feature.t())

        return scores
