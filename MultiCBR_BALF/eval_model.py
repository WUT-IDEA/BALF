# eval.py
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import powerlaw
import yaml
from itertools import product
from scipy.linalg import svd
from utility import Datasets
from scipy.special import zeta
import scipy.sparse as sp
import gc
import torch.nn as nn
import psutil
import argparse


def eval(model, device, dataloader, Test=False):
    model.eval()

    popular_bundles, unpopular_bundles = group_bundles_by_popularity(model.ub_graph, model.ui_graph, model.bi_graph)  # train

    rs = model.get_multi_modal_representations(test=True)
    top_20_recs = {}
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)

        if Test:
            top_20_scores, top_20_indices = torch.topk(pred_b, 20, dim=1)
            for user, top_bundles, scores in zip(users, top_20_indices, top_20_scores):
                # 每个用户存bundle的索引和预测分数
                top_20_recs[user.item()] = [
                    (bundle.item(), score.item()) for bundle, score in zip(top_bundles, scores)
                ]

    print("="*100)
    print("Proportion of Popular Bundle")
    proportion_of_popular_bundle(top_20_recs, popular_bundles)

    print("="*100)
    print("Average Recommendation Popularity")
    average_recommendation_popularity(top_20_recs, model.ub_graph)

    print("="*100)
    print("Gini Index")
    gini_index(top_20_recs, model.ub_graph)


def calculate_gini(popularity):
    """计算给定流行度数组的基尼系数"""
    # 移除零值并排序
    nonzero_pop = popularity[popularity > 0]
    sorted_pop = np.sort(nonzero_pop)
    n = len(sorted_pop)

    if n == 0:
        return 0.0  # 避免除零错误
    
    # 计算累积值
    cum_pop = np.cumsum(sorted_pop, dtype=float)
    total_pop = cum_pop[-1]

    # 计算基尼系数 (使用离散公式)
    indices = np.arange(1, n + 1)
    numerator = np.sum((2 * indices - n - 1) * sorted_pop)
    denominator = n * total_pop
    
    return numerator / denominator if denominator > 0 else 0.0


def gini_index(top_20_recs, ub_graph):
    """
    计算基于交互数据和推荐结果的基尼系数
    参数:
    top_20_recs (dict): 模型推荐结果 {user_id: [(bundle_id, score), ...]}
    ub_graph (torch.Tensor): 用户-bundle交互矩阵 [n_users, n_bundles]
    """
    # 1. 计算交互数据的基尼系数 (基于训练集)
    # 计算每个bundle的流行度 (交互用户数)
    bundle_popularity = torch.as_tensor(ub_graph.sum(axis=0)).flatten()
    gini_train = calculate_gini(bundle_popularity)
    
    # 2. 计算推荐结果的基尼系数 (基于模型输出)
    # 统计每个bundle被推荐的总次数
    n_bundles = ub_graph.shape[1]
    rec_counts = np.zeros(n_bundles)
    
    for user_recs in top_20_recs.values():
        for bundle_id, _ in user_recs:
            rec_counts[bundle_id] += 1
    
    gini_rec = calculate_gini(rec_counts)

    print(f"\033[91mGini Index of Training Data: {gini_train:.4f}\033[0m")
    print(f"\033[91mGini Index of Recommendation: {gini_rec:.4f}\033[0m")


def average_recommendation_popularity(top_20_recs, ub_graph):
    bundle_popularity = torch.as_tensor(ub_graph.sum(axis=0)).flatten()
    total_popularity = 0
    total_recommendation = 0
    for user, top_20_bundles in top_20_recs.items():
        for bundle, score in top_20_bundles:
            total_popularity += bundle_popularity[bundle]
            total_recommendation += 1
    average_popularity = total_popularity / total_recommendation
    print(f"\033[91mAverage Recommendation Popularity (ARP): {average_popularity:.4f}\033[0m")


def proportion_of_popular_bundle(top_20_recs, popular_bundles):
    """
    计算推荐列表中popular bundle的比例
    """
    total_pop_count = 0
    total_recs = 0
    
    # 遍历每个用户的推荐列表
    for user_id, recommendations in top_20_recs.items():
        # 计算该用户推荐列表中popular bundle的数量
        pop_count = sum(1 for bundle_id, _ in recommendations if bundle_id in popular_bundles)
        total_pop_count += pop_count
        total_recs += len(recommendations)
    
    # 计算整体占比
    popular_ratio = total_pop_count / total_recs

    print(f"\033[91mProportion of Popular Bundle: {popular_ratio:.4f}\033[0m")


def group_bundles_by_popularity(ub_graph, ui_graph, bi_graph, num_groups=3):
    """
    将bundle按流行度分5组，每组的bundle popularity总和相同，第一组的bundle为popular bundle，其余为unpopular bundle
    """
    # bundle level bundle popularity
    bundle_popularity = torch.as_tensor(ub_graph.sum(axis=0)).flatten()

    # item level bundle popularity
    # item_popularity = torch.as_tensor(ui_graph.sum(axis=0)).flatten()
    # bi_tensor = torch.from_numpy(bi_graph.toarray()).float()

    # # 计算每个 bundle 包含多少 item
    # bundle_item_counts = bi_tensor.sum(dim=1)  # shape: (num_bundles,)

    # # 避免除以 0
    # bundle_item_counts[bundle_item_counts == 0] = 1

    # # 使用加权和除以数量，得到平均值
    # bundle_popularity = (bi_tensor @ item_popularity) / bundle_item_counts
 
    popularity_sorted, sorted_indices = torch.sort(bundle_popularity, descending=True)  # popularity值，bundle的id
    # print(f'Max popularity: {popularity_sorted.max()}')
    # print(f'Min popularity: {popularity_sorted.min()}')

    total_popularity = bundle_popularity.sum().item()
    # print(f"total_popularity: {total_popularity}")
    target_group_popularity = total_popularity / num_groups
    # print(f"target_group_popularity: {target_group_popularity}")
    groups = []  # 每组bundle的index
    current_group = []  # 当前组bundle的index
    current_sum = 0.0  # 当前组bundle popularity总和
    group_id = 0  # 当前组id

    # 遍历sorted后的bundle
    for i, pop in enumerate(popularity_sorted):
        current_sum += pop.item()
        current_group.append(sorted_indices[i].item())
        if current_sum >= target_group_popularity and group_id < num_groups - 1:
            groups.append(current_group)
            current_group = []
            current_sum = 0.0
            group_id += 1
    
    if current_group:
        groups.append(current_group)

    # 定义popular bundle和unpopular bundle
    popular_bundles = groups[0]  # bundle的id
    unpopular_bundles = []
    for grp in groups[1:]:
        unpopular_bundles.extend(grp)

    # print(f"len(popular_bundles): {len(popular_bundles)}")
    # print(f"len(unpopular_bundles): {len(unpopular_bundles)}")
    # print(f"popular_bundles: {popular_bundles[:10]}")

    return popular_bundles, unpopular_bundles