import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


# 读取数据文件
def load_data(file_path):
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            user_id, bundle_id = line.strip().split()
            interactions.append((user_id, bundle_id))
    return interactions


# 计算每个 bundle 的 popularity（交互次数）
def calculate_popularity(data):
    return Counter(bundle_id for _, bundle_id in data)


# 可视化函数，增加了坐标范围参数
def visualization(train_data, test_data, ax, type, ref_ax=None):
    train_popularity = calculate_popularity(train_data)
    test_popularity = calculate_popularity(test_data)

    # 找出所有出现的 bundle_id
    all_bundle_ids = set(train_popularity.keys()).union(test_popularity.keys())

    # 获取训练集和测试集每个 bundle 的 popularity
    train_values = [train_popularity.get(bundle_id, 0) for bundle_id in all_bundle_ids]
    test_values = [test_popularity.get(bundle_id, 0) for bundle_id in all_bundle_ids]

    # 转换为对数坐标, 只考虑值 >= 1 的点（log10(1) = 0），过滤掉小于1的点
    train_values_log = np.log10([v if v >= 1 else np.nan for v in train_values])
    test_values_log = np.log10([v if v >= 1 else np.nan for v in test_values])

    # 设置过滤的阈值，过滤掉低于 log(10) 的点
    threshold = np.log10(1)

    # 过滤掉低于阈值的点
    filtered_data = [(train, test) for train, test in zip(train_values_log, test_values_log) 
                     if not np.isnan(train) and not np.isnan(test) and train >= threshold and test >= threshold]

    # 分开过滤后的训练集和测试集
    if filtered_data:
        filtered_train_values_log, filtered_test_values_log = zip(*filtered_data)

        # 创建散点图
        ax.scatter(filtered_train_values_log, filtered_test_values_log, alpha=0.5)

        # 设置坐标轴标签和标题
        ax.set_xlabel(f"Log of Training Set Bundle Popularity ({type})")
        ax.set_ylabel(f"Log of Test Set Bundle Popularity ({type})")
        ax.set_title(f"Bundle Popularity in Training vs Test Set ({type})")

        # 设置对数坐标轴
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 添加网格线
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        # 如果提供了参考坐标轴（例如REG的坐标轴），同步坐标轴范围
        if ref_ax:
            ax.set_xlim(ref_ax.get_xlim())
            ax.set_ylim(ref_ax.get_ylim())
    else:
        print(f"No data to plot for {type}!")


# 主函数
def main():
    # 读取IID数据
    iid_train_data = load_data('../datasets/Youshu_IID/user_bundle_train.txt')
    iid_test_data = load_data('../datasets/Youshu_IID/user_bundle_test.txt')
    iid_tune_data = load_data('../datasets/Youshu_IID/user_bundle_tune.txt')

    # 创建图形，设置为1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 可视化IID数据
    visualization(iid_train_data, iid_test_data, ax1, 'Youshu_IID')

    # 读取OOD数据
    ood_train_data = load_data('../datasets/Youshu/user_bundle_train.txt')
    ood_tune_data = load_data('../datasets/Youshu/user_bundle_tune.txt')
    ood_test_data = load_data('../datasets/Youshu/user_bundle_test.txt')

    # 可视化OOD数据并同步坐标轴
    visualization(ood_train_data, ood_test_data, ax2, 'Youshu_OOD', ref_ax=ax1)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片
    plt.savefig('Youshu_bundle_popularity_scatter_IID_vs_OOD_Train_Test.png')

    print('Picture saved as Youshu_bundle_popularity_scatter_IID_vs_OOD_Train_Test.png')


if __name__ == "__main__":
    main()
