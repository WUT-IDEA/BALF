import os
from typing_extensions import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 全局绘图风格（兼容不同 matplotlib 版本）
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.edgecolor": "#bcbcbc",
        "legend.facecolor": "white",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.color": "#d6dde5",
        "axes.linewidth": 0.9,
        "legend.title_fontsize": 12,
    }
)

# 平滑参数
SMOOTHING_WINDOW = 5  # 设置为 1 可关闭平滑
SMOOTHING_METHOD: Literal["rolling", "ewm", "none"] = "rolling"


def smooth_series(series: pd.Series, window: int, method: str) -> pd.Series:
    """平滑序列，使曲线更平滑易读。"""
    if window <= 1 or method == "none":
        return series

    window = min(window, len(series))

    if method == "rolling":
        return series.rolling(window, min_periods=1, center=True).mean()
    if method == "ewm":
        alpha = 2 / (window + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()

    return series

# 四个模型的 CSV 文件路径
# Meal
dataset_name = 'MealRec+'
csv_files = {
    "CrossCBR": "../results_OOD/Meal/CrossCBR/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
    "CrossCBR+BALF (w/o IL)": "../results_OOD/Meal/CrossCBR_BALF_BL/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
    "CrossCBR+BALF (w/o BL)": "../results_OOD/Meal/CrossCBR_BALF_IL/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
    "CrossCBR+BALF": "../results_OOD/Meal/CrossCBR_BALF/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv"
}

# Youshu
# dataset_name = 'Youshu'
# csv_files = {
#     "CrossCBR": "../results_OOD/Youshu/CrossCBR/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
#     "CrossCBR+BALF (w/o IL)": "../results_OOD/Youshu/CrossCBR_BALF_BL/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
#     "CrossCBR+BALF (w/o BL)": "../results_OOD/Youshu/CrossCBR_BALF_IL/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv",
#     "CrossCBR+BALF": "../results_OOD/Youshu/CrossCBR_BALF/0.2_0.2_0.2/epoch_wise_fairness_metrics.csv"
# }

# 要绘制的三个指标
metrics = ["entropy", "hhi", "AF"]
metric_map = {
    'entropy': 'Entropy',
    'hhi': 'HHI',
    'AF': 'AF'
}

# 输出目录
output_dir = f"fairness_plots/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

# 颜色循环，保证不同模型颜色区分度高
color_cycle = ["#66a61e", "#d95f02", "#7570b3", "#1b9e77"]
line_styles = [":", "--", "-.", "-"]
marker_styles = ["^", "s", "D", "o"]

# 全局字体大小
legend_fontsize = 14
label_fontsize = 14
title_fontsize = 15
tick_fontsize = 13

# 逐个指标绘图
for metric in metrics:
    plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    for idx, ((model_name, csv_path), color) in enumerate(zip(csv_files.items(), color_cycle)):
        df = pd.read_csv(csv_path)
        if "epoch" not in df.columns:
            raise ValueError(f"{csv_path} does not contain 'epoch' column")
        if metric not in df.columns:
            raise ValueError(f"{csv_path} does not contain metric '{metric}'")

        smoothed_metric = smooth_series(df[metric], SMOOTHING_WINDOW, SMOOTHING_METHOD)
        style = line_styles[idx % len(line_styles)]
        marker = marker_styles[idx % len(marker_styles)]
        markevery = max(1, len(df) // 18)

        ax.plot(
            df["epoch"],
            smoothed_metric,
            label=model_name,
            linewidth=2.5,
            color=color,
            linestyle=style,
            marker=marker,
            markevery=markevery,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )

    metric_name = metric_map.get(metric, metric)  # 找不到就用原名

    plt.xlabel("Epoch", fontweight="bold", fontsize=label_fontsize)
    plt.ylabel(metric_name, fontweight="bold", fontsize=label_fontsize)
    plt.title(f"{metric_name} on {dataset_name}", fontweight="bold", fontsize=title_fontsize)

    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="0.9",
        prop={"weight": "bold", "size": legend_fontsize},
        labelcolor="#111111",
        facecolor="white"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    ax.set_facecolor("#f8f9fb")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"All figures saved into: {output_dir}")
