import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) 数据（与原图近似）
# -----------------------
x = np.arange(10, 101, 10)
data = {
    "Proposed": {"y": [10, 55, 60, 82, 90, 91, 90, 90, 91, 92],
                 "e": [3, 15, 20, 4, 8, 5, 8, 6, 6, 4]},
    "SAC+BC": {"y": [10, 50, 60, 52, 62, 65, 68, 72, 75, 80],
               "e": [2, 25, 20, 25, 20, 20, 25, 10, 5, 10]},
    "TD3+BC": {"y": [5, 50, 58, 60, 70, 68, 72, 70, 72, 74],
               "e": [5, 10, 15, 25, 15, 20, 10, 15, 10, 10]},
    "DSAC": {"y": [10, 35, 42, 50, 53, 59, 62, 66, 72, 78],
             "e": [3, 10, 10, 15, 15, 12, 18, 15, 15, 10]},
    "REDQ": {"y": [10, 25, 35, 40, 50, 51, 56, 60, 65, 70],
             "e": [2, 20, 15, 20, 18, 15, 16, 20, 15, 20]},
    "DDPG": {"y": [10, 27, 30, 41, 50, 53, 55, 60, 58, 59],
             "e": [2, 25, 20, 25, 15, 15, 15, 20, 15, 10]},
}

# -----------------------
# 2) 全局风格（论文风）
# -----------------------
plt.rcParams.update({
    "font.family": "Times New Roman",   # 如需其他字体可改
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# -----------------------
# 3) 作图
# -----------------------
fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
axes = axes.ravel()

line_color = "#2a6fbb"
err_color = "#b0c4de"

for ax, (title, d) in zip(axes, data.items()):
    y = d["y"]
    e = d["e"]

    ax.errorbar(
        x, y, yerr=e, fmt='-s',
        color=line_color, markerfacecolor="white",
        markeredgecolor=line_color, markersize=5,
        ecolor=err_color, elinewidth=1.8, capsize=4
    )

    ax.set_title(title, pad=6)
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 120)

    ax.set_xlabel("Episode (×1000)")
    ax.set_ylabel("Episode average reward")

    ax.axhline(100, color="red", linestyle="--", linewidth=1.0)
    ax.text(2, 104, "Optimal", color="red", fontsize=11)

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{t}%" for t in [0, 20, 40, 60, 80, 100]])

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

fig.tight_layout()
plt.show()