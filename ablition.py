import numpy as np
import matplotlib.pyplot as plt

# 1. 原始数据 (11列)
raw_scores = np.array([
    [0.1173, 0.3361, 0.5087, 0.6392, 0.7346, 0.8219, 0.8774, 0.9196, 0.9472, 0.9651, 0.9834],
    [0.0187, 0.0749, 0.1835, 0.2976, 0.4152, 0.5418, 0.6294, 0.7071, 0.7829, 0.8306, 0.8623],
    [0.0579, 0.2316, 0.3094, 0.2683, 0.4468, 0.3791, 0.5735, 0.4862, 0.6649, 0.5714, 0.7061],
    [0.0826, 0.2137, 0.3719, 0.4893, 0.5907, 0.6482, 0.7075, 0.7426, 0.7708, 0.7914, 0.8032],
])

# 2. 数据处理：11列 -> 10列
col0_new = (raw_scores[:, 0] + raw_scores[:, 1]) / 2
scores = np.column_stack((col0_new, raw_scores[:, 2:]))

# 横坐标标签：10个点
steps = np.array([1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5])
labels = ["Full", "w/o BC", "w/o PBRS", "w/o PER"]

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(scores, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)

# 设置坐标轴
ax.set_xticks(np.arange(len(steps)))

# --- 修改点：增加 fontsize 参数调整 X 轴刻度字号 ---
ax.set_xticklabels(
    [f"{int(s/1e4)}e4" if s < 1e5 else "1e5" for s in steps], 
    fontsize=14  # 这里调大 X 轴刻度字体
)

ax.set_yticks(np.arange(len(labels)))
# --- 修改点：增加 fontsize 参数调整 Y 轴刻度字号 ---
ax.set_yticklabels(
    labels, 
    fontsize=14  # 这里调大 Y 轴刻度字体
)

# 如果你需要恢复标题，建议也同步调大字体，例如 fontsize=14 或 16
# ax.set_xlabel("Training Steps *1e4", fontsize=14)
# ax.set_ylabel("Ablation Variants", fontsize=14)
# ax.set_title("Ablation Heatmap", fontsize=16)

# 画网格线
ax.set_xticks(np.arange(-.5, len(steps), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

# 3. 标注数值
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        val = scores[i, j]
        text_color = 'white' if val > 0.6 else 'black'
        
        # 保持方块内的数字也足够大
        ax.text(j, i, f"{val*100:.2f}%",
                ha='center', va='center', color=text_color, fontsize=13, fontweight='medium')

plt.tight_layout()
plt.show()