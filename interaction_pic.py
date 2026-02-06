import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

# --- 配色方案 ---
SCI_COLOR_ORIG = '#666666'   # 深灰色
SCI_COLOR_CURVE = '#004488'  # 深蓝色
SCI_COLOR_FINAL = '#D95F02'  # 赭橙色
GRID_COLOR = '#EAEAEA'       # 极淡灰色

# --- 数据处理 ---

def read_region_file(filepath):
    points = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            try:
                x = float(parts[0][1:])
                y = float(parts[1][1:])
                z = float(parts[2][1:])
                points.append((x, y, z))
            except ValueError:
                continue
    return np.array(points)

def arclength_parameterize(points):
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(seg_len)])
    return s

def resample_linear(points, num=200):
    if len(points) < 2: return points
    s = arclength_parameterize(points)
    if s[-1] == 0: return points.copy()
    s_new = np.linspace(0, s[-1], num)
    x = np.interp(s_new, s, points[:,0])
    y = np.interp(s_new, s, points[:,1])
    z = np.interp(s_new, s, points[:,2])
    return np.stack([x, y, z], axis=1)

# --- S-G 平滑 ---
def smooth_savgol(points, window_length=15, polyorder=3):
    n = len(points)
    wl = window_length
    if wl >= n: wl = n if n % 2 == 1 else n - 1
    if wl < polyorder + 2: return points.copy()
    x = savgol_filter(points[:,0], wl, polyorder)
    y = savgol_filter(points[:,1], wl, polyorder)
    z = savgol_filter(points[:,2], wl, polyorder)
    return np.stack([x, y, z], axis=1)

# --- B-Spline 平滑曲线生成 ---
def generate_bspline_dense(points, num_dense=1000):
    t_points = points.T
    try:
        tck, u = splprep(t_points, u=None, s=0.0, k=3) 
        u_new = np.linspace(u.min(), u.max(), num_dense)
        x_new, y_new, z_new = splev(u_new, tck, der=0)
        return np.stack([x_new, y_new, z_new], axis=1)
    except:
        return resample_linear(points, num_dense)

# --- 绘图函数 ---

def plot_trajectory_synced_large_ticks(p_orig, p_curve_dense, p_final_sparse):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, constrained_layout=True)
    
    orig_y, orig_z = p_orig[:, 1], p_orig[:, 2]
    curve_y, curve_z = p_curve_dense[:, 1], p_curve_dense[:, 2]
    final_y, final_z = p_final_sparse[:, 1], p_final_sparse[:, 2]

    def style_ax(ax):
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, which='major', color=GRID_COLOR, linestyle='--', linewidth=1.0)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6, direction='out', labelleft=True, left=True, bottom=True)

    # 1. 原轨迹
    ax1 = axes[0]
    ax1.plot(orig_y, orig_z, color=SCI_COLOR_ORIG, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.scatter(orig_y, orig_z, color=SCI_COLOR_ORIG, s=15, alpha=0.5)
    style_ax(ax1)

    # 2. 平滑曲线
    ax2 = axes[1]
    ax2.plot(curve_y, curve_z, color=SCI_COLOR_CURVE, linestyle="-", linewidth=2.0, alpha=0.9)
    style_ax(ax2)

    # 3. 最终结果
    ax3 = axes[2]
    ax3.plot(final_y, final_z, color=SCI_COLOR_FINAL, linestyle="-", linewidth=1.0, alpha=0.7)
    ax3.scatter(final_y, final_z, color=SCI_COLOR_FINAL, s=30, alpha=1.0)
    style_ax(ax3)

    plt.show()

def visualize_folder(folder):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found.")
        return

    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    if not files:
        print("No .txt files found.")
        return

    # 参数
    N_ORIG = 200    
    N_CURVE = 1000  # 这里定义了密集的程度
    SG_WINDOW = 15  
    SG_POLY = 3

    for filename in files:
        filepath = os.path.join(folder, filename)
        pts = read_region_file(filepath)
        if len(pts) == 0: continue

        print(f"\n{'='*20} Processing {filename} {'='*20}")

        pts_orig = resample_linear(pts, num=N_ORIG)
        pts_final = smooth_savgol(pts_orig, window_length=SG_WINDOW, polyorder=SG_POLY)
        
        # 生成密集曲线 (这就是中间那条非常平滑的线)
        pts_curve_dense = generate_bspline_dense(pts_final, num_dense=N_CURVE)
        
        # ==========================================================
        # 计算密集点之间的微小变化量 (Dense Deltas)
        # ==========================================================
        
        # 计算差分：Delta = Point[i+1] - Point[i]
        # 这里的 pts_curve_dense 包含了您指定的从 Start 到 End 之间的所有点
        deltas = np.diff(pts_curve_dense, axis=0) 
        
        print(f"\n>>> [DENSE Trajectory Increments] (B-Spline Interpolation, N={len(deltas)})")
        print(f"Format: dX... dY... dZ...")
        
        for d in deltas:
            # 格式：dX0.000000 dY0.000000 dZ0.000000
            # 数值会非常小，因为点非常密集
            print(f"dX{d[0]:.6f} dY{d[1]:.6f} dZ{d[2]:.6f}")
            
        print("=" * 60)
        
        plot_trajectory_synced_large_ticks(pts_orig, pts_curve_dense, pts_final)

if __name__ == "__main__":
    visualize_folder("top_curvature_regions")