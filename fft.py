import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.fft import rfft, rfftfreq  # [新增] 导入FFT库

def read_region_file(filepath):
    points = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 假设数据格式为 "x12.34" 或类似格式，根据原代码逻辑切片
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

def resample_by_arclength(points, num=2000):
    s = arclength_parameterize(points)
    if len(s) == 0 or s[-1] == 0:
        return points.copy()
    s_new = np.linspace(0, s[-1], num)
    x = np.interp(s_new, s, points[:,0])
    y = np.interp(s_new, s, points[:,1])
    z = np.interp(s_new, s, points[:,2])
    return np.stack([x, y, z], axis=1)

def smooth_savgol(points, window_length=7, polyorder=3):
    wl = min(window_length, len(points) if len(points)%2==1 else len(points)-1)
    if wl < 3:
        return points.copy()
    x = savgol_filter(points[:,0], wl, polyorder)
    y = savgol_filter(points[:,1], wl, polyorder)
    z = savgol_filter(points[:,2], wl, polyorder)
    return np.stack([x, y, z], axis=1)

def smooth_spline(points, smooth_factor=None):
    n = len(points)
    if n < 4: return points # Spline usually needs enough points
    t = np.linspace(0, 1, n)
    if smooth_factor is None:
        smooth_factor = n * 0.01
    try:
        sx = UnivariateSpline(t, points[:,0], s=smooth_factor)
        sy = UnivariateSpline(t, points[:,1], s=smooth_factor)
        sz = UnivariateSpline(t, points[:,2], s=smooth_factor)
        x_s = sx(t)
        y_s = sy(t)
        z_s = sz(t)
        return np.stack([x_s, y_s, z_s], axis=1)
    except:
        return points

def smooth_gaussian(points, sigma=1.0):
    x = gaussian_filter1d(points[:,0], sigma=sigma)
    y = gaussian_filter1d(points[:,1], sigma=sigma)
    z = gaussian_filter1d(points[:,2], sigma=sigma)
    return np.stack([x, y, z], axis=1)

def smooth_lowess(points, frac=0.05, it=1):
    n = len(points)
    t = np.linspace(0, 1, n)
    # return_sorted=False 保持原顺序
    x = lowess(points[:,0], t, frac=frac, it=it, return_sorted=False)
    y = lowess(points[:,1], t, frac=frac, it=it, return_sorted=False)
    z = lowess(points[:,2], t, frac=frac, it=it, return_sorted=False)
    return np.stack([x, y, z], axis=1)

def compute_derivatives(points, dt=0.01):
    # 使用 gradient 计算一阶和二阶差分
    v = np.gradient(points, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    # 返回标量模长 (Speed, Acc magnitude, Jerk magnitude)
    return (np.linalg.norm(v, axis=1),
            np.linalg.norm(a, axis=1),
            np.linalg.norm(j, axis=1))

def compute_curvatures(points):
    curvatures = np.zeros(len(points))
    for i in range(1, len(points)-1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        a = p2 - p1
        b = p3 - p1
        cross = np.cross(a, b)
        cross_norm = np.linalg.norm(cross)
        la = np.linalg.norm(a)
        lb = np.linalg.norm(p3 - p2)
        lc = np.linalg.norm(b)
        if la == 0 or lb == 0 or lc == 0:
            curvatures[i] = 0.0
        else:
            curvatures[i] = (2 * cross_norm) / (la * lb * lc)
    return curvatures

# [新增] 计算频域数据
def compute_spectrum(signal, dt):
    """
    对输入信号进行FFT变换
    :param signal: 1D 数组 (加速度标量)
    :param dt: 采样时间间隔
    :return: (频率数组, 幅度数组)
    """
    n = len(signal)
    # rfft 针对实数输入，只计算正频率部分
    yf = rfft(signal)
    xf = rfftfreq(n, dt)
    # 归一化幅值
    magnitude = np.abs(yf) / n
    return xf, magnitude

def stats(arr):
    return np.max(arr), np.mean(arr)

def error_stats(ref_points, test_points):
    errors = np.linalg.norm(test_points - ref_points, axis=1)
    return np.max(errors), np.mean(errors)

def plot_trajectory_comparison(p0, p1, p2, p3, p4, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # 为了图表清晰，这里只画线，减少scatter的点的大小
    ax.plot(p0[:,0], p0[:,1], p0[:,2], color="black", label="Original", alpha=0.6)
    ax.plot(p1[:,0], p1[:,1], p1[:,2], color="blue", label="S-G")
    ax.plot(p2[:,0], p2[:,1], p2[:,2], color="red", label="Spline")
    ax.plot(p3[:,0], p3[:,1], p3[:,2], color="green", label="Gaussian")
    ax.plot(p4[:,0], p4[:,1], p4[:,2], color="purple", label="LOWESS")

    ax.set_title(title + " Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def plot_derivative_comparison(v0,a0,j0, v1,a1,j1, v2,a2,j2, v3,a3,j3, v4,a4,j4, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 速度
    axes[0].plot(v0, label="Original", color="black", alpha=0.5)
    axes[0].plot(v1, label="S-G", color="blue")
    axes[0].plot(v2, label="Spline", color="red")
    axes[0].plot(v3, label="Gaussian", color="green")
    axes[0].plot(v4, label="LOWESS", color="purple")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Time Domain Analysis")
    axes[0].legend(loc='upper right')

    # 加速度
    axes[1].plot(a0, color="black", alpha=0.5)
    axes[1].plot(a1, color="blue")
    axes[1].plot(a2, color="red")
    axes[1].plot(a3, color="green")
    axes[1].plot(a4, color="purple")
    axes[1].set_ylabel("Acceleration (m/s²)")

    # 加加速度 (Jerk)
    axes[2].plot(j0, color="black", alpha=0.5)
    axes[2].plot(j1, color="blue")
    axes[2].plot(j2, color="red")
    axes[2].plot(j3, color="green")
    axes[2].plot(j4, color="purple")
    axes[2].set_ylabel("Jerk (m/s³)")
    axes[2].set_xlabel("Time Step Index")

    fig.suptitle(title + " Kinematics")
    plt.tight_layout()
    plt.show()

# [新增] 绘制加速度频域分析图
def plot_acceleration_spectrum(a0, a1, a2, a3, a4, dt, title):
    # 计算所有方法的频谱
    xf0, yf0 = compute_spectrum(a0, dt)
    xf1, yf1 = compute_spectrum(a1, dt)
    xf2, yf2 = compute_spectrum(a2, dt)
    xf3, yf3 = compute_spectrum(a3, dt)
    xf4, yf4 = compute_spectrum(a4, dt)

    plt.figure(figsize=(10, 6))
    
    # 使用半对数坐标 (semilogy) 能更好地展示高频噪声幅值的衰减情况
    # 也可以使用 plt.plot() 看线性坐标
    plt.semilogy(xf0, yf0, label="Original", color="black", alpha=0.5)
    plt.semilogy(xf1, yf1, label="S-G", color="blue")
    plt.semilogy(xf2, yf2, label="Spline", color="red")
    plt.semilogy(xf3, yf3, label="Gaussian", color="green")
    plt.semilogy(xf4, yf4, label="LOWESS", color="purple")

    plt.title(f"{title} - Acceleration Frequency Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def log_stats(filename, method, kmax, kmean, vmax, vmean, amax, amean, jmax, jmean, errmax, time_val):
    print(
        f"{filename} | {method:<10} | "
        f"Kmax={kmax:.4f} "
        f"Amax={amax:.4f} Amean={amean:.4f} "
        f"Jmean={jmean:.4f} "
        f"ErrMax={errmax:.4f} Time={time_val:.2f}ms"
    )

def visualize_folder(folder, dt=0.01, csv_path="stats_results.csv"):
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    if not files:
        print("No .txt files found.")
        return

    rows = []
    
    # 为了演示方便，这里对每个文件进行处理
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)
        pts = read_region_file(filepath)
        
        if len(pts) < 10:
            print(f"Skipping {filename}: too few points.")
            continue

        pts_resampled = resample_by_arclength(pts, num=200)

        # 1. 原始 (Original)
        t0 = time.perf_counter()
        v0,a0,j0 = compute_derivatives(pts_resampled, dt)
        curv0 = compute_curvatures(pts_resampled)
        t_original = (time.perf_counter() - t0) * 1000

        # 2. Savitzky-Golay (S-G)
        t0 = time.perf_counter()
        pts_sg = smooth_savgol(pts_resampled, window_length=11, polyorder=3)
        v1,a1,j1 = compute_derivatives(pts_sg, dt)
        curv1 = compute_curvatures(pts_sg)
        max_err_sg, _ = error_stats(pts_resampled, pts_sg)
        t_sg = (time.perf_counter() - t0) * 1000

        # 3. B-Spline
        t0 = time.perf_counter()
        pts_spline = smooth_spline(pts_resampled, smooth_factor=len(pts_resampled) * 0.0005)
        v2,a2,j2 = compute_derivatives(pts_spline, dt)
        curv2 = compute_curvatures(pts_spline)
        max_err_sp, _ = error_stats(pts_resampled, pts_spline)
        t_spline = (time.perf_counter() - t0) * 1000

        # 4. Gaussian
        t0 = time.perf_counter()
        pts_gauss = smooth_gaussian(pts_resampled, sigma=2.0)
        v3,a3,j3 = compute_derivatives(pts_gauss, dt)
        curv3 = compute_curvatures(pts_gauss)
        max_err_gs, _ = error_stats(pts_resampled, pts_gauss)
        t_gauss = (time.perf_counter() - t0) * 1000

        # 5. LOWESS
        t0 = time.perf_counter()
        pts_lowess = smooth_lowess(pts_resampled, frac=0.06, it=0)
        # LOWESS 输出后需要重新采样以保持对齐
        pts_lowess = resample_by_arclength(pts_lowess, num=len(pts_resampled))
        v4,a4,j4 = compute_derivatives(pts_lowess, dt)
        curv4 = compute_curvatures(pts_lowess)
        max_err_lw, _ = error_stats(pts_resampled, pts_lowess)
        t_lowess = (time.perf_counter() - t0) * 1000

        # 收集统计数据
        datasets = [
            ("Original", curv0, v0, a0, j0, 0.0, t_original),
            ("S-G", curv1, v1, a1, j1, max_err_sg, t_sg),
            ("Spline", curv2, v2, a2, j2, max_err_sp, t_spline),
            ("Gaussian", curv3, v3, a3, j3, max_err_gs, t_gauss),
            ("LOWESS", curv4, v4, a4, j4, max_err_lw, t_lowess),
        ]

        for method, k_arr, v_arr, a_arr, j_arr, err, t_val in datasets:
            max_k, mean_k = stats(k_arr)
            max_v, mean_v = stats(v_arr)
            max_a, mean_a = stats(a_arr)
            max_j, mean_j = stats(j_arr)
            
            rows.append([
                filename, method, max_k, mean_k, max_v, mean_v, 
                max_a, mean_a, max_j, mean_j, err, t_val
            ])
            log_stats(filename, method, max_k, mean_k, max_v, mean_v, max_a, mean_a, max_j, mean_j, err, t_val)

        title = f"{filename} (Region {i})"
        
        # 绘图 1: 3D 轨迹
        plot_trajectory_comparison(pts_resampled, pts_sg, pts_spline, pts_gauss, pts_lowess, title)
        
        # 绘图 2: 时域导数 (v, a, j)
        plot_derivative_comparison(v0,a0,j0, v1,a1,j1, v2,a2,j2, v3,a3,j3, v4,a4,j4, title)
        
        # 绘图 3: [新增] 频域分析 (Acceleration Spectrum)
        plot_acceleration_spectrum(a0, a1, a2, a3, a4, dt, title)

    # 保存CSV
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "method", "Kmax", "Kmean", "Vmax", "Vmean",
                "Amax", "Amean", "Jmax", "Jmean", "ErrMax", "Time_ms"
            ])
            writer.writerows(rows)
        print(f"\nStats saved to {csv_path}")
    except PermissionError:
        print(f"\nError: Could not write to {csv_path}. File might be open.")

if __name__ == "__main__":
    # 请确保当前目录下有 "top_curvature_regions" 文件夹，且里面包含数据文件
    visualize_folder("top_curvature_regions", dt=0.002, csv_path="stats_results_fft.csv")