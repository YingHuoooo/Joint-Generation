import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess

def read_region_file(filepath):
    points = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            x = float(parts[0][1:])
            y = float(parts[1][1:])
            z = float(parts[2][1:])
            points.append((x, y, z))
    return np.array(points)

def arclength_parameterize(points):
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(seg_len)])
    return s

def resample_by_arclength(points, num=200):
    s = arclength_parameterize(points)
    if s[-1] == 0:
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
    t = np.linspace(0, 1, n)
    if smooth_factor is None:
        smooth_factor = n * 0.01
    sx = UnivariateSpline(t, points[:,0], s=smooth_factor)
    sy = UnivariateSpline(t, points[:,1], s=smooth_factor)
    sz = UnivariateSpline(t, points[:,2], s=smooth_factor)
    x_s = sx(t)
    y_s = sy(t)
    z_s = sz(t)
    return np.stack([x_s, y_s, z_s], axis=1)

def smooth_gaussian(points, sigma=1.0):
    x = gaussian_filter1d(points[:,0], sigma=sigma)
    y = gaussian_filter1d(points[:,1], sigma=sigma)
    z = gaussian_filter1d(points[:,2], sigma=sigma)
    return np.stack([x, y, z], axis=1)

def smooth_lowess(points, frac=0.05, it=1):
    n = len(points)
    t = np.linspace(0, 1, n)
    x = lowess(points[:,0], t, frac=frac, it=it, return_sorted=False)
    y = lowess(points[:,1], t, frac=frac, it=it, return_sorted=False)
    z = lowess(points[:,2], t, frac=frac, it=it, return_sorted=False)
    return np.stack([x, y, z], axis=1)

def compute_derivatives(points, dt=0.01):
    v = np.gradient(points, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
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

def stats(arr):
    return np.max(arr), np.mean(arr)

def error_stats(ref_points, test_points):
    errors = np.linalg.norm(test_points - ref_points, axis=1)
    return np.max(errors), np.mean(errors)

def plot_trajectory_comparison(p0, p1, p2, p3, p4, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p0[:,0], p0[:,1], p0[:,2], color="black", label="Original")
    ax.scatter(p0[:,0], p0[:,1], p0[:,2], color="black", s=8)

    ax.plot(p1[:,0], p1[:,1], p1[:,2], color="blue", label="S-G")
    ax.scatter(p1[:,0], p1[:,1], p1[:,2], color="blue", s=8)

    ax.plot(p2[:,0], p2[:,1], p2[:,2], color="red", label="Spline")
    ax.scatter(p2[:,0], p2[:,1], p2[:,2], color="red", s=8)

    ax.plot(p3[:,0], p3[:,1], p3[:,2], color="green", label="Gaussian")
    ax.scatter(p3[:,0], p3[:,1], p3[:,2], color="green", s=8)

    ax.plot(p4[:,0], p4[:,1], p4[:,2], color="purple", label="LOWESS")
    ax.scatter(p4[:,0], p4[:,1], p4[:,2], color="purple", s=8)

    ax.set_title(title + " Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def plot_derivative_comparison(v0,a0,j0, v1,a1,j1, v2,a2,j2, v3,a3,j3, v4,a4,j4, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(v0, label="Original", color="black")
    axes[0].plot(v1, label="S-G", color="blue")
    axes[0].plot(v2, label="Spline", color="red")
    axes[0].plot(v3, label="Gaussian", color="green")
    axes[0].plot(v4, label="LOWESS", color="purple")
    axes[0].set_ylabel("Velocity")
    axes[0].legend()

    axes[1].plot(a0, color="black")
    axes[1].plot(a1, color="blue")
    axes[1].plot(a2, color="red")
    axes[1].plot(a3, color="green")
    axes[1].plot(a4, color="purple")
    axes[1].set_ylabel("Acceleration")

    axes[2].plot(j0, color="black")
    axes[2].plot(j1, color="blue")
    axes[2].plot(j2, color="red")
    axes[2].plot(j3, color="green")
    axes[2].plot(j4, color="purple")
    axes[2].set_ylabel("Jerk")
    axes[2].set_xlabel("Index")

    fig.suptitle(title + " v/a/j")
    plt.tight_layout()
    plt.show()

def log_stats(filename, method, kmax, kmean, vmax, vmean, amax, amean, jmax, jmean, errmax, time_val):
    print(
        f"{filename} | {method} | "
        f"Kmax={kmax:.6f} Kmean={kmean:.6f} "
        f"Vmax={vmax:.6f} Vmean={vmean:.6f} "
        f"Amax={amax:.6f} Amean={amean:.6f} "
        f"Jmax={jmax:.6f} Jmean={jmean:.6f} "
        f"ErrMax={errmax:.6f} Time={time_val:.6f}"
    )

def visualize_folder(folder, dt=0.01, csv_path="stats_results.csv"):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    rows = []
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)
        pts = read_region_file(filepath)

        pts_resampled = resample_by_arclength(pts, num=200)

        # 原始
        t0 = time.perf_counter()
        v0,a0,j0 = compute_derivatives(pts_resampled, dt)
        curv0 = compute_curvatures(pts_resampled)
        t_original = (time.perf_counter() - t0) * 100

        # S-G
        t0 = time.perf_counter()
        pts_sg = smooth_savgol(pts_resampled, window_length=7, polyorder=3)
        v1,a1,j1 = compute_derivatives(pts_sg, dt)
        curv1 = compute_curvatures(pts_sg)
        max_err_sg, _ = error_stats(pts_resampled, pts_sg)
        t_sg = (time.perf_counter() - t0) * 100

        # Spline
        t0 = time.perf_counter()
        pts_spline = smooth_spline(pts_resampled, smooth_factor=len(pts_resampled) * 0.0002)
        v2,a2,j2 = compute_derivatives(pts_spline, dt)
        curv2 = compute_curvatures(pts_spline)
        max_err_sp, _ = error_stats(pts_resampled, pts_spline)
        t_spline = (time.perf_counter() - t0) * 100

        # Gaussian
        t0 = time.perf_counter()
        pts_gauss = smooth_gaussian(pts_resampled, sigma=1.0)
        v3,a3,j3 = compute_derivatives(pts_gauss, dt)
        curv3 = compute_curvatures(pts_gauss)
        max_err_gs, _ = error_stats(pts_resampled, pts_gauss)
        t_gauss = (time.perf_counter() - t0) * 100

        # LOWESS
        t0 = time.perf_counter()
        #pts_resampled = resample_by_arclength(pts, num=120)
        pts_lowess = smooth_lowess(pts_resampled, frac=0.06, it=0)
        pts_lowess = resample_by_arclength(pts_lowess, num=len(pts_resampled))
        #pts_lowess = smooth_lowess(pts_resampled, frac=0.15, it=1)
        v4,a4,j4 = compute_derivatives(pts_lowess, dt)
        curv4 = compute_curvatures(pts_lowess)
        max_err_lw, _ = error_stats(pts_resampled, pts_lowess)
        t_lowess = (time.perf_counter() - t0) * 100

        max_k0, mean_k0 = stats(curv0)
        max_k1, mean_k1 = stats(curv1)
        max_k2, mean_k2 = stats(curv2)
        max_k3, mean_k3 = stats(curv3)
        max_k4, mean_k4 = stats(curv4)

        max_v0, mean_v0 = stats(v0)
        max_a0, mean_a0 = stats(a0)
        max_j0, mean_j0 = stats(j0)

        max_v1, mean_v1 = stats(v1)
        max_a1, mean_a1 = stats(a1)
        max_j1, mean_j1 = stats(j1)

        max_v2, mean_v2 = stats(v2)
        max_a2, mean_a2 = stats(a2)
        max_j2, mean_j2 = stats(j2)

        max_v3, mean_v3 = stats(v3)
        max_a3, mean_a3 = stats(a3)
        max_j3, mean_j3 = stats(j3)

        max_v4, mean_v4 = stats(v4)
        max_a4, mean_a4 = stats(a4)
        max_j4, mean_j4 = stats(j4)

        rows += [
            [filename, "Original", max_k0, mean_k0, max_v0, mean_v0, max_a0, mean_a0, max_j0, mean_j0, 0.0, t_original],
            [filename, "S-G", max_k1, mean_k1, max_v1, mean_v1, max_a1, mean_a1, max_j1, mean_j1, max_err_sg, t_sg],
            [filename, "Spline", max_k2, mean_k2, max_v2, mean_v2, max_a2, mean_a2, max_j2, mean_j2, max_err_sp, t_spline],
            [filename, "Gaussian", max_k3, mean_k3, max_v3, mean_v3, max_a3, mean_a3, max_j3, mean_j3, max_err_gs, t_gauss],
            [filename, "LOWESS", max_k4, mean_k4, max_v4, mean_v4, max_a4, mean_a4, max_j4, mean_j4, max_err_lw, t_lowess],
        ]

        log_stats(filename, "Original", max_k0, mean_k0, max_v0, mean_v0, max_a0, mean_a0, max_j0, mean_j0, 0.0, t_original)
        log_stats(filename, "S-G", max_k1, mean_k1, max_v1, mean_v1, max_a1, mean_a1, max_j1, mean_j1, max_err_sg, t_sg)
        log_stats(filename, "Spline", max_k2, mean_k2, max_v2, mean_v2, max_a2, mean_a2, max_j2, mean_j2, max_err_sp, t_spline)
        log_stats(filename, "Gaussian", max_k3, mean_k3, max_v3, mean_v3, max_a3, mean_a3, max_j3, mean_j3, max_err_gs, t_gauss)
        log_stats(filename, "LOWESS", max_k4, mean_k4, max_v4, mean_v4, max_a4, mean_a4, max_j4, mean_j4, max_err_lw, t_lowess)

        title = f"{filename} (Region {i})"
        plot_trajectory_comparison(pts_resampled, pts_sg, pts_spline, pts_gauss, pts_lowess, title)
        plot_derivative_comparison(v0,a0,j0, v1,a1,j1, v2,a2,j2, v3,a3,j3, v4,a4,j4, title)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "method", "Kmax", "Kmean", "Vmax", "Vmean",
            "Amax", "Amean", "Jmax", "Jmean", "ErrMax", "Time"
        ])
        writer.writerows(rows)

if __name__ == "__main__":
    visualize_folder("top_curvature_regions", dt=0.01, csv_path="stats_results.csv")