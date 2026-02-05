import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ====== 参数 ======
dt = 0.1               # 固定采样间隔
num_samples = 500       # 插值采样点数
smooth_s = 0.5          # B样条平滑度
window = 3              # 最大曲率点前后展示的点数（原始轨迹索引窗口）

# ====== 自动读取脚本同目录下的 gcode.txt ======
base_dir = os.path.dirname(os.path.abspath(__file__))
gcode_path = os.path.join(base_dir, "gcode.txt")

# ====== 读取 GCode ======
pattern = re.compile(r"G01\s+X([-\d.]+)\s+Y([-\d.]+)\s+Z([-\d.]+)")
points = []

with open(gcode_path, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line.strip())
        if m:
            points.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])

if len(points) < 2:
    raise ValueError("有效G代码点太少，无法处理")

pts = np.array(points)

# ====== 去掉连续重复点 ======
unique_pts = [pts[0]]
for p in pts[1:]:
    if not np.allclose(p, unique_pts[-1]):
        unique_pts.append(p)
pts = np.array(unique_pts)

if len(pts) < 2:
    raise ValueError("去重后有效点太少，无法处理")

x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

# ====== 曲率计算（离散点）=====
def curvature(points):
    curv = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = np.linalg.norm(np.cross(v1, v2))
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v1 + v2) + 1e-12)
        curv.append(cross / denom)
    return np.array(curv)

curv = curvature(pts)
if len(curv) > 0:
    max_idx = np.argmax(curv) + 1
    max_point = pts[max_idx]
else:
    max_idx = 0
    max_point = pts[0]

# ====== 样条插值轨迹 ======
k = min(3, len(pts) - 1)
if k < 1:
    raise ValueError("有效点数不足，无法进行样条插值")

tck, u = splprep([x, y, z], s=1e-6, k=k)
u_new = np.linspace(0, 1, num_samples)
x_i, y_i, z_i = splev(u_new, tck)

# ====== B样条平滑轨迹 ======
tck_smooth, u_smooth = splprep([x, y, z], s=smooth_s, k=k)
x_s, y_s, z_s = splev(u_new, tck_smooth)

# ====== 计算速度/加速度/加加速度 ======
def kinematics(x, y, z, dt):
    pos = np.vstack([x, y, z]).T
    v = np.gradient(pos, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    speed = np.linalg.norm(v, axis=1)
    accel = np.linalg.norm(a, axis=1)
    jerk = np.linalg.norm(j, axis=1)
    return speed, accel, jerk

speed_raw, accel_raw, jerk_raw = kinematics(x, y, z, dt)
speed_i, accel_i, jerk_i = kinematics(x_i, y_i, z_i, dt)
speed_s, accel_s, jerk_s = kinematics(x_s, y_s, z_s, dt)

# ====== 统计指标 ======
def stats(name, speed, accel, jerk, dt):
    max_speed = np.max(speed)
    avg_speed = np.mean(speed)
    max_accel = np.max(accel)
    avg_accel = np.mean(accel)
    max_jerk = np.max(jerk)
    avg_jerk = np.mean(jerk)
    process_time = len(speed) * dt

    print(f"\n==== {name} ====")
    print(f"最大速度: {max_speed:.6f}")
    print(f"平均速度: {avg_speed:.6f}")
    print(f"最大加速度: {max_accel:.6f}")
    print(f"平均加速度: {avg_accel:.6f}")
    print(f"最大加加速度: {max_jerk:.6f}")
    print(f"平均加加速度: {avg_jerk:.6f}")
    print(f"加工时间: {process_time:.6f} s")

stats("Raw", speed_raw, accel_raw, jerk_raw, dt)
stats("Spline Interp", speed_i, accel_i, jerk_i, dt)
stats("B-spline Smooth", speed_s, accel_s, jerk_s, dt)

# ====== 只显示最大曲率点附近的局部轨迹 ======
start = max(0, max_idx - window)
end = min(len(pts), max_idx + window + 1)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[start:end], y[start:end], z[start:end], 'o-', label="Raw (local)")
ax.scatter(max_point[0], max_point[1], max_point[2], c='r', s=80, label="Max Curvature")

# 在平滑轨迹中找对应参数位置
u_max = u[max_idx] if len(u) == len(pts) else max_idx / (len(pts) - 1)
idx_smooth = int(u_max * (num_samples - 1))
win_s = max(0, idx_smooth - window*10)
win_e = min(num_samples, idx_smooth + window*10 + 1)

ax.plot(x_i[win_s:win_e], y_i[win_s:win_e], z_i[win_s:win_e], '-', label="Spline Interp (local)")
ax.plot(x_s[win_s:win_e], y_s[win_s:win_e], z_s[win_s:win_e], '-', label="B-spline Smooth (local)")

ax.set_title("Local Trajectory Around Max Curvature")
ax.legend()
plt.show()

# ====== 三种轨迹速度对比 ======
t_raw = np.arange(len(speed_raw)) * dt
t_i = np.arange(len(speed_i)) * dt
t_s = np.arange(len(speed_s)) * dt

plt.figure(figsize=(10, 6))
plt.plot(t_raw, speed_raw, label="Raw Speed")
plt.plot(t_i, speed_i, label="Spline Speed")
plt.plot(t_s, speed_s, label="B-spline Speed")
plt.title("Speed Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Speed")
plt.legend()
plt.grid(True)
plt.show()

# ====== 三种轨迹加速度对比 ======
plt.figure(figsize=(10, 6))
plt.plot(t_raw, accel_raw, label="Raw Accel")
plt.plot(t_i, accel_i, label="Spline Accel")
plt.plot(t_s, accel_s, label="B-spline Accel")
plt.title("Acceleration Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.legend()
plt.grid(True)
plt.show()

# ====== 三种轨迹加加速度对比 ======
plt.figure(figsize=(10, 6))
plt.plot(t_raw, jerk_raw, label="Raw Jerk")
plt.plot(t_i, jerk_i, label="Spline Jerk")
plt.plot(t_s, jerk_s, label="B-spline Jerk")
plt.title("Jerk Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Jerk")
plt.legend()
plt.grid(True)
plt.show()

print("最大曲率点索引:", max_idx)
print("最大曲率点坐标:", max_point)