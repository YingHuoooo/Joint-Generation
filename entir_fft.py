import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

def analyze_gcode_acceleration(filename='gcode.txt', fs=2000):
    """
    读取 G01 G代码文件并进行加速度频域分析
    filename: G代码文件名
    fs: 重采样频率 (Hz)。为了捕捉尖峰，建议使用较高的频率，例如 2000Hz 或更高。
    """
    
    # --- 1. 检查文件是否存在 ---
    if not os.path.exists(filename):
        print(f"错误: 找不到文件 '{filename}'")
        return

    print(f"正在读取 {filename} ...")

    # --- 2. 解析 G代码 (仅针对 G01) ---
    coords = []  # 存储 [x, y, z]
    feeds = []   # 存储 F 值
    
    current_x, current_y, current_z = 0.0, 0.0, 0.0
    current_f = 200.0 # 默认初始速度
    
    coords.append([current_x, current_y, current_z])
    feeds.append(current_f) 

    pattern = re.compile(r'([XYZF])([-+]?[0-9]*\.?[0-9]+)')

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().upper().split(';')[0].split('(')[0]
            if not line: continue

            matches = pattern.findall(line)
            if not matches: continue

            updated_pos = False
            for axis, val in matches:
                val = float(val)
                if axis == 'X': current_x, updated_pos = val, True
                elif axis == 'Y': current_y, updated_pos = val, True
                elif axis == 'Z': current_z, updated_pos = val, True
                elif axis == 'F': current_f = val
            
            if updated_pos:
                coords.append([current_x, current_y, current_z])
                feeds.append(current_f)

    coords = np.array(coords)
    feeds = np.array(feeds)

    if len(coords) < 2:
        print("错误: 数据点不足。")
        return

    # --- 3. 物理运动学计算 (空间 -> 时间) ---
    print("正在构建时间轴...")
    deltas = np.diff(coords, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    # F 为 mm/min 转为 mm/s
    segment_velocities = feeds[1:] / 60.0
    segment_velocities = np.where(segment_velocities <= 1e-9, 1e-6, segment_velocities)
    segment_durations = distances / segment_velocities
    timestamps = np.concatenate(([0], np.cumsum(segment_durations)))
    total_time = timestamps[-1]
    
    if total_time <= 1e-9:
        print("总加工时间几乎为 0，无法分析。")
        return

    # --- 4. 均匀重采样与微分 (计算速度和加速度) ---
    print(f"正在以 {fs}Hz 进行重采样和微分计算...")
    
    # 生成均匀时间轴
    num_samples = int(total_time * fs)
    # 保证至少有足够的点进行两次微分
    if num_samples < 5: num_samples = 5 
    t_uniform = np.linspace(0, total_time, num_samples)
    dt = t_uniform[1] - t_uniform[0]
    
    # A) 位置插值 (使用线性插值，忠实反映 G01 的直线特性)
    interp_x = interp1d(timestamps, coords[:, 0], kind='linear')(t_uniform)
    interp_y = interp1d(timestamps, coords[:, 1], kind='linear')(t_uniform)
    interp_z = interp1d(timestamps, coords[:, 2], kind='linear')(t_uniform)

    # B) 一次微分：计算速度向量分量 (mm/s)
    vx = np.gradient(interp_x, dt)
    vy = np.gradient(interp_y, dt)
    vz = np.gradient(interp_z, dt)
    # 合成线速度 (用于检查)
    # v_combined = np.sqrt(vx**2 + vy**2 + vz**2)

    # C) 二次微分：计算加速度向量分量 (mm/s^2)
    # 对速度分量再次求导
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)

    # 计算合成加速度的大小 (Tangential Acceleration Magnitude)
    # 这是我们需要分析的主要信号
    a_combined = np.sqrt(ax**2 + ay**2 + az**2)
    
    # 数据清洗：由于数值微分会放大噪声，特别是在起点和终点，
    # 可以简单去除极其巨大的异常值(可选，这里暂时保留以展示真实数学结果)
    # a_combined = np.clip(a_combined, 0, np.percentile(a_combined, 99.9))

    # --- 5. FFT 频域分析 (针对加速度) ---
    print("正在进行加速度 FFT 分析...")
    n = len(t_uniform)
    
    # 去除直流分量 (关注加速度的波动情况)
    a_fluctuation = a_combined - np.mean(a_combined)
    
    fft_val = np.fft.fft(a_fluctuation)
    fft_freq = np.fft.fftfreq(n, d=dt)
    
    mask = fft_freq > 0
    freqs = fft_freq[mask]
    # 归一化幅值
    magnitudes = 2.0 * np.abs(fft_val[mask]) / n 

    # --- 6. 绘图 ---
    plt.figure(figsize=(14, 10))

    # 子图1: 时域加速度
    plt.subplot(2, 1, 1)
    # 使用半透明红色描绘加速度，因为它们通常是尖峰
    plt.plot(t_uniform, a_combined, color='red', alpha=0.7, linewidth=1, label='Acceleration Magnitude')
    plt.title(f'Time Domain: Acceleration (Resampled @ {fs}Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration ($mm/s^2$)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # 由于尖峰可能非常高，可以适当限制Y轴视野来观察细节
    # y_max_view = np.percentile(a_combined, 99.5) * 1.5
    # plt.ylim(0, y_max_view)

    # 子图2: 加速度频谱
    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitudes, color='purple')
    plt.title('Frequency Domain: Acceleration Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (Amplitude)')
    plt.grid(True, alpha=0.3)
    
    # 限制频率显示范围，通常高频部分是采样噪声
    plt.xlim(0, fs/4) # 显示到奈奎斯特频率的一半

    plt.tight_layout()
    print("分析完成，正在显示图像。")
    plt.show()

# ==========================================
# 运行程序
# ==========================================
if __name__ == "__main__":
    # 创建一个简单的测试用 Gcode 文件用于演示
    test_filename = 'gcode_acc_test.txt'
    with open(test_filename, 'w') as f:
        # 创建一个往复运动，速度在 F1000 和 F2000 之间切换
        f.write("G01 X0 Y0 F1000\n")
        for i in range(10):
            f.write(f"G01 X{10*(i+1)} Y0 F2000\n") # 加速段
            f.write(f"G01 X{10*(i+1)+5} Y0 F1000\n") # 减速段
    
    # 运行分析。提高采样率到 2000Hz 以更好地捕捉加速度尖峰
    analyze_gcode_acceleration("top_curvature_regions/region_01.txt", fs=1000)