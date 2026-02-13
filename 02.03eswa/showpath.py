import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def plot_gcode_clean(file_path):
    # 初始化数据存储
    # 为了连贯性，我们将所有点存在一个序列里，或者分段存储
    x_coords, y_coords, z_coords = [0.0], [0.0], [0.0]
    
    # 当前状态
    curr_pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 预处理：转大写，去注释
                line = line.upper().split(';')[0]
                line = re.sub(r'\(.*?\)', '', line).strip()
                
                if not line:
                    continue

                # 提取坐标
                found = False
                for axis in ['X', 'Y', 'Z']:
                    match = re.search(rf'{axis}([-+]?\d*\.\d+|\d+)', line)
                    if match:
                        curr_pos[axis] = float(match.group(1))
                        found = True
                
                # 如果这一行产生了位移，记录点
                if found:
                    x_coords.append(curr_pos['X'])
                    y_coords.append(curr_pos['Y'])
                    z_coords.append(curr_pos['Z'])

        # 绘图设置
        fig = plt.figure("G-Code Trajectory", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制实线路径，并加上轨迹点 (marker='o')
        # markersize 可以控制点的大小，linewidth 控制线的粗细
        ax.plot(x_coords, y_coords, z_coords, 
                color='#1f77b4',     # 科技感蓝
                linestyle='-',       # 实线
                linewidth=1,         # 线宽
                marker='o',          # 点的形状
                markersize=2,        # 点的大小
                markerfacecolor='red',# 点的颜色
                markeredgecolor='red',
                label='Toolpath')

        # 隐藏坐标轴的数值 (Tick Labels)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # 移除坐标轴的刻度线
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # 设置标题和标签
        #ax.set_title('CNC Trajectory Visualization (No Values)', pad=20)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # 调整视觉角度
        ax.view_init(elev=30, azim=45)

        plt.savefig('trajectory_vector.svg', bbox_inches='tight')



        plt.show()

    except FileNotFoundError:
        print(f"找不到文件: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
plot_gcode_clean('gcode.txt')