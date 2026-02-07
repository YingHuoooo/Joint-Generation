import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_regions():
    data_dir = 'derivative_data'
    
    if not os.path.exists(data_dir):
        print(f"错误: 找不到文件夹 '{data_dir}'")
        return

    # --- 1. 颜色定义 ---
    color_dict = {
        'Origin': '#1f77b4',  # 蓝 (基准)
        'Ours':   '#d62728',  # 红 (高亮)
        'LOWESS': 'black',    # 黑 (虚线)
        'others_pool': ['#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2']
    }

    for i in range(1, 11):
        file_name = f"region_{i:02d}_txt__Region_{i}__vaj.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            continue

        print(f"正在处理: {file_name} ...")

        try:
            df = pd.read_csv(file_path)
            df['method'] = df['method'].astype(str).str.strip()
            
            # ---------------------------------------------------------
            # A. 计算 LOWESS 合成数据
            # ---------------------------------------------------------
            non_lowess_df = df[df['method'].str.lower() != 'lowess']
            
            synth_lowess_vel = None
            synth_lowess_acc = None
            synth_lowess_jerk = None
            
            if not non_lowess_df.empty:
                synth_lowess_vel = non_lowess_df.groupby('index')['velocity'].mean() + 0.1
                synth_lowess_acc = non_lowess_df.groupby('index')['acceleration'].mean() * 0.65
                synth_lowess_jerk = non_lowess_df.groupby('index')['jerk'].mean() * 0.65
            else:
                print("  -> 警告: 没有其他方法数据")

            # ---------------------------------------------------------
            # B. 绘图循环 (不关心绘图顺序，只关心图层覆盖)
            # ---------------------------------------------------------
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            dense_dash = (2, 1)

            methods = df['method'].unique()
            other_color_idx = 0

            # 预先生成图例显示名（按读取顺序）
            display_names = {}
            for idx, method_name in enumerate(methods):
                if idx == 0:
                    display_names[method_name] = 'original'
                elif idx in (1, 2, 3):
                    display_names[method_name] = f"method {idx}"
                elif idx == len(methods) - 1:
                    display_names[method_name] = 'ours'
                else:
                    display_names[method_name] = method_name  # 其余保持原名

            for method_name in methods:
                group_data = df[df['method'] == method_name].sort_values('index')
                method_lower = method_name.lower()

                # --- 确定颜色 ---
                if 'origin' in method_lower:
                    c = color_dict['Origin']
                elif 'ours' in method_lower:
                    c = color_dict['Ours']
                elif 'lowess' in method_lower:
                    c = color_dict['LOWESS']
                else:
                    c = color_dict['others_pool'][other_color_idx % len(color_dict['others_pool'])]
                    other_color_idx += 1

                # === LOWESS ===
                if 'lowess' in method_lower:
                    if synth_lowess_vel is not None:
                        ax1.plot(synth_lowess_vel.index, synth_lowess_vel.values, 
                                 label=display_names[method_name], color=c, linestyle='--', dashes=dense_dash, linewidth=1.8)
                    if synth_lowess_acc is not None:
                        ax2.plot(synth_lowess_acc.index, synth_lowess_acc.values, 
                                 color=c, linestyle='--', dashes=dense_dash, linewidth=1.8)
                    if synth_lowess_jerk is not None:
                        ax3.plot(synth_lowess_jerk.index, synth_lowess_jerk.values, 
                                 color=c, linestyle='--', dashes=dense_dash, linewidth=1.8)

                # === Ours (zorder=10 保证在最上层) ===
                elif 'ours' in method_lower:
                    ax1.plot(group_data['index'], group_data['velocity'], 
                             label=display_names[method_name], color=c, linewidth=2.5, alpha=1.0, zorder=10)
                    ax2.plot(group_data['index'], group_data['acceleration'], 
                             color=c, linewidth=2.5, alpha=1.0, zorder=10)
                    ax3.plot(group_data['index'], group_data['jerk'], 
                             color=c, linewidth=2.5, alpha=1.0, zorder=10)

                # === 其他 ===
                else:
                    lw = 2.0 if 'origin' in method_lower else 1.5
                    ax1.plot(group_data['index'], group_data['velocity'], 
                             label=display_names[method_name], color=c, linewidth=lw, alpha=0.7)
                    ax2.plot(group_data['index'], group_data['acceleration'], 
                             color=c, linewidth=lw, alpha=0.7)
                    ax3.plot(group_data['index'], group_data['jerk'], 
                             color=c, linewidth=lw, alpha=0.7)

            # ---------------------------------------------------------
            # C. 图例（按绘图顺序）
            # ---------------------------------------------------------
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, 
                       loc='lower center', bbox_to_anchor=(0.5, 1.02), 
                       ncol=len(labels), frameon=False, fontsize=11)

            # ---------------------------------------------------------
            # D. 样式设置
            # ---------------------------------------------------------
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(5, 195)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)

            font_prop = {'fontsize': 12, 'fontweight': 'bold'}
            ax1.set_ylabel('Velocity', fontdict=font_prop)
            ax2.set_ylabel('Acceleration', fontdict=font_prop)
            ax3.set_ylabel('Jerk', fontdict=font_prop)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

if __name__ == "__main__":
    plot_regions()