import torch
import matplotlib
# matplotlib.use('TkAgg') # 如果在本地运行遇到显示问题，请尝试取消注释这行
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 设置中文字体支持 (可选，如果不需要中文标题可注释掉)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def visualize_dataset(pt_file_path):
    print(f"Loading {pt_file_path}...")
    try:
        data = torch.load(pt_file_path, map_location='cpu')
    except FileNotFoundError:
        print(f"[Error] File not found: {pt_file_path}")
        print("Please run preprocess.py first to generate the data file.")
        return

    inputs = data['inputs']   # [N, 3, 32, 32, 10] (bool or float)
    targets = data['targets'] # [N, 4]
    cfg = data['config']
    
    N = len(inputs)
    print(f"Loaded {N} samples successfully.")
    
    # 交互式查看
    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 用于交互的全局变量
    state = {'current_idx': 0}
    
    def render(idx):
        ax.clear()
        
        # 1. 准备数据 (确保转为 numpy bool 数组用于绘图)
        grid = inputs[idx].numpy() > 0.5 # 转为 bool
        target = targets[idx].numpy()
        
        # Channel Mapping
        obs_mask = grid[0]      # Obstacle (障碍物)
        front_mask = grid[1]    # Frontier (前沿)
        free_mask = grid[2]     # Free Space (自由空间)

        # 2. 绘制 Voxel (注意绘制顺序，后画的在上面)
        
        # A. Free Space (自由空间) - 最底层，极淡的绿色，高透明
        # 使用 hex 颜色码加 alpha 值 (#RRGGBBAA)，A0为十六进制透明度
        if np.any(free_mask):
            # 淡绿色，透明度很高，边缘非常细
            ax.voxels(free_mask, facecolors='#90EE9020', edgecolors='#90EE9040', linewidth=0.1, shade=False)

        # B. Obstacle (障碍物) - 中间层，深蓝色，半透明
        if np.any(obs_mask):
            ax.voxels(obs_mask, facecolors='#1f77b440', edgecolors='#1f77b480', linewidth=0.3, shade=True)
            
        # C. Frontier (前沿) - 最上层，青色高亮，不透明度较高
        if np.any(front_mask):
            ax.voxels(front_mask, facecolors='#00FFFF95', edgecolors='#008B8B', linewidth=0.6, shade=False)

        # 3. 绘制 Label (最佳视点)
        # 反归一化坐标
        dim_x, dim_y, dim_z = cfg['target_dim']
        
        tx = (target[0] + 1.0) * (dim_x / 2.0)
        ty = (target[1] + 1.0) * (dim_y / 2.0)
        tz = (target[2] + 1.0) * (dim_z / 2.0)
        yaw = target[3] * np.pi
        
        # 绘制 Agent 位置 (鲜绿色三角)
        ax.scatter([tx], [ty], [tz], c='#32CD32', s=250, marker='^', edgecolors='white', linewidth=1.5, zorder=20, label='Best View Agent')
        
        # 绘制朝向箭头 (已缩短)
        arrow_len = 2.5  # <--- 修改这里改变箭头长度 (之前是 4.0)
        u = arrow_len * np.cos(yaw)
        v = arrow_len * np.sin(yaw)
        w = 0 # 简化 Z 轴朝向，只看水平面
        # 使用 quiver 画箭头，颜色鲜亮
        ax.quiver(tx, ty, tz, u, v, w, color='#32CD32', linewidth=3, length=arrow_len, normalize=True, arrow_length_ratio=0.3, zorder=20)

        # 4. 设置视图属性
        # 设置轴范围略大于 grid，留点边距
        ax.set_xlim(-1, dim_x + 1)
        ax.set_ylim(-1, dim_y + 1)
        ax.set_zlim(-1, dim_z + 1)
        
        ax.set_xlabel(f'X axis ({cfg["res"]}m/grid)')
        ax.set_ylabel(f'Y axis ({cfg["res"]}m/grid)')
        ax.set_zlabel(f'Z axis ({cfg["res"]}m/grid)')
        
        # 保持物理比例 (关键! 否则扁平的盒子会被拉成正方体)
        ax.set_box_aspect((dim_x, dim_y, dim_z)) 
        
        # 隐藏背景网格和轴背景色，让显示更干净
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # 设置背景色为深灰，突出显示体素
        ax.set_facecolor('#333333') 
        fig.patch.set_facecolor('#333333')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

        
        title_str = f"Sample ID: {idx}/{N-1}\nObs: {np.sum(obs_mask)} | Free: {np.sum(free_mask)} | Frontier: {np.sum(front_mask)}"
        ax.set_title(title_str, color='white')
        
        # 自定义图例
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.4, label='障碍物 (Obstacle)'),
            Patch(facecolor='#90EE90', alpha=0.2, label='自由空间 (Free)'),
            Patch(facecolor='#00FFFF', alpha=0.9, label='前沿表面 (Frontier)'),
            plt.Line2D([0], [0], marker='^', color='w', label='最佳视点 (Best View)',
                          markerfacecolor='#32CD32', markersize=15, markeredgecolor='w')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#555555', labelcolor='white')
        
        fig.canvas.draw_idle()

    # 事件处理函数
    def on_key(event):
        if event.key == 'right' or event.key == 'enter':
            state['current_idx'] = (state['current_idx'] + 1) % N
            render(state['current_idx'])
        elif event.key == 'left':
            state['current_idx'] = (state['current_idx'] - 1 + N) % N
            render(state['current_idx'])
        elif event.key == 'q' or event.key == 'escape':
            plt.close(fig)

    # 注册键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 初始渲染
    render(state['current_idx'])
    
    print("\n--- Controls ---")
    print("Press [Right Arrow] or [Enter] for next sample")
    print("Press [Left Arrow] for previous sample")
    print("Press [Q] or [Esc] to quit")
    print("Use mouse to rotate and zoom.")
    
    plt.show()

if __name__ == "__main__":
    # 生成的 pt 文件路径
    PT_PATH = "./processed_data.pt"
    visualize_dataset(PT_PATH)