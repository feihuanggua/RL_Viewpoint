import matplotlib
matplotlib.use('TkAgg') # 保持使用 TkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

class VoxelVisualizer:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        print("正在初始化可视化窗口...")
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    def render(self, gt_grid, known_mask, robot_pos, robot_yaw, hit_points=None, pause_time=0.1):
        """
        gt_grid: 真实障碍物地图 (0/1)
        known_mask: 当前已探索区域 (0=Unknown, 1=Known)
        """
        self.ax.clear()
        
        # --- 1. 计算要显示的体素 ---
        # A. 已知的障碍物 (Known Obstacles) -> 蓝色
        # 逻辑：实际上是障碍物 AND 我们已经看见了它
        # 注意：为了演示效果，我们也可以一直显示所有障碍物(半透明)，但高亮显示的已知障碍物
        known_obs = (gt_grid == 1) & (known_mask == 1)
        
        # B. 已知的空闲区域 (Known Free) -> 绿色 (代表已探索)
        known_free = (gt_grid == 0) & (known_mask == 1)
        
        # C. 未知的障碍物 (Unknown Obstacles) -> 灰色 (上帝视角，稍微显示一点方便调试)
        unknown_obs = (gt_grid == 1) & (known_mask == 0)

        # --- 2. 绘制 ---
        
        # 绘制已知障碍 (蓝色，不透明)
        if np.any(known_obs):
            self.ax.voxels(known_obs, facecolors='#1f77b4', edgecolors='k', linewidth=0.5, alpha=0.8, label='Known Obs')
            
        # 绘制未知障碍 (灰色，半透明，作为背景参考)
        if np.any(unknown_obs):
            self.ax.voxels(unknown_obs, facecolors='#bfbfbf', edgecolors='gray', linewidth=0.2, alpha=0.15)

        # 绘制已知空闲区域 (绿色，非常透明) -> 这就是“由于探索而扩大的区域”
        if np.any(known_free):
            self.ax.voxels(known_free, facecolors='#2ca02c', edgecolors=None, alpha=0.08)

        # --- 3. 绘制无人机 ---
        rx, ry, rz = robot_pos
        arrow_len = 3.0
        dx = np.cos(robot_yaw) * arrow_len
        dy = np.sin(robot_yaw) * arrow_len
        
        self.ax.scatter([rx], [ry], [rz], color='red', s=100, label='Drone')
        self.ax.quiver(rx, ry, rz, dx, dy, 0, color='red', linewidth=2, length=arrow_len)
        
        # --- 4. 绘制射线击中点 ---
        if hit_points is not None and len(hit_points) > 0:
            # 降采样
            step = max(1, len(hit_points) // 30) 
            for i in range(0, len(hit_points), step):
                hx, hy, hz = hit_points[i]
                self.ax.plot([rx, hx], [ry, hy], [rz, hz], color='orange', alpha=0.4, linewidth=0.5)

        # --- 5. 设置视图 ---
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        self.ax.set_title(f"Exploration Progress\nGreen Bubble = Explored Area")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # time.sleep(pause_time) # 有时候 flush_events 已经够慢了，不需要 sleep

if __name__ == "__main__":
    try:
        from env import OfflineFrontierEnv
    except ImportError:
        print("缺少 env.py")
        exit()

    env = OfflineFrontierEnv()
    viz = VoxelVisualizer(grid_size=32)
    
    print("开始演示探索过程...")
    print("观察【绿色雾气】的扩散，那就是前沿被覆盖的过程！")
    
    try:
        for episode in range(3):
            print(f"--- Episode {episode+1} ---")
            env.reset() # 注意 reset 返回值，这里为了能在 loop 里拿到 mask，我们直接用 env 成员变量
            
            for step in range(30): # 增加步数，看清楚探索过程
                
                # 动作: 螺旋上升并旋转，模拟全向扫描
                yaw_change = 0.5 
                z_change = 0.05 if step < 15 else -0.05
                action = [0.0, 0.0, z_change, 0.1] 
                
                _, reward, done, _ = env.step(action)
                
                # 获取数据
                gt_map = env.gt_map.grid
                known_mask = env.current_known_mask # <--- 关键：获取已知掩码
                robot_pos = env.robot_pos
                robot_yaw = env.robot_yaw
                
                # 重新扫描获取射线用于画图
                _, hit_points = env.raycaster.scan(env.gt_map, robot_pos, robot_yaw)
                
                # 渲染 (传入 known_mask)
                viz.render(gt_map, known_mask, robot_pos, robot_yaw, hit_points, pause_time=0.01)
                
                print(f"Step {step}: Explored Cells={np.sum(known_mask)}")
                
    except KeyboardInterrupt:
        print("Stop")
    
    input("Finished. Press Enter to exit.")