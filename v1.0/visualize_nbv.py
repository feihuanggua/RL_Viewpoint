import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import OfflineFrontierEnv 

class PenetrationVisualizer:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        print("初始化穿透视效窗口...")
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    def list_to_grid(self, point_list):
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
        if point_list is None or len(point_list) == 0:
            return grid
        
        # 使用 numpy 高级索引加速
        pts = np.array(list(point_list))
        if pts.shape[0] > 0:
            # 边界过滤
            mask = (pts[:,0]>=0) & (pts[:,0]<self.grid_size) & \
                   (pts[:,1]>=0) & (pts[:,1]<self.grid_size) & \
                   (pts[:,2]>=0) & (pts[:,2]<self.grid_size)
            valid_pts = pts[mask]
            grid[valid_pts[:,0], valid_pts[:,1], valid_pts[:,2]] = True
        return grid

    def render(self, obstacle_grid, frontier_surface, known_volume, camera_pose):
        self.ax.clear()
        
        # --- 1. 绘制障碍物 (环境参考) ---
        grid_obs = obstacle_grid.astype(bool)
        if np.any(grid_obs):
            self.ax.voxels(grid_obs, facecolors='#1f77b415', edgecolors='#AAAAAA', linewidth=0.1)

        # --- 2. 绘制前沿面 (The Thin Wall) ---
        # 这是一层厚度为1的墙，作为"未知"与"已知"的分界线
        # 我们把它画成像玻璃一样的灰色半透明薄片
        grid_frontier = self.list_to_grid(frontier_surface)
        if np.any(grid_frontier):
            self.ax.voxels(grid_frontier, 
                           facecolors='#FFFFFF10', # 极度透明的白/灰
                           edgecolors='#888888',   # 灰色边框勾勒轮廓
                           linewidth=0.3,
                           alpha=0.2,
                           label='Frontier Surface')

        # --- 3. 绘制探索后的体积 (The Gold Volume) ---
        # 这是射线穿过前沿后，照亮的背后区域
        grid_gold = self.list_to_grid(known_volume)
        
        if np.any(grid_gold):
            # 这里的金色代表"Unknown -> Known"的增益体积
            self.ax.voxels(grid_gold, 
                           facecolors='#FFD700', # 金色
                           edgecolors='#B8860b', # 深金边框
                           linewidth=0.5,
                           alpha=0.9,            # 不透明，强调体积感
                           shade=True,
                           label='Discovered Volume')

        # --- 4. 绘制 Agent ---
        cx, cy, cz, cyaw = camera_pose
        self.ax.scatter([cx], [cy], [cz], c='#00FF00', s=200, marker='^', zorder=10)
        
        # 视线引导线
        arrow_len = 8.0
        self.ax.quiver(cx, cy, cz, np.cos(cyaw)*arrow_len, np.sin(cyaw)*arrow_len, 0, 
                       color='#00FF00', linewidth=2)

        # --- 设置 ---
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        
        # 统计体积
        vol_count = len(known_volume) if known_volume else 0
        self.ax.set_title(f"Volumetric Exploration\nGold Volume: {vol_count} voxels")

        # 自定义图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DDDDDD', edgecolor='#888888', alpha=0.3, label='Frontier Wall (Interface)'),
            Patch(facecolor='#FFD700', label='Discovered Volume (Behind Wall)'),
            Patch(facecolor='#00FF00', label='Agent')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# --- 场景生成：薄墙与背后的未知空间 ---
def generate_thin_wall_scenario(env):
    env.reset()
    # 1. 定义墙的位置 X=20
    wall_x = 20
    
    # 2. 清空中间区域
    env.gt_map.grid[5:28, 5:28, 5:28] = 0
    
    # 3. 放置真正的物理障碍物 (比如在 X=30 处放一个背板，或者在中间放一个柱子)
    # 这样射线就不会无限延伸，会形成形状
    env.gt_map.grid[30, :, :] = 1 # 背板
    env.gt_map.grid[24:28, 14:18, 10:22] = 1 # 墙后藏着一个柱子障碍物
    
    # 4. 生成前沿面 (Frontier Surface) List
    # 这就是那层厚度为1的墙
    frontier_surface = []
    for y in range(8, 24):
        for z in range(8, 24):
            frontier_surface.append((wall_x, y, z))
            
    return frontier_surface

# --- 核心逻辑：计算穿透后的体积 ---
def calculate_penetrated_volume(env, frontier_surface, camera_pose):
    cx, cy, cz, cyaw = camera_pose
    wall_x = 20
    
    # 1. 执行常规 Raycast
    # scan 会返回所有射线经过的"非障碍物"点 (covered_set)
    # 注意：我们的 RayCaster 需要能返回"经过的所有空点"，而不仅仅是击中点
    # 标准的 scan 返回的是 covered_voxels (set of tuples)
    covered_set, _ = env.raycaster.scan(env.gt_map, [cx, cy, cz], cyaw)
    
    # 2. 筛选：只保留位于前沿墙"后面"的点
    # 这就是"探索到的未知区域"
    gold_volume = []
    
    for (vx, vy, vz) in covered_set:
        # 逻辑：
        # 1. 点必须在墙的后面 (vx > wall_x)
        # 2. 并且这个点在 Y, Z 方向上处于我们定义的"未知区域范围"内 (可选，避免画出整个世界的空区域)
        if vx > wall_x: 
            # 简单的裁切一下范围，只看墙后面的核心区域
            if 5 <= vy < 27 and 5 <= vz < 27:
                gold_volume.append((vx, vy, vz))
                
    return gold_volume

# --- 简单的最佳视点选择 ---
def select_best_volumetric_view(env, frontier_surface):
    best_score = -1
    best_data = None # (pose, volume)
    
    # 在墙前采样
    for _ in range(15):
        cx = np.random.uniform(5, 18) # 离墙越近，FOV 覆盖越小但越清晰；离远了覆盖大
        cy = np.random.uniform(8, 24)
        cz = np.random.uniform(8, 24)
        
        # 指向墙的中心
        dx, dy, dz = np.array([20, 16, 16]) - np.array([cx, cy, cz])
        cyaw = np.arctan2(dy, dx)
        
        pose = [cx, cy, cz, cyaw]
        
        # 计算体积
        gold_vol = calculate_penetrated_volume(env, frontier_surface, pose)
        score = len(gold_vol)
        
        if score > best_score:
            best_score = score
            best_data = (pose, gold_vol)
            
    return best_data

if __name__ == "__main__":
    env = OfflineFrontierEnv()
    viz = PenetrationVisualizer(grid_size=32)
    
    print("开始穿透体积演示...")
    print("灰色薄面 = 前沿墙 (Frontier)")
    print("金色实体 = 探索到的背后体积 (Explored Volume)")
    
    for i in range(10):
        # 1. 生成场景
        ft_surface = generate_thin_wall_scenario(env)
        
        # 2. 选一个能看到最多背后体积的视点
        print(f"Case {i+1}: 寻找最佳穿透视点...")
        pose, gold_volume = select_best_volumetric_view(env, ft_surface)
        
        # 3. 渲染
        viz.render(env.gt_map.grid, ft_surface, gold_volume, pose)
        
        input(f"Case {i+1} 完成. 按 [Enter] 继续...")