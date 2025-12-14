import gym
import numpy as np
from config import Config
from utils import RayCaster, FrontierProcessor

class PenetrationEnv(gym.Env):
    def __init__(self):
        self.cfg = Config()
        self.grid_size = self.cfg.GRID_SIZE
        
        # 动作空间: [x, y, z, yaw] (-1 ~ 1)
        # 代表相对于局部坐标系中心的位移和偏航角偏移
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 状态空间: [3, 32, 32, 32]
        # Channel 0: 已知障碍物 (Known Obstacles)
        # Channel 1: 前沿面 (Frontier Surface)
        # Channel 2: 已知/安全区域 (Known/Safe Space Mask)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 32, 32, 32), dtype=np.float32)
        
        # 工具类初始化
        self.raycaster = RayCaster(map_size=self.grid_size)
        self.processor = FrontierProcessor(grid_size=self.grid_size)
        
        # 内部变量 (Ground Truth & Logic)
        self.gt_grid = None         # 真值地图 (含隐形障碍物)
        self.known_obstacles = None # Agent 可见的障碍物
        self.known_mask = None      # 安全飞行区域掩码
        self.frontier_pts = None    # 前沿点列表
        self.frontier_set = set()   # 前沿点集合 (加速查找)
        self.unknown_mask = None    # 真值未知区域 (用于计算 Volume Reward)
        
        self.current_center = None  # 当前前沿簇的几何中心 (全局)
        self.current_yaw = None     # 当前前沿簇的主方向 Yaw (全局)

    def reset(self):
        # 1. 初始化地图全 0
        self.gt_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int8)
        self.unknown_mask = np.zeros_like(self.gt_grid, dtype=bool)
        
        # 2. 生成薄墙 (Frontier Surface)
        # 位置随机在 X=14~18 之间，带有一定的斜率
        wall_x = np.random.randint(14, 18)
        self.frontier_pts = []
        self.frontier_set = set()
        slope = np.random.uniform(-0.5, 0.5)
        
        # 生成较大面积的墙，测试 FOV 覆盖能力
        for y in range(6, 26):
            for z in range(6, 26):
                x = int(wall_x + (z - 16) * slope) 
                if 0 <= x < self.grid_size:
                    pt = (x, y, z)
                    self.frontier_pts.append(pt)
                    self.frontier_set.add(pt)
                    
                    # 定义: 墙和墙后面是未知的
                    if x < self.grid_size:
                        self.unknown_mask[x:, y, z] = True

        self.frontier_pts = np.array(self.frontier_pts)
        
        # 3. 生成障碍物
        # A. 墙后面的障碍物 (隐形，待探测)
        num_obs_back = np.random.randint(2, 6)
        for _ in range(num_obs_back):
            ox = np.random.randint(wall_x+1, 28)
            oy = np.random.randint(8, 24)
            oz = np.random.randint(8, 24)
            # 边界检查防止越界赋值
            x_end = min(ox+3, self.grid_size)
            y_end = min(oy+3, self.grid_size)
            z_end = min(oz+3, self.grid_size)
            self.gt_grid[ox:x_end, oy:y_end, oz:z_end] = 1

        # B. 墙前面的障碍物 (已知，需要避障)
        num_obs_front = np.random.randint(0, 3)
        for _ in range(num_obs_front):
            ox = np.random.randint(2, max(3, wall_x-2))
            oy = np.random.randint(8, 24)
            oz = np.random.randint(8, 24)
            x_end = min(ox+2, self.grid_size)
            y_end = min(oy+2, self.grid_size)
            z_end = min(oz+2, self.grid_size)
            self.gt_grid[ox:x_end, oy:y_end, oz:z_end] = 1

        # 4. 构建 Known Mask (安全区)
        # 逻辑: 非未知区域 (Known) 且 非障碍物 (Free)
        self.known_mask = (~self.unknown_mask) & (self.gt_grid == 0)
        
        # Agent 只能看到安全区内的障碍物
        self.known_obstacles = self.gt_grid.copy()
        self.known_obstacles[self.unknown_mask] = 0 

        return self._get_observation()

    def _get_observation(self):
        # 1. 对齐前沿 (PCA Yaw)
        aligned_ft, center, yaw = self.processor.align_cluster_horizontal(self.frontier_pts)
        self.current_center = center
        self.current_yaw = yaw
        
        # 预计算旋转矩阵 (Global -> Local)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # 2. 对齐障碍物
        obs_indices = np.argwhere(self.known_obstacles == 1)
        aligned_obs = []
        if len(obs_indices) > 0:
            centered_obs = obs_indices - center
            aligned_obs = centered_obs @ R.T

        # 3. 对齐已知区域 (Known Mask) -> Channel 2
        known_indices = np.argwhere(self.known_mask)
        aligned_known = []
        if len(known_indices) > 0:
            centered_known = known_indices - center
            aligned_known = centered_known @ R.T

        # 4. 体素化 (3 Channels)
        state = self.processor.voxelize_aligned_cluster(aligned_ft, aligned_obs, aligned_known)
        return state

    def step(self, action):
        # 1. 解析动作 (Local -> Global)
        scale = 12.0 # 动作范围缩放
        lx, ly, lz = action[0]*scale, action[1]*scale, action[2]*scale
        local_yaw_off = action[3] * np.pi 
        
        c, s = np.cos(self.current_yaw), np.sin(self.current_yaw)
        gx = lx * c - ly * s
        gy = lx * s + ly * c
        gz = lz
        
        target_pos = self.current_center + np.array([gx, gy, gz])
        target_yaw = self.current_yaw + local_yaw_off
        
        ix, iy, iz = int(target_pos[0]), int(target_pos[1]), int(target_pos[2])
        
        # --- 2. 安全性检查 (软约束) ---
        
        # A. 越界检查
        if not (0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size):
             return self._get_observation(), -0.5, True, {'status': 'out_of_bounds'}
             
        # B. 碰撞检查 (撞已知障碍物)
        if self.known_obstacles[ix, iy, iz] == 1:
             return self._get_observation(), -2.0, True, {'status': 'collision'}
             
        # C. 未知区域检查 (飞出安全区)
        # 如果当前位置不在 Known Mask 内，视为极度危险
        if not self.known_mask[ix, iy, iz]:
             return self._get_observation(), -2.0, True, {'status': 'unsafe_area'}

        # --- 3. 合法位置：计算收益 ---
        
        # Raycast 扫描
        covered_set = self.raycaster.scan(self.gt_grid, target_pos, target_yaw)
        
        surface_hits = 0
        volume_hits = 0
        
        for p in covered_set:
            # 收益 A: 覆盖前沿表面 (Surface Coverage) - 权重高
            if p in self.frontier_set:
                surface_hits += 1
            # 收益 B: 穿透未知体积 (Volume Penetration) - 权重低
            elif self.unknown_mask[p]:
                volume_hits += 1
        
        # 原始奖励值
        raw_reward = (surface_hits * 1.0) + (volume_hits * 0.05)
        
        # 距离惩罚 (正则项，防止飞太远)
        dist_penalty = 0.005 * np.linalg.norm([lx, ly, lz])
        
        # 【关键】奖励缩放：将几百的分数缩放到个位数，防止梯度爆炸
        # 例如: hits=200 -> reward=2.0. 这样与惩罚 -2.0 量级相当
        reward = (raw_reward * 0.01) - dist_penalty
        
        return self._get_observation(), reward, True, {'status': 'success'}