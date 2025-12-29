import gym
import numpy as np
from config import Config
from utils import RayCaster, FrontierProcessor

class PenetrationEnv(gym.Env):
    def __init__(self):
        # ... (初始化部分不变) ...
        self.cfg = Config()
        self.grid_size = self.cfg.GRID_SIZE
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 32, 32, 32), dtype=np.float32)
        
        self.raycaster = RayCaster(map_size=self.grid_size)
        self.processor = FrontierProcessor(grid_size=self.grid_size)
        
        self.gt_grid = None
        self.known_obstacles = None
        self.known_mask = None
        self.frontier_pts = None 
        self.frontier_set = set()
        self.unknown_mask = None
        
        self.current_center = None
        self.current_yaw = None
        
        # 记录上一帧的 Robot 位置，用于辅助判断方向
        self.last_robot_pos = None

    def reset(self):
        # ... (前半部分生成地图、障碍物的逻辑完全不变) ...
        # ... (请保留原有的地图生成代码) ...
        
        # 1. 地图初始化 (简写)
        self.gt_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int8)
        self.unknown_mask = np.zeros_like(self.gt_grid, dtype=bool)
        
        wall_x = np.random.randint(14, 18)
        self.frontier_pts = []
        self.frontier_set = set()
        slope = np.random.uniform(-0.5, 0.5)
        for y in range(6, 26):
            for z in range(6, 26):
                x = int(wall_x + (z - 16) * slope) 
                if 0 <= x < self.grid_size:
                    pt = (x, y, z)
                    self.frontier_pts.append(pt)
                    self.frontier_set.add(pt)
                    if x < self.grid_size: self.unknown_mask[x:, y, z] = True
        self.frontier_pts = np.array(self.frontier_pts)

        # 2. 障碍物 (简写)
        for _ in range(np.random.randint(2, 6)): # Back
            ox, oy, oz = np.random.randint(wall_x+1, 28), np.random.randint(8, 24), np.random.randint(8, 24)
            self.gt_grid[ox:ox+3, oy:oy+3, oz:oz+3] = 1
        for _ in range(np.random.randint(0, 3)): # Front
            ox, oy, oz = np.random.randint(2, max(3, wall_x-2)), np.random.randint(8, 24), np.random.randint(8, 24)
            self.gt_grid[ox:ox+2, oy:oy+2, oz:oz+2] = 1

        # 3. Mask & Robot Pos
        self.known_mask = (~self.unknown_mask) & (self.gt_grid == 0)
        self.known_obstacles = self.gt_grid.copy()
        self.known_obstacles[self.unknown_mask] = 0 
        
        # 4. 设置初始 Robot 位置 (必须在墙前面)
        # 我们假设墙前 10 格是安全的初始点
        self.last_robot_pos = np.array([float(wall_x - 10), 16.0, 16.0])

        return self._get_observation()

    def _get_observation(self):
        # 【修改】传入 robot_pos 以便校正法向量方向
        aligned_ft, center, yaw = self.processor.align_cluster_horizontal(self.frontier_pts, self.last_robot_pos)
        self.current_center = center
        self.current_yaw = yaw
        
        # ... (后续体素化逻辑不变) ...
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        obs_indices = np.argwhere(self.known_obstacles == 1)
        aligned_obs = []
        if len(obs_indices) > 0:
            aligned_obs = (obs_indices - center) @ R.T

        known_indices = np.argwhere(self.known_mask)
        aligned_known = []
        if len(known_indices) > 0:
            aligned_known = (known_indices - center) @ R.T

        state = self.processor.voxelize_aligned_cluster(aligned_ft, aligned_obs, aligned_known)
        return state

    def step(self, action):
        # 1. 解析动作
        scale = 15.0 
        lx, ly, lz = action[0]*scale, action[1]*scale, action[2]*scale
        local_yaw_off = action[3] * np.pi 
        
        # ... (坐标转换逻辑不变) ...
        c, s = np.cos(self.current_yaw), np.sin(self.current_yaw)
        gx = lx * c - ly * s
        gy = lx * s + ly * c
        gz = lz
        
        target_pos = self.current_center + np.array([gx, gy, gz])
        target_yaw = self.current_yaw + local_yaw_off
        
        ix, iy, iz = int(target_pos[0]), int(target_pos[1]), int(target_pos[2])
        
        # --- 2. 关键修改：正面探索约束 ---
        
        # 计算 Agent 相对于墙中心的向量 (Global Frame)
        vec_to_agent = target_pos - self.current_center
        # 墙的法向量 (Global Frame, 已校正指向墙前)
        wall_normal = np.array([np.cos(self.current_yaw), np.sin(self.current_yaw), 0])
        
        # 投影：判断 Agent 在墙前还是墙后
        # dot > 0: 在墙前 (Known Side) -> Good
        # dot < 0: 在墙后 (Unknown Side) -> Bad
        dist_in_front = np.dot(vec_to_agent, wall_normal)
        
        # 判定：是否在"背后"
        is_behind_wall = dist_in_front < -1.0 # 留1米的容差，防止贴太近误判
        
        # --- 安全性惩罚 ---
        if not (0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size):
             return self._get_observation(), -1.0, True, {'status': 'out_of_bounds'}
        if self.known_obstacles[ix, iy, iz] == 1:
             return self._get_observation(), -2.0, True, {'status': 'collision'}
        if not self.known_mask[ix, iy, iz]:
             return self._get_observation(), -2.0, True, {'status': 'unsafe_area'}
             
        # 【新增】背后偷窥惩罚
        if is_behind_wall:
             # 如果跑到了墙后面，这在实际中是很难发生的（因为是未知区），
             # 而且从后面看前面是没有探索价值的。
             return self._get_observation(), -5.0, True, {'status': 'behind_wall'}

        # --- 3. 奖励计算 ---
        covered_set = self.raycaster.scan(self.gt_grid, target_pos, target_yaw)
        
        surface_hits = 0
        volume_hits = 0
        
        for p in covered_set:
            if p in self.frontier_set:
                surface_hits += 1
            elif self.unknown_mask[p]:
                volume_hits += 1
        
        # 覆盖率奖励
        total_frontiers = max(1, len(self.frontier_pts))
        coverage_ratio = surface_hits / total_frontiers
        r_coverage = coverage_ratio * 5.0
        
        # 体积奖励
        r_volume = volume_hits * 0.005
        
        # 【新增】正视奖励 (Orthogonality Reward)
        # 鼓励视线方向 与 墙面法向 相反 (即：正对着墙看)
        # Agent View Vector
        view_vec = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
        # dot should be close to -1 (antiparallel)
        ortho_score = -np.dot(view_vec, wall_normal) # Range [-1, 1], 1 is best
        # 只有在看到了东西的情况下才给这个奖励
        if surface_hits > 0:
            r_ortho = max(0, ortho_score) * 1.0
        else:
            r_ortho = 0
            
        # 距离引导 (保持在正面 8 米左右)
        r_dist_guide = 0
        if dist_in_front > 0: # 只有在前面才引导距离
            r_dist_guide = 0.5 * np.exp(-((dist_in_front - 8.0)**2) / (2 * 4.0**2))

        # 总奖励
        reward = r_coverage + r_volume + r_ortho + r_dist_guide
        
        return self._get_observation(), reward, True, {'status': 'success'}