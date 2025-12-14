import gym
from gym import spaces
import numpy as np
import torch
from raycast_utils import VoxelMap, RayCaster  # 导入刚才写的工具

class OfflineFrontierEnv(gym.Env):
    def __init__(self, dataset_path=None, grid_size=32):
        super(OfflineFrontierEnv, self).__init__()
        
        # 动作: dx, dy, dz, dyaw (-1 ~ 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.grid_size = grid_size
        
        # 初始化 RayCaster (模拟 RealSense)
        self.raycaster = RayCaster(map_size=grid_size, max_range=15.0)
        self.gt_map = VoxelMap(size=grid_size)
        
        # 状态管理
        self.current_known_mask = None # 0: Unknown, 1: Known
        self.frontier_center = np.array([16.0, 16.0, 16.0]) # 假设簇中心在正中央

    def reset(self):
        """
        生成/重置一个场景
        """
        # 1. 生成一个新的随机障碍物地图 (Ground Truth)
        # 实际工程中这里应该 load_from_disk
        obs_mask = self._generate_random_obstacles()
        self.gt_map.load_scenario(obs_mask)
        
        # 2. 初始化"已知区域" (Known Mask)
        # 假设一开始我们什么都不知道 (全是 Unknown=0)，或者只知道无人机附近是空的
        self.current_known_mask = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int8)
        
        # 3. 随机放置无人机 (在非障碍物区域)
        self.robot_pos = self._find_safe_start_pos()
        self.robot_yaw = np.random.uniform(0, 2*np.pi)
        
        # 4. 构建 State
        # State = (当前已知的障碍物图, 无人机相对向量)
        # 注意：网络只能看到"已知"的障碍物，看不到未知的
        known_obstacles = self.gt_map.grid * self.current_known_mask
        
        # 归一化位置向量
        robot_vec = (self.robot_pos - self.frontier_center) / float(self.grid_size)
        
        return known_obstacles[np.newaxis, ...], robot_vec

    def step(self, action):
        # 1. 解析动作并更新位姿
        scale_pos = 5.0 # 移动范围
        dx, dy, dz = action[0]*scale_pos, action[1]*scale_pos, action[2]*scale_pos
        dyaw = action[3] * np.pi
        
        # 计算新视点 (在当前位置基础上的偏移，或者是相对于簇中心的绝对位置，这里选用相对于簇中心)
        # 策略：动作直接定义为"目标位置相对于簇中心的偏移"
        target_pos = self.frontier_center + np.array([dx, dy, dz])
        target_yaw = self.robot_yaw + dyaw # 或者直接指定绝对 Yaw
        
        # 2. 检查安全性 (Collision Check)
        ix, iy, iz = int(target_pos[0]), int(target_pos[1]), int(target_pos[2])
        if not self.gt_map.is_valid(ix, iy, iz) or self.gt_map.is_occupied(ix, iy, iz):
            # 撞墙或出界，给大惩罚，结束
            return self._get_state(), -10.0, True, {}
            
        # 3. 执行 Ray-Cast 扫描 (核心逻辑)
        # 计算在这个新位置，能看到哪些体素
        covered_voxels, _ = self.raycaster.scan(self.gt_map, target_pos, target_yaw)
        
        # 4. 计算 Reward (Information Gain)
        new_info_count = 0
        frontier_gain_count = 0
        
        for (vx, vy, vz) in covered_voxels:
            # 如果这个体素之前是未知的 (Known=0)
            if self.current_known_mask[vx, vy, vz] == 0:
                self.current_known_mask[vx, vy, vz] = 1 # 标记为已知
                new_info_count += 1
                
                # 如果这是个前沿附近的空区域（这里简化模拟，假设空的就是好的）
                if self.gt_map.grid[vx, vy, vz] == 0: 
                    frontier_gain_count += 1

        # Reward 设计
        # 鼓励看未知区域，稍微惩罚移动距离
        dist_cost = 0.05 * np.linalg.norm(target_pos - self.robot_pos)
        
        # 如果啥都没看到，给一点点负分，防止懒惰
        reward = (new_info_count * 0.1) + (frontier_gain_count * 0.5) - dist_cost
        
        # 更新位置
        self.robot_pos = target_pos
        self.robot_yaw = target_yaw
        
        # 单步任务：做完一次扫描就结束
        done = True 
        
        return self._get_state(), reward, done, {}

    def _get_state(self):
        # 网络只能看到"已知"部分的地图
        # 输入: Known_Obstacles (0:unknown/free, 1:occupied)
        # 实际上我们可能需要更复杂的编码: 0:free, 1:occupied, -1:unknown
        # 为了适配之前的 VoxNet (输入0/1)，我们可以把 Unknown 当作 Free 处理，或者用多通道
        # 这里简化：只返回已知障碍物
        known_obs = self.gt_map.grid * self.current_known_mask
        robot_vec = (self.robot_pos - self.frontier_center) / float(self.grid_size)
        return known_obs[np.newaxis, ...], robot_vec

    def _generate_random_obstacles(self):
        # 生成一些随机块作为障碍物
        obs = np.zeros((32, 32, 32))
        num_obstacles = np.random.randint(3, 8)
        for _ in range(num_obstacles):
            cx, cy, cz = np.random.randint(0, 32, 3), np.random.randint(0, 32, 3), np.random.randint(0, 32, 3)
            w, h, d = np.random.randint(2, 8, 3)
            # 简单的矩形块
            x_end = min(cx[0]+w, 32)
            y_end = min(cy[1]+h, 32)
            z_end = min(cz[2]+d, 32)
            obs[cx[0]:x_end, cy[1]:y_end, cz[2]:z_end] = 1
        return obs

    def _find_safe_start_pos(self):
        # 随机找一个空闲位置
        while True:
            pos = np.random.randint(0, 32, 3).astype(np.float32)
            if not self.gt_map.is_occupied(int(pos[0]), int(pos[1]), int(pos[2])):
                return pos