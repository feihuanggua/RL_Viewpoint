import numpy as np

class VoxelMap:
    def __init__(self, size=32):
        self.size = size
        # 0: Free, 1: Occupied, -1: Unknown
        # 这里为了简化，假设离线数据里只有 Free(0) 和 Occupied(1)，
        # 我们在这个基础上模拟"未知区域"的发现过程。
        self.grid = np.zeros((size, size, size), dtype=np.int8)

    def load_scenario(self, obstacle_mask):
        """
        加载一个场景（真值地图）
        obstacle_mask: 32x32x32 的 0/1 数组
        """
        self.grid = obstacle_mask.astype(np.int8)

    def is_valid(self, x, y, z):
        return 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size

    def is_occupied(self, x, y, z):
        if not self.is_valid(x, y, z): return True # 出界视为障碍
        return self.grid[x, y, z] == 1

class RayCaster:
    def __init__(self, map_size=32, fov_h=80, fov_v=60, max_range=10.0, resolution=2.0):
        """
        resolution: 射线的密度 (度)
        """
        self.map_size = map_size
        self.max_range = max_range
        
        # 预计算射线的方向向量 (在相机坐标系下)
        # 这种预计算能极大提升 step() 的速度
        h_angles = np.deg2rad(np.linspace(-fov_h/2, fov_h/2, int(fov_h/resolution)))
        v_angles = np.deg2rad(np.linspace(-fov_v/2, fov_v/2, int(fov_v/resolution)))
        
        # 生成所有射线的 (x, y, z) 方向向量
        # 相机坐标系: x前方, y左方, z上方
        self.ray_dirs = []
        for v in v_angles:
            for h in h_angles:
                # 球坐标转笛卡尔坐标
                x = np.cos(v) * np.cos(h)
                y = np.cos(v) * np.sin(h)
                z = np.sin(v)
                self.ray_dirs.append(np.array([x, y, z]))
        self.ray_dirs = np.array(self.ray_dirs) # [N_rays, 3]

    def scan(self, voxel_map, camera_pos, camera_yaw):
        """
        执行一次扫描
        camera_pos: [x, y, z] (体素坐标系，可以是浮点数)
        camera_yaw: 偏航角 (弧度)
        返回:
            hit_points: 击中障碍物的点列表 (用于可视化或安全检查)
            covered_voxels: 这一帧扫描覆盖到的所有体素集合 (set of tuples)
        """
        covered_voxels = set()
        hit_points = []
        
        # 1. 旋转射线向量到世界坐标系
        # 简单的绕 Z 轴旋转矩阵
        c, s = np.cos(camera_yaw), np.sin(camera_yaw)
        rot_matrix = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # 批量旋转所有射线
        world_ray_dirs = self.ray_dirs @ rot_matrix.T
        
        # 2. 射线步进 (Ray Marching)
        # 为了纯 Python 速度，这里用定长步进 (Step Size) 而非精确的 Bresenham
        step_size = 0.8 # 步长小于1以避免穿墙
        
        for ray_dir in world_ray_dirs:
            # 当前射线的位置
            curr_pos = np.array(camera_pos, dtype=np.float32)
            
            dist = 0
            while dist < self.max_range:
                # 离散化为整数坐标
                ix, iy, iz = int(curr_pos[0]), int(curr_pos[1]), int(curr_pos[2])
                
                # 检查是否出界
                if not voxel_map.is_valid(ix, iy, iz):
                    break
                
                # 记录覆盖的体素
                covered_voxels.add((ix, iy, iz))
                
                # 检查碰撞
                if voxel_map.is_occupied(ix, iy, iz):
                    hit_points.append((ix, iy, iz))
                    break # 视线被挡住，停止这条射线
                
                # 移动一步
                curr_pos += ray_dir * step_size
                dist += step_size
                
        return covered_voxels, hit_points