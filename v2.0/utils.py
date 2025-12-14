import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import torch

# --- 1. RayCaster (CPU 模拟) ---
class RayCaster:
    def __init__(self, map_size=32, fov_h=90, fov_v=60, max_range=20.0, resolution=3.0):
        self.map_size = map_size
        self.max_range = max_range
        
        # 预计算射线方向
        h_angles = np.deg2rad(np.linspace(-fov_h/2, fov_h/2, int(fov_h/resolution)))
        v_angles = np.deg2rad(np.linspace(-fov_v/2, fov_v/2, int(fov_v/resolution)))
        
        self.ray_dirs = []
        for v in v_angles:
            for h in h_angles:
                x = np.cos(v) * np.cos(h)
                y = np.cos(v) * np.sin(h)
                z = np.sin(v)
                self.ray_dirs.append(np.array([x, y, z]))
        self.ray_dirs = np.array(self.ray_dirs)

    def scan(self, obstacle_grid, camera_pos, camera_yaw):
        """
        返回: covered_voxels (set of tuples) - 所有被射线穿过的非障碍物体素
        """
        covered_voxels = set()
        
        # 旋转矩阵 (Yaw only)
        c, s = np.cos(camera_yaw), np.sin(camera_yaw)
        rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        world_ray_dirs = self.ray_dirs @ rot_matrix.T
        
        step_size = 0.8
        
        for ray_dir in world_ray_dirs:
            curr_pos = np.array(camera_pos, dtype=np.float32)
            dist = 0
            while dist < self.max_range:
                ix, iy, iz = int(curr_pos[0]), int(curr_pos[1]), int(curr_pos[2])
                
                # 出界检查
                if not (0 <= ix < self.map_size and 0 <= iy < self.map_size and 0 <= iz < self.map_size):
                    break
                
                # 记录
                covered_voxels.add((ix, iy, iz))
                
                # 障碍物检查 (视线阻挡)
                if obstacle_grid[ix, iy, iz] == 1:
                    break
                
                curr_pos += ray_dir * step_size
                dist += step_size
                
        return covered_voxels

# --- 2. 前沿处理器 (PCA & Voxelization) ---
class FrontierProcessor:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def align_cluster_horizontal(self, cluster_points):
        """
        只在 XY 平面上做 PCA，保留 Z 轴倾斜特征。
        返回: aligned_points (N, 3), center (3,), yaw_angle
        """
        if len(cluster_points) < 3:
            return cluster_points, np.mean(cluster_points, axis=0), 0.0

        center = np.mean(cluster_points, axis=0)
        centered = cluster_points - center
        
        # 只取 XY 做 PCA
        points_2d = centered[:, :2]
        pca = PCA(n_components=2).fit(points_2d)
        
        # 假设最小方差方向是法向 (指向墙外)
        normal_2d = pca.components_[1] 
        yaw = np.arctan2(normal_2d[1], normal_2d[0])
        
        # 逆旋转矩阵 (把墙转正)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        aligned_points = centered @ R.T
        return aligned_points, center, yaw

    def voxelize_aligned_cluster(self, aligned_points, aligned_obstacles, aligned_known_mask_pts=None):
        """
        修改为 3 通道输出:
        Ch0: 障碍物
        Ch1: 前沿面
        Ch2: 已知区域 (1=Known/Safe, 0=Unknown/Danger)
        """
        # grid shape 变为 (3, ...)
        grid = np.zeros((3, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        offset = self.grid_size / 2.0
        
        # Channel 0: 障碍物
        if aligned_obstacles is not None:
             for p in aligned_obstacles:
                ix, iy, iz = int(p[0] + offset), int(p[1] + offset), int(p[2] + offset)
                if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size:
                    grid[0, ix, iy, iz] = 1.0
                    
        # Channel 1: 前沿面
        for p in aligned_points:
            ix, iy, iz = int(p[0] + offset), int(p[1] + offset), int(p[2] + offset)
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size:
                grid[1, ix, iy, iz] = 1.0

        # Channel 2: 已知区域 (Known Space)
        # 输入的 aligned_known_mask_pts 应该是所有 Known=True 的点坐标列表
        if aligned_known_mask_pts is not None:
            for p in aligned_known_mask_pts:
                ix, iy, iz = int(p[0] + offset), int(p[1] + offset), int(p[2] + offset)
                if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size:
                    grid[2, ix, iy, iz] = 1.0
                
        return grid