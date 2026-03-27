import torch
import numpy as np
from torch.utils.data import Dataset

class FUELAugmentedDataset(Dataset):
    def __init__(self, pt_path, device='cpu'):
        """
        支持 4 倍旋转增强的数据集
        """
        print(f"[Dataset] Loading data from {pt_path} ...")
        # 使用 map_location 节省显存，或者直接加载到 RAM
        data = torch.load(pt_path, map_location=device)
        
        # 原始数据
        self.inputs = data['inputs'].to(dtype=torch.float32) # [N, 3, 32, 32, 10]
        self.targets = data['targets']                       # [N, 4] (x, y, z, yaw)
        self.config = data['config']
        
        self.original_len = len(self.inputs)
        print(f"[Dataset] Original samples: {self.original_len}")
        print(f"[Dataset] Augmented samples: {self.original_len * 4} (Rotation 0, 90, 180, 270)")

    def __len__(self):
        # 长度翻 4 倍
        return self.original_len * 4

    def __getitem__(self, idx):
        # 1. 映射回原始索引
        original_idx = idx // 4
        rot_times = idx % 4  # 0, 1, 2, 3 (分别代表旋转 0, 90, 180, 270 度)
        
        # 2. 获取原始数据
        grid = self.inputs[original_idx]  # [3, X, Y, Z]
        target = self.targets[original_idx] # [x, y, z, yaw]
        
        # 如果不需要旋转，直接返回
        if rot_times == 0:
            return grid, target

        # 3. 数据增强：旋转体素网格
        # grid shape: [Channels, X, Y, Z]
        # 我们需要在 XY 平面旋转，即维度 1 和 2
        # k=rot_times: 旋转 k * 90 度 (逆时针)
        aug_grid = torch.rot90(grid, k=rot_times, dims=[1, 2])
        
        # 4. 数据增强：旋转 Label (x, y, z, yaw)
        # Target 是归一化到 [-1, 1] 的坐标
        x, y, z, yaw = target[0], target[1], target[2], target[3]
        
        # 计算旋转后的 (x, y)
        # 逆时针旋转公式:
        # 90度:  (x, y) -> (-y, x)
        # 180度: (x, y) -> (-x, -y)
        # 270度: (x, y) -> (y, -x)
        
        if rot_times == 1:   # 90 deg
            new_x, new_y = -y, x
        elif rot_times == 2: # 180 deg
            new_x, new_y = -x, -y
        elif rot_times == 3: # 270 deg
            new_x, new_y = y, -x
        else:
            new_x, new_y = x, y
            
        # 计算旋转后的 Yaw
        # yaw 是归一化到 [-1, 1] 的 (对应 -PI ~ PI)
        # 旋转 90度 对应 yaw + 0.5
        new_yaw = yaw + (rot_times * 0.5)
        
        # 处理 Yaw 的周期性 (-1 ~ 1)
        while new_yaw > 1.0:
            new_yaw -= 2.0
        while new_yaw < -1.0:
            new_yaw += 2.0
            
        aug_target = torch.tensor([new_x, new_y, z, new_yaw], dtype=torch.float32, device=target.device)
        
        return aug_grid, aug_target