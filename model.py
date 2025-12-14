import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VoxNetFeatureExtractor(nn.Module):
    """
    VoxNet 风格的 3D CNN 特征提取器
    输入: [Batch, 1, D, H, W] (例如 1x32x32x32 的局部体素)
    输出: Flattened Feature Vector
    """
    def __init__(self, output_dim=128):
        super(VoxNetFeatureExtractor, self).__init__()
        # 3D 卷积层: Conv3d(in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2) 
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d(2)
        
        # 计算 Flatten 后的维度 (假设输入是 32x32x32)
        # 32 -> (5,2) -> 14 -> (3,1) -> 12 -> pool(2) -> 6
        # 最终特征图大小: 32 * 6 * 6 * 6
        self.flat_dim = 32 * 6 * 6 * 6
        self.fc = nn.Linear(self.flat_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        return x

class ActorCritic(nn.Module):
    def __init__(self, action_dim=4, state_vec_dim=3):
        super(ActorCritic, self).__init__()
        
        # 1. 3D 特征提取器 (处理地图)
        self.vox_net = VoxNetFeatureExtractor(output_dim=128)
        
        # 2. 状态融合层 (地图特征 + 无人机相对位置向量)
        self.fusion_dim = 128 + state_vec_dim
        self.common_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # 3. Actor Head (输出动作均值) -> x, y, z, yaw
        self.actor_mean = nn.Linear(64, action_dim)
        # 动作的对数标准差 (可学习参数)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # 4. Critic Head (输出价值)
        self.critic = nn.Linear(64, 1)

    def forward(self, voxel_map, robot_vec):
        """
        voxel_map: [B, 1, 32, 32, 32]
        robot_vec: [B, 3] (无人机相对于前沿中心的归一化位置)
        """
        map_feat = self.vox_net(voxel_map)
        
        # 拼接特征
        combined = torch.cat([map_feat, robot_vec], dim=1)
        x = self.common_mlp(combined)
        
        # Actor
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic
        value = self.critic(x)
        
        return action_mean, action_std, value