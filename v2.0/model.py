import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 模块 1: 3D 空间注意力模块 (部署优化版) ---
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        # 将通道压缩为 2 (Max + Avg)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, D, H, W]
        # 沿着通道维度做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积 + Sigmoid
        weight_map = self.sigmoid(self.conv1(x_cat))
        return x * weight_map

# --- 模块 2: 通道注意力模块 (部署优化版) ---
class ChannelAttention3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention3D, self).__init__()
        # [部署优化] 移除 AdaptiveAvgPool3d，改用 torch.mean
        # self.avg_pool = nn.AdaptiveAvgPool3d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # [部署优化] 手动计算 Global Average Pooling
        # dim=(2,3,4) 对应 D, H, W 维度
        y = x.mean(dim=(2, 3, 4)).view(b, c)
        
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# --- 模块 3: 残差全连接块 ---
class ResMlpBlock(nn.Module):
    def __init__(self, dim):
        super(ResMlpBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual # Residual connection
        out = self.act(out)
        return out

# --- 主模型: Attention-Enhanced Actor-Critic ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim=4):
        super(ActorCritic, self).__init__()
        
        # === 1. Backbone (特征提取) ===
        # Input: [B, 3, 32, 32, 32] (3通道: Obs, Frontier, Known)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=5, stride=2, padding=2) # -> 32 x 16^3
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1) # -> 64 x 8^3
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1) # -> 128 x 4^3
        self.bn3 = nn.BatchNorm3d(128)
        
        # === 2. Attention Modules (特征增强) ===
        self.channel_att = ChannelAttention3D(128)
        self.spatial_att = SpatialAttention3D()
        
        # Flatten dim: 128 * 4 * 4 * 4 = 8192
        self.flat_dim = 128 * 4 * 4 * 4
        
        # Bottleneck Embedding
        self.fc_embed = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU()
        )
        
        # === 3. Decoupled Heads (解耦决策头) ===
        
        # --- Stream A: Position Head ---
        self.pos_net = nn.Sequential(
            ResMlpBlock(512),
            ResMlpBlock(512)
        )
        self.actor_pos_mean = nn.Linear(512, 3) # x, y, z
        
        # --- Stream B: Orientation Head ---
        # 接收 feature + pos 作为输入
        self.yaw_net = nn.Sequential(
            nn.Linear(512 + 3, 256), 
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            ResMlpBlock(256)
        )
        self.actor_yaw_mean = nn.Linear(256, 1) # yaw
        
        # Log Std (Parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # --- Critic Head ---
        self.critic_net = nn.Sequential(
            ResMlpBlock(512),
            ResMlpBlock(512),
            nn.Linear(512, 1)
        )
        
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Backbone
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        # 2. Attention
        x = self.channel_att(x)
        x = self.spatial_att(x)
        
        # 3. Flatten
        x = x.view(x.size(0), -1)
        shared_feat = self.fc_embed(x)
        
        # 4. Decoupled Actor
        # 4.1 Pos
        pos_feat = self.pos_net(shared_feat)
        pos_mean = torch.tanh(self.actor_pos_mean(pos_feat))
        
        # 4.2 Yaw (Conditioned on Pos)
        yaw_input = torch.cat([shared_feat, pos_mean.detach()], dim=1)
        yaw_feat = self.yaw_net(yaw_input)
        yaw_mean = torch.tanh(self.actor_yaw_mean(yaw_feat))
        
        # Combine
        action_mean = torch.cat([pos_mean, yaw_mean], dim=1)
        
        # Std
        log_std = self.actor_log_std.expand_as(action_mean)
        std = torch.exp(log_std)
        
        # 5. Critic
        val_feat = self.critic_net(shared_feat)
        
        return action_mean, std, val_feat

    def evaluate(self, state, action):
        mean, std, val = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)
        
        return action_logprobs, val, dist_entropy