import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplorationActorCritic(nn.Module):
    def __init__(self):
        super(ExplorationActorCritic, self).__init__()
        
        # 输入: [Batch, 3, 32, 32, 10]
        # Channel 0: Obstacle, 1: Frontier, 2: Free
        
        # === Backbone: Asymmetric 3D CNN ===
        self.backbone = nn.Sequential(
            # Layer 1: 仅压缩 XY，保留 Z 轴分辨率
            # [3, 32, 32, 10] -> [32, 16, 16, 10]
            nn.Conv3d(3, 32, kernel_size=3, stride=(2, 2, 1), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 2: XYZ 同时压缩
            # [32, 16, 16, 10] -> [64, 8, 8, 5]
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 3: 仅压缩 XY，Z 轴保持 (5层已经很少了，不宜再除)
            # [64, 8, 8, 5] -> [128, 4, 4, 5]
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # Layer 4: 最终压缩
            # [128, 4, 4, 5] -> [256, 2, 2, 3]
            # Z轴计算: (5 + 2*1 - 3)/2 + 1 = 3
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Flatten 维度计算: 256 * 2 * 2 * 3 = 3072
        self.feature_dim = 256 * 2 * 2 * 3
        
        # === Shared Embedding ===
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.1) # 防止过拟合
        )
        
        # === Actor Head (Action) ===
        # 输出: [x, y, z, yaw] (范围 -1 ~ 1)
        self.actor_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 4),
            nn.Tanh() # 关键: 强制映射到 [-1, 1]
        )
        
        # === Critic Head (Value) ===
        # PPO 训练用，BC 阶段仅作为占位或辅助 Loss
        self.critic_net = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1)
        )
        
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [B, 3, 32, 32, 10]
        features = self.backbone(x)
        embed = self.embedding(features)
        
        action = self.actor_net(embed)
        value = self.critic_net(embed)
        
        return action, value

if __name__ == "__main__":
    # 维度测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplorationActorCritic().to(device)
    dummy_in = torch.randn(2, 3, 32, 32, 10).to(device)
    act, val = model(dummy_in)
    print(f"Input: {dummy_in.shape}")
    print(f"Action: {act.shape}") # 应为 [2, 4]
    print(f"Value: {val.shape}")  # 应为 [2, 1]