import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, action_dim=4):
        super(ActorCritic, self).__init__()
        
        # --- 3D CNN Backbone ---
        # Input: [B, 2, 32, 32, 32]
        self.conv1 = nn.Conv3d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2) # -> 32 x 6 x 6 x 6
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1) # -> 64 x 4 x 4 x 4
        self.bn3 = nn.BatchNorm3d(64)
        
        flat_dim = 64 * 4 * 4 * 4
        
        # --- Heads ---
        self.fc_common = nn.Linear(flat_dim, 256)
        
        # Actor
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) # 可学习的标准差
        
        # Critic
        self.critic_val = nn.Linear(256, 1)
        
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, 2, 32, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc_common(x))
        
        # Actor
        mean = torch.tanh(self.actor_mean(x)) # 限制动作在 -1~1
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        # Critic
        val = self.critic_val(x)
        
        return mean, std, val
    
    def evaluate(self, state, action):
        mean, std, val = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)
        
        return action_logprobs, val, dist_entropy