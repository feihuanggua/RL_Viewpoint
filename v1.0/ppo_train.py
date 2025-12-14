import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from model import ActorCritic
from env import OfflineFrontierEnv
import torch.nn.functional as F

# --- 超参数 ---
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4        # 每次 update 更新几次网络
BATCH_SIZE = 108
MAX_EPISODES = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """计算广义优势估计 (GAE) - 虽然是单步任务，保持这个结构方便未来扩展"""
    # 针对单步任务 (Contextual Bandit)，GAE 会退化为 R - V(s)
    # 这里简化处理：returns = reward
    returns = []
    for r in rewards:
        returns.append(r)
    return torch.tensor(returns, dtype=torch.float32).to(DEVICE)

def train():
    # 1. 初始化
    env = OfflineFrontierEnv(dataset_path="./data.pkl")
    policy = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    print(f"Start Training on {DEVICE}...")

    # 2. 训练主循环
    for i_episode in range(MAX_EPISODES):
        
        # 存储轨迹数据的 Buffer
        buffer_voxels = []
        buffer_vecs = []
        buffer_actions = []
        buffer_logprobs = []
        buffer_rewards = []
        buffer_values = []
        
        # --- 收集数据 (Rollout) ---
        # 每次收集 BATCH_SIZE 个样本 (单步任务)
        for _ in range(BATCH_SIZE):
            # Reset 环境 (获取新的地图场景)
            voxel, robot_vec = env.reset()
            
            # 转 Tensor
            t_voxel = torch.FloatTensor(voxel).unsqueeze(0).to(DEVICE) # [1, 1, 32, 32, 32]
            t_vec = torch.FloatTensor(robot_vec).unsqueeze(0).to(DEVICE)
            
            # 不需要梯度，只做推理
            with torch.no_grad():
                mu, std, value = policy(t_voxel, t_vec)
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)
            
            # 与环境交互
            action_np = action.cpu().numpy()[0]
            _, reward, done, _ = env.step(action_np)
            
            # 存入 Buffer
            buffer_voxels.append(t_voxel)
            buffer_vecs.append(t_vec)
            buffer_actions.append(action)
            buffer_logprobs.append(log_prob)
            buffer_rewards.append(reward)
            buffer_values.append(value.item())

        # --- PPO 更新 ---
        
        # 整理数据
        # 既然是单步任务，Return 就是 Reward 本身
        old_voxels = torch.cat(buffer_voxels, dim=0).detach()
        old_vecs = torch.cat(buffer_vecs, dim=0).detach()
        old_actions = torch.cat(buffer_actions, dim=0).detach()
        old_logprobs = torch.tensor(buffer_logprobs).to(DEVICE).detach()
        rewards = torch.tensor(buffer_rewards).to(DEVICE).float()
        old_values = torch.tensor(buffer_values).to(DEVICE).float()
        
        # 计算 Advantage (简单版: R - V)
        advantages = rewards - old_values
        
        # PPO 多次 Epoch 更新
        for _ in range(K_EPOCHS):
            # 重新计算分布和 Value
            mu, std, values = policy(old_voxels, old_vecs)
            dist = Normal(mu, std)
            
            # 新的 log_prob 和 entropy
            new_logprobs = dist.log_prob(old_actions).sum(axis=-1)
            dist_entropy = dist.entropy().sum(axis=-1)
            
            # Ratio for clipping
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            # Total Loss = Actor Loss + Critic Loss - Entropy Bonus
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * F.mse_loss(values.squeeze(), rewards) - \
                   0.01 * dist_entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}, Avg Reward: {rewards.mean().item():.2f}")

    # 3. 保存模型
    torch.save(policy.state_dict(), "ppo_viewpoint_generator.pth")
    print("Training Finished & Model Saved.")

if __name__ == '__main__':
    train()