import os
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from config import Config
from env import PenetrationEnv
from model import ActorCritic

def train():
    cfg = Config()
    print(f"Running on {cfg.DEVICE}")
    
    # 1. 初始化
    env = PenetrationEnv()
    policy = ActorCritic().to(cfg.DEVICE)
    
    # --- 【修复部分】优化器定义 ---
    # 1. 提取 Actor 独有的参数 (Head)
    actor_params = list(policy.actor_mean.parameters()) + [policy.actor_log_std]
    actor_param_ids = list(map(id, actor_params))
    
    # 2. 提取 Critic 独有的参数 (Head)
    critic_params = list(policy.critic_val.parameters())
    critic_param_ids = list(map(id, critic_params))
    
    # 3. 剩下的都是共享层参数 (Backbone: Conv, BN, FC_Common)
    # 逻辑: 所有参数 - Actor参数 - Critic参数
    backbone_params = filter(lambda p: id(p) not in actor_param_ids + critic_param_ids, 
                             policy.parameters())
    
    # 4. 定义优化器
    optimizer = optim.Adam([
        {'params': actor_params, 'lr': cfg.LR_ACTOR},      # Actor Head
        {'params': critic_params, 'lr': cfg.LR_CRITIC},    # Critic Head
        {'params': backbone_params, 'lr': cfg.LR_CRITIC}   # Backbone (共享层用 Critic 的学习率)
    ])
    
    start_episode = 0
    
    # 2. 断点续训 (Load Checkpoint)
    if cfg.LOAD_MODEL and os.path.exists(cfg.CKPT_PATH):
        print(f"发现检查点 {cfg.CKPT_PATH}，正在加载...")
        checkpoint = torch.load(cfg.CKPT_PATH, map_location=cfg.DEVICE)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f"成功加载，从 Episode {start_episode} 继续训练。")
    else:
        print("未发现检查点或设置为不加载，重新开始训练。")

    # PPO Buffer
    buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
    
    # 3. 训练循环
    try:
        for i_episode in range(start_episode, cfg.MAX_EPISODES + 1):
            
            state = env.reset()
            current_ep_reward = 0
            
            # --- Collection Phase ---
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(cfg.DEVICE)
            
            with torch.no_grad():
                mean, std, _ = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                action_logprob = dist.log_prob(action).sum()
            
            # Interact
            action_np = action.cpu().numpy()[0]
            _, reward, done, _ = env.step(action_np)
            
            # Store
            buffer['states'].append(state_tensor) 
            buffer['actions'].append(action)
            buffer['logprobs'].append(action_logprob)
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)
            
            current_ep_reward = reward
            
            # --- Update Phase ---
            # 只有当 Buffer 攒够了 UPDATE_TIMESTEP 个样本才更新
            # 注意：因为你的 env 是单步任务，每个 episode 只有 1 个 step
            # 所以这里 i_episode 实际上就是样本计数
            if (i_episode % cfg.UPDATE_TIMESTEP == 0) and (i_episode > 0):
                print(f"Episode {i_episode}: Updating PPO...")
                update_ppo(policy, optimizer, buffer, cfg)
                # Clear buffer
                buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
            
            # --- Logging & Saving ---
            if i_episode % 10 == 0:
                print(f"Ep {i_episode} | Reward: {current_ep_reward:.3f}")
                
            if i_episode % cfg.SAVE_INTERVAL == 0:
                save_checkpoint(policy, optimizer, i_episode, cfg.CKPT_PATH)

    except KeyboardInterrupt:
        print("\n检测到中断 (Ctrl+C)！正在保存当前模型参数...")
        save_checkpoint(policy, optimizer, i_episode, cfg.CKPT_PATH)
        print("保存成功，程序退出。")

def update_ppo(policy, optimizer, buffer, cfg):
    # Stack data
    if len(buffer['states']) == 0: return # 防止空 buffer

    old_states = torch.cat(buffer['states'], dim=0).detach()
    old_actions = torch.stack(buffer['actions']).squeeze(1).detach()
    old_logprobs = torch.stack(buffer['logprobs']).detach()
    rewards = torch.tensor(buffer['rewards'], dtype=torch.float32).to(cfg.DEVICE)
    
    for _ in range(cfg.K_EPOCHS):
        # Evaluate
        logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
        state_values = state_values.squeeze()
        
        # Calculate Advantage
        advantages = rewards - state_values.detach()
        
        # Normalize Advantage
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # PPO Ratio
        ratios = torch.exp(logprobs - old_logprobs)
        
        # Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-cfg.EPS_CLIP, 1+cfg.EPS_CLIP) * advantages
        
        loss = -torch.min(surr1, surr2).mean() + \
               0.5 * torch.nn.functional.mse_loss(state_values, rewards) - \
               0.01 * dist_entropy.mean()
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def save_checkpoint(policy, optimizer, episode, path):
    torch.save({
        'episode': episode,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    train()