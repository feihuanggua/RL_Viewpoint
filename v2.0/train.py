import os
import torch
import torch.optim as optim
import numpy as np
import csv  # <--- 新增
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
    
    # --- 智能参数分组 ---
    # A. Critic 参数
    critic_modules = [policy.critic_net]
    critic_params = []
    for m in critic_modules:
        critic_params += list(m.parameters())
    critic_ids = list(map(id, critic_params))
    
    # B. Actor 参数
    actor_modules = [
        policy.pos_net, policy.actor_pos_mean,
        policy.yaw_net, policy.actor_yaw_mean
    ]
    actor_params = []
    for m in actor_modules:
        actor_params += list(m.parameters())
    actor_params.append(policy.actor_log_std)
    actor_ids = list(map(id, actor_params))
    
    # C. Backbone 参数
    backbone_params = filter(lambda p: id(p) not in critic_ids + actor_ids, policy.parameters())
    
    # 定义优化器
    optimizer = optim.Adam([
        {'params': actor_params, 'lr': cfg.LR_ACTOR},
        {'params': critic_params, 'lr': cfg.LR_CRITIC},
        {'params': backbone_params, 'lr': cfg.LR_CRITIC}
    ])
    
    start_episode = 0
    
    # 2. 断点续训
    if cfg.LOAD_MODEL and os.path.exists(cfg.CKPT_PATH):
        try:
            print(f"发现检查点 {cfg.CKPT_PATH}，正在加载...")
            checkpoint = torch.load(cfg.CKPT_PATH, map_location=cfg.DEVICE, weights_only=False)
            policy.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            print(f"成功加载，从 Episode {start_episode} 继续训练。")
        except Exception as e:
            print(f"加载失败 ({e})，将重新开始训练。")
    else:
        print("重新开始训练。")

    # --- 初始化日志记录 ---
    log_filename = "training_log.csv"
    file_exists = os.path.isfile(log_filename)
    # 如果是重新开始(start_episode=0)，则覆盖模式'w'，否则追加模式'a'
    mode = 'a' if file_exists and start_episode > 0 else 'w'
    
    log_file = open(log_filename, mode, newline='')
    writer = csv.writer(log_file)
    
    # 如果是新文件，写入表头
    if mode == 'w':
        writer.writerow(['episode', 'reward', 'loss'])
    
    current_loss = 0.0 # 用于记录最近一次的 loss

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
            if (i_episode % cfg.UPDATE_TIMESTEP == 0) and (i_episode > 0):
                print(f"Episode {i_episode}: Updating PPO...")
                # 获取 update 返回的 loss
                current_loss = update_ppo(policy, optimizer, buffer, cfg)
                buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
            
            # --- Logging & Saving ---
            # 每一回合都记录 Reward，Loss 记录最近一次更新的值
            writer.writerow([i_episode, current_ep_reward, current_loss])
            if i_episode % 10 == 0:
                log_file.flush() # 刷新缓冲区，确保数据写入磁盘
                print(f"Ep {i_episode} | Reward: {current_ep_reward:.3f} | Loss: {current_loss:.4f}")
                
            if i_episode % cfg.SAVE_INTERVAL == 0:
                save_checkpoint(policy, optimizer, i_episode, cfg.CKPT_PATH)

    except KeyboardInterrupt:
        print("\n检测到中断 (Ctrl+C)！正在保存当前模型参数...")
        save_checkpoint(policy, optimizer, i_episode, cfg.CKPT_PATH)
        log_file.close()
        print("保存成功，程序退出。")
    finally:
        if not log_file.closed:
            log_file.close()

def update_ppo(policy, optimizer, buffer, cfg):
    if len(buffer['states']) == 0: return 0.0

    old_states = torch.cat(buffer['states'], dim=0).detach()
    old_actions = torch.stack(buffer['actions']).squeeze(1).detach()
    old_logprobs = torch.stack(buffer['logprobs']).detach()
    rewards = torch.tensor(buffer['rewards'], dtype=torch.float32).to(cfg.DEVICE)
    
    avg_loss = 0
    
    for _ in range(cfg.K_EPOCHS):
        # Evaluate
        logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
        state_values = state_values.squeeze()
        
        # Advantage
        advantages = rewards - state_values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Ratio
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
        
        avg_loss += loss.item()
        
    return avg_loss / cfg.K_EPOCHS # 返回平均 Loss

def save_checkpoint(policy, optimizer, episode, path):
    torch.save({
        'episode': episode,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    train()