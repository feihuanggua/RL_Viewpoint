import torch

class Config:
    # --- 基础设置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRID_SIZE = 32
    
    # --- 训练超参数 ---
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    K_EPOCHS = 4          # 每次 PPO 更新循环次数
    EPS_CLIP = 0.2        # PPO 裁剪阈值
    
    BATCH_SIZE = 64       # 显存敏感，32^3 的 voxel 很吃显存
    UPDATE_TIMESTEP = 2000 # 每收集多少个样本进行一次更新
    MAX_EPISODES = 100000 # 总训练回合数
    
    # --- 检查点 (Checkpoint) ---
    CKPT_PATH = "ppo_penetration_checkpoint.pth"
    SAVE_INTERVAL = 50    # 每多少个 Episode 保存一次
    LOAD_MODEL = True     # 是否加载已有模型继续训练
    
    # --- 环境设置 ---
    REWARD_SCALE = 0.01   # 缩放 Reward，防止数值过大