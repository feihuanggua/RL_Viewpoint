import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_curves(log_file="training_log.csv"):
    if not os.path.exists(log_file):
        print(f"错误: 找不到日志文件 {log_file}。请先运行 train.py 进行训练。")
        return

    # 读取数据
    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    episodes = data['episode']
    rewards = data['reward']
    losses = data['loss']

    # 计算 Reward 的移动平均线，让曲线更平滑易读
    window_size = 50
    reward_smooth = rewards.rolling(window=window_size, min_periods=1).mean()

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- 绘制 Reward ---
    ax1.plot(episodes, rewards, alpha=0.3, color='gray', label='Raw Reward')
    ax1.plot(episodes, reward_smooth, color='blue', linewidth=2, label=f'Avg Reward (win={window_size})')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 绘制 Loss ---
    # 由于 Loss 只在更新时变化，且可能为 0，我们过滤一下非零点或者直接画
    # 这里直接画，通常 Loss 曲线在 PPO 中波动较大是正常的
    ax2.plot(episodes, losses, color='red', alpha=0.8, linewidth=1, label='PPO Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 保存图片
    plt.savefig('training_curves.png')
    print("曲线图已保存至 training_curves.png")
    plt.show()

if __name__ == "__main__":
    # 确保安装了 pandas: pip install pandas
    plot_curves()