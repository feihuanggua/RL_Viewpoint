import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
import time # 确保导入了 time 模块

# 导入模块 (确保 dataset.py 和 model.py 在同一目录下)
from dataset import FUELAugmentedDataset
from model import ExplorationActorCritic

# === 配置 ===
CONFIG = {
    "data_path": "./processed_data.pt",
    "save_dir": "./checkpoints",
    "result_path": "./inference_results.pt", # 可视化结果保存路径
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 100,
    "num_workers": 4, # Windows 系统如果报错请改为 0
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def save_visualization_data(model, val_loader, save_path, device):
    """
    运行推理并保存 Input, Target, Prediction 用于后续可视化
    """
    print(f"\n[Info] Generating visualization results to {save_path}...")
    model.eval()
    
    inputs_list = []
    targets_list = []
    preds_list = []
    
    # 只取前 2 个 batch 就够看了，不用全部保存
    max_batches = 2
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= max_batches: break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds, _ = model(inputs)
            
            # 转回 CPU 保存
            inputs_list.append(inputs.cpu())
            targets_list.append(targets.cpu())
            preds_list.append(preds.cpu())

    if len(inputs_list) > 0:
        # 获取原始 Dataset 对象以访问配置
        # val_loader.dataset 是一个 Subset，.dataset 才是原始的 FUELAugmentedDataset
        raw_dataset = val_loader.dataset.dataset
        
        data = {
            'inputs': torch.cat(inputs_list),
            'targets': torch.cat(targets_list),
            'preds': torch.cat(preds_list),
            'config': raw_dataset.config # 获取原始配置
        }
        torch.save(data, save_path)
        print(f"[Info] Saved {len(data['inputs'])} samples for visualization.")

def train(args):
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    print(f"Using device: {CONFIG['device']}")

    # 1. 加载增强数据集
    if not os.path.exists(CONFIG["data_path"]):
        print(f"Error: Data {CONFIG['data_path']} not found.")
        return

    # 注意：device="cpu" 只是把数据加载到内存，训练时再 to(cuda)
    full_dataset = FUELAugmentedDataset(CONFIG["data_path"], device="cpu")
    
    # 划分验证集 (保持随机种子以便复现)
    generator = torch.Generator().manual_seed(42)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    # 验证集不 Shuffle，方便对比
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, 
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # 2. 初始化模型
    model = ExplorationActorCritic().to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    # 3. 断点续训逻辑
    if args.resume:
        ckpt_path = args.resume if args.resume != "latest" else os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth")
        if os.path.exists(ckpt_path):
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=CONFIG["device"])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('loss', float('inf'))
            # 尝试加载历史 loss 数据（如果之前保存过）
            history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
            print(f"[Resume] Resuming from epoch {start_epoch}")
        else:
            print(f"[Resume] Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    # 4. 训练循环 (带异常捕获)
    try:
        for epoch in range(start_epoch, CONFIG["epochs"]):
            model.train()
            train_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
            for inputs, targets in loop:
                inputs = inputs.to(CONFIG["device"])
                targets = targets.to(CONFIG["device"])
                
                optimizer.zero_grad()
                pred_actions, _ = model(inputs)
                loss = criterion(pred_actions, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(CONFIG["device"])
                    targets = targets.to(CONFIG["device"])
                    pred_actions, _ = model(inputs)
                    loss = criterion(pred_actions, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            print(f"Summary Ep {epoch+1}: Train {avg_train_loss:.6f} | Val {avg_val_loss:.6f}")
            
            # 保存 Latest Checkpoint (用于续训，加入 history)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'history': history # 保存历史 Loss 数据
            }, os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth"))

            # 保存 Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
                print("--> Best Model Saved!")

    except KeyboardInterrupt:
        print("\n[Info] Training paused by user (Ctrl+C). Saving checkpoint...")
        # 保存中断时的状态
        torch.save({
            'epoch': epoch if 'epoch' in locals() else start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss if 'avg_val_loss' in locals() else best_val_loss,
            'history': history
        }, os.path.join(CONFIG["save_dir"], "interrupted_checkpoint.pth"))
        print("[Info] Progress saved to 'interrupted_checkpoint.pth'.")
    
    finally:
        # --- 无论正常结束还是被中断，都执行以下操作 ---
        
        # 1. 保存可视化结果
        save_visualization_data(model, val_loader, CONFIG["result_path"], CONFIG["device"])
        
        # 2. 绘制并保存带时间戳的 Loss 曲线
        if len(history['train_loss']) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Imitation Learning Training Curve')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            
            # 生成时间戳字符串 (例如: 20231027_153045)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_filename = f"loss_curve_{timestamp}.png"
            plot_path = os.path.join(CONFIG["save_dir"], plot_filename)
            
            plt.savefig(plot_path)
            print(f"[Info] Loss curve saved to {plot_path}")
            plt.close() # 关闭图表释放内存

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 用法: python train_bc.py --resume (默认找 latest)
    # 用法: python train_bc.py --resume ./checkpoints/interrupted_checkpoint.pth
    parser.add_argument('--resume', nargs='?', const='latest', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(args)