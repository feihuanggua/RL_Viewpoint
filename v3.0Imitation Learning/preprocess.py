import os
import glob
import numpy as np
import torch
from tqdm import tqdm

# === 配置参数 ===
CONFIG = {
    "raw_data_dir": r"C:\Users\96290\Desktop\reinforce\v3.0Imitation Learning\fuel_dataset",  # C++ 生成的 txt 根目录
    "output_path": "./processed_data.pt",         # 输出文件路径
    
    # 必须与 C++ saveTrainingData 保持一致
    "res": 0.2,
    "roi_xy": 3.2,  # [-3.2, 3.2] -> 6.4m -> 32 grids
    "roi_z": 1.0,   # [-1.0, 1.0] -> 2.0m -> 10 grids
    "target_dim": (32, 32, 10)
}

def parse_txt_file(path):
    center = None
    label_raw = None
    points = []
    
    # 从文件名解析 ID (可选)
    file_id = os.path.basename(path).split('_')[-1].replace('.txt', '')
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    mode = "META"
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line == "VOXEL_DATA_START":
            mode = "DATA"
            continue
        if line == "META_END" or line == "VOXEL_DATA_END":
            continue

        parts = line.split()
        if mode == "META":
            if parts[0] == "CENTER":
                center = np.array([float(x) for x in parts[1:4]])
            elif parts[0] == "LABEL":
                # x, y, z, yaw
                label_raw = np.array([float(x) for x in parts[1:5]])
        elif mode == "DATA":
            if len(parts) == 4:
                # x, y, z, type
                points.append([float(x) for x in parts])
                
    return center, label_raw, np.array(points), file_id

def convert_to_tensor(center, label_raw, points, cfg):
    # 1. 准备空 Tensor [3, 32, 32, 10]
    # Channel 0: Obstacle, 1: Frontier, 2: Free
    dim_x, dim_y, dim_z = cfg['target_dim']
    grid = torch.zeros((3, dim_x, dim_y, dim_z), dtype=torch.bool) # 使用 bool 节省空间
    
    if len(points) == 0 or center is None:
        return None, None

    # 2. 坐标变换: Global -> Local -> Grid Index
    local_pts = points[:, :3] - center
    
    # 计算索引 (加偏移 -> 除分辨率 -> 取整)
    # XY: [-3.2, 3.2] -> [0, 6.4] -> [0, 32]
    # Z:  [-1.0, 1.0] -> [0, 2.0] -> [0, 10]
    idx_x = np.floor((local_pts[:, 0] + cfg['roi_xy']) / cfg['res']).astype(int)
    idx_y = np.floor((local_pts[:, 1] + cfg['roi_xy']) / cfg['res']).astype(int)
    idx_z = np.floor((local_pts[:, 2] + cfg['roi_z']) / cfg['res']).astype(int)
    
    # 边界过滤
    valid_mask = (
        (idx_x >= 0) & (idx_x < dim_x) &
        (idx_y >= 0) & (idx_y < dim_y) &
        (idx_z >= 0) & (idx_z < dim_z)
    )
    
    if not np.any(valid_mask):
        return None, None
        
    ix = idx_x[valid_mask]
    iy = idx_y[valid_mask]
    iz = idx_z[valid_mask]
    types = points[valid_mask, 3].astype(int)
    
    # 填充 Grid
    # Obs (Type 1) -> Ch 0
    grid[0, ix[types==1], iy[types==1], iz[types==1]] = True
    # Frontier (Type 3) -> Ch 1
    grid[1, ix[types==3], iy[types==3], iz[types==3]] = True
    # Free (Type 2) -> Ch 2
    grid[2, ix[types==2], iy[types==2], iz[types==2]] = True
    
    # 3. 处理 Label
    if label_raw is not None:
        local_label = label_raw[:3] - center
        # 归一化到 [-1, 1]
        norm_x = local_label[0] / cfg['roi_xy']
        norm_y = local_label[1] / cfg['roi_xy']
        norm_z = local_label[2] / cfg['roi_z']
        norm_yaw = label_raw[3] / np.pi
        
        target = torch.tensor([norm_x, norm_y, norm_z, norm_yaw], dtype=torch.float32)
    else:
        target = torch.zeros(4, dtype=torch.float32)
        
    # 将 grid 转回 float 以供训练 (或保持 bool 在 Dataset 中转)
    # 这里为了文件体积，建议保存为 bool 或 uint8
    return grid, target

def main():
    files = glob.glob(os.path.join(CONFIG['raw_data_dir'], "**", "*.txt"), recursive=True)
    print(f"Found {len(files)} raw files.")
    
    data_list = []
    target_list = []
    
    skipped = 0
    
    for fpath in tqdm(files, desc="Processing"):
        try:
            center, label, points, fid = parse_txt_file(fpath)
            grid, target = convert_to_tensor(center, label, points, CONFIG)
            
            if grid is not None:
                # 压缩：将 bool 转为 pack bits 可以极致压缩，但这里先存 uint8 够用了
                # 或者直接存 list，最后 stack
                data_list.append(grid) # Keep as bool tensor
                target_list.append(target)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            skipped += 1

    print(f"Conversion done. Valid: {len(data_list)}, Skipped: {skipped}")
    
    if len(data_list) > 0:
        # Stack 可能会爆内存如果数据量极大(>10万)。如果爆内存，请改用分块保存。
        # 32*32*10*3 bytes * 10000 ~ 300MB, 没问题。
        print("Stacking tensors...")
        all_data = torch.stack(data_list)    # [N, 3, 32, 32, 10] (bool)
        all_targets = torch.stack(target_list) # [N, 4] (float32)
        
        print(f"Saving to {CONFIG['output_path']}...")
        torch.save({
            "inputs": all_data, 
            "targets": all_targets,
            "config": CONFIG
        }, CONFIG['output_path'])
        print("Done!")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()