import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from config import Config
from env import PenetrationEnv
from model import ActorCritic

class ResultVisualizer:
    def __init__(self):
        self.cfg = Config()
        self.grid_size = 32
        
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

        self.policy = ActorCritic().to(self.cfg.DEVICE)
        self.model_loaded = False
        try:
            # 尝试加载模型
            checkpoint = torch.load(self.cfg.CKPT_PATH, map_location=self.cfg.DEVICE, weights_only=False)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.eval()
            print(f"Loaded Model Ep: {checkpoint['episode']}")
            self.model_loaded = True
        except Exception as e:
            print(f"Using Random Policy (Reason: {e})")

    def list_to_grid(self, points, shape):
        grid = np.zeros(shape, dtype=bool)
        if len(points) > 0:
            pts = np.array(list(points))
            mask = (pts[:,0]>=0) & (pts[:,0]<shape[0]) & \
                   (pts[:,1]>=0) & (pts[:,1]<shape[1]) & \
                   (pts[:,2]>=0) & (pts[:,2]<shape[2])
            pts = pts[mask]
            if len(pts) > 0:
                grid[pts[:,0], pts[:,1], pts[:,2]] = True
        return grid

    def render_episode(self, env):
        state = env.reset()
        
        # Get Action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.cfg.DEVICE)
        with torch.no_grad():
            if self.model_loaded:
                action_mean, _, _ = self.policy(state_tensor)
                action = action_mean.cpu().numpy()[0]
            else:
                action = env.action_space.sample()

        # Parse Action
        scale = 12.0
        lx, ly, lz = action[0]*scale, action[1]*scale, action[2]*scale
        local_yaw_off = action[3] * np.pi
        
        c, s = np.cos(env.current_yaw), np.sin(env.current_yaw)
        gx = lx * c - ly * s
        gy = lx * s + ly * c
        gz = lz
        
        target_pos = env.current_center + np.array([gx, gy, gz])
        target_yaw = env.current_yaw + local_yaw_off
        
        ix, iy, iz = int(target_pos[0]), int(target_pos[1]), int(target_pos[2])
        
        # --- 判定合法性 ---
        is_valid = False
        status_msg = "Unknown"
        if not (0 <= ix < 32 and 0 <= iy < 32 and 0 <= iz < 32):
            status_msg = "Out of Bounds"
        elif env.known_obstacles[ix, iy, iz] == 1:
            status_msg = "Collision (Known Obs)"
        elif not env.known_mask[ix, iy, iz]:
            status_msg = "Invalid (Unknown Area)"
        else:
            is_valid = True
            status_msg = "Valid"

        # --- 核心计算 ---
        seen_surface_set = set()
        seen_volume_list = [] # 用于存体积点
        
        if is_valid:
            covered_set = env.raycaster.scan(env.gt_grid, target_pos, target_yaw)
            
            for p in covered_set:
                # 1. 检查是否覆盖了墙面
                if p in env.frontier_set:
                    seen_surface_set.add(p)
                
                # 2. 【新增】检查是否穿透到了未知区域 (Volume)
                # 使用 unknown_mask 判定
                if env.unknown_mask[p]:
                    seen_volume_list.append(p)

        # 计算漏掉的墙面
        missed_surface = []
        for p in env.frontier_pts:
            if tuple(p) not in seen_surface_set:
                missed_surface.append(p)
                
        # --- 绘图 ---
        self.ax.clear()
        
        # A. 障碍物
        grid_obs = env.known_obstacles.astype(bool)
        if np.any(grid_obs):
            self.ax.voxels(grid_obs, facecolors='#1f77b430', edgecolors='#1f77b4', linewidth=0.2)

        # B. 墙面 - 看到的 (青色)
        if len(seen_surface_set) > 0:
            grid_seen = self.list_to_grid(seen_surface_set, (32,32,32))
            if np.any(grid_seen):
                self.ax.voxels(grid_seen, facecolors='#00FFFF', edgecolors='#008B8B', alpha=0.9)
        
        # C. 墙面 - 没看到的 (灰色)
        if len(missed_surface) > 0:
            grid_missed = self.list_to_grid(missed_surface, (32,32,32))
            if np.any(grid_missed):
                self.ax.voxels(grid_missed, facecolors='#999999', edgecolors='gray', alpha=0.2)

        # D. 【新增】穿透的体积 (淡金色)
        # 用极高的透明度，防止遮挡前面的墙
        if len(seen_volume_list) > 0:
            grid_vol = self.list_to_grid(seen_volume_list, (32,32,32))
            if np.any(grid_vol):
                self.ax.voxels(grid_vol, facecolors='#FFD700', linewidth=0, alpha=0.15)

        # E. Agent
        cx, cy, cz = target_pos
        agent_color = '#00FF00' if is_valid else '#FF0000'
        marker = '^' if is_valid else 'x'
        
        self.ax.scatter([cx], [cy], [cz], c=agent_color, s=200, marker=marker, zorder=10)
        
        if is_valid:
            arrow_len = 5.0
            self.ax.quiver(cx, cy, cz, np.cos(target_yaw)*arrow_len, np.sin(target_yaw)*arrow_len, 0, color='#00FF00')

        # Title
        total_surf = len(env.frontier_pts)
        seen_count = len(seen_surface_set)
        vol_count = len(seen_volume_list)
        ratio = seen_count / max(1, total_surf) * 100
        
        title_str = f"Status: {status_msg}\nSurface Cov: {seen_count}/{total_surf} ({ratio:.1f}%)\nVolume Penetrated: {vol_count} voxels"
        self.ax.set_title(title_str)
        self.ax.set_xlim(0, 32); self.ax.set_ylim(0, 32); self.ax.set_zlim(0, 32)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#00FFFF', label='Surface (Cyan)'),
            Patch(facecolor='#FFD700', alpha=0.3, label='Volume (Gold)'),
            Patch(facecolor='#999999', alpha=0.3, label='Missed'),
            Patch(facecolor=agent_color, label='Agent')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    env = PenetrationEnv()
    viz = ResultVisualizer()
    print("\n--- 按 [Enter] 键查看下一个测试场景 ---")
    for i in range(50):
        viz.render_episode(env)
        input(f"Case {i+1} Done. [Enter]")