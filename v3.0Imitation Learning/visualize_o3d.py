import open3d as o3d
import torch
import numpy as np
import warnings

# 忽略 PyTorch 警告
warnings.filterwarnings("ignore", category=FutureWarning)

def create_voxel_meshes_fast(grid_indices, color, box_scale=0.95):
    """ [实体构建] 用于障碍物和前沿 """
    if len(grid_indices) == 0: return None, None
    N = len(grid_indices)
    d = box_scale / 2.0
    
    # 标准立方体数据
    base_vertices = np.array([[-d, -d, -d], [ d, -d, -d], [ d,  d, -d], [-d,  d, -d],
                              [-d, -d,  d], [ d, -d,  d], [ d,  d,  d], [-d,  d,  d]])
    base_triangles = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                               [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                               [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    base_lines = np.array([[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4],
                           [0,4], [1,5], [2,6], [3,7]])
    
    # 广播计算
    centers = grid_indices
    all_vertices = (base_vertices[None, :, :] + centers[:, None, :]).reshape(-1, 3)
    offsets = np.arange(N) * 8
    all_triangles = (base_triangles[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
    all_lines = (base_lines[None, :, :] + offsets[:, None, None]).reshape(-1, 2)
    
    # 构建 Mesh & LineSet
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_vertices)
    lineset.lines = o3d.utility.Vector2iVector(all_lines)
    lineset.paint_uniform_color([0.1, 0.1, 0.1])
    
    return mesh, lineset

def create_fog_point_cloud(grid_indices, color):
    """ [点云构建] 用于自由空间 """
    if len(grid_indices) == 0: return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_indices.astype(float))
    pcd.paint_uniform_color(color)
    return pcd

def create_agent_actor(norm_pose, dims, color, scale=1.0):
    """
    创建一个 Agent 组合体 (球 + 箭头)
    norm_pose: [x, y, z, yaw] (归一化值)
    dims: [dim_x, dim_y, dim_z]
    """
    dim_x, dim_y, dim_z = dims
    # 反归一化坐标
    tx = (norm_pose[0] + 1.0) * (dim_x / 2.0)
    ty = (norm_pose[1] + 1.0) * (dim_y / 2.0)
    tz = (norm_pose[2] + 1.0) * (dim_z / 2.0)
    yaw = norm_pose[3] * np.pi
    
    geoms = []
    
    # 1. 位置球
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.6 * scale)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate([tx, ty, tz])
    geoms.append(sphere)
    
    # 2. 朝向箭头
    arrow_len = 3.0 * scale
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.15*scale, cone_radius=0.3*scale, 
        cylinder_height=arrow_len*0.7, cone_height=arrow_len*0.3
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)
    
    # 旋转修正 (Open3D 箭头默认朝 Z -> 转到 X -> 再转 Yaw)
    R_fix = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0]) 
    arrow.rotate(R_fix, center=[0,0,0])
    R_yaw = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
    arrow.rotate(R_yaw, center=[0,0,0])
    
    arrow.translate([tx, ty, tz])
    geoms.append(arrow)
    
    return geoms, np.array([tx, ty, tz])

def create_error_line(start_pos, end_pos):
    """ 画一条连接真值和预测值的线 """
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector([start_pos, end_pos])
    lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
    lineset.paint_uniform_color([1.0, 1.0, 0.0]) # 黄色连线
    return lineset

def visualize_with_open3d(pt_path):
    print(f"Loading {pt_path}...")
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    inputs = data['inputs']
    targets = data['targets']
    
    # 检查是否包含预测结果
    preds = None
    if 'preds' in data:
        print("[Info] Found Prediction data! Visualization mode: Comparative.")
        preds = data['preds']
    else:
        print("[Info] No Prediction data found. Visualization mode: Ground Truth only.")

    cfg = data['config']
    dim_x, dim_y, dim_z = cfg['target_dim']
    N = len(inputs)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='FUEL Result Viewer', width=1280, height=720)

    # 渲染设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15])
    opt.light_on = True
    opt.point_size = 3.0
    
    state = {'idx': 0, 'geometries': [], 'show_free': True}

    def render_idx(idx):
        for geom in state['geometries']:
            vis.remove_geometry(geom, reset_bounding_box=False)
        state['geometries'] = []
        
        # 1. 解析数据
        grid = inputs[idx].numpy() > 0.5
        target_pose = targets[idx].numpy()
        pred_pose = preds[idx].numpy() if preds is not None else None
        
        geoms_to_add = []
        
        # A. 地图渲染
        obs_idx = np.argwhere(grid[0])
        if len(obs_idx) > 0:
            m, l = create_voxel_meshes_fast(obs_idx, color=[0.2, 0.4, 0.9])
            geoms_to_add.extend([m, l])

        if state['show_free']:
            free_idx = np.argwhere(grid[2])
            if len(free_idx) > 0:
                geoms_to_add.append(create_fog_point_cloud(free_idx, color=[0.6, 0.9, 0.6]))

        front_idx = np.argwhere(grid[1])
        if len(front_idx) > 0:
            m_f, l_f = create_voxel_meshes_fast(front_idx, color=[0.0, 1.0, 1.0], box_scale=0.95)
            geoms_to_add.extend([m_f, l_f])

        # B. 绘制视点
        
        # 1. Ground Truth (绿色)
        gt_geoms, gt_pos = create_agent_actor(target_pose, [dim_x, dim_y, dim_z], color=[0.0, 1.0, 0.0])
        geoms_to_add.extend(gt_geoms)
        
        # 2. Prediction (紫红色)
        if pred_pose is not None:
            pred_geoms, pred_pos = create_agent_actor(pred_pose, [dim_x, dim_y, dim_z], color=[1.0, 0.0, 1.0])
            geoms_to_add.extend(pred_geoms)
            
            # 3. 误差连线 (黄色虚线效果)
            err_line = create_error_line(gt_pos, pred_pos)
            geoms_to_add.append(err_line)
            
            # 计算误差文本
            dist_err = np.linalg.norm(gt_pos - pred_pos) * 0.2 # 乘分辨率转为米
            print(f"Sample {idx} | Pos Error: {dist_err:.2f}m")
        else:
            print(f"Sample {idx}")

        # 添加到视图
        for geom in geoms_to_add:
            vis.add_geometry(geom, reset_bounding_box=(idx==0))
            state['geometries'].append(geom)

    # 回调函数
    def next_sample(vis):
        state['idx'] = (state['idx'] + 1) % N
        render_idx(state['idx'])
        return False
    def prev_sample(vis):
        state['idx'] = (state['idx'] - 1 + N) % N
        render_idx(state['idx'])
        return False
    def toggle_free(vis):
        state['show_free'] = not state['show_free']
        render_idx(state['idx'])
        return False

    vis.register_key_callback(ord('N'), next_sample)
    vis.register_key_callback(ord('P'), prev_sample)
    vis.register_key_callback(ord('F'), toggle_free)
    vis.register_key_callback(ord('Q'), lambda v: v.close())
    
    print("渲染中...")
    render_idx(0)
    vis.reset_view_point(True)
    
    print("\n=== 图例 ===")
    print("  🟩 绿色箭头: 标签 (Ground Truth)")
    print("  🟪 紫色箭头: 预测 (Model Prediction)")
    print("  🟨 黄色连线: 误差距离")
    print("  🟦 蓝色方块: 障碍物")
    print("  🟦 青色方块: 前沿")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 这里加载推理结果文件
    # 如果你想看原始数据，改回 processed_data.pt 即可
    RESULT_PATH = "./inference_results.pt" 
    visualize_with_open3d(RESULT_PATH)