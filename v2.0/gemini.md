# Project Context: UAV Exploration Viewpoint Generation via DRL

## 1. 项目核心目标 (Project Objective)

将无人机自主探索中的“视点选择”模块，从传统的**启发式射线投射（Heuristic Raycasting）**升级为**深度强化学习（Deep Reinforcement Learning, DRL）**方法。

* **输入**：局部的栅格地图（ESDF/Occupancy Grid）、前沿簇（Frontier Cluster）、已知区域掩码（Known Mask）。
* **输出**：最佳观测位姿 $(x, y, z, yaw)$。
* **目标**：最大化前沿表面的覆盖率（Surface Coverage）和未知区域的穿透体积（Penetrated Volume），同时确保飞行安全。

---

## 2. 核心算法思路 (Methodology)

### 2.1 整体架构

采用 **"Next Best View (NBV)"** 的单步决策模式进行训练：

1. **场景生成**：随机生成一面“前沿墙”，墙后是未知区域和隐形障碍物。
2. **预处理**：对全局地图进行 **DBSCAN 聚类** 和 **PCA 局部对齐**。
3. **推理**：Agent 根据标准化后的局部体素地图，输出一个观测视点。
4. **评估**：计算视点的 FOV 覆盖收益（表面积 + 体积）作为 Reward。

### 2.2 关键技术点

* **PCA 对齐 (Canonicalization)**：
  * 为了提高泛化性，不直接输入全局地图。
  * **逻辑**：只计算 XY 平面的主方向（Yaw），进行旋转和平移，保留 Z 轴特征（如斜坡）。
  * **结果**：网络看到的永远是“正前方有一面前沿”，输出动作也是相对于前沿中心的局部坐标。
* **双重收益机制**：
  * 单纯追求“穿透体积”会导致 Agent 贴脸观测。
  * **修正**：引入“前沿表面积覆盖”作为主奖励，利用 FOV 几何特性迫使 Agent 后退，以获得更大的视野截面。
* **安全性约束 (Safety)**：
  * **放弃**了 KD-Tree 强制投影的硬约束。
  * **采用**软约束（Soft Constraint）：在 Reward 函数中对“飞入未知区域”或“撞墙”施加巨额惩罚（-2.0 ~ -10.0），让 Agent 自主学会待在 Known Mask 内。

---

## 3. 环境设计 (`env.py`)

### 3.1 状态空间 (Observation Space)

维度：`[3, 32, 32, 32]` (3D Voxel Grid)

* **Channel 0**: `Known Obstacles` (已知障碍物，避障用)
* **Channel 1**: `Frontier Surface` (前沿面，目标)
* **Channel 2**: `Known Mask` (安全飞行区域，越界即死)

### 3.2 动作空间 (Action Space)

维度：`[4]` (Continuous)

* `[lx, ly, lz, lyaw]`：相对于 PCA 局部坐标系中心的位移和偏航角偏移。
* 范围：`[-1, 1]`，在 `step` 中会被缩放（如 `scale=12.0`）。

### 3.3 奖励函数 (Reward Function)

$$
R = (\alpha \cdot N_{surface} + \beta \cdot N_{volume}) \times \text{Scale} - \text{Penalty}_{dist}
$$

* **Surface Hits ($N_{surface}$)**: 射线击中的前沿点数量（权重 $\alpha=1.0$）。
* **Volume Hits ($N_{volume}$)**: 射线穿透到未知区域的点数量（权重 $\beta=0.05$）。
* **Reward Scale**: `0.01` (将数百的 Hits 缩放到个位数，防止梯度爆炸)。
* **惩罚**:
  * 越界/撞墙/飞入未知区：给予 `-2.0` (缩放后量级) 并结束回合。

---

## 4. 网络模型 (`model.py`)

* **架构**：Actor-Critic (PPO)。
* **Backbone**：3D CNN (VoxNet 变体)。
  * Input: `(B, 3, 32, 32, 32)`
  * Layers: 3层 Conv3d + BatchNorm3d + ReLU。
* **Heads**：
  * Actor: 输出 Mean (Tanh激活) + 可学习的 LogStd。
  * Critic: 输出 Value。

---

## 5. 训练配置与细节 (`train.py` & `config.py`)

### 5.1 优化器分组 (Critical fix)

**重要**：必须正确处理 PyTorch 优化器参数组，防止漏掉 BatchNorm 参数。

```python
# 推荐写法
optimizer = optim.Adam([
    {'params': actor_params, 'lr': 5e-5},
    {'params': critic_params, 'lr': 1e-4},
    {'params': backbone_params, 'lr': 1e-4} # 包含 Conv 和 BN
])
```
