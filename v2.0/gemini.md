
# Project Context: UAV Exploration Viewpoint Generation via DRL (Advanced Version)

## 1. 项目核心目标 (Project Objective)

构建一个基于深度强化学习（DRL）的端到端视点生成模型，用于无人机自主探索。

* **输入**：局部 3D 体素地图（3通道）。
* **输出**：最佳观测位姿 $(x, y, z, yaw)$。
* **核心升级**：从简单的 CNN+MLP 升级为 **Attention-Enhanced Residual Network**，并针对 **边缘端部署 (NPU/TensorRT)** 进行了算子优化。

---

## 2. 核心算法思路 (Methodology)

### 2.1 整体架构 (NBV via DRL)

* **场景**：生成“前沿墙 + 背后未知区域 + 隐形障碍物”的训练场景。
* **对齐**：使用 **DBSCAN** 聚类前沿，通过 **PCA (仅 Yaw 旋转)** 将局部地图标准化。
* **决策**：网络根据标准化体素，输出相对于前沿中心的局部坐标和偏航角。
* **约束**：采用 **软约束 (Soft Constraint)**，通过 Known Mask 通道和惩罚项，教会 Agent 待在安全区。

### 2.2 网络架构升级 (Model Evolution)

为了解决简单 MLP 无法处理复杂空间关系的问题，引入了以下机制：

* **3D Attention**：引入 Spatial & Channel Attention，让网络忽略空白区域，聚焦几何特征。
* **Residual Connections**：使用 ResMLP Block 防止梯度消失，加深网络。
* **Decoupled Heads**：将位置 ($x,y,z$) 和朝向 ($yaw$) 解耦。采用**自回归逻辑**：先决定站位，再根据站位决定看哪里。
* **Deployment Friendly**：移除了 `AdaptiveAvgPool3d` 等 NPU 不友好的算子，全线使用标准算子。

---

## 3. 环境设计 (`env.py`)

### 3.1 状态空间 (Observation Space)

维度：`[3, 32, 32, 32]` (3D Voxel Grid)

* **Channel 0**: `Known Obstacles` (已知障碍物，避障用)
* **Channel 1**: `Frontier Surface` (前沿面，目标)
* **Channel 2**: `Known Mask` (安全飞行区域，**核心约束**，出界即死)

### 3.2 动作空间 (Action Space)

维度：`[4]` (Continuous, Range `[-1, 1]`)

* 物理含义：相对于局部坐标系中心的位移和偏航角偏移。
* 缩放系数：`scale=12.0`。

### 3.3 奖励函数 (Reward Function)

$$
R = (\alpha \cdot N_{surface} + \beta \cdot N_{volume}) \times 0.01 - \text{Penalty}
$$

* **收益**：表面积覆盖 ($N_{surface}$) + 体积穿透 ($N_{volume}$)。
* **缩放**：整体乘以 `0.01`，将数值从 ~200 降至 ~2.0，防止梯度爆炸。
* **惩罚**：
  * 越界/撞墙/飞入未知区：给予 `-2.0` (与收益量级相当) 并结束回合。

---

## 4. 网络模型细节 (`model.py`)

### 4.1 Backbone

* **Input**: `(B, 3, 32, 32, 32)`
* **Structure**: 3层 `Conv3d` + `BatchNorm3d` + `LeakyReLU`。
* **Attention**:
  * `ChannelAttention3D`: 使用 `torch.mean` 代替 AdaptivePool。
  * `SpatialAttention3D`: Max + Avg pooling -> Conv3d -> Sigmoid。

### 4.2 Decoupled Heads

* **Shared Embedding**: Flatten -> Linear -> LayerNorm.
* **Stream A (Position)**: ResMLP -> `Linear(3)` -> Tanh.
* **Stream B (Yaw)**: Input(`Shared` + `Pos_Output`) -> ResMLP -> `Linear(1)` -> Tanh.
* **Critic**: ResMLP -> Value.

---

## 5. 训练配置 (`train.py` & `config.py`)

### 5.1 智能参数分组 (Auto-Grouping)

优化器定义不再硬编码层名称，而是通过逻辑自动提取：

1. **Actor Params**: `pos_net`, `yaw_net`, `log_std`.
2. **Critic Params**: `critic_net`.
3. **Backbone Params**: `filter(remaining parameters)`.

* **优势**：修改模型结构后无需修改训练脚本，自动包含 BN、Attention 等所有层。

### 5.2 监控与日志

* **CSV Logging**: 记录 `episode`, `reward`, `loss` 到 `training_log.csv`。
* **Plotting**: 提供 `plot_training.py` 绘制平滑后的训练曲线。

### 5.3 超参数 (Fine-tuning phase)

* `LR_ACTOR`: 5e-5 (低学习率，稳定策略)
* `LR_CRITIC`: 1e-4
* `BATCH_SIZE`: 64
* `CKPT_PATH`: `ppo_penetration_checkpoint.pth` (需注意模型结构变更需删旧档)

---

## 6. 文件结构清单

1. **`config.py`**: 全局配置。
2. **`utils.py`**:
   * `RayCaster`: 射线投射。
   * `FrontierProcessor`: DBSCAN, PCA (Yaw only), Voxelization (3-Channel)。
3. **`env.py`**: 环境逻辑 (3通道输入, Reward Scaling, Soft Constraints)。
4. **`model.py`**: **[部署优化版]** Attention-Enhanced Actor-Critic。
5. **`train.py`**: **[智能分组版]** PPO 训练循环 + CSV Logger。
6. **`visualize_result.py`**: 结果可视化 (Cyan Surface, Gold Volume, Red/Green Agent)。
7. **`plot_training.py`**: 训练曲线绘图工具。

---

## 7. 部署注意事项 (Deployment)

* **算子兼容性**：模型已移除 `AdaptiveAvgPool3d`，所有算子（Conv3d, Mean, Max, MatMul）均支持 ONNX opset 11+。
* **目标平台**：Nvidia Jetson (TensorRT) 或 Rockchip RK3588 (RKNN)。
* **输入尺寸**：固定为 `32x32x32`，计算量极小，适合边缘端实时推理。
