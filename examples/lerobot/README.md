# LeRobot 示例

使用 LeRobot 开源框架进行遥操作、数据采集和 AI 策略训练/推理。

## 前提

```bash
# 安装 Onero 机械臂的 LeRobot 插件
pip install -e ../../lerobot/lerobot_robot_onero
pip install -e ../../lerobot/lerobot_teleoperator_onero

# 安装 LeRobot 框架 (如需训练)
pip install lerobot
```

## 示例列表

| # | 文件 | 说明 | 适用场景 |
|---|------|------|----------|
| 1 | `01_lerobot_teleoperate.py` | 双臂 Leader-Follower | 遥操作入门 |
| 2 | `02_lerobot_record.py` | 数据采集录制 | 采集训练数据 |
| 3 | `03_lerobot_replay.py` | 数据回放验证 | 验证数据质量 |
| 4 | `04_lerobot_policy.py` | 策略推理 (ACT等) | 模型部署 |
| 5 | `05_cross_arm_teleop.py` | 跨臂遥操作 (SO101→A1) | 异构臂控制 |

## 典型工作流

```
遥操作验证 → 采集数据 → 验证回放 → 训练模型 → 真机推理
   01            02          03       (离线)       04
```

## 命令行工具

安装插件后可直接使用 LeRobot CLI:

```bash
# 遥操作
lerobot-teleoperate --robot.type=onero --robot.can_channel=can0 \
    --teleop.type=onero_teleop --teleop.leader_channel=can1

# 数据采集
lerobot-record --robot.type=onero --robot.can_channel=can0 \
    --teleop.type=onero_teleop --teleop.leader_channel=can1 \
    --fps=50 --repo-id=your-name/dataset-name --num-episodes=50

# 数据回放
lerobot-replay --robot.type=onero --robot.can_channel=can0 \
    --repo-id=your-name/dataset-name --episode=0

# 策略推理
lerobot-infer --robot.type=onero --robot.can_channel=can0 \
    --policy.type=act --policy.path=path/to/checkpoint
```

## 支持的算法

通过 LeRobot 框架，Onero A1 支持以下算法的训练和推理:

- **ACT** (Action Chunking with Transformers) — 推荐入门
- **Diffusion Policy** — 高质量连续动作生成
- **TDMPC** — 基于模型的强化学习
- **VLA** (Vision-Language-Action) — 语言指令引导的操作

## 跨臂控制说明

`05_cross_arm_teleop.py` 演示了用 SO101 (6-DOF) 控制 A1 (7-DOF) 的方案。核心原理是**任务空间映射**:

- 不映射关节角度（臂形不同，关节角没有对应关系）
- 映射末端位移（缩放 + 坐标系对齐）和旋转（直接传递）
- 需要配置的参数:
  - `POSITION_SCALE_XYZ`: 位移缩放系数（补偿臂长差异）
  - `AXIS_MAP`: 坐标轴方向对齐
  - `WORKSPACE_MIN/MAX`: 工作空间限制
