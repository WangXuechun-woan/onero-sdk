# BEAVR-Bot Onero A1 适配器

将 Onero A1 机械臂接入 [BEAVR-Bot](https://github.com/ARCLab-MIT/beavr-bot/) VR 遥操作框架。

## 功能

- Meta Quest 3 VR 手柄 → Onero A1 末端实时跟随
- 自动坐标系变换 (VR Y-up → 机械臂坐标系)
- 位移缩放 (适配 VR 活动空间 vs 机械臂工作空间)
- 多种安装方式预设 (侧装/倒装/人形肩装)
- 低通滤波消除手部抖动
- 速度/工作空间安全限制
- ZeroMQ 通信，与 BEAVR 标准架构一致
- 数据输出兼容 LeRobot 格式

## 安装

```bash
# 先安装 BEAVR
git clone https://github.com/ARCLab-MIT/beavr-bot.git
cd beavr-bot && uv sync

# 安装本适配器
cd /path/to/a1_sdk/beavr
pip install -e .
```

## 使用

### 快速启动

```bash
# 1. 启动 BEAVR VR 服务 (在 BEAVR 目录)
python -m beavr.teleop.vr_detector

# 2. 启动 Onero A1 适配器
beavr-onero --can-channel=can0 --mounting=side_mount_left
```

### 参数说明

```bash
beavr-onero \
    --can-channel=can0 \         # CAN 通道
    --mounting=side_mount_left \ # 安装方式 (见下方)
    --host=localhost \           # VR ZMQ host
    --vr-port=5555 \             # VR ZMQ 端口
    --vr-topic=hand_right \      # VR 手 (hand_right/hand_left)
    --scale=0.8 \                # 位移缩放
    --scale-xyz=0.8,0.8,0.7 \   # 各轴独立缩放
    --control-freq=60 \          # 控制频率
    --smoothing=0.3 \            # 滤波系数 (0=重滤波, 1=无滤波)
    --max-frame-age=0.2 \        # VR 断流超时秒数
    --max-linear-speed=0.5 \     # 最大末端线速度 m/s
    --max-angular-speed=2.0      # 最大末端角速度 rad/s
```

### 安装方式

| 预设 | 说明 | 场景 |
|------|------|------|
| `side_mount_left` | 基座 Z 朝左 | 桌面支架, 臂在操作员右侧 |
| `side_mount_right` | 基座 Z 朝右 | 桌面支架, 臂在操作员左侧 |
| `humanoid_left` | 肩装左臂 | 人形机器人 |
| `humanoid_right` | 肩装右臂 | 人形机器人 |
| `upright` | 标准 Z-up | 调试/测试 |
| `ceiling` | 倒装 Z-down | 龙门架/天花板 |

### VR 操作方式

| 按键 | 功能 |
|------|------|
| Trigger (食指) | 按住 = 激活遥操作, 松开 = 暂停 |
| Grip (中指) | 按住 = 夹爪关闭, 松开 = 打开 |
| Menu | 复位到初始位 |

首次按下 Trigger 时自动校准 VR 原点 (当前手柄位置 = A1 当前末端位置)。

## 稳定性与安全策略

- ZMQ 输入/输出默认设置低水位队列和 `LINGER=0`，控制链路优先使用最新帧，避免执行积压旧动作。
- 超过 `--max-frame-age` 没有收到新 VR 帧时自动暂停遥操作并 hold，重新按 Trigger 后重新校准原点。
- VR 位姿会经过工作空间裁剪、线速度限幅、角速度限幅、旋转矩阵合法化和低通滤波。
- `--mounting` 会传入 Onero SDK 的 `Arm(..., mounting=...)`，重力补偿和坐标变换会使用实际安装方式。
- Grip 默认不控制夹爪，也不会因为误按中断遥操作；接入真实末端执行器后再启用夹爪控制。

## 架构

```
Meta Quest 3
    │
    │  WebXR / Oculus SDK
    ▼
BEAVR VR Detector (port 5555)
    │
    │  ZMQ pub: "hand_right" + 6-DOF pose
    ▼
BEAVR Transform (稳定化)
    │
    │  ZMQ pub: 稳定 6-DOF
    ▼
┌─────────────────────────────────────────┐
│  beavr_onero (本适配器)                  │
│                                         │
│  OneroA1Operator                        │
│    ├── 接收 VR pose                     │
│    ├── 坐标变换 (VR Y-up → A1)         │
│    ├── 位移缩放                         │
│    ├── 低通滤波                         │
│    └── 安全限制                         │
│                                         │
│  OneroA1Robot (RobotWrapper)            │
│    ├── arm.servol(target_pose)          │
│    └── 发布状态 (ZMQ port 5556)         │
└─────────────────────────────────────────┘
    │
    │  CAN bus (1Mbps)
    ▼
Onero A1 硬件 (200Hz MIT impedance)
```

## 在 BEAVR 中注册

如果要集成到 BEAVR 的标准启动流程中：

```python
# beavr/teleop/robots/__init__.py 中添加:
from beavr_onero import OneroA1Robot, OneroA1Config

ROBOT_REGISTRY["onero_a1"] = {
    "wrapper_class": OneroA1Robot,
    "config_class": OneroA1Config,
}
```

然后可以用：
```bash
python -m beavr.teleop.run --robot=onero_a1 --laterality=right
```

## 与 LeRobot 数据采集配合

BEAVR 原生支持 LeRobot 格式数据输出。用 VR 遥操作采集的数据可以直接用于训练：

```bash
# BEAVR 采集 (自动保存为 LeRobot dataset)
python -m beavr.teleop.run --robot=onero_a1 --record --repo-id=user/vr_pick_place

# 训练 ACT
python lerobot/scripts/train.py --policy.type=act --dataset.repo_id=user/vr_pick_place
```
