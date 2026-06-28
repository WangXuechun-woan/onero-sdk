# LeRobot 使用指南

本文档介绍如何使用 LeRobot 框架配合 Onero A1 机械臂进行遥操作、数据采集和 AI 策略训练。

LeRobot 是 Hugging Face 推出的开源具身智能框架，支持多种算法（ACT、Diffusion Policy、VLA 等），是目前最流行的机器人学习框架之一。

---

## 目录

1. [环境准备](#1-环境准备)
2. [安装 Onero 插件](#2-安装-onero-插件)
3. [遥操作 (Teleoperation)](#3-遥操作-teleoperation)
4. [数据采集 (Recording)](#4-数据采集-recording)
5. [数据回放与验证](#5-数据回放与验证)
6. [训练策略 (Training)](#6-训练策略-training)
7. [策略推理与部署](#7-策略推理与部署)
8. [跨臂遥操作 (异构臂)](#8-跨臂遥操作-异构臂)
9. [相机配置](#9-相机配置)
10. [常见问题](#10-常见问题)

---

## 1. 环境准备

### 系统要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| GPU | NVIDIA GPU (训练需要, 推理可 CPU) |
| CUDA | 11.8+ (训练需要) |
| 内存 | 16GB+ (训练建议 32GB) |

### 安装 LeRobot 框架

```bash
# 方式 1: pip 安装
pip install lerobot

# 方式 2: 从源码 (最新功能)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[all]"
```

### 安装 onero-sdk

```bash
pip install -e /path/to/a1_sdk
```

---

## 2. 安装 Onero 插件

LeRobot 通过插件机制支持不同硬件。安装后会被 CLI 工具自动发现。

```bash
# 安装 Robot 插件 (从臂/follower)
cd /path/to/a1_sdk/lerobot/lerobot_robot_onero
pip install -e .

# 安装 Teleoperator 插件 (主臂/leader)
cd /path/to/a1_sdk/lerobot/lerobot_teleoperator_onero
pip install -e .
```

验证安装：
```bash
# 应能看到 onero 类型
python -c "from lerobot_robot_onero import OneroRobot; print('Robot plugin OK')"
python -c "from lerobot_teleoperator_onero import OneroTeleop; print('Teleop plugin OK')"
```

---

## 3. 遥操作 (Teleoperation)

遥操作是最基础的功能：用一台臂（leader/主手）控制另一台（follower/从手）。

### 硬件连接

```
操作员
  │
  ├── Leader 臂 (can1): freedrive 模式, 手动拖动
  │
  └── Follower 臂 (can0): 实时跟随 leader 的动作
```

两台臂通过不同的 CAN 通道连接到同一台电脑。

### 启动遥操作

```bash
# 命令行方式 (推荐)
lerobot-teleoperate \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --teleop.type=onero_teleop \
    --teleop.leader_channel=can1

# Python 方式
python examples/lerobot/01_lerobot_teleoperate.py
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--robot.type` | 从臂类型 | onero |
| `--robot.can_channel` | 从臂 CAN 通道 | can0 |
| `--robot.model` | 机械臂型号 | A1 |
| `--teleop.type` | 主臂类型 | onero_teleop |
| `--teleop.leader_channel` | 主臂 CAN 通道 | can1 |
| `--teleop.damping` | 主臂阻尼 (0=无阻力, 1=重) | 0.3 |

### 操作步骤

1. 确保两台臂 CAN 连接正常
2. 运行命令
3. Leader 臂进入 freedrive（可自由拖动）
4. 拖动 leader，follower 实时跟随
5. `Ctrl+C` 停止

---

## 4. 数据采集 (Recording)

遥操作的同时录制数据，用于后续训练。

### 快速开始

```bash
lerobot-record \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --teleop.type=onero_teleop \
    --teleop.leader_channel=can1 \
    --fps=50 \
    --repo-id=your-username/onero_pick_place \
    --num-episodes=50 \
    --episode-time-s=10
```

### 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--fps` | 采集帧率 | 50 (平衡精度和存储) |
| `--repo-id` | HuggingFace 数据集 ID | `username/task_name` |
| `--num-episodes` | 采集多少个 episode | 50-200 |
| `--episode-time-s` | 每个 episode 最长时间 | 10-30s |
| `--push-to-hub` | 采集完自动上传 HuggingFace | 1 |
| `--video` | 录制视频 | 1 |

### 采集流程

```
对于每个 episode:
  1. [自动] 机械臂回到起始位
  2. [提示] "准备好后按 Enter 开始录制"
  3. [操作] 操作员拖动 leader 完成任务
  4. [提示] "按 Enter 结束本次录制" (或到时间自动结束)
  5. [提示] "这次采集满意吗? (y/n/再来一次)"
  6. [自动] 保存数据, 进入下一个 episode
```

### 采集技巧

**采集数量:**
- ACT: 最少 50 episodes, 建议 100-200
- Diffusion Policy: 50-100 即可
- 越难的任务需要越多数据

**采集质量:**
- 保持一致性: 每次操作的路径和速度尽量相似
- 覆盖变化: 物体位置稍有变化，增加泛化性
- 避免犹豫: 动作干脆利落，不要中途停顿

**带相机:**
```bash
lerobot-record \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --robot.cameras='{"wrist": {"type": "opencv", "index": 0, "width": 640, "height": 480}}' \
    --teleop.type=onero_teleop \
    --teleop.leader_channel=can1 \
    --fps=50 \
    --repo-id=your-username/onero_pick_place_cam
```

---

## 5. 数据回放与验证

采集完毕后，先回放验证数据质量再训练。

```bash
# 回放第 0 个 episode
lerobot-replay \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --repo-id=your-username/onero_pick_place \
    --episode=0

# Python
python examples/lerobot/03_lerobot_replay.py --dataset=your-username/onero_pick_place
```

### 验证要点

- ✅ 动作流畅，没有突变或跳跃
- ✅ 任务成功完成（物体被正确抓取/放置）
- ✅ 速度合理，没有过快或过慢的片段
- ❌ 有抖动 → 检查 CAN 通信或 leader 阻尼设置
- ❌ 动作不连续 → 帧率太低，增加 fps
- ❌ 任务失败 → 删除该 episode，重新采集

### 可视化数据

```bash
# 查看数据集统计
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('your-username/onero_pick_place')
print(f'Episodes: {ds.num_episodes}')
print(f'Total frames: {len(ds)}')
print(f'FPS: {ds.fps}')
print(f'Features: {list(ds.features.keys())}')
"
```

---

## 6. 训练策略 (Training)

### ACT (推荐入门)

```bash
# 训练 ACT 策略
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=your-username/onero_pick_place \
    --dataset.episodes=null \
    --training.num_epochs=2000 \
    --training.batch_size=8 \
    --training.lr=1e-5 \
    --output_dir=outputs/train/act_onero_pick

# 使用 GPU
CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/train.py ...
```

### Diffusion Policy

```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=your-username/onero_pick_place \
    --training.num_epochs=5000 \
    --training.batch_size=64 \
    --output_dir=outputs/train/diffusion_onero_pick
```

### 训练参数参考

| 算法 | epochs | batch_size | 训练时间 (A100) | 数据量建议 |
|------|--------|-----------|-----------------|-----------|
| ACT | 2000 | 8 | ~2h | 50-200 eps |
| Diffusion | 5000 | 64 | ~8h | 50-100 eps |
| TDMPC | 10000 | 256 | ~12h | 100+ eps |

### 监控训练

```bash
# TensorBoard
tensorboard --logdir=outputs/train/act_onero_pick

# WandB (如果配置了)
# 训练时添加 --wandb.enable=true --wandb.project=onero
```

### 训练完成

模型保存在：
```
outputs/train/act_onero_pick/
├── checkpoints/
│   ├── 000500/pretrained_model/   # 中间 checkpoint
│   ├── 001000/pretrained_model/
│   └── last/pretrained_model/     # 最终模型 ← 推理用这个
├── config.yaml
└── train_log.csv
```

---

## 7. 策略推理与部署

### 命令行推理

```bash
lerobot-infer \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --policy.type=act \
    --policy.path=outputs/train/act_onero_pick/checkpoints/last/pretrained_model \
    --fps=50
```

### Python 推理

```bash
python examples/lerobot/04_lerobot_policy.py \
    --policy=act \
    --ckpt=outputs/train/act_onero_pick/checkpoints/last/pretrained_model \
    --fps=50 \
    --max-steps=500
```

### 推理流程

```
每一步 (50Hz):
  1. 读取观测 (关节状态 + 相机图像)
  2. 模型推理 → 输出动作序列 (action chunk)
  3. 执行第一个动作
  4. 重复
```

### 安全注意事项

⚠️ **首次推理请缓慢观察:**
- 先用 `--fps=10` 低帧率测试
- 手放在急停按钮上
- 确认动作范围合理后再提高帧率
- 设置合理的 `--max-steps` 避免无限执行

---

## 8. 跨臂遥操作 (异构臂)

如果手边没有两台 A1，可以用其他品牌的臂（如 SO101）作为 leader。

### 原理

不映射关节角度（臂形不同），而是映射末端位移和旋转：

```
SO101 末端位移 × 缩放系数 × 坐标系变换 → A1 末端位移
SO101 末端旋转 (无缩放)                 → A1 末端旋转
```

### 使用

```bash
python examples/lerobot/05_cross_arm_teleop.py
```

### 配置要点

```python
# 1. 位移缩放 (臂长比)
# SO101 臂展 ~25cm, A1 臂展 ~55cm → 比例 2.2
POSITION_SCALE_XYZ = [2.2, 2.2, 2.0]

# 2. 坐标系对齐 (SO101的X方向对应A1的哪个方向)
AXIS_MAP = {
    'x': ('+x',),  # SO101 +x → A1 +x
    'y': ('+y',),
    'z': ('+z',),
}
# 如果方向不同, 比如 SO101 面朝你, A1 侧面朝你:
# AXIS_MAP = {'x': '-y', 'y': '+x', 'z': '+z'}

# 3. 工作空间限制 (防止超出可达范围)
WORKSPACE_MIN = [0.05, -0.40, 0.02]
WORKSPACE_MAX = [0.60,  0.40, 0.60]
```

### 校准步骤

1. 将两台臂都放到舒适的操作位
2. 运行程序，按 Enter 记录初始位姿
3. 移动 SO101，观察 A1 是否正确跟随
4. 如果方向不对 → 调整 `AXIS_MAP`
5. 如果幅度不对 → 调整 `POSITION_SCALE_XYZ`

---

## 9. 相机配置

视觉策略（如 ACT with images）需要配置相机。

### 支持的相机类型

- USB 摄像头 (UVC 协议)
- Intel RealSense
- 自定义相机 (通过回调)

### 配置腕部相机

```bash
# 查找相机设备号
ls /dev/video*
# 或
v4l2-ctl --list-devices

# 在 LeRobot 中配置
lerobot-teleoperate \
    --robot.type=onero \
    --robot.can_channel=can0 \
    --robot.cameras='{"wrist_cam": {"type": "opencv", "index": 0, "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=onero_teleop \
    --teleop.leader_channel=can1
```

### 多相机配置

```bash
# 腕部相机 + 全局相机
--robot.cameras='{
    "wrist": {"type": "opencv", "index": 0, "width": 640, "height": 480},
    "overhead": {"type": "opencv", "index": 2, "width": 640, "height": 480}
}'
```

### RealSense 相机

```bash
pip install pyrealsense2

--robot.cameras='{
    "wrist_rgb": {"type": "realsense", "serial": "123456789", "width": 640, "height": 480},
    "wrist_depth": {"type": "realsense", "serial": "123456789", "width": 640, "height": 480, "stream": "depth"}
}'
```

---

## 10. 常见问题

### 安装问题

**Q: `lerobot-teleoperate` 命令找不到?**
```bash
# 确认 lerobot 已正确安装
pip show lerobot
# 确认 scripts 在 PATH 中
which lerobot-teleoperate
# 或用 python -m
python -m lerobot.teleoperate ...
```

**Q: `Unknown robot type: onero`?**
```bash
# 插件未安装或安装到了错误的 Python 环境
pip install -e /path/to/lerobot/lerobot_robot_onero
# 确认在同一个 Python 环境
which python
pip show lerobot-robot-onero
```

### 遥操作问题

**Q: Leader 臂拖动很重?**
- 降低阻尼: `--teleop.damping=0.1`
- 检查 CAN 通信是否正常

**Q: Follower 跟随有延迟?**
- 延迟 >50ms: 检查 CAN 总线负载（`canbusload can0`）
- 延迟 >200ms: 可能 USB-CAN 适配器带宽不够，换原生 CAN 接口

**Q: 遥操作时 follower 抖动?**
- 增大 leader 阻尼: `--teleop.damping=0.5`
- 降低 follower 刚度: SDK 中调低 kp

### 数据采集问题

**Q: 采集的数据在哪?**
```bash
# 本地缓存
ls ~/.cache/huggingface/datasets/your-username/

# 或自定义路径
lerobot-record --root=/path/to/save ...
```

**Q: 如何删除某个失败的 episode?**
```python
# 目前 LeRobot 不支持单独删除
# 最佳实践: 采集时选 "n" (不保留) 即可
```

### 训练问题

**Q: 训练 loss 不下降?**
- 检查数据质量: 回放几个 episode 看动作是否合理
- 降低学习率: `--training.lr=5e-6`
- 增加数据量: 至少 50 个有效 episode

**Q: GPU 显存不够?**
- 降低 batch_size: `--training.batch_size=4`
- 降低图片分辨率: 采集时用 320x240
- 使用 gradient checkpointing

### 推理问题

**Q: 推理时机械臂不动?**
- 检查模型加载是否成功
- 检查 action 输出范围是否正确
- 先打印 action 看值是否合理

**Q: 推理时动作粗暴/不准?**
- 训练不够: 增加 epochs
- 数据不够: 增加 episodes
- 帧率不匹配: 推理 fps 要和采集 fps 一致

---

## 完整工作流总结

```
┌─────────────────────────────────────────────────────────┐
│                    LeRobot 工作流                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 硬件准备                                             │
│     └── CAN 配置 → pip install 插件 → 验证连接           │
│                                                         │
│  2. 遥操作验证                                           │
│     └── lerobot-teleoperate → 确认跟随正常               │
│                                                         │
│  3. 数据采集                                             │
│     └── lerobot-record → 50~200 episodes                │
│                                                         │
│  4. 数据验证                                             │
│     └── lerobot-replay → 确认数据质量                    │
│                                                         │
│  5. 训练模型                                             │
│     └── python train.py → 等待收敛 (几小时)              │
│                                                         │
│  6. 推理部署                                             │
│     └── lerobot-infer → 真机验证效果                     │
│                                                         │
│  7. 迭代优化                                             │
│     └── 效果不好? → 回到 3 采集更多/更好的数据            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
