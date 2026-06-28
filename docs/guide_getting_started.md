# Onero SDK 新手指南

本指南面向初次接触机械臂 SDK 的用户，帮助你在 5 分钟内理解核心概念并运行第一个程序。

## 安装

```bash
pip install onero-sdk[sim]       # 仿真 (MuJoCo)
pip install onero-sdk[dynamics]  # 重力补偿/freedrive (pinocchio)
pip install onero-sdk[full]      # 全部功能
```

## 5 分钟快速开始

```python
from onero_sdk import Arm
from onero_sdk.mujoco_backend import MujocoBackend

# 无需硬件 — 仿真后端
arm = Arm("A1R", backend=MujocoBackend(model_name="A1R"))
arm.enable()

# 关节空间运动
arm.movej([0, -0.5, 0.3, -1.0, 0, 0.5, 0], duration=2.0)

# 笛卡尔空间运动 (x, y, z, rx, ry, rz)
arm.movel([0.3, 0, 0.2, 0, 0, 0], duration=2.0)

# 获取当前位姿
pos, rot = arm.get_pose()
print(f"末端位置: {pos}")

arm.disable()
```

连接真实硬件时只需改一行：

```python
arm = Arm("A1R", channel="can0")  # 替换 backend 参数
```

## 核心概念

### 坐标系

```
世界坐标系 (World Frame)
    X = 前方 (远离操作员)
    Y = 左方
    Z = 上方

基座坐标系 (Base Frame)
    机械臂基座的固有坐标系
    取决于安装方式 (mounting)

法兰坐标系 (Flange Frame)
    末端法兰中心
    FK/IK 的原生输出

TCP 坐标系 (Tool Center Point)
    工具尖端 = 法兰 + 偏移
```

坐标系链路：`世界 ←R_world_base→ 基座 ←FK→ 法兰 ←TCP偏移→ 工具`

### 安装方式 (Mounting)

A1 通常**侧装**在支架上 (Z 轴朝左或朝右)，而非直立。安装方式影响：

1. **重力补偿方向** — freedrive/力控是否正确
2. **世界坐标系映射** — "向前移动" 对应哪个基座轴

```python
# 最常见：侧装左臂
arm = Arm("A1R", channel="can0", mounting="side_mount_left")

# 所有预设
from onero_sdk import MountingPreset
# UPRIGHT       — 直立 (Z朝上)
# SIDE_MOUNT_LEFT  — 侧装左臂 (Z朝左)
# SIDE_MOUNT_RIGHT — 侧装右臂 (Z朝右)
# HUMANOID_LEFT    — 人形肩部左臂
# HUMANOID_RIGHT   — 人形肩部右臂
# CEILING          — 天花板/龙门架倒装
```

### 世界坐标系模式

默认情况下，`get_pose()`/`movel()` 在基座坐标系下工作。开启 `world_frame` 后，所有笛卡尔 API 自动转换到世界坐标系：

```python
arm = Arm("A1R", channel="can0",
          mounting="side_mount_left",
          world_frame=True)

pos, rot = arm.get_pose()   # 世界坐标系 (X前 Y左 Z上)
arm.movel([0.3, 0, 0.2, 0, 0, 0])  # "向前 0.3m, 向上 0.2m"
```

### TCP (工具偏移)

安装夹爪后，设置 TCP 使所有 API 以工具尖端为参考点：

```python
arm.set_tcp([0, 0, 0.15])          # 法兰 Z 方向 15cm 处
pos, rot = arm.get_pose()           # 返回夹爪尖端位置
arm.movel([0.3, 0, 0.1, 0, 0, 0])  # 目标是夹爪尖端位置
arm.clear_tcp()                     # 恢复法兰坐标
```

### Freedrive (零力拖动/示教)

```python
arm.freedrive_enable(damping=0.5)
# 用手拖动机械臂 — 重力自动补偿
# 按 Ctrl+C 或 ...
arm.freedrive_disable()
```

重力补偿依赖正确的 `mounting` 设置。如果安装方式错误，松手后关节会下垂。

### 力控模式

```python
# Z 方向施加 5N 向下的力，其他方向保持位置
arm.force_mode(
    selection=[0, 0, 1, 0, 0, 0],   # Z 轴力控
    wrench=[0, 0, -5.0, 0, 0, 0],   # -5N (向下)
    limits=[0, 0, 0.1, 0, 0, 0],
)
```

## 常用 API 速查

| 方法 | 功能 | 空间 |
|------|------|------|
| `movej(joints, duration)` | 关节空间运动 | 关节 |
| `movel(pos, rot, duration)` | 直线运动 | 笛卡尔 |
| `movep(pos, rot, duration)` | 点到点 (IK+关节插值) | 笛卡尔 |
| `movec(via, target, duration)` | 圆弧运动 | 笛卡尔 |
| `servoj(joints)` | 实时关节指令 (单帧) | 关节 |
| `servop(pos, rot)` | 实时笛卡尔指令 (单帧) | 笛卡尔 |
| `speedl(twist, duration)` | 速度控制 | 笛卡尔 |
| `get_pose()` | 获取末端位姿 | 笛卡尔 |
| `freedrive_enable()` | 零力拖动 | — |
| `force_mode(...)` | 力/位混合控制 | 笛卡尔 |

## 示例文件

按编号从简到繁：

| # | 文件 | 主题 |
|---|------|------|
| 01 | `01_quickstart_sim.py` | 仿真快速入门 |
| 02 | `02_multi_waypoint.py` | 多路点轨迹 |
| 03 | `03_error_handling.py` | 异常处理 |
| 04 | `04_circular_motion.py` | 圆弧运动 |
| 05 | `05_stiffness.py` | 刚度调节 |
| 08 | `08_force_control.py` | 力控 |
| 11 | `11_teach_replay.py` | 示教回放 |
| 15 | `15_mounting.py` | 安装方式配置 |

## 常见问题

**Q: freedrive 松手后关节下垂？**
检查 `mounting` 参数是否正确设置。错误的安装方式 → 重力方向错误 → 补偿力矩方向错误。

**Q: movel 运动方向和预期相反？**
可能是坐标系问题。开启 `world_frame=True` 后，X/Y/Z 对应直觉的前/左/上方向。

**Q: 需要 pinocchio 吗？**
- `movej`/`movel`: 不需要 (纯运动学)
- `freedrive`/`force_mode`/`calibrate`: 需要 (`pip install onero-sdk[dynamics]`)

**Q: 仿真和真机代码一样吗？**
是的。只需将 `backend=MujocoBackend(...)` 替换为 `channel="can0"`，其他代码完全不变。
