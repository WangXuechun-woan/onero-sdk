# ROS2 使用指南

本文档介绍如何通过 ROS2 控制 Onero A1 机械臂。

---

## 目录

1. [环境准备](#1-环境准备)
2. [安装 onero_ros2 包](#2-安装-onero_ros2-包)
3. [启动机械臂节点](#3-启动机械臂节点)
4. [基础操作](#4-基础操作)
5. [ROS2 接口详解](#5-ros2-接口详解)
6. [与其他节点协作](#6-与其他节点协作)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 系统要求

| 项目 | 要求 |
|------|------|
| OS | Ubuntu 22.04 / 24.04 |
| ROS2 | Humble (22.04) 或 Jazzy (24.04) |
| Python | 3.10+ |
| CAN 接口 | SocketCAN (USB-CAN 或 原生 CAN) |

### 安装 ROS2

如果尚未安装 ROS2：

```bash
# Ubuntu 22.04 → ROS2 Humble
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop

# 环境变量
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 安装依赖

```bash
sudo apt install ros-${ROS_DISTRO}-control-msgs \
    ros-${ROS_DISTRO}-controller-manager \
    ros-${ROS_DISTRO}-joint-state-broadcaster \
    python3-colcon-common-extensions
```

---

## 2. 安装 onero_ros2 包

```bash
# 创建工作空间 (如果还没有)
mkdir -p ~/onero_ws/src
cd ~/onero_ws/src

# 克隆或链接 onero_ros2 包
ln -s /path/to/a1_sdk/ros2/onero_ros2 .
ln -s /path/to/a1_sdk/ros2/onero_msgs .
# 或
# cp -r /path/to/a1_sdk/ros2/onero_ros2 .
# cp -r /path/to/a1_sdk/ros2/onero_msgs .

# 安装 onero SDK (onero_ros2 的依赖)
pip install -e /path/to/a1_sdk

# 编译
cd ~/onero_ws
colcon build --symlink-install
source install/setup.bash
```

验证安装：
```bash
ros2 pkg list | grep onero
# 应输出:
# onero_ros2
# onero_msgs
```

---

## 3. 启动机械臂节点

### Step 1: 配置 CAN 总线

```bash
# 设置 CAN 接口 (每次开机需执行一次)
sudo ip link set can0 up type can bitrate 1000000
# 验证
ip link show can0
# 应显示: can0: <NOARP,UP,LOWER_UP,ECHO> ...
```

### Step 2: 启动节点

```bash
# 基础启动
ros2 launch onero_ros2 bringup.launch.py can_channel:=can0

# 带参数启动
ros2 launch onero_ros2 bringup.launch.py \
    can_channel:=can0 \
    model:=A1R \
    control_freq:=200 \
    kp:=80.0 \
    kd:=2.0
```

启动后终端会输出：
```
[INFO] [onero_arm_driver]: Connected to CAN 'can0', model=A1R
[INFO] [onero_arm_driver]: Onero arm driver started (call ~/enable to activate)
```

### Step 3: 使能机械臂

```bash
# 使能 (机械臂通电、锁定)
ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger

# 去使能 (机械臂掉电、可自由拖动)
ros2 service call /onero_arm_driver/disable std_srvs/srv/Trigger
```

### Step 4: 验证状态

```bash
# 查看关节状态
ros2 topic echo /joint_states --once

# 查看话题列表
ros2 topic list

# 查看服务列表
ros2 service list
```

---

## 4. 基础操作

### 读取关节状态

```bash
# 命令行
ros2 topic echo /joint_states

# Python
python examples/ros2/01_ros2_joint_states.py
```

### 发送轨迹

```bash
# Python (推荐)
python examples/ros2/03_ros2_trajectory.py

# 命令行 (通过 action)
ros2 action send_goal /arm_controller/follow_joint_trajectory \
    control_msgs/action/FollowJointTrajectory \
    "{trajectory: {joint_names: ['joint1-a1_r','joint2-a1_r','joint3-a1_r','joint4-a1_r','joint5-a1_r','joint6-a1_r','joint7-a1_r'], \
    points: [{positions: [0,0,0,0,0,0,0], time_from_start: {sec: 2}}]}}"
```

---

## 5. ROS2 接口详解

### 话题 (Topics)

| 话题名 | 类型 | 方向 | 频率 | 说明 |
|--------|------|------|------|------|
| `/joint_states` | sensor_msgs/JointState | 发布 | 200Hz | 关节角度、速度、力矩 |

### 服务 (Services)

| 服务名 | 类型 | 说明 |
|--------|------|------|
| `/onero_arm_driver/enable` | std_srvs/Trigger | 使能电机 |
| `/onero_arm_driver/disable` | std_srvs/Trigger | 去使能电机 |

### 动作 (Actions)

| 动作名 | 类型 | 说明 |
|--------|------|------|
| `/arm_controller/follow_joint_trajectory` | control_msgs/FollowJointTrajectory | 轨迹执行 (兼容 MoveIt2) |

### 关节名

节点发布的关节名为 `joint1-a1_r` ~ `joint7-a1_r`（与 URDF 一致）。发送轨迹时必须使用这些名称。

### 参数 (Parameters)

```bash
# 查看所有参数
ros2 param list /onero_arm_driver

# 动态调参
ros2 param set /onero_arm_driver kp 100.0
ros2 param set /onero_arm_driver kd 3.0
```

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `can_channel` | string | "can0" | CAN 接口名 |
| `model` | string | "A1R" | 机械臂型号 |
| `control_rate` | double | 200.0 | 控制频率 (Hz) |
| `kp` | double | 50.0 | 位置增益 |
| `kd` | double | 1.0 | 速度增益 |

---

## 6. 与其他节点协作

### 通过 Action 发送轨迹

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = [
    "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
    "joint5-a1_r", "joint6-a1_r", "joint7-a1_r",
]

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self._action_client.wait_for_server()

    def go_to(self, positions, duration_sec=2.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=int(duration_sec))
        goal.trajectory.points = [pt]
        return self._action_client.send_goal(goal)
```

### 与视觉节点配合

```python
from geometry_msgs.msg import PoseStamped

class VisualGrasping(Node):
    def __init__(self):
        super().__init__('visual_grasping')
        self.create_subscription(
            PoseStamped, '/detection/target_pose',
            self.on_target_detected, 10)
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

    def on_target_detected(self, msg):
        # 通过 SDK 做 IK 解算后，用 action 发送关节轨迹
        # (需要在节点中嵌入 SDK 的 IK 求解器)
        raise NotImplementedError("需要接入真实 IK 求解器和轨迹 action 后再运行")
```

### 录制与回放 (rosbag)

```bash
# 录制关节状态
ros2 bag record /joint_states -o my_recording

# 回放
ros2 bag play my_recording
```

---

## 7. 常见问题

### CAN 连接失败

```
错误: Failed to open CAN channel: can0
```

解决:
```bash
# 检查 CAN 接口是否 UP
ip link show can0

# 重新配置
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000

# 检查 USB-CAN 设备
lsusb | grep -i can
dmesg | tail -20
```

### 节点启动后无 /joint_states

- 确认机械臂已上电
- 确认 CAN 总线波特率正确 (1Mbps)
- 检查: `candump can0` 能否收到报文

### 使能失败

- 检查是否有硬件报错: `ros2 topic echo /onero_arm_driver/status`
- 机械臂可能处于保护状态，需要断电重启

### 控制频率不够

- 确认不是 WiFi 远程连接（延迟大）
- 使用本地运行或有线连接
- 检查 CPU 负载: `htop`
