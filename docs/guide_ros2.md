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
    model:=A1 \
    control_freq:=200 \
    kp:=80.0 \
    kd:=2.0
```

启动后终端会输出：
```
[INFO] [onero_arm_driver]: Onero A1 node started
[INFO] [onero_arm_driver]: CAN channel: can0
[INFO] [onero_arm_driver]: Control freq: 200 Hz
[INFO] [onero_arm_driver]: Waiting for enable...
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

### 发送 movej 指令

```bash
# 命令行
ros2 service call /onero_arm_driver/movej onero_msgs/srv/MoveJ \
    "{positions: [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], speed: 1.0, acceleration: 2.0}"

# Python
python examples/ros2/02_ros2_movej_service.py
```

### 发送轨迹

```bash
# Python (推荐)
python examples/ros2/03_ros2_trajectory.py

# 命令行 (通过 action)
ros2 action send_goal /onero_arm_driver/follow_joint_trajectory \
    control_msgs/action/FollowJointTrajectory \
    "{trajectory: {joint_names: ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7'], \
    points: [{positions: [0,0,0,0,0,0,0], time_from_start: {sec: 2}}]}}"
```

---

## 5. ROS2 接口详解

### 话题 (Topics)

| 话题名 | 类型 | 方向 | 频率 | 说明 |
|--------|------|------|------|------|
| `/joint_states` | sensor_msgs/JointState | 发布 | 200Hz | 关节角度、速度、力矩 |
| `/onero_arm_driver/tcp_pose` | geometry_msgs/PoseStamped | 发布 | 200Hz | 末端位姿 |
| `/onero_arm_driver/status` | onero_msgs/ArmStatus | 发布 | 10Hz | 运行状态、温度、错误码 |

### 服务 (Services)

| 服务名 | 类型 | 说明 |
|--------|------|------|
| `/onero_arm_driver/enable` | std_srvs/Trigger | 使能电机 |
| `/onero_arm_driver/disable` | std_srvs/Trigger | 去使能电机 |
| `/onero_arm_driver/movej` | onero_msgs/MoveJ | 关节运动 |
| `/onero_arm_driver/movel` | onero_msgs/MoveL | 直线运动 |
| `/onero_arm_driver/stop` | std_srvs/Trigger | 停止运动 |
| `/onero_arm_driver/set_tcp` | onero_msgs/SetTCP | 设置工具坐标系 |
| `/onero_arm_driver/freedrive_enable` | std_srvs/Trigger | 开启零力拖动 |
| `/onero_arm_driver/freedrive_disable` | std_srvs/Trigger | 关闭零力拖动 |

### 动作 (Actions)

| 动作名 | 类型 | 说明 |
|--------|------|------|
| `/onero_arm_driver/follow_joint_trajectory` | control_msgs/FollowJointTrajectory | 轨迹执行 |

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
| `model` | string | "A1" | 机械臂型号 |
| `control_freq` | int | 200 | 控制频率 (Hz) |
| `kp` | double | 80.0 | 位置增益 |
| `kd` | double | 2.0 | 速度增益 |

---

## 6. 与其他节点协作

### 在自己的节点中调用

```python
import rclpy
from rclpy.node import Node
from onero_msgs.srv import MoveJ

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.movej_client = self.create_client(MoveJ, '/onero_arm_driver/movej')
        self.movej_client.wait_for_service()

    def go_to(self, positions, speed=1.0):
        req = MoveJ.Request()
        req.positions = positions
        req.speed = speed
        req.acceleration = 2.0
        future = self.movej_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success
```

### 与视觉节点配合

```python
# 订阅相机检测结果, 机械臂去抓取
from geometry_msgs.msg import PoseStamped

class VisualGrasping(Node):
    def __init__(self):
        super().__init__('visual_grasping')
        self.create_subscription(
            PoseStamped, '/detection/target_pose',
            self.on_target_detected, 10)
        self.movel_client = self.create_client(MoveL, '/onero_arm_driver/movel')

    def on_target_detected(self, msg):
        # msg.pose 是目标在基座坐标系下的位姿
        req = MoveL.Request()
        req.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.movel_client.call_async(req)
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
