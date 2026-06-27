# Onero ROS2 — ros2_control Hardware Interface

ROS2 Control hardware interface for Onero robot arms, enabling integration with the standard ROS2 control ecosystem (MoveIt2, joint_trajectory_controller, etc.).

## 依赖

- ROS2 Humble 或更高版本
- `onero-sdk` (pip install)
- ros2_control, ros2_controllers
- xacro, robot_state_publisher

## 安装

```bash
# 1. 安装 Onero SDK
cd /path/to/a1_sdk
pip install -e .

# 2. 编译 ROS2 包
cd ros2
colcon build --packages-select onero_ros2
source install/setup.bash
```

## 使用

### 真机启动

```bash
# 确保 CAN 接口已配置
sudo ip link set can0 up type can bitrate 1000000

# 启动 (默认 can0, kp=50, kd=1)
ros2 launch onero_ros2 bringup.launch.py

# 自定义参数
ros2 launch onero_ros2 bringup.launch.py can_channel:=can1 kp:=80.0 kd:=2.0
```

### Mock 模式（无硬件）

```bash
ros2 launch onero_ros2 fake.launch.py
```

### 验证

```bash
# 查看关节状态
ros2 topic echo /joint_states

# 查看可用 controllers
ros2 control list_controllers

# 发送关节轨迹
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: ['joint1-a1_r','joint2-a1_r','joint3-a1_r','joint4-a1_r','joint5-a1_r','joint6-a1_r','joint7-a1_r'], points: [{positions: [0.5, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], time_from_start: {sec: 3}}]}}"
```

## 架构

```
ros2_control_node
  └── OneroHardwareInterface (plugin)
        └── CanBackend (onero_sdk)
              └── SocketCAN (/dev/can0)

Controllers:
  - joint_state_broadcaster → /joint_states
  - arm_controller (JointTrajectoryController) → position commands
  - forward_velocity_controller → velocity commands
  - forward_effort_controller → torque commands
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `can_channel` | `can0` | SocketCAN 接口名 |
| `model` | `A1` | 机器人型号 |
| `kp` | `50.0` | MIT 阻抗位置增益 |
| `kd` | `1.0` | MIT 阻抗阻尼增益 |

## 控制模式

Hardware interface 暴露三种命令接口：

- **position**: 通过 MIT 阻抗控制跟踪目标位置（默认，MoveIt2 使用此模式）
- **velocity**: 直接速度指令
- **effort**: 直接力矩指令

切换控制器即切换模式：
```bash
# 切换到速度控制
ros2 control switch_controllers --activate forward_velocity_controller --deactivate arm_controller
```
