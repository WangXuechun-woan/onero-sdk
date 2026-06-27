# Onero ROS2 — 机械臂驱动节点

ROS2 驱动节点，将 Onero SDK 桥接到 ROS2 生态。兼容 MoveIt2 的 FollowJointTrajectory action 接口。

## 架构

```
onero_arm_node (standalone Python node)
  ├── 200Hz 控制循环: read → command → publish
  ├── /joint_states (sensor_msgs/JointState)
  ├── /arm_controller/follow_joint_trajectory (action)
  ├── ~/enable (service)
  └── ~/disable (service)
           │
           ▼
      CanBackend (onero_sdk)
           │
           ▼
      SocketCAN (/dev/can0)
```

不依赖 ros2_control（pluginlib 仅支持 C++ 插件）。使用 standalone node 方案，
对外暴露的接口与 ros2_control 的 JointTrajectoryController 完全一致，MoveIt2 可直接对接。

## 前置条件

```bash
# 安装 Onero SDK
cd /path/to/a1_sdk
pip install -e .

# 配置 CAN 接口（真机使用）
sudo ip link set can0 up type can bitrate 1000000

# 确认 CAN 权限（避免每次 sudo）
sudo usermod -aG dialout $USER
```

## 安装

```bash
cd ros2
colcon build --packages-select onero_ros2
source install/setup.bash
```

## 使用

### 真机启动

```bash
# 默认参数
ros2 launch onero_ros2 bringup.launch.py

# 自定义
ros2 launch onero_ros2 bringup.launch.py can_channel:=can1 kp:=80.0 kd:=2.0
```

启动后需要先 enable：
```bash
ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger
```

### Mock 模式（无硬件）

```bash
ros2 launch onero_ros2 fake.launch.py
```
使用 joint_state_publisher_gui 手动拖动关节，用于 RViz 可视化调试。

### 验证

```bash
# 查看关节状态
ros2 topic echo /joint_states

# 发送轨迹
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: ['joint1-a1_r','joint2-a1_r','joint3-a1_r','joint4-a1_r','joint5-a1_r','joint6-a1_r','joint7-a1_r'], points: [{positions: [0.5, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], time_from_start: {sec: 3}}]}}"
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `can_channel` | `can0` | SocketCAN 接口名 |
| `model` | `A1` | 机器人型号 |
| `kp` | `50.0` | MIT 阻抗位置增益 |
| `kd` | `1.0` | MIT 阻抗阻尼增益 |
| `control_rate` | `200.0` | 控制频率 (Hz) |

## 话题和服务

| 接口 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | sensor_msgs/JointState | 关节状态 (200Hz) |
| `/arm_controller/follow_joint_trajectory` | control_msgs/FollowJointTrajectory | 轨迹执行 (MoveIt2 兼容) |
| `~/enable` | std_srvs/Trigger | 使能电机 |
| `~/disable` | std_srvs/Trigger | 失能电机 |

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `Failed to connect: No such device` | CAN 接口未 UP：`sudo ip link set can0 up type can bitrate 1000000` |
| `Permission denied` | 加用户组：`sudo usermod -aG dialout $USER` 并重新登录 |
| `Trajectory rejected` | 先调用 `~/enable` 服务使能电机 |
| 关节不动 | 检查 `ros2 topic echo /joint_states` 是否有数据 |
