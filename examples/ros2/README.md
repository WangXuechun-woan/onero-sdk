# ROS2 示例

通过 ROS2 接口控制 Onero A1 机械臂。

## 前提

```bash
# 1. 启动机械臂节点
ros2 launch onero_ros2 bringup.launch.py can_channel:=can0

# 2. 使能
ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger
```

## 示例列表

| # | 文件 | 说明 | 适用场景 |
|---|------|------|----------|
| 1 | `01_ros2_joint_states.py` | 订阅关节状态话题 | 基础连通测试 |
| 2 | `02_ros2_movej_service.py` | 通过 Service 发送 movej | 外部程序控制 |
| 3 | `03_ros2_trajectory.py` | FollowJointTrajectory action | 执行规划轨迹 |

## ROS2 接口一览

```
话题 (Topic):
  /joint_states              [sensor_msgs/JointState]     关节状态 (200Hz)

服务 (Service):
  /onero_arm_driver/enable   [std_srvs/Trigger]          使能
  /onero_arm_driver/disable  [std_srvs/Trigger]          去使能
  /onero_arm_driver/movej    [onero_msgs/MoveJ]          关节运动

动作 (Action):
  /onero_arm_driver/follow_joint_trajectory
                             [control_msgs/FollowJointTrajectory]  轨迹执行
```
