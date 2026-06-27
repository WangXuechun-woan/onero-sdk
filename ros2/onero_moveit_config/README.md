# Onero MoveIt2 Configuration

MoveIt2 配置包，用于 Onero A1 7-DOF 机械臂的运动规划。

## 依赖

- `onero_ros2` 包（hardware interface + URDF）
- MoveIt2 (moveit_ros, moveit_planners_ompl)
- RViz2

## 安装

```bash
cd ros2
colcon build --packages-select onero_moveit_config
source install/setup.bash
```

## 使用

### 1. 先启动硬件驱动

```bash
# 真机
ros2 launch onero_ros2 bringup.launch.py

# 或 mock 模式
ros2 launch onero_ros2 fake.launch.py
```

### 2. 启动 MoveIt2 + RViz

```bash
ros2 launch onero_moveit_config move_group.launch.py
```

在 RViz 中即可通过 MoveIt 插件拖拽规划和执行轨迹。

### 3. Python 编程接口

```python
import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_commander import MoveGroupCommander

rclpy.init()
node = rclpy.create_node('moveit_example')

# 使用 MoveIt Python API
move_group = MoveGroupCommander("arm", node=node)
move_group.set_named_target("ready")
move_group.go(wait=True)

# 笛卡尔直线运动
from geometry_msgs.msg import Pose
pose = Pose()
pose.position.x = 0.3
pose.position.y = 0.0
pose.position.z = 0.4
pose.orientation.w = 1.0
move_group.set_pose_target(pose)
move_group.go(wait=True)
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `a1.srdf` | 语义描述：planning group、named poses、碰撞矩阵 |
| `kinematics.yaml` | IK 求解器配置（KDL） |
| `joint_limits.yaml` | 规划用速度/加速度限制（比硬件限制更保守） |
| `moveit_controllers.yaml` | MoveIt → ros2_control 映射 |
| `ompl_planning.yaml` | OMPL 规划器配置（RRTConnect 等） |

## Named Poses

| 名称 | 说明 |
|------|------|
| `home` | 所有关节零位 |
| `ready` | 预备姿态（臂弯曲，适合抓取） |

## 规划器

默认使用 RRTConnect（最快），也可选：
- RRTstar: 渐近最优
- PRM: 适合重复查询
- BiTRRT: 考虑代价的规划
