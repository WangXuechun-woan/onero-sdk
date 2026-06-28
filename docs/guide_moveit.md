# MoveIt2 使用指南

本文档介绍如何使用 MoveIt2 运动规划框架控制 Onero A1 机械臂。

MoveIt 提供：碰撞检测、运动学求解、轨迹优化、场景管理。适合需要避障或复杂路径规划的场景。

---

## 目录

1. [环境准备](#1-环境准备)
2. [安装 MoveIt 配置包](#2-安装-moveit-配置包)
3. [启动 MoveIt](#3-启动-moveit)
4. [使用 RViz 可视化操作](#4-使用-rviz-可视化操作)
5. [Python API 编程](#5-python-api-编程)
6. [添加碰撞物体](#6-添加碰撞物体)
7. [笛卡尔路径规划](#7-笛卡尔路径规划)
8. [与夹爪配合](#8-与夹爪配合)
9. [高级配置](#9-高级配置)
10. [常见问题](#10-常见问题)

---

## 1. 环境准备

### 依赖安装

```bash
# MoveIt2 核心
sudo apt install ros-${ROS_DISTRO}-moveit

# MoveIt Python 绑定
sudo apt install ros-${ROS_DISTRO}-moveit-py

# 可视化
sudo apt install ros-${ROS_DISTRO}-rviz2 ros-${ROS_DISTRO}-moveit-visual-tools

# 规划器 (可选, OMPL 是默认规划器)
sudo apt install ros-${ROS_DISTRO}-moveit-planners-ompl
```

### 确认 onero_ros2 已安装

MoveIt 通过 onero_ros2 节点与硬件通信：
```bash
cd ~/onero_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg list | grep onero
```

---

## 2. 安装 MoveIt 配置包

```bash
cd ~/onero_ws/src

# 链接 MoveIt 配置包
ln -s /path/to/a1_sdk/ros2/onero_moveit_config .

# 编译
cd ~/onero_ws
colcon build --symlink-install --packages-select onero_moveit_config
source install/setup.bash
```

配置包结构：
```
onero_moveit_config/
├── config/
│   ├── onero_a1.srdf          # 运动组、末端执行器定义
│   ├── joint_limits.yaml      # 关节限制
│   ├── kinematics.yaml        # IK 求解器配置
│   ├── ompl_planning.yaml     # OMPL 规划器参数
│   └── controllers.yaml       # 控制器映射
├── launch/
│   ├── move_group.launch.py   # MoveIt 主启动文件
│   └── demo.launch.py         # 仿真演示 (无硬件)
└── urdf/
    └── onero_a1.urdf.xacro    # 机器人描述
```

---

## 3. 启动 MoveIt

### 真机模式

```bash
# 终端 1: 启动机械臂节点
ros2 launch onero_ros2 bringup.launch.py can_channel:=can0

# 终端 2: 启动 MoveIt + RViz
ros2 launch onero_moveit_config move_group.launch.py
```

### 仿真演示模式 (无硬件)

```bash
# 单条命令, 使用 fake controllers
ros2 launch onero_moveit_config demo.launch.py
```

启动成功后会打开 RViz 窗口，显示机械臂 3D 模型。

---

## 4. 使用 RViz 可视化操作

RViz 中可以交互式地规划和执行运动：

### 基本操作

1. **拖动末端球体** → 设置目标位姿（橙色臂显示目标）
2. **Plan** 按钮 → 计算轨迹（绿色轨迹线）
3. **Execute** 按钮 → 发送轨迹到真机执行
4. **Plan & Execute** → 一步到位

### 界面说明

```
MotionPlanning 面板:
├── Planning (规划标签)
│   ├── Planning Group: arm         ← 选择运动组
│   ├── Start State: current        ← 起始状态
│   └── Goal State: [拖动/手输]     ← 目标状态
├── Scene Objects (场景物体)
│   └── 导入碰撞物体 (.stl/.obj)
└── Planning (轨迹显示)
    ├── Show Trail: ☑              ← 显示轨迹
    └── Loop Animation: ☑          ← 循环播放
```

### 使用命名姿态

在 Planning 标签中 Goal State 下拉选择：
- `home` — 默认工作位
- `zero` — 零位（所有关节 0°）

---

## 5. Python API 编程

### 基础模板

```python
import rclpy
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

rclpy.init()
moveit = MoveItPy(node_name="my_moveit_script")
arm = moveit.get_planning_component("arm")

# --- 规划到命名姿态 ---
arm.set_start_state_to_current_state()
arm.set_goal_state(configuration_name="home")
result = arm.plan()
if result:
    arm.execute()

# --- 规划到指定关节角 ---
robot_state = RobotState(moveit.get_robot_model())
robot_state.set_joint_group_positions("arm", [0.5, -0.3, 0.2, -1.2, 0.0, 0.8, 0.0])
arm.set_start_state_to_current_state()
arm.set_goal_state(robot_state=robot_state)
result = arm.plan()
if result:
    arm.execute()

# --- 规划到末端位姿 ---
from geometry_msgs.msg import PoseStamped
target = PoseStamped()
target.header.frame_id = "base_link"
target.pose.position.x = 0.3
target.pose.position.y = 0.0
target.pose.position.z = 0.2
target.pose.orientation.x = 0.0
target.pose.orientation.y = 1.0  # 朝下
target.pose.orientation.z = 0.0
target.pose.orientation.w = 0.0

arm.set_start_state_to_current_state()
arm.set_goal_state(pose_stamped_msg=target, pose_link="tool0")
result = arm.plan()
if result:
    arm.execute()

moveit.shutdown()
rclpy.shutdown()
```

### 运行示例

```bash
# 确保 MoveIt 已启动
python examples/moveit/01_moveit_basic.py
python examples/moveit/02_moveit_cartesian.py
python examples/moveit/03_moveit_pick_place.py
```

---

## 6. 添加碰撞物体

在规划场景中加入障碍物，MoveIt 会自动避开。

```python
from moveit.core.planning_scene import PlanningScene
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive

scene = moveit.get_planning_scene_monitor()

# 添加一个桌子 (长方体)
with scene.read_write() as rw_scene:
    collision_object = rw_scene.make_collision_object("table")
    collision_object.header.frame_id = "base_link"

    # 桌面: 1m x 0.6m x 0.02m, 在 Z=0 处
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.0, 0.6, 0.02]

    pose = Pose()
    pose.position.x = 0.4
    pose.position.y = 0.0
    pose.position.z = -0.01  # 桌面略低于基座

    collision_object.primitives.append(box)
    collision_object.primitive_poses.append(pose)
    rw_scene.apply_collision_object(collision_object)

print("桌子已添加到规划场景")
# 现在规划会自动避开桌面
```

### 从文件加载碰撞模型

```python
# 导入 STL/OBJ 网格文件作为碰撞物体
from shape_msgs.msg import Mesh
# scene.add_mesh("shelf", mesh_path="path/to/shelf.stl", pose=...)
```

---

## 7. 笛卡尔路径规划

让末端沿精确的直线或曲线运动（不仅仅是 point-to-point）：

```python
from geometry_msgs.msg import Pose, Point, Quaternion

# 定义路径点 (末端沿直线经过这些点)
waypoints = [
    Pose(position=Point(x=0.3, y=-0.05, z=0.2),
         orientation=Quaternion(x=0, y=1, z=0, w=0)),
    Pose(position=Point(x=0.3, y=0.05, z=0.2),
         orientation=Quaternion(x=0, y=1, z=0, w=0)),
    Pose(position=Point(x=0.3, y=0.05, z=0.3),
         orientation=Quaternion(x=0, y=1, z=0, w=0)),
]

arm.set_start_state_to_current_state()
fraction = arm.compute_cartesian_path(
    waypoints=waypoints,
    max_step=0.005,       # 插值步长 5mm (越小越精确, 越慢)
    jump_threshold=0.0,   # 0 = 禁止关节突变
)

if fraction > 0.95:
    print(f"路径规划成功: {fraction*100:.1f}% 可达")
    arm.execute()
else:
    print(f"路径规划不完整: 仅 {fraction*100:.1f}% 可达")
    # 尝试调整姿态或路径
```

**关键参数:**
- `max_step`: 路径插值步长（米）。0.001=1mm 极精确但慢，0.01=1cm 快但粗糙
- `jump_threshold`: 关节空间跳变检测。设 0 禁止任何突变（安全），设 >0 允许一定的关节跳变

---

## 8. 与夹爪配合

### 配置末端执行器

在 `onero_a1.srdf` 中定义夹爪：
```xml
<end_effector name="gripper" parent_link="tool0" group="gripper"/>
<group name="gripper">
    <joint name="gripper_joint"/>
</group>
```

### 抓取-放置流程

```python
# 完整的 pick & place
# 参见 examples/moveit/03_moveit_pick_place.py

# 1. 到预抓取位 (目标上方)
arm.set_goal_state(pose_stamped_msg=pre_grasp_pose, pose_link="tool0")
arm.plan() and arm.execute()

# 2. 直线下降到抓取点
arm.compute_cartesian_path(waypoints=[grasp_pose], ...)
arm.execute()

# 3. 闭合夹爪
gripper.set_goal_state(configuration_name="close")
gripper.plan() and gripper.execute()

# 4. 直线抬起
arm.compute_cartesian_path(waypoints=[lift_pose], ...)
arm.execute()

# 5. 移动到放置位
arm.set_goal_state(pose_stamped_msg=place_pose, pose_link="tool0")
arm.plan() and arm.execute()

# 6. 打开夹爪
gripper.set_goal_state(configuration_name="open")
gripper.plan() and gripper.execute()
```

---

## 9. 高级配置

### 更换规划器

```python
# 默认: RRTConnect (快, 路径不最优)
arm.set_planner_id("RRTConnectkConfigDefault")

# 路径质量更好 (慢):
arm.set_planner_id("RRTstarkConfigDefault")

# 多约束场景:
arm.set_planner_id("PRMstarkConfigDefault")
```

### 规划时间

```python
# 复杂场景可能需要更多时间
arm.set_planning_time(10.0)       # 默认 5s
arm.set_num_planning_attempts(5)  # 多次尝试取最优
```

### 速度缩放

```python
# 降低执行速度 (安全调试时)
arm.set_max_velocity_scaling_factor(0.3)      # 30% 速度
arm.set_max_acceleration_scaling_factor(0.2)  # 20% 加速度
```

### 约束规划

```python
from moveit_msgs.msg import Constraints, OrientationConstraint

# 保持末端朝下 (不翻转)
oc = OrientationConstraint()
oc.link_name = "tool0"
oc.header.frame_id = "base_link"
oc.orientation.y = 1.0  # 朝下
oc.absolute_x_axis_tolerance = 0.3
oc.absolute_y_axis_tolerance = 0.3
oc.absolute_z_axis_tolerance = 3.14  # Z 轴自由
oc.weight = 1.0

constraints = Constraints()
constraints.orientation_constraints.append(oc)
arm.set_path_constraints(constraints)
```

---

## 10. 常见问题

### 规划总是失败

| 可能原因 | 解决方法 |
|----------|----------|
| 目标不可达 | 检查目标是否在工作空间内 |
| 碰撞检测拦截 | RViz 中查看碰撞物体位置是否正确 |
| IK 无解 | 调整末端姿态（7-DOF 有冗余，换个姿态试试） |
| 规划时间不够 | `set_planning_time(20.0)` |
| 起始状态有碰撞 | 先手动 movej 离开碰撞位 |

### 笛卡尔路径覆盖率低

- 路径经过奇异点（肘部完全伸直/折叠） → 绕过奇异点
- max_step 太大 → 减小到 0.002
- 中间有障碍物 → 分段规划

### 执行时机械臂抖动

- `kp` 过高 → 降低到 50~80
- 轨迹点太密 → `max_step` 适当增大
- 控制频率不匹配 → 确保 onero_ros2 节点跑在 200Hz

### MoveIt 与 onero_ros2 不通信

```bash
# 检查 controller 是否正确加载
ros2 control list_controllers

# 检查 joint_states 话题是否有数据
ros2 topic hz /joint_states
# 应为 ~200 Hz
```
