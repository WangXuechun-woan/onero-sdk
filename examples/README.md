# Onero A1 SDK 示例集

本目录包含 Onero A1 机械臂的完整示例，覆盖从入门到进阶的各种使用场景。

---

## 快速导航

| 我想要... | 看哪个示例 |
|-----------|-----------|
| 快速跑起来试试 | [01_quickstart_sim.py](#01-仿真快速开始) |
| 让臂走多个点位 | [02_multi_waypoint.py](#02-多路点运动) |
| 处理异常和错误 | [03_error_handling.py](#03-异常处理) |
| 画圆/画弧线 | [04_circular_motion.py](#04-圆弧运动) |
| 调整刚度/柔顺 | [05_stiffness.py](#05-刚度调节) |
| 读取关节状态 | [06_state_monitoring.py](#06-状态监控) |
| 紧急停止/安全 | [07_safe_stop.py](#07-安全停止) |
| 力控/力矩控制 | [08_force_control.py](#08-力控) |
| 实时伺服(高频) | [09_realtime_servo.py](#09-实时伺服) |
| 了解坐标系 | [10_coordinate_systems.py](#10-坐标系) |
| 示教录制与回放 | [11_teach_replay.py](#11-示教回放) |
| 连接真实硬件 | [12_hardware_setup.py](#12-硬件配置) |
| 夹爪/工具坐标 | [13_tool_frame.py](#13-工具坐标系) |
| 相机标定/手眼标定 | [14_camera_calibration.py](#14-相机与手眼标定) |
| 通过 ROS2 控制 | [ros2/](#ros2-示例) |
| MoveIt 运动规划 | [moveit/](#moveit-示例) |
| LeRobot 遥操作/AI | [lerobot/](#lerobot-示例) |

---

## SDK 直接调用示例

适用场景：直接通过 Python SDK 控制机械臂，无需 ROS2 或其他中间层。适合快速验证、嵌入式部署、自定义控制。

### 01 仿真快速开始
**文件:** `01_quickstart_sim.py`

零硬件依赖，使用 MuJoCo 仿真环境快速体验 SDK 接口。适合初次接触时运行。

**适用场景:** 刚拿到 SDK、没有硬件时的第一步。

### 02 多路点运动
**文件:** `02_multi_waypoint.py`

让机械臂依次经过多个目标点位，包括路径平滑（blending）。

**适用场景:** 搬运、码垛、巡检等需要走多个固定点的任务。

### 03 异常处理
**文件:** `03_error_handling.py`

演示 SDK 可能抛出的各种异常及正确处理方式。

**适用场景:** 编写生产级代码时参考，确保程序不会因异常而失控。

### 04 圆弧运动
**文件:** `04_circular_motion.py`

末端沿圆弧路径运动（movec），通过 3 个点定义一段圆弧。

**适用场景:** 涂胶、焊接、打磨等需要曲线路径的任务。

### 05 刚度调节
**文件:** `05_stiffness.py`

动态调整关节刚度（kp/kd），实现从"硬"到"软"的切换。

**适用场景:** 人机协作、柔顺装配、力位混合控制。

### 06 状态监控
**文件:** `06_state_monitoring.py`

实时读取关节角度、速度、力矩等状态信息。

**适用场景:** 调试、数据记录、状态可视化。

### 07 安全停止
**文件:** `07_safe_stop.py`

多线程下安全停止运动中的机械臂，包括减速停止和急停。

**适用场景:** 安全系统集成、紧急情况处理、多线程控制架构。

### 08 力控
**文件:** `08_force_control.py`

力/力矩控制模式，末端在指定方向上施加恒力。

**适用场景:** 打磨（恒力接触）、插装（力引导）、表面探测。

### 09 实时伺服
**文件:** `09_realtime_servo.py`

200Hz 实时伺服（servoj），直接发送关节角命令，无内部插值。

**适用场景:** 自定义轨迹生成器、视觉伺服闭环、高频控制算法。

### 10 坐标系
**文件:** `10_coordinate_systems.py`

关节空间 vs 笛卡尔空间，正运动学/逆运动学，位姿表示方式。

**适用场景:** 理解基础概念、关节运动与末端运动的转换关系。

### 11 示教回放
**文件:** `11_teach_replay.py`

Freedrive（零力拖动）模式下录制轨迹，然后精确回放。

**适用场景:** 快速编程（无需手写坐标）、演示录制、轨迹采集。

### 12 硬件配置
**文件:** `12_hardware_setup.py`

连接真实硬件的完整流程：CAN 总线配置、使能、状态检查。

**适用场景:** 首次连接硬件、排查连接问题。

### 13 工具坐标系
**文件:** `13_tool_frame.py`

设置 TCP（工具中心点）偏移，使运动指令以工具尖端为参考。

**适用场景:** 安装了夹爪、灵巧手、吸盘、焊枪等工具后的坐标系配置。

### 14 相机与手眼标定
**文件:** `14_camera_calibration.py`

腕部相机坐标系变换、手眼标定流程、相机检测结果转换到机械臂坐标。

**适用场景:**
- 腕部安装相机后，需要知道"相机看到的目标在哪"
- 视觉抓取（相机检测 → 坐标变换 → 机械臂抓取）
- Eye-in-hand / Eye-to-hand 标定

---

## ROS2 示例

适用场景：通过 ROS2 中间层控制机械臂，适合多机器人系统、与其他 ROS 节点协作、使用 ROS 生态工具（rviz, rosbag 等）。

**前提:** 启动 onero_ros2 节点
```bash
ros2 launch onero_ros2 bringup.launch.py can_channel:=can0
ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger
```

### ros2/01 关节状态订阅
**文件:** `ros2/01_ros2_joint_states.py`

订阅 `/joint_states` 话题，实时显示关节信息。

**适用场景:** 基础连通测试、了解 ROS2 话题接口。

### ros2/02 服务调用 movej
**文件:** `ros2/02_ros2_movej_service.py`

通过 ROS2 Service 发送 movej 指令。

**适用场景:** 外部程序（C++/Python/其他语言）通过 ROS2 接口控制机械臂。

### ros2/03 轨迹执行
**文件:** `ros2/03_ros2_trajectory.py`

发送 `FollowJointTrajectory` action，兼容 MoveIt 输出格式。

**适用场景:** 执行 MoveIt 或自定义规划器生成的轨迹。

---

## MoveIt 示例

适用场景：使用 MoveIt2 运动规划框架，自动处理碰撞检测、逆运动学、轨迹优化。适合复杂环境中的运动规划。

**前提:** 启动 MoveIt
```bash
ros2 launch onero_moveit_config move_group.launch.py
```

### moveit/01 基础规划
**文件:** `moveit/01_moveit_basic.py`

关节空间规划与执行，运动到命名姿态或指定关节角。

**适用场景:** MoveIt 入门、简单的 point-to-point 运动。

### moveit/02 笛卡尔路径
**文件:** `moveit/02_moveit_cartesian.py`

笛卡尔空间直线/圆弧规划，末端沿精确路径运动。

**适用场景:** 焊接、涂胶、切割等对路径精度有要求的任务。

### moveit/03 抓取放置
**文件:** `moveit/03_moveit_pick_place.py`

完整的 Pick & Place 流程：预抓取 → 下降 → 夹取 → 抬起 → 移动 → 放置。

**适用场景:** 物流分拣、装配线上的取放操作。

---

## LeRobot 示例

适用场景：使用 LeRobot 开源框架进行遥操作数据采集和 AI 策略训练/推理。适合具身智能研究、VLA 算法验证。

**前提:** 安装插件
```bash
pip install -e lerobot/lerobot_robot_onero
pip install -e lerobot/lerobot_teleoperator_onero
```

### lerobot/01 遥操作
**文件:** `lerobot/01_lerobot_teleoperate.py`

双臂 Leader-Follower 遥操作，主手拖动、从手跟随。

**适用场景:** 最基础的遥操作体验、验证硬件连通性。

### lerobot/02 数据采集
**文件:** `lerobot/02_lerobot_record.py`

遥操作同时录制 episode 数据，保存为 HuggingFace Dataset 格式。

**适用场景:** 为策略训练（ACT/Diffusion Policy）采集演示数据。

### lerobot/03 数据回放
**文件:** `lerobot/03_lerobot_replay.py`

加载已采集的 dataset，在真机上回放验证数据质量。

**适用场景:** 采集后验证数据是否正确、动作是否流畅。

### lerobot/04 策略推理
**文件:** `lerobot/04_lerobot_policy.py`

加载训练好的模型（ACT / Diffusion Policy / VLA），在真机上自主执行。

**适用场景:** 模型训练完成后的真机验证和部署。

### lerobot/05 跨臂遥操作
**文件:** `lerobot/05_cross_arm_teleop.py`

用不同型号的机械臂（如 SO101）作为主臂，控制 Onero A1 作为从臂。通过任务空间映射解决臂形差异。

**适用场景:**
- 手边没有两台 A1，但有一台 SO101 可以当主手
- 不同品牌/自由度的臂之间做遥操作
- 末端位移缩放、坐标系对齐

---

## 典型工作流

### 从零开始
```
01_quickstart_sim → 12_hardware_setup → 02_multi_waypoint
```

### 视觉抓取
```
13_tool_frame → 14_camera_calibration → moveit/03_pick_place
```

### AI 训练数据采集
```
lerobot/01_teleoperate → lerobot/02_record → lerobot/03_replay → lerobot/04_policy
```

### ROS2 集成
```
ros2/01_joint_states → ros2/02_movej_service → ros2/03_trajectory → moveit/01_basic
```

---

## 其他示例

以下是一些独立的功能演示脚本:

| 文件 | 说明 |
|------|------|
| `basic_movej.py` | 最简 movej 调用 |
| `basic_movel.py` | 最简 movel 调用 |
| `blending.py` | 路径平滑过渡 |
| `dual_arm.py` | 双臂协同控制 |
| `freedrive_teach.py` | 零力示教模式 |
| `gripper_pick_place.py` | 夹爪抓取 |
| `mit_control.py` | MIT 阻抗底层控制 |
| `mujoco_demo.py` | 单臂 MuJoCo 仿真演示 |
| `dual_arm_mujoco_demo.py` | 双臂共享场景仿真；通过公开 `DualArm` 控制链并检查桌面接触 |
| `position_control.py` | 位置控制模式 |
| `servo_control.py` | 伺服控制 |
| `speed_control.py` | 速度控制模式 |
| `trajectory_planning_demo.py` | 轨迹规划演示 |
| `calibration_demo.py` | 运动学标定 |
