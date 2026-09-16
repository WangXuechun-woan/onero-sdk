# Hardware / Control Alignment 交付说明

本文档记录当前 `master` 相对 `origin/master` 的待推送变更、验证结果和已知边界。行为规范以代码、README 和专题指南为准；本文用于发布复核，不替代真机安全流程。

## 1. 本次交付范围

### 1.1 安全关闭语义

- `Arm.close()`、上下文管理器退出和解释器退出只释放通信资源，默认**不发送失能命令**。
- 这是为了避免重力负载下突然失去力矩而掉臂。
- 只有机械臂得到可靠物理支撑后，才允许显式调用：

```python
arm.disable()
arm.close(disable=True)
```

关闭通信后 SDK 不再持续发送保持帧；驱动器自身看门狗行为必须单独确认。

### 1.2 SpeedJ / SpeedL 安全加固

- 使用积分位置参考 `q_ref += qdot * dt`，并限制参考超前量。
- 从实测速度开始进行加减速，不再假设初速度为零。
- 异常时尽力发送零速度位置保持帧。
- SpeedL 对整组关节速度按同一比例缩放，保持笛卡尔方向。
- 增加下一步关节限位保护、奇异点减速和硬阈值停止。
- A1 输出关节速度上限：J1–J4 为 `3 rad/s`，J5–J7 为 `6 rad/s`。
- 这些控制改动已通过仿真和单元测试，仍需在可靠支撑、低速和现场监护下完成真机验收。

### 1.3 坐标系与安装矩阵

统一世界坐标约定：

```text
+X = forward
+Y = left
+Z = up
```

URDF 内部根坐标仍为 `base_link`，安装差异只通过 `R_world_base` 和 `base_position` 表达：

```text
p_world = R_world_base · p_base + base_position
```

人形左右肩权威矩阵：

```text
humanoid_left                 humanoid_right
[[0, 1, 0],                  [[ 0,-1, 0],
 [0, 0, 1],                   [ 0, 0,-1],
 [1, 0, 0]]                   [ 1, 0, 0]]
```

`DualArm.create()` 默认使用 `humanoid_left/right`。`a1_vertical` 仅保留为兼容预设，其水平 yaw 等同于左臂，不能代替右肩矩阵。

### 1.4 双臂 MuJoCo 与演示

- `DualMujocoBackend.arm_backends()` 为同一物理场景提供左右两个 `ArmBackend` 视图。
- 演示通过 `DualArm -> Arm -> SafetyBackend -> MuJoCo ArmBackend` 执行，不再直接调用后端运动函数绕过 SDK。
- 镜像运动使用 `q_left = -q_right`。
- 最新 GIF 的 1600 个控制周期中，末端镜像误差最大 `1.698 mm`、P95 `1.497 mm`。
- 演示生成过程逐周期检查桌面接触；当前轨迹桌面接触周期为 `0`。

`DualArm` 是两个 `Arm` 的便利封装，不是通用协调规划器：

- 普通阻塞动作由两个线程并行执行，不保证硬实时同步。
- SDK 不提供双臂之间或桌面/墙体等环境几何体的自动避碰。
- MuJoCo 接触可阻止继续穿透，但不等于预碰撞路径规划。
- 需要通用几何避障时，应使用 MoveIt 或应用层规划器。

### 1.5 Debug UI

- 支持 MoveJ、MoveL、MoveP、Servo、Freedrive、目标拖拽、预览和显式 Stop。
- MoveL/MoveP 拖拽在 world 坐标中预览，松手后只提交一次阻塞动作。
- 阻塞动作与 Servo/Freedrive 互斥；完整 Cartesian IK 路径在发送前预检。
- 仿真模式只关闭硬件 tracking-error latch；真机安全策略不受影响。
- UI 中显示的桌面和双臂不是碰撞规划场景，操作者仍需确认真机目标安全。

建议启动命令：

```bash
# 双臂仿真
python -m onero_sdk.debugui --arm both --sim

# 人形左肩真机
python -m onero_sdk.debugui --arm left --channel can0 \
  --mount humanoid_left --gripper-compensation on
```

### 1.6 CAN transport

- Linux 原生 SocketCAN 仍为默认路径。
- 可选 `PythonCanBus` 支持 `python-can` 和直接 SLCAN 串口。
- 每个 `Arm` 独占一个 transport；`backend`、`channel`、`can_bus` 互斥。
- 用户显式选择 `A1L` / `A1R`；CAN 接口名和串口编号不推断机械臂侧别。
- 不允许 `slcand` 与直接 Python SLCAN 同时打开同一串口。

## 2. 验证结果

### 当前核心仿真

```text
46 passed in 91.97s
```

覆盖：

- 左右肩安装矩阵与零位方向
- 公共 DualArm 控制链
- 双臂镜像误差
- 双臂桌面无接触路径
- MuJoCo、MoveJ/MoveL/MoveC
- SpeedJ/SpeedL 安全约束

### 最近完整非硬件回归

```text
999 passed, 4 skipped, 1 failed
```

唯一失败是已知的 SafeBackend rate-limiter baseline 策略冲突：

```text
TestSafeBackendIntegration::test_reconcile_uses_feedback_not_command
baseline 0.3000 rad, feedback 0.2925 rad
```

此失败未隐藏。硬件测试因需要现场 fixture 与物理设备，不包含在非硬件回归中。

## 3. 推送前仍需人工确认

1. 在可靠支撑下低速验证真实 A1L/A1R 的 `+world X/+Y/+Z` 方向。
2. 校准真实 `base_position`、电机零偏和 TCP。
3. 真机验证 SpeedJ/SpeedL 的噪声、加速度、停止保持和关节限位。
4. 直接 Python SLCAN 尚需安装可选依赖并完成 200 Hz 单/双适配器基准。
5. 确认上述已知 SafeBackend 测试失败是否接受随本次版本保留，或在推送前单独修复。

## 4. 相关文档

- [README.md](../README.md)
- [README_en.md](../README_en.md)
- [坐标系指南](guide_coordinate_frames.md)
- [Debug UI 指南](guide_debugui.md)
- [中文快速开始](guide_getting_started.md)
- [English Getting Started](guide_getting_started_en.md)
- [MoveIt 指南](guide_moveit.md)
