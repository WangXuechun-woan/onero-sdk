# onero SDK

用于控制 onero A1 机械臂的 Python SDK。

## 功能特性

- 高级运动 API：`movej`、`movel`、`movep`、`movec`、`servoj`、`servop`
- 多种轨迹曲线：余弦、梯形速度、S 曲线（jerk-limited）
- 速度控制：`speedj`、`speedl`
- 示教模式 / freedrive（基于 pinocchio 重力补偿）
- 力控：混合力/位置模式
- 动力学原语：雅可比矩阵、重力补偿力矩、科氏力力矩
- 动力学参数标定（辨识）
- 运动学标定（单点约束法，提高绝对定位精度）
- 碰撞检测与末端外力估计
- 高级轨迹规划：Ruckig（在线 jerk-limited）、TOPP-RA（时间最优）
- 可配置负载（影响动力学精度）
- 轨迹录制与回放
- 多路点轨迹 + blending 平滑过渡
- 可插拔后端架构（真实 CAN 硬件或 MuJoCo 仿真）
- MuJoCo 双臂仿真（人体躯干 + 桌面场景）
- 末端执行器支持（夹爪、灵巧手）
- 内置安全限制（速度钳位、关节限位、超时保护）
- 模型注册表 — 通过名称和通道即可创建机械臂实例
- 轻量 FK（基于 URDF，无外部依赖）
- 可选 IK（pinocchio / pink）
- 直接 SocketCAN 通信

## 安装

```bash
pip install -e .              # 核心（CAN + FK + 电机控制）
pip install -e ".[ik]"        # + 逆运动学（pinocchio、pink）
pip install -e ".[dynamics]"  # + 动力学（示教模式、力控）
pip install -e ".[sim]"       # + MuJoCo 仿真
pip install -e ".[trajectory]" # + 高级规划（Ruckig、TOPP-RA）
pip install -e ".[full]"      # 全部安装
```

## 快速开始

### 关节运动

```python
from onero_sdk import Arm

with Arm("A1", channel="can0") as arm:
    # 指定运动时间
    arm.movej([0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0], duration=3.0)

    # 或指定速度/加速度（UR 兼容风格）
    arm.movej([0.3, 0.8, 0.0, -0.5, 0.0, 0.0, 0.0], speed=1.5, acceleration=1.4)

    arm.home()
```

### 末端直线运动

```python
import numpy as np
from onero_sdk import Arm

with Arm("A1", channel="can0") as arm:
    # 指定时间
    arm.movel(position=[0.3, 0.0, 0.2], rotation=np.eye(3), duration=3.0)

    # 指定末端速度 (m/s) — 自动计算运动时间
    arm.movel([0.3, 0.1, 0.3, 0, 0, 0], speed=0.25, acceleration=1.2)
```

### 实时伺服控制（100Hz）

```python
import time, math
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

try:
    for step in range(500):
        t = step / 100.0
        joints = [0.0] * 7
        joints[1] = 0.5 * math.sin(2 * math.pi * 0.5 * t)
        arm.servoj(joints)
        time.sleep(1/100)
finally:
    arm.disable()
```

### 速度控制

```python
from onero_sdk import Arm

with Arm("A1", channel="can0") as arm:
    # 关节1以0.5 rad/s运动3秒（带加速斜坡）
    arm.speedj([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], acceleration=1.0, duration=3.0)

    # 笛卡尔速度（需要dynamics）
    arm.speedl([0.0, 0.0, -0.05, 0.0, 0.0, 0.0], acceleration=0.5, duration=2.0)
```

### 示教模式（freedrive）+ 录制

```python
import time
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

# 示教模式下录制（全轴自由）
arm.start_recording()
arm.freedrive_enable(damping=0.3)
time.sleep(10.0)  # 用户引导机械臂
arm.freedrive_disable()
recording = arm.stop_recording()

# 保存并回放
recording.save("trajectory.json")
arm.replay(recording, speed_factor=1.0)

# 约束示教：只允许 Z 轴方向移动
arm.freedrive_enable(damping=0.5, free_axes=[0, 0, 1, 0, 0, 0])

arm.disable()
```

### 多路点轨迹 + blending

```python
from onero_sdk import Arm, TrajectoryPlanner, Waypoint

planner = TrajectoryPlanner(num_joints=7, control_freq=100.0)
planner.add_waypoint(Waypoint(joints=[0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0], duration=2.0, blend=0.3))
planner.add_waypoint(Waypoint(joints=[0.3, 0.8, 0.2, -0.5, 0.0, 0.1, 0.0], duration=2.0, blend=0.3))
planner.add_waypoint(Waypoint(joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0))

with Arm("A1", channel="can0") as arm:
    frames = planner.compute(arm.state.joints)
    arm.execute_trajectory(frames)
```

### 力控模式（混合力/位置）

```python
import time
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

try:
    # 在z方向施加10N向下力，x/y保持位置
    # task_frame: 力方向的参考坐标系 [x, y, z, rx, ry, rz]
    # limits: 力控方向的速度上限 (m/s, rad/s)
    for _ in range(1000):
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],       # z = 力控
            wrench=[0, 0, -10.0, 0, 0, 0],       # 10N向下
            task_frame=[0, 0, 0, 0, 0, 0],       # 基坐标系
            limits=[0, 0, 0.1, 0, 0, 0],         # z方向最大0.1 m/s
        )
        time.sleep(1/100)

    arm.force_mode_disable()
finally:
    arm.disable()
```

### 圆弧运动（movec）

```python
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

try:
    # 末端沿圆弧运动
    # 位姿格式: [x, y, z, rx, ry, rz]（米，旋转向量，单位rad）
    via_pose = [0.3, 0.1, 0.4, 0, 0, 0]
    target_pose = [0.3, -0.1, 0.4, 0, 0, 0]

    arm.movec(via_pose, target_pose, duration=3.0)
finally:
    arm.disable()
```

### 碰撞检测

```python
from onero_sdk import Arm
from onero_sdk.safety import SafetyConfig

# 配置碰撞阈值 (Nm)
safety = SafetyConfig(collision_threshold=5.0, collision_stop=True)
arm = Arm("A1", channel="can0", safety=safety)
arm.enable()

# 手动检查（collision_stop=False 时有用）
collision = arm.check_collision()
if collision:
    print("检测到外部接触!")
```

### 负载配置

```python
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

# 设置工具负载以提高动力学精度
# （影响示教模式、碰撞检测、外力估计）
arm.set_payload(mass=0.5, cog=[0.0, 0.0, 0.05])
```

### 末端外力估计

```python
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

# 估计末端执行器受到的外力/力矩
wrench = arm.get_external_wrench()
# wrench = [fx, fy, fz, tx, ty, tz]，单位 N 和 Nm
print(f"力: {wrench[:3]}, 力矩: {wrench[3:]}")
```

### 动力学原语（雅可比、重力补偿、科氏力）

```python
import numpy as np
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

# 末端雅可比矩阵 (6x7)，世界坐标系
J = arm.jacobian()

# 重力补偿力矩
g = arm.gravity_torque()

# 科氏力 + 离心力力矩（静止时约为零）
c = arm.coriolis_torque()

# 示例：笛卡尔空间阻抗控制
Kp = np.diag([200, 200, 200, 50, 50, 50])
Kd = np.diag([20, 20, 20, 5, 5, 5])
x_error = x_desired - x_current   # 6D 位姿误差
dx_error = dx_desired - dx_current # 6D 速度误差
F = Kp @ x_error + Kd @ dx_error
tau = J.T @ F + np.array(g)
arm.servoj(arm.state.joints, kp=0, kd=0, tau_ff=tau.tolist())
```

### 动力学参数标定

```python
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

# 标定动力学参数（提高重力补偿精度）
result = arm.calibrate_dynamics(num_configs=40, settle_time=0.5)
print(f"RMS 残差: {result.residual_rms:.4f} Nm")
print(f"库仑摩擦: {result.friction_coulomb}")

# 标定后，示教模式 / 外力估计精度显著提升
arm.disable()
```

### 运动学标定（提高绝对定位精度）

```python
from onero_sdk import Arm, KinematicCalibrator

arm = Arm("A1", channel="can0")
arm.enable()

# 从不同构型触碰同一个固定参考点（如桌角）
calibrator = KinematicCalibrator.from_arm(arm)
for i in range(20):
    input(f"触碰参考点（第 {i+1}/20 次），按 Enter 记录")
    calibrator.record_touch(arm=arm)

# 解算运动学修正
result = calibrator.solve()
print(f"残差: {result.residual_mm:.2f} mm")
result.save("kinematic_cal.json")

# 应用修正 — 后续所有 FK/IK 使用校正后的参数
arm.apply_kinematic_calibration(result)

arm.disable()
```

### 高级轨迹规划

```python
from onero_sdk import Arm, TrapezoidalProfile, RuckigPlanner, TimeOptimalPlanner

arm = Arm("A1", channel="can0")
arm.enable()

# 1. 梯形 / S 曲线（内置，无额外依赖）
arm.movej(target, speed=2.0, acceleration=5.0)                # 梯形（自动）
arm.movej(target, speed=2.0, acceleration=5.0, profile='s_curve')  # jerk-limited

# 2. Ruckig — 在线 jerk-limited（pip install ruckig）
planner = RuckigPlanner(num_joints=7)
planner.set_limits(v_max=[2.5]*7, a_max=[5.0]*7, j_max=[30.0]*7)
frames = planner.plan(start, target, dt=0.01)
arm.execute_trajectory(planner.to_trajectory_frames(frames, dt=0.01))

# 3. TOPP-RA — 时间最优多路点（pip install toppra）
opt = TimeOptimalPlanner(num_joints=7)
opt.set_velocity_limits([2.5]*7)
opt.set_acceleration_limits([5.0]*7)
frames = opt.plan(waypoints, dt=0.01)  # 约束内最快轨迹
arm.execute_trajectory(frames)

arm.disable()
```

### MuJoCo 仿真

```python
from onero_sdk.dual_mujoco_backend import DualMujocoBackend
import mujoco.viewer

# 双臂仿真：人体躯干 + 桌面场景
backend = DualMujocoBackend(model_name='A1', realtime=True)
viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
backend._viewer = viewer

# 对称运动：反号 = 镜像动作
import numpy as np
r_forward = np.array([1.5, 0, 0, 0, 0, 0, 0])
l_forward = -r_forward  # 镜像
backend.movej(r_forward, l_forward, duration=2.0)
```

### 双臂控制

```python
from onero_sdk import DualArm

with DualArm.create("A1", left_channel="can0", right_channel="can1") as dual:
    dual.movej(
        left_joints=[0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0],
        right_joints=[0.0, -0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
        duration=3.0,
    )
```

### 末端执行器（夹爪 / 灵巧手）

SDK 提供抽象接口（`Gripper`、`DexterousHand`），您需要为具体硬件实现通信协议：

```python
from onero_sdk import Arm, Gripper
from onero_sdk.can_bus import CanBus

class MyGripper(Gripper):
    def __init__(self, bus, can_id=0x08):
        self._bus = bus
        self._can_id = can_id
        self._pos = 0xFF

    def enable(self):    self.open()
    def disable(self):   pass
    def shutdown(self):  pass
    def open(self):      self._send(0xFF)
    def grasp(self):     self._send(0x00)
    def set_position(self, position: float):
        self._send(int(position * 0xFF))
    def get_position(self) -> float:
        return self._pos / 0xFF
    def _send(self, pos):
        data = bytes([0x00, pos, 0xFF, 0x80, 0xFF, 0xFF, 0x00, 0x00])
        self._bus.send(self._can_id, data)
        self._pos = pos

# 挂载到机械臂 — enable/disable 自动管理
bus = CanBus("can0")
gripper = MyGripper(bus, can_id=0x08)

with Arm("A1", channel="can0", end_effector=gripper) as arm:
    arm.movej([0.0, 0.8, 0.0, -1.0, 0.0, 0.5, 0.0], duration=2.0)
    arm.gripper.open()
    arm.movej([0.0, 1.0, 0.0, -1.2, 0.0, 0.5, 0.0], duration=1.5)
    arm.gripper.grasp()
```

完整抓取工作流参见 `examples/gripper_pick_place.py`。

## 硬件配置

### 1. USB-CAN 适配器权限（一次性配置）

```bash
sudo cp configs/99-onero-can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout,plugdev $USER
sudo setcap cap_net_admin+ep $(which slcand)
sudo setcap cap_net_admin+ep $(which ip)
# 注销后重新登录
```

### 2. 启动 CAN 接口

```bash
# 使用提供的脚本：
# 用法: onero-can-up.sh [device] [channel] [bitrate]
#   device:  串口设备路径（默认: /dev/ttyACM0）
#   channel: CAN 接口名称（默认: can0）
#   bitrate: slcand 速率标志（默认: s8 = 1Mbit/s）
#            s4=125k, s5=250k, s6=500k, s7=800k, s8=1M
sudo bash configs/onero-can-up.sh                           # 默认: /dev/ttyACM0 can0 s8
sudo bash configs/onero-can-up.sh /dev/ttyACM1 can1 s8     # 第二条臂使用 can1

# 或者手动操作:
sudo slcand -o -s8 /dev/ttyACM0 can0 && sudo ip link set up can0
```

## 配置文件

最小配置 (`configs/a1.yaml`):

```yaml
model: "A1"
channel: "can0"
```

或在代码中直接创建：

```python
arm = Arm("A1", channel="can0")
arm = Arm.from_config("configs/a1.yaml")
```

## 架构

```
onero_sdk/
├── arm.py              # 高级 API（movej/movel/speedj/freedrive/force_mode/...）
├── dual_arm.py         # DualArm 双臂便利封装
├── backend.py          # ArmBackend 抽象基类（可插拔传输层）
├── can_backend.py      # CAN 硬件后端实现
├── safe_backend.py     # SafeBackend 装饰器 — 在后端接缝透明施加安全钳位
├── motion_executor.py  # MotionExecutor — 帧生成器的统一控制循环
├── clock.py            # ControlClock — 基于截止时间的定时器，带漂移补偿
├── geometry.py         # 纯数学函数（SLERP、旋转向量、圆拟合、Rodrigues）
├── end_effector.py     # EndEffector / Gripper / DexterousHand 抽象基类
├── trajectory.py       # 录制、TrajectoryPlanner、blending（五次多项式）
├── trajectory_profiles.py # 梯形、S 曲线、五次多项式过渡曲线
├── ruckig_planner.py   # 在线 jerk-limited 轨迹（Ruckig）[可选]
├── toppra_planner.py   # 时间最优路径参数化（TOPP-RA）[可选]
├── dynamics.py         # 动力学模型（pinocchio RNEA + 雅可比）[可选]
├── calibration.py      # 动力学参数辨识 [可选]
├── kinematic_calibration.py # 运动学标定（单点约束法）[可选]
├── mujoco_backend.py   # 单臂 MuJoCo 仿真后端 [可选]
├── dual_mujoco_backend.py # 双臂 MuJoCo 仿真（人体躯干）[可选]
├── fk.py              # 轻量 FK（仅 numpy，从 URDF 解析）
├── ik_solver.py        # IK: DLS (pinocchio) + QP (pink) [可选]
├── safety.py           # 安全限制与超时保护
├── model_registry.py   # 机械臂模型 → 关节配置查询
├── exceptions.py       # OneroError 异常体系
├── protocol.py         # CAN 帧编解码
├── can_bus.py          # SocketCAN 原始套接字
├── motor.py            # 单电机控制器（底层）
└── config.py           # YAML 配置加载
```

## 运动 API

| 命令 | 空间 | 模式 | 阻塞 | 用途 |
|------|------|------|------|------|
| `movej` | 关节 | 位置 | 是 | 快速关节运动 |
| `movel` | 笛卡尔 | 位置 | 是 | 直线运动（装配、涂胶） |
| `movep` | 笛卡尔 | 位置 | 是 | 到达位姿，路径任意 |
| `movec` | 笛卡尔 | 位置 | 是 | 圆弧运动（焊接、抛光） |
| `speedj` | 关节 | 速度 | 是 | 恒定关节速度 |
| `speedl` | 笛卡尔 | 速度 | 是 | 恒定末端速度 |
| `servoj` | 关节 | 实时 | 否 | 实时关节控制 |
| `servop` | 笛卡尔 | 实时 | 否 | 实时笛卡尔控制 |
| `freedrive_step` | — | 力矩 | 否 | 手动引导（逐帧） |
| `force_mode` | 笛卡尔 | 力控 | 否 | 混合力/位置控制 |

运动指令（`movej`、`movel`、`movec`、`movep`）支持两种参数风格：
- **时间风格**：`arm.movej(target, duration=3.0)` — 指定总运动时间
- **速度风格**（UR 兼容）：`arm.movej(target, speed=1.5, acceleration=1.4)` — 指定最大速度，自动根据梯形速度曲线计算运动时间

默认曲线为**梯形速度曲线**（行业标准）。使用 `profile='cosine'` 恢复旧版行为，或 `profile='s_curve'` 获得 jerk-limited 平滑运动。

## 底层 API

用于硬件调试或自定义控制模式：

```python
from onero_sdk import CanBus, Motor, MotorLimits

with CanBus("can0") as bus:
    motor = Motor(bus, can_id=0x01, mst_id=0x11,
                  limits=MotorLimits(p_max=12.5, v_max=30.0, t_max=15.0))
    motor.enable()
    state = motor.set_mit(position=1.0, kp=50.0, kd=1.0)
    motor.disable()
```

## 依赖

- **必需**：`numpy`、`pyyaml`
- **可选（IK）**：`pin`（pinocchio）、`pink`
- **可选（动力学）**：`pin`（pinocchio）— 用于示教模式、力控、speedl
- **可选（轨迹规划）**：`ruckig`、`toppra` — Ruckig 和 TOPP-RA 高级规划
- **可选（仿真）**：`mujoco`
- **系统要求**：Linux + SocketCAN 支持

## 测试

```bash
# 协议单元测试（无需硬件）
python3 tests/test_protocol.py

# 虚拟 CAN 集成测试
sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
python3 -c "from onero_sdk import Arm; arm = Arm('A1', channel='vcan0'); print('OK')"
```

## 诊断工具

运行内置诊断工具验证配置：

```bash
python -m onero_sdk.diag          # 默认检查 can0
python -m onero_sdk.diag can1     # 检查指定接口
```

检查项目：CAN 接口状态、URDF 文件、pinocchio/pink 可用性。

## 日志

SDK 使用 Python 标准 `logging` 模块。启用 debug 输出可查看控制循环内部、CAN 帧时序和安全监控器活动：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或仅启用 SDK 日志:
logging.getLogger("onero_sdk").setLevel(logging.DEBUG)
```

日志级别：
- `DEBUG`：逐帧时序、CAN 帧详情
- `INFO`：生命周期事件（enable/disable）、运动开始/结束
- `WARNING`：CAN 帧丢失（低于阈值）、接近限位
- `ERROR`：急停触发、硬件故障

## 示例

- `examples/basic_movej.py` — 关节运动
- `examples/basic_movel.py` — 笛卡尔直线运动
- `examples/servo_control.py` — 实时 100Hz 伺服循环
- `examples/speed_control.py` — 速度模式（speedj）
- `examples/freedrive_teach.py` — 示教 + 录制 + 回放
- `examples/blending.py` — 多路点轨迹 + blending
- `examples/gripper_pick_place.py` — 机械臂 + 夹爪抓取
- `examples/dual_arm.py` — 双臂控制
- `examples/mit_control.py` — 底层单电机 MIT 控制
- `examples/position_control.py` — 位置-速度模式
- `examples/calibration_demo.py` — 动力学参数辨识（MuJoCo 仿真）
- `examples/kinematic_calibration_demo.py` — 运动学标定（单点约束法）
- `examples/trajectory_planning_demo.py` — 轨迹规划对比（梯形、S曲线、Ruckig、TOPP-RA）
- `examples/mujoco_trajectory_comparison.py` — MuJoCo 轨迹验证（自动化正确性检查）
- `examples/dual_arm_mujoco_demo.py` — 双臂仿真（人体躯干 + 桌面）
