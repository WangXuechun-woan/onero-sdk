# Debug UI 调试面板

`onero_sdk.debugui` 是 SDK 内置的轻量 Web 调试工具，无需额外安装依赖（标准库实现），打开浏览器即可对真机或仿真进行交互式调试。

---

## 启动方式

### 命令行（推荐）

```bash
# 真机 — 单右臂
python -m onero_sdk.debugui --arm right --channel slcan0

# 真机 — 单左臂
python -m onero_sdk.debugui --arm left --channel slcan1

# 真机 — 人形竖直安装且带夹爪补偿的单左臂
python -m onero_sdk.debugui --arm left --channel can0 \
  --mount a1_vertical --gripper-compensation on

# 真机 — 双臂
python -m onero_sdk.debugui --arm both \
  --left-channel slcan1 --right-channel slcan0

# MuJoCo 仿真（无需硬件）
python -m onero_sdk.debugui --arm both --sim

# 指定端口和绑定地址
python -m onero_sdk.debugui --arm right --channel slcan0 --port 8090 --host 0.0.0.0
```

启动后打开 `http://<机器IP>:8090` 即可访问。

### 嵌入到脚本

如果已有 `Arm` 或 `DualArm` 实例，可以直接传入，避免重复初始化：

```python
from onero_sdk import Arm
from onero_sdk.debugui import start_lab

arm = Arm("A1R", channel="slcan0")
arm.enable()

# block=False: 后台启动，不阻塞脚本
server = start_lab(arm, port=8090, block=False)

# 继续执行其他逻辑 …
arm.movej([0] * 7)

# 关闭时
server.close()
```

双臂版本：

```python
from onero_sdk import DualArm
from onero_sdk.debugui import start_lab_dual

dual = DualArm.create(
    left_channel="slcan1",
    right_channel="slcan0",
    arm_spacing=0.36,
)
start_lab_dual(dual, port=8090, block=True)  # block=True 前台运行
```

仿真后端：

```python
from onero_sdk import Arm
from onero_sdk.debugui import start_lab
from onero_sdk.mujoco_backend import MujocoBackend

arm = Arm("A1R", backend=MujocoBackend())
arm.enable()
start_lab(arm, port=8090, block=True)
```

---

## 界面功能

### 3D 视口（左侧）

- 使用 Three.js + URDF Loader 渲染真实网格（从 `onero_sdk/urdf/` 和 `onero_sdk/meshes/` 加载）
- **实线机械臂**：实时跟踪硬件反馈的关节角度
- **轴向幽灵**（Ghost）：跟踪当前 UI 滑条的预览位置，可通过 Ghost 勾选框开关
- **末端球**（MoveL 模式）：拖动彩色球驱动末端运动
- **当前末端指示点**（半透明）：显示硬件反馈的末端位置

鼠标操作：
- 左键拖动：旋转视角
- 右键拖动 / 中键：平移
- 滚轮：缩放
- 左键点击末端球后拖动：驱动 MoveL

### 控制面板（右侧）

每条机械臂对应一个面板，包含以下区域：

#### 刚度预设（Stiffness）

快速切换 `stiff / medium / soft` 三档增益，调用 `arm.set_stiffness()`。适合根据负载和任务快速调整响应特性。

#### 基础控制按钮

| 按钮 | 对应 SDK 调用 | 说明 |
|------|--------------|------|
| **Enable** | `arm.enable()` | 使能电机 |
| **Disable** | `arm.disable()` | 失能电机 |
| **Stop** | `arm.stop()` | 减速停止，中断当前运动 |
| **Home** | `arm.home()` | 所有关节归零位 |

#### MoveJ 标签页

关节空间运动，对应 `arm.movej(joints, speed=…)`。

- 7 个滑条对应 7 个关节角（rad），范围从 URDF 限位读取
- **Auto**：打勾后拖动滑条自动发送（300ms 节流），松手立即发送
- **Apply**：手动触发一次发送
- **Ghost**：开关幽灵预览
- **speed**：运动速度（m/s），传给 `arm.movej(speed=…)`

#### Servo 标签页

100 Hz 实时关节空间伺服，对应 `arm.servoj(joints, kp, kd)`。

- **Start**：启动 100 Hz 伺服循环，记录当前位置为初始目标
- **Stop**：停止伺服循环
- **kp / kd**：增益参数，直接传入 `arm.servoj(kp=…, kd=…)`
- 伺服激活后拖动滑条实时更新目标，绿点脉冲表示循环运行中

> ⚠️ 伺服模式与 MoveJ / MoveL / MoveP / Home 互斥。服务端在接收阻塞运动前会先停止 Servo，避免两个线程同时向后端发送命令。

#### MoveL 标签页

笛卡尔直线运动，对应 `arm.movel(position, speed=…)`。

- 可直接输入目标 X/Y/Z，或点击 **Current** 填入当前末端位置
- **Drag Ball**：在 3D 视口中显示末端控制球；拖动过程只预览，松手后发送一次 MoveL
- **Apply**：提交一次完整直线运动，不会在拖动期间反复取消和重启轨迹
- **Stop**：停止当前运动并隐藏控制球
- 每个坐标轴单次最大变化 2.5 cm
- 未指定姿态时保持命令开始前的当前姿态
- **speed**：工具速度（m/s）

#### MoveP 标签页

笛卡尔点到点运动，对应 `arm.movep(position, speed=…)`。

- 输入目标 X/Y/Z，或点击 **Current** 填入当前末端位置
- **Drag Ball**：与 MoveL 一样可拖动末端球预览目标，松手后发送一次 MoveP
- **Apply**：目标位姿先解 IK，再执行关节空间插值
- 路径不保证为直线；需要直线路径时使用 MoveL
- 每个坐标轴单次最大变化 2.5 cm
- 未指定姿态时保持命令开始前的当前姿态
- **speed**：关节速度（rad/s）

#### Freedrive 标签页

零重力示教模式，对应 `arm.freedrive_enable(damping=…)` 和 `arm.freedrive_disable()`。

- **Enable**：进入 freedrive，可手动引导机械臂运动
- **Disable**：退出 freedrive，机械臂保持当前位置
- **damping**：关节阻尼（0=无阻力/感觉失重，0.5=轻阻尼，5.0=重阻尼）

> 需要安装动力学依赖：`pip install onero-sdk[dynamics]`

---

## HTTP API（供二次开发）

Debug UI 暴露了完整的 REST API，可从其他工具调用：

### 单臂模式（`--arm right` 或 `--arm left`）

所有端点均以 `/api/` 为前缀。

### 双臂模式（`--arm both`）

左臂使用 `/left/api/`，右臂使用 `/right/api/`。

### 端点列表

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| `GET` | `/api/state` | — | 完整状态 JSON |
| `GET` | `/api/stream?hz=N` | `hz`=1–50 | SSE 状态流 |
| `POST` | `/api/enable` | — | 使能 |
| `POST` | `/api/disable` | — | 失能 |
| `POST` | `/api/stop` | — | 减速停止 |
| `POST` | `/api/home` | — | 归零位 |
| `POST` | `/api/set_stiffness` | `{"stiffness":"stiff"}` | 刚度预设 |
| `POST` | `/api/movej` | `{"joints":[…],"speed":0.5}` | 关节运动 |
| `POST` | `/api/movel` | `{"position":[x,y,z],"speed":0.1}` | 末端直线运动 |
| `POST` | `/api/movep` | `{"position":[x,y,z],"speed":0.5}` | 末端点到点运动 |
| `POST` | `/api/servo/start` | `{"kp":50,"kd":1.0}` | 启动 100 Hz 伺服 |
| `POST` | `/api/servo/target` | `{"joints":[…],"velocities":[…]}` | 更新伺服目标 |
| `POST` | `/api/servo/stop` | — | 停止伺服 |
| `POST` | `/api/freedrive/start` | `{"damping":0.5}` | 进入 freedrive |
| `POST` | `/api/freedrive/stop` | — | 退出 freedrive |

### 状态 JSON 结构

```json
{
  "side": "right",
  "model": "A1R",
  "num_joints": 7,
  "enabled": true,
  "moving": false,
  "freedrive": false,
  "servo_active": false,
  "joints":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "torques": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "pose": {
    "pos":  [0.0, 0.0, 0.45],
    "quat": [0.0, 0.0, 0.0, 1.0]
  },
  "limits": [[-1.04, 3.14], …],
  "msg":   "movej done",
  "fault": ""
}
```

SSE 流（`/api/stream`）的每条消息是包含双侧的 JSON：

```json
{
  "left":  { … },
  "right": { … }
}
```

---

## CLI 参数完整列表

```
python -m onero_sdk.debugui --help

  --arm {right,left,both}   控制哪侧机械臂（默认: right）
  --channel CHANNEL         单臂 CAN 通道（默认: slcan0/slcan1 随侧）
  --left-channel CHANNEL    双臂左臂 CAN 通道（默认: slcan1）
  --right-channel CHANNEL   双臂右臂 CAN 通道（默认: slcan0）
  --arm-spacing METERS      双臂基座间距（默认: 0.36 m）
  --mount PRESET            安装方向；default 保留原版单/双臂默认值
  --gripper-compensation {off,on}
                            是否启用夹爪重力补偿（默认: off）
  --sim                     使用 MuJoCo 仿真后端
  --host HOST               绑定地址（默认: 0.0.0.0）
  --port PORT               TCP 端口（默认: 8090）
```

---

## 快速测试（无硬件）

```bash
# 启动仿真模式
python -m onero_sdk.debugui --arm both --sim --port 8090

# 浏览器打开
open http://localhost:8090
```

仿真模式下 MuJoCo 提供完整物理仿真，所有 UI 功能均可正常使用。

---

## 依赖说明

| 功能 | 依赖 | 安装 |
|------|------|------|
| 基础 UI（MoveJ / MoveL / Servo） | 标准库 + numpy | 核心包已包含 |
| Freedrive | pinocchio | `pip install onero-sdk[dynamics]` |
| 仿真后端 | mujoco | `pip install onero-sdk[sim]` |
| 3D URDF 渲染 | Three.js（CDN） | 无需本地安装 |
