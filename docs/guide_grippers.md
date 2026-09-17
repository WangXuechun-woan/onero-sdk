# 夹爪与工具模型指南

SDK 将以下配置刻意分开，修改其中一项不会隐式改变其他项：

1. 浏览器中的工具外观（`--gripper-model`）
2. 实体夹爪 CAN 协议（`--gripper-protocol`）
3. TCP（`arm.set_tcp()`）
4. payload（`arm.set_payload()`）
5. 经验重力补偿档位（`with_gripper=True` 或 `--gripper-compensation on`）

工具模型作为 Link7 附件保存于 `onero_sdk/tools/`，不会修改 A1L/A1R 的标准 URDF。

## 公共位置约定

所有夹爪统一使用归一化开度：

```text
0.0 = 全闭
1.0 = 全开
```

`open()` 等价于 `set_position(1.0)`；`grasp()` 等价于 `set_position(0.0)`。

## 支持的实体协议

### OmniPicker

`OmniPickerGripper` 使用旧款 8 字节协议：

```text
D0 reserved
D1 position
D2 force
D3 velocity
D4 acceleration
D5 deceleration
D6-D7 reserved
```

默认夹持力为保守值 `0x20/255`。该协议没有独立 enable/disable 帧，因此 lifecycle 调用不会为了“失能”而自动张开夹爪。

### Onero / Woan Picker

`OneroGripper` 使用 DM 兼容 MIT 协议：

```text
command CAN ID  = 0x08
feedback CAN ID = 0x18
kp = 20, kd = 1
```

协议编码范围 `[-10, 55]` 大于机械结构的实际有效行程，因此 SDK 使用左右实测端点：

| 型号 | 全开原始值 | 全闭原始值 |
|---|---:|---:|
| A1L | 37/255 | 239/255 |
| A1R | 37/255 | 238/255 |

电机位置增大方向与公共“开度”方向相反，转换在驱动内部完成。传给 `Arm` 时，SDK 根据明确的 `A1L`/`A1R` 型号选择端点；独立使用驱动时必须传入 `model=`。

DM 状态 `0`/`1` 分别表示失能/使能，不作为故障；`8..14` 才通过 `fault_code` 报告为驱动故障。

## Python 用法

```python
from onero_sdk import Arm, OneroGripper

# Arm 会使用明确的 A1L 型号选择左夹爪端点。
gripper = OneroGripper()
arm = Arm(
    "A1L",
    channel="can0",          # 通道映射由部署者明确指定
    end_effector=gripper,
    with_gripper=True,       # 仅选择经验重力补偿档位
)

try:
    gripper.enable()          # 只使能夹爪，不使能机械臂关节
    gripper.set_position(0.5)
    print(gripper.state)
    gripper.disable()
finally:
    arm.close()               # 默认只释放 transport，不自动卸载机械臂力矩
```

独立诊断时可显式绑定型号和 transport：

```python
from onero_sdk import CanBus, OneroGripper

bus = CanBus("can0")
gripper = OneroGripper(bus=bus, model="A1L")
```

一个 `Arm` 只拥有一个 CAN transport。机械臂和夹爪共享该 transport 及 I/O 锁，避免重复打开 SLCAN 串口或并发读取错配反馈帧。不要同时让 `slcand` 和直接 Python SLCAN 打开同一个串口设备。

## Debug UI

实体 Woan/Onero 夹爪示例：

```bash
python -m onero_sdk.debugui \
  --arm both \
  --left-channel can0 \
  --right-channel can1 \
  --gripper-model woan_picker \
  --gripper-protocol onero \
  --gripper-compensation on
```

`can0/can1` 与左右臂的关系取决于现场接线，SDK 不根据通道名称推断左右。协议必须显式选择；`none` 不发送任何夹爪 CAN 帧，也不会自动探测或回退协议。

Gripper 页面的 Enable/Disable 只控制夹爪，不控制机械臂关节。拖动滑块时仅做本地预览，释放后发送一次位置命令；Onero 状态轮询随后用真实反馈更新开度。

## 安全与限制

- Debug UI 不是安全等级 E-stop，也不执行环境碰撞规划。
- 最大力高速闭合可能夹伤人员或冲击机构。
- 工具外观不会自动设置 TCP；未调用 `set_tcp()` 时，笛卡尔目标仍指 Link7/法兰。
- `close()`、上下文退出和解释器退出不会自动失能承受重力的机械臂。
- 当前 OmniPicker 来源 URDF 将机构关节固定，网页动画是近似效果；Woan Picker 使用 H1 v1.3 的八关节视觉机构。
