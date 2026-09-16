# Onero A1 坐标系与关节正方向

本文说明 A1 单臂/双臂安装时的坐标系、左右臂模型选择，以及 7 个关节正方向的定义。

本文所有俯视描述统一采用“前方 = world +X，左方 = world +Y，上方 = world +Z”。

## 推荐表达方式

成熟机械臂文档通常把这几件事分开写：

1. **世界坐标系 / 基座坐标系 / 工具坐标系**：先定义用户编程时看到的 frame。
2. **安装姿态**：说明 base frame 如何映射到 world frame。
3. **关节正方向**：用 URDF joint axis + 右手定则定义，不用“向内/向外”这类依赖视角的说法。
4. **双臂镜像规则**：给一个明确的左右臂姿态映射规则。

## 两层坐标系：URDF 内部与装配后 world

SDK 明确区分两层：

1. **URDF base frame**：机械臂内部运动学坐标，左右模型各自固定，不随安装改变。
2. **装配后 world frame**：用户编程坐标，统一约定 `X=前、Y=左、Z=上`。

二者只通过安装变换连接，不修改 URDF：

```text
p_world = R_world_base · p_base + base_position
R_world_tcp = R_world_base · R_base_tcp
```

`R_world_base` 的三列分别是 URDF `base +X/+Y/+Z` 在 world 中的方向。

### URDF 内部 base frame

| 项目 | A1L | A1R |
|---|---:|---:|
| 根坐标 | `base_link` | `base_link` |
| J1 原点 | `[0,0,0.0315] m` | `[0,0,0.0315] m` |
| J1 正轴 | `base +Z` | `base +Z` |
| 零位 Link7 原点 | `[-0.5665,-0.0001,0.1582] m` | `[-0.5665,0,0.1590] m` |
| 零位主伸展方向 | `base -X` | `base -X` |

因此 URDF 的 `base +X/+Y/+Z` 只是 CAD/运动学固有轴，不能直接解释成真实环境中的前/左/上。

### 装配后统一 world frame

```text
world +X = forward / 前方
world +Y = left    / 左方
world +Z = up      / 上方
```

垂直肩部装配使用唯一的左右矩阵：

```text
R_world_base_left =
[[0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]

R_world_base_right =
[[ 0,-1, 0],
 [ 0, 0,-1],
 [ 1, 0, 0]]
```

轴映射与零位结果：

| 机械臂 | 安装预设 | base +X | base +Y | base +Z | 零位 Link7（相对肩部，world） | 重力（base） |
|---|---|---|---|---|---|---|
| 左臂 A1L | `humanoid_left` | 上 `+Zw` | 前 `+Xw` | 左/外 `+Yw` | `[≈0,+0.158,-0.566] m` | `[-9.81,0,0]` |
| 右臂 A1R | `humanoid_right` | 上 `+Zw` | 后 `-Xw` | 右/外 `-Yw` | `[≈0,-0.159,-0.566] m` | `[-9.81,0,0]` |

这两个矩阵是 `mounting.py`、`DualArm.create()` 和 `DualMujocoBackend` 的共同权威来源。`a1_vertical` 仅保留为旧的、不区分左右侧的兼容预设：它的重力方向正确，但水平 yaw 等同于左臂；新双臂代码不要用它替代右臂矩阵。

### 默认行为

| 模式 | 构造方式 | 默认 frame | 默认安装姿态 | 默认原点 |
|---|---|---|---|---|
| 单臂原始 URDF | `Arm(...)` | `base` | `upright`（`mounting=None`） | URDF `base_link` 原点 |
| 左单臂装配系 | `Arm("A1L", mounting="humanoid_left", world_frame=True, ...)` | `world` | 左垂直肩部 | `base_position` |
| 右单臂装配系 | `Arm("A1R", mounting="humanoid_right", world_frame=True, ...)` | `world` | 右垂直肩部 | `base_position` |
| 双臂 | `DualArm.create(...)` | `world` | 左 `humanoid_left`，右 `humanoid_right` | `arm_spacing` 或左右 `base_position` |

若 `arm_spacing=d`，world 原点位于两臂 base 中点：

```text
left base_position  = [0,+d/2,0]
right base_position = [0,-d/2,0]
```

推荐单臂和双臂物理控制都面向 world 编程：

```python
left = Arm("A1L", channel="can0", mounting="humanoid_left",
           world_frame=True, base_position=[0, 0.15, 0])
right = Arm("A1R", channel="can1", mounting="humanoid_right",
            world_frame=True, base_position=[0, -0.15, 0])
```

```python
dual = DualArm.create(left_channel="can0", right_channel="can1",
                      arm_spacing=0.30)
```

对任意笛卡尔 API，显式 `frame=` 会覆盖默认值；`arm.default_frame` 可查询当前值。`get_pose()` 默认描述 flange；调用 `set_tcp()` 后描述 TCP；`frame="tool"` 表示相对当前 TCP 的增量运动。

## 关节正方向定义

关节正方向由 URDF 决定：每个 revolute joint 的 `<axis xyz="...">` 是该关节的正旋转轴。正角度遵循右手定则：右手拇指指向正轴方向，四指弯曲方向就是正转方向。

下表是从 `onero_sdk/urdf/A1/a1_l.urdf` 和 `onero_sdk/urdf/A1/a1_r.urdf` 计算得到的零位 joint axis，均表达在各自 base frame 下。

| Joint | A1L 正轴 | A1R 正轴 | 说明 |
|---|---:|---:|---|
| J1 | +Zb | +Zb | 肩部/基座 yaw |
| J2 | -Yb | -Yb | 肩部抬升/展开 |
| J3 | +Xb | -Xb | 左右镜像 |
| J4 | +Zb | +Zb | 肘/腕链中的旋转轴 |
| J5 | +Xb | -Xb | 左右镜像 |
| J6 | -Zb | -Zb | 腕部旋转轴 |
| J7 | +Xb | -Xb | 左右镜像 |

## 应该给正还是给负？

单臂控制时，不要靠“从图片看顺时针/逆时针”判断；按下面流程判断：

1. 先确认你在控制 `A1L` 还是 `A1R`。
2. 查上表得到该 joint 的正轴。
3. 沿正轴方向看过去，逆时针就是正角度，顺时针就是负角度。
4. 如果只是做双臂对称动作，优先使用镜像规则：

```python
right_pose = [1.2, -0.5, 0.0, -1.0, 0.0, 0.3, 0.0]
left_pose = [-q for q in right_pose]
```

这条规则与现有双臂 MuJoCo demo 的姿态约定一致：`q_left = -q_right` 表示对称镜像动作。

## 文档中建议统一使用的句式

建议文档里避免写：

```text
J2 正方向是向外展开。
```

因为“向外”依赖安装方向、左右臂、观察视角。建议写成：

```text
J2 的正轴为 base -Y。正角度按右手定则定义。对于双臂对称展开，使用 q_left = -q_right。
```

或者写成用户更容易操作的形式：

```text
如果你在写单臂程序，用 joint-axis 表判断正负；如果你在写双臂对称动作，先定义右臂姿态，再令左臂姿态等于逐元素取反。
```
