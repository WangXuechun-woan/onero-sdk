# Onero A1 奇异点处理

奇异点（singularity）是机械臂丢失一个笛卡尔自由度的关节配置。在这些位置附近，把末端速度反解为关节速度会出现"小的末端速度需要巨大关节速度"的现象，直接下发会造成过冲、抖动、超速甚至硬件损伤。本文说明 SDK 目前的处理策略，以及与 UR / Franka / ABB / KUKA / MoveIt 等成熟方案的差异。

## 一、7-DOF 三类奇异点

Onero A1 是人形 7-DOF 拟人臂，有三类经典奇异：

| 类型 | 判据 | 现象 |
|---|---|---|
| **腕部（wrist）** | `J5 ≈ 0` 或 `±π` | J4 与 J6 轴共线，丢失一个姿态 DOF |
| **肘部（elbow）** | `J3 ≈ 0` 或 `±π` | 手臂完全伸直/折叠，径向位置在可达边界 |
| **肩部（shoulder）** | 末端落在 J0 轴（基座 +Z）上 | J0 转动不改变末端位置 |

SDK 用 `arm.singularity_info()` 一次性给出三类奇异的分类与严重度：

```python
info = arm.singularity_info()
# {'w': 0.0083, 'sigma_min': 0.019, 'type': 'wrist', 'severity': 0.87}
```

## 二、SDK 的三层保护

### 第 1 层：DLS 自适应阻尼（永远在线，默认开启）

所有涉及雅可比反解的路径（`speedl`、`solve_ik`、`estimate_external_wrench`）都使用 **Damped Least Squares**，阻尼系数随 σ_min 平滑上升：

```
σ_min > σ_threshold  →  λ² = 0（无阻尼，全精度）
σ_min ≤ σ_threshold  →  λ² = λ_max² · (1 − (σ_min/σ_threshold)²)
```

默认 `σ_threshold = 0.02`，`λ_max = 0.1`。含义：越接近奇异，阻尼越大，反解出的关节速度会自动"变钝"而不是发散。这一层完全透明，用户无需配置。

### 第 2 层：两阈值策略（可选，需显式开启）

对 `speedl` 这类连续 Cartesian 速度控制，光靠阻尼不够——阻尼虽然不发散，但仍会让末端**慢慢滑向**奇异点，最后卡住。为此 SDK 提供 `SingularityPolicy`（模型来自 MoveIt Servo 与 UR e-Series 的做法）：

```
σ_min > warn_threshold        →  原速通过
hard_threshold < σ_min ≤ warn →  按 (σ_min − hard)/(warn − hard) 线性降速，记录一次告警
σ_min ≤ hard_threshold        →  抛 OneroSafetyError（默认）或钳位到 0（clamp 模式）
```

启用方式：

```python
arm.set_singularity_policy(
    warn_threshold=0.03,   # 进入警戒带
    hard_threshold=0.005,  # 硬停/钳位
    abort_on_hard=True,    # False → 保持接近但钳零，适合遥操作
)
```

告警和错误消息里会附带分类信息，例如：

```
Cartesian motion aborted: sigma_min=0.0042 ≤ hard_threshold=0.005.
Arm is at a singularity (type=wrist, severity=0.91).
```

### 第 3 层：零空间冗余利用（IK 层）

7-DOF 有一个冗余 DOF，可以在保持末端位姿不变的前提下调整肘部姿态。`solve_ik_with_nullspace` 通过零空间投影把关节配置往 `q_ref`（通常是"舒展的中位姿"）拉：

```
N = I − J⁺ J
dq_null = k · N · (q_ref − q)
```

这不是"避奇异"的直接手段，但通过让 IK 优先选择远离肘/腕奇异的解，间接降低进入奇异点的概率。适合**长轨迹规划**类场景，实时伺服里目前默认走非零空间路径。

## 三、与成熟机械臂对比

| 厂家/框架 | 分类识别 | 自适应阻尼 | 降速警告 | 强制中止 | 备注 |
|---|---|---|---|---|---|
| **Onero A1（本 SDK）** | ✅ wrist/elbow/shoulder + severity | ✅ DLS 自适应 λ | ✅ 两阈值线性降速 | ✅ OneroSafetyError | 全开源，可调阈值 |
| **UR e-Series** | ✅ 3 类 | ✅（黑盒） | ✅ 自动降速 | ✅ Protective Stop | 商业实现，用户不能改阈值 |
| **Franka Panda** | ❌ 不暴露 | ✅（黑盒） | ❌ 无 | ✅ Cartesian 模式直接拒绝启动 | 遇奇异必须切回 joint mode |
| **ABB IRC5** | ✅ RAPID 里可查 | 依 `SingArea` 指令 | ✅ `SingArea\Wrist`/`\Off` | ✅ | 需要在代码里显式声明 |
| **KUKA KRC** | ✅ 提示 | 弱 | ❌ | ✅ 直接 E-Stop | 靠离线仿真规避为主 |
| **MoveIt Servo** | ❌ 只看 σ_min | ✅ | ✅ 两阈值 | ✅ | 我们的策略层与它对齐 |
| **Kinova Gen3** | 部分 | ✅ | ✅ 自动降速 | ⚠ 会硬停 | 靠冗余关节回避为主 |

**总结**：本 SDK 的策略在架构上最接近 UR（分类 + 降速 + 中止）与 MoveIt Servo（两阈值），但保留了所有阈值的用户可见性和可配置性。这是刻意选择——工业臂厂家把这些当作黑盒是因为他们要保证"任何用户任何场景都安全"，而本 SDK 面向研究/开发场景，暴露参数换来了在特定任务上做取舍的能力（例如遥操作愿意接受更靠近奇异以获得更大工作空间，可以调低 `warn_threshold`）。

## 四、上层应用应该做什么

即便 SDK 有三层保护，上层规划器/遥操作栈仍需要：

1. **规划阶段避开奇异**：`arm.singularity_info()` 可用来评价候选轨迹或 IK 解的健康度，选 σ_min 最大的分支。
2. **在遥操作里降低发生概率**：把主手工作空间限制到臂展的 [0.3, 0.85] 倍范围内。
3. **响应 `OneroSafetyError`**：切到关节空间的恢复动作，而不是重发相同的 Cartesian 命令。

如果上层完全不处理：
- 未开启 `SingularityPolicy`：`speedl` 会因为 DLS 逐渐把末端速度打折，观感是"越靠近奇异越慢，最后停在附近晃动"——不危险，但达不到目标。
- 已开启 `SingularityPolicy`（推荐）：进入 hard 阈值时直接抛异常，上层必须处理。这是本 SDK 推荐的做法，因为"逐渐失控"比"明确失败"更难诊断。

## 五、参数调优建议

| 场景 | `warn_threshold` | `hard_threshold` | `abort_on_hard` |
|---|---|---|---|
| 精密插拔、力控 | 0.05 | 0.02 | True |
| 通用点到点 | 0.03（默认） | 0.005（默认） | True |
| 主从遥操作 | 0.02 | 0.005 | False（钳零而非中止） |
| 演示 / 教学 | 0.04 | 0.01 | False |

调节原则：`warn - hard` 之间是"降速带"，太窄会突然降为 0（观感是抖动），太宽会过早失速。默认 6× 比例是通用起点。

## 六、参考文献

- Yoshikawa, T. (1985). *Manipulability of robotic mechanisms.* IJRR 4(2).
- Nakamura & Hanafusa (1986). *Inverse kinematic solutions with singularity robustness for robot manipulator control.* Trans. ASME.
- Chiaverini (1997). *Singularity-robust task-priority redundancy resolution.* IEEE T-RA.
- MoveIt Servo: <https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html>
- UR Universal Robots User Manual — Safety functions & Singularity.
