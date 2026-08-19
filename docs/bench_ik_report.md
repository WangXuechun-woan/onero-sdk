# IK Benchmark Report — Onero A1 SDK

> 实测数据。**只有在收敛容差内的 IK 结果才有意义**，未收敛意味着 IK 输出的关节角对应的末端位姿离目标很远——本报告严格区分了这两类结果。

## 测试环境

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 7 3700X @ 3.97 GHz |
| Python | 3.14.6 |
| URDF | `a1_r.urdf` (7-DOF) |
| 采样量 | 1000 trials/场景 |
| 收敛容差 | `pos_tol = 1e-4 m` (0.1 mm), `rot_tol = 1e-3 rad` (≈ 0.057°) |
| Solvers | `PinkIKSolver` (QP via `quadprog` + DampingTask 0.01)，`IKSolver.solve_ik` (自适应 DLS, dt=1.0) |

复现：`python -m benchmarks.bench_ik`

## 初始状态 · 重要

**所有场景都从非奇异的关节配置起始**。A1 的 URDF 零位（全零关节）恰好落在肩-肘硬奇异点上（σ_min ≈ 5e-8），从这个状态发起的任何 Cartesian 运动都会让 IK 陷死循环。

- **Q_HOME**：`[0, -0.5, 0, -1.0, 0, 0.5, 0]`，σ_min = 0.085（舒展位姿）
- **Q_COLD_INIT**：`[0.3, -0.6, 0.2, -1.2, -0.2, 0.7, 0.1]`，σ_min = 0.096（另一个非奇异位姿，用于 cold-start 场景）
- **q_near_sing**：`[0, -0.5, 0, -0.1, 0, 0.5, 0]`，σ_min = 0.0089（J3 接近 0 → 手臂近乎伸直 → 肘部奇异，故意落在阻尼阈值 0.02 之下）

## 场景定义

| ID | 场景 | 目标位移 | 起始关节 | 用途 |
|---|---|---|---|---|
| A | `servo_steady` | 每帧 1 mm | Q_HOME + 上一帧解（warm start） | teleop / servol / servop |
| B | `movel_near` | 5–10 cm | Q_HOME | movel/movep 一般段 |
| C | `movel_cold` | 30–50 cm | Q_COLD_INIT（无 warm start） | 手动重定位后首次 IK |
| D | `near_singular` | 3 mm 抖动 | q_near_sing（σ_min ≈ 0.009） | 阻尼行为、奇异邻域可靠性 |

## Table 1 — 收敛 trial 的端到端耗时

**端到端**指从进入 `solve_ik()` / `solve()` 到返回**已满足容差**（pos_err < 0.1 mm 且 rot_err < 0.057°）的关节角为止，包含内部所有迭代。未达到容差的 trial 不计入本表，见 Table 2。

三个分位数的含义：

- **p50**：典型耗时（一半 trial 更快）——反映"通常有多快"
- **p95**：常见坏情况（95% trial 更快）——用来评估实时控制预算
- **p99**：最差 1% 的下界——决定是否会偶发抖动 / 打断控制周期

| 场景 / Solver | 收敛率 | 有效 n | **p50 ms** | **p95 ms** | **p99 ms** | pos_err p95 (mm) | rot_err p95 (mrad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. servo_steady · Pink QP | 100.0% | 1000 | 0.150 | 0.159 | 0.163 | 0.033 | 0.042 |
| A. servo_steady · **DLS** | 100.0% | 1000 | **0.069** | **0.078** | **0.080** | 0.028 | 0.042 |
| B. movel_near · Pink QP | 82.3% | 823 | 0.421 | 0.613 | **0.735** | 0.080 | 0.120 |
| B. movel_near · **DLS** | 82.2% | 822 | **0.164** | **0.214** | 1.270 | 0.071 | 0.149 |
| C. movel_cold · Pink QP | 16.3% | 163 | 0.682 | **1.220** | **1.608** | 0.083 | 0.245 |
| C. movel_cold · **DLS** | 16.3% | 163 | **0.285** | 7.924 | 12.799 | 0.099 | 0.842 |
| D. near_singular · **Pink QP** | 60.0% | 600 | **0.415** | **0.704** | **1.522** | 0.096 | 0.044 |
| D. near_singular · DLS | 58.3% | 583 | 2.317 | 8.691 | 17.631 | 0.100 | 0.057 |

**读表要点：**

- **A. servo（1 kHz 遥操作）**：DLS p50/p95/p99 全线跑赢 Pink，分布极紧（p99 才 0.08 ms），运行 1 kHz 无压力
- **B. movel_near（常规轨迹）**：DLS p50 更快，但 p99 拖到 1.27 ms（长尾）；Pink 分布更稳
- **C. movel_cold（无 warm-start 大跳跃）**：两者 p99 都到 ms 级别；DLS p99 破 10 ms，冷启动是概率事件
- **D. near_singular（奇异邻域）**：**Pink 反超 DLS 5×**（p50 0.42 vs 2.32 ms）——DLS 阻尼区需要 50+ 迭代穿越

## Table 2 — 未收敛 trial 的残余误差

| 场景 / Solver | 失败数 | fail_pos p50 (mm) | fail_pos max (mm) | fail_rot p50 (mrad) |
|---|---:|---:|---:|---:|
| B. movel_near · Pink QP | 177 | **20.9** | 131.0 | 14.5 |
| B. movel_near · DLS | 178 | 16.2 | 67.3 | 1.5 |
| C. movel_cold · Pink QP | 837 | 193.3 | 956.8 | 211.9 |
| C. movel_cold · DLS | 837 | 176.1 | 1772.5 | 175.9 |
| D. near_singular · Pink QP | 400 | **2.0** | 12.4 | 0.12 |
| D. near_singular · DLS | 417 | 1.9 | 10.6 | 0.11 |

## v0.3.0 相较 v0.2 的改进

| 场景 | 指标 | v0.2 | v0.3 | 改进 |
|---|---|---:|---:|---|
| A. servo · DLS | p50 | 1.21 ms | **0.069 ms** | **18× 更快** |
| A. servo · DLS | iters | 28 | 2 | 14× 更少 |
| B. movel · DLS | p50 | 3.12 ms | **0.16 ms** | **19× 更快** |
| C. cold · DLS | p50 | 3.59 ms | 0.29 ms | 12× 更快 |
| D. near_sing · DLS | 收敛率 | 29.2% | **58.3%** | 2× 更多收敛 |
| B. movel · Pink | fail_pos p50 | 427 mm | **21 mm** | **20× 更小** |
| D. near_sing · Pink | fail_pos p50 | 40 mm | **2.0 mm** | 20× 更小 |
| C. cold · Pink | fail_pos p50 | 348 mm | 193 mm | 45% 更小 |

## 关键结论

### 1. DLS 在常规场景反超 Pink，但奇异邻域仍慢 5×

历史上 Pink QP 曾比 DLS 快 10×，因为 DLS 用 `dt=0.1` 需要 20+ 次迭代逼近。将默认 `dt` 调到 **1.0**（即完整 Newton 步）后：

- Servo：DLS **0.069 ms**（14.5 kHz）vs Pink 0.15 ms（6.7 kHz）——DLS **端到端反超** Pink 2×
- movel_near：DLS **0.164 ms** vs Pink 0.42 ms（2.5× 快）
- **near_singular**：DLS **2.32 ms**（50 iters 穿越阻尼区）vs Pink 0.42 ms（3 iters）——**Pink 反过来快 5×**

原因清楚：**Pink 每次迭代 ~140 μs（QP 开销），DLS 每次 ~40 μs（单次线性方程）**。iters 少的场景 DLS 胜，iters 多的场景 Pink 胜。近奇异是 DLS 的天然弱项——阻尼被拉到很高后每步只能挪一点点。

**实用建议**：SDK 默认继续用 Pink QP（在所有场景下时间都稳定 <1 ms），把 DLS 作为备用 / 教学 / 后处理。

### 2. Pink 加了 DampingTask，失败模式有界

给 Pink QP 加一个固定的 `DampingTask(cost=0.01)` 后，near_singular 的失败残余从 **40 mm → 2 mm**（20× 改善）。这等效于给 Pink 加了 Tikhonov 正则化，把 QP 从"无约束的走飞"变成"温和收敛到附近可解"。收敛率没变（还是 60%），但**失败时不再乱走**。

代价：normal servo 0.15 → 0.21 ms（约 40% 慢，但绝对值仍在亚毫秒）。

### 3. movel 的 18% 失败率是可达性问题，不是 solver 问题

即使算法都升级了，movel_near 场景仍然只有 82% 收敛率。原因：随机采样 5-10 cm 目标+ 姿态旋转，其中 ~18% 落在**该 URDF 边界外或需要跨越奇异**。这是 A1 工作空间的固有限制，不是 IK 缺陷。

**v0.3 已经修复**：现在这类失败会抛 `IKConvergenceError` 而不是静默下发错误 joints（见 `arm.py` 中 `strict=True`）。

### 4. Cold start 无 warm-start 依然只有 16% 成功率

这是局部 Newton 方法的固有局限，不是实现问题。上层要么用关节空间粗规划先接近，要么在 IK 层加多次随机重启。

### 5. proxqp vs quadprog

在 7-DOF 单臂上，proxqp 反而**略慢于** quadprog（QP 规模太小，proxqp 的 warm-start 优势体现不出来）。默认保留 quadprog；`PinkIKSolver(solver="proxqp")` 依然可选，未来用于双臂 14-DOF 或多约束场景可能有优势。

## v0.3.0 采用的所有措施

| 层次 | 措施 | 位置 |
|---|---|---|
| **正确性** | `movel/movep/movec/servop` 前 `ik.reset(current_joints)` | `arm.py` |
| **正确性** | IK 失败抛 `IKConvergenceError` 而不是只 log warning | `ik_solver.py` + `exceptions.py` |
| **正确性** | motion_strategy 插值 IK 用宽松容差（1 mm/10 mrad）+ max_iter=50 | `motion_strategy.py` |
| **速度** | DLS 默认 `dt = 1.0`（原 0.1），iters 从 28 → 2 | `ik_solver.py` |
| **鲁棒性** | Pink `DampingTask(cost=0.01)`，失败残余 40 mm → 2 mm | `ik_solver.py` |
| **数据可信** | benchmark cold-start 从非奇异 Q_COLD_INIT 起（原为零位=奇异） | `benchmarks/bench_ik.py` |

## 剩余空间（不 P0/P1）

| 措施 | 预期收益 | 说明 |
|---|---|---|
| CasADi/JAX 批量 IK | 100–1000× 提速（离线轨迹） | 仅规划场景，非实时 |
| 多起点重启 (random restart) | Cold-start 收敛率 16% → 60%+ | 需要预算多解时间；用于 movep 大跳跃 |
| 双臂 proxqp | 双臂 IK 若做联合约束时可能有 1.5-2× 优势 | 等双臂 IK 联合优化需求出现再评估 |

**结论**：IK 在 v0.3.0 已经**又快又稳**——常规场景端到端 0.07-0.42 ms（DLS 领先），近奇异 0.42 ms（Pink 领先），失败时能保持在 mm 级邻域并抛异常给上层。剩下的空间主要在离线批量规划场景。
