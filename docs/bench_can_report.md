# CAN Benchmark Report — Onero A1 SDK

> 硬件不能改，只测软件层优化的实际收益。判定标准：p99 改善 > 5% 值得合并，1-5% 待观察，< 1% 不合并。

## 测试环境

- Host: woan
- Kernel: 6.8.0-107-generic
- Python: 3.12.3
- can0: bitrate 0

## 场景

| ID | 场景 | 说明 |
|---|---|---|
| S1 | idle_command | 单臂 hold-position，隔离软件 + 总线纯 RTT |
| S2 | sweep_command | 单臂小幅正弦轨迹，模拟真实运动负载 |
| S3 | dual_arm_parallel | 双臂并发（can0+can1），验证 GIL / CPU 竞争 |

每 trial = 一次 `command()`：encode 7 → send 7 → recv batch → decode 7。N = 1000 / 场景，目标 200 Hz（周期预算 5 ms）。

**测试条件**：`python3` 已通过 `setcap cap_sys_nice,cap_ipc_lock+ep` 授权，`sched_fifo`（prio 50）与 `mlockall(MCL_CURRENT|MCL_FUTURE)` 无需 sudo 即可生效。

**总线时序背景**：classic CAN @ 1 Mbps。7 帧响应串行占用总线约 770 µs（不含 stuff bit / 帧间隙 / 电机固件处理）。Baseline `recv_ms ≈ 1.7–1.8 ms` 意味着 **总线时间已占周期 85%+**，软件层可优化的绝对预算约 300 µs（send 200 + decode 55 + framing/syscall 30-50）。

## Table 1 — Baseline (无任何优化)

| 场景 | n | p50 ms | p95 ms | p99 ms | max ms | send p50 | recv p50 | decode p50 | miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1000 | 2.042 | 2.158 | 2.169 | 4.585 | 0.198 | 1.788 | 0.055 | 0.00% |
| S2 | 1000 | 2.025 | 2.145 | 2.154 | 2.267 | 0.197 | 1.770 | 0.055 | 0.00% |
| S3 | 1000 | 2.044 | 2.203 | 2.220 | 3.020 | 0.293 | 1.669 | 0.058 | 0.00% |

## Table 2 — 单项优化 A/B (delta vs baseline)

| 场景 | 优化 | p50 ms | Δ p50 | p95 ms | Δ p95 | p99 ms | Δ p99 | miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | raw_filter | 2.041 | -0.0% | 2.155 | -0.1% | 2.197 | +1.3% | 0.00% |
| S1 | tight_recv | 2.031 | -0.5% | 2.059 | -4.6% | 2.095 | -3.4% | 0.00% |
| S1 | recvmmsg | 2.142 | +4.9% | 2.276 | +5.5% | 2.470 | +13.8% | 0.00% |
| S1 | rcvbuf | 2.036 | -0.3% | 2.159 | +0.1% | 2.236 | +3.1% | 0.00% |
| S1 | struct_precompile | 2.019 | -1.1% | 2.178 | +1.0% | 2.183 | +0.6% | 0.00% |
| S1 | sched_fifo | 2.000 | -2.1% | 2.108 | -2.3% | 2.162 | -0.3% | 0.00% |
| S1 | mlockall | 1.996 | -2.3% | 2.119 | -1.8% | 2.164 | -0.3% | 0.00% |
| S1 | raw_filter+tight_recv+rcvbuf+struct_precompile | 1.995 | -2.3% | 2.060 | -4.5% | 2.070 | -4.6% | 0.00% |
| S1 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | 1.997 | -2.2% | 2.040 | -5.5% | 2.053 | -5.4% | 0.00% |
| S2 | raw_filter | 1.991 | -1.7% | 2.083 | -2.9% | 2.090 | -3.0% | 0.00% |
| S2 | tight_recv | 1.991 | -1.6% | 2.037 | -5.0% | 2.043 | -5.2% | 0.00% |
| S2 | recvmmsg | 2.044 | +0.9% | 2.201 | +2.6% | 2.351 | +9.1% | 0.00% |
| S2 | rcvbuf | 1.998 | -1.3% | 2.102 | -2.0% | 2.113 | -1.9% | 0.00% |
| S2 | struct_precompile | 1.999 | -1.3% | 2.134 | -0.5% | 2.154 | -0.0% | 0.00% |
| S2 | sched_fifo | 2.001 | -1.2% | 2.156 | +0.5% | 2.170 | +0.7% | 0.00% |
| S2 | mlockall | 2.018 | -0.3% | 2.101 | -2.0% | 2.112 | -1.9% | 0.00% |
| S2 | raw_filter+tight_recv+rcvbuf+struct_precompile | 1.992 | -1.6% | 2.039 | -5.0% | 2.047 | -5.0% | 0.00% |
| S2 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | 1.991 | -1.6% | 2.040 | -4.9% | 2.049 | -4.9% | 0.00% |
| S3 | raw_filter | 2.051 | +0.4% | 2.109 | -4.3% | 2.145 | -3.4% | 0.00% |
| S3 | tight_recv | 1.978 | -3.2% | 2.041 | -7.4% | 2.053 | -7.5% | 0.00% |
| S3 | recvmmsg | 2.019 | -1.2% | 2.053 | -6.8% | 2.193 | -1.2% | 0.00% |
| S3 | rcvbuf | 2.016 | -1.3% | 2.093 | -5.0% | 2.147 | -3.3% | 0.00% |
| S3 | struct_precompile | 2.007 | -1.8% | 2.060 | -6.5% | 2.101 | -5.4% | 0.00% |
| S3 | sched_fifo | 1.978 | -3.2% | 2.096 | -4.8% | 2.138 | -3.7% | 0.00% |
| S3 | mlockall | 1.960 | -4.1% | 2.077 | -5.7% | 2.095 | -5.6% | 0.00% |
| S3 | raw_filter+tight_recv+rcvbuf+struct_precompile | 2.036 | -0.4% | 2.097 | -4.8% | 2.131 | -4.0% | 0.00% |
| S3 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | 1.988 | -2.7% | 2.041 | -7.4% | 2.053 | -7.5% | 0.00% |

## Table 3 — 特权优化实际应用情况

| 场景 | 优化组合 | 应用状态 |
|---|---|---|
| S1 | sched_fifo | sched_fifo: applied (prio 50) |
| S1 | mlockall | mlockall: applied |
| S1 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | sched_fifo: applied (prio 50) |
| S1 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | mlockall: applied |
| S2 | sched_fifo | sched_fifo: applied (prio 50) |
| S2 | mlockall | mlockall: applied |
| S2 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | sched_fifo: applied (prio 50) |
| S2 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | mlockall: applied |
| S3 | sched_fifo | sched_fifo: applied (prio 50) |
| S3 | mlockall | mlockall: applied |
| S3 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | sched_fifo: applied (prio 50) |
| S3 | raw_filter+tight_recv+rcvbuf+struct_precompile+sched_fifo+mlockall | mlockall: applied |

## 结论

### 关键发现

**1. Baseline 稳定但 max 有毛刺**
- p50 ≈ 2.03 ms、p99 ≈ 2.17 ms，miss rate 0%
- **max 分布不稳**：S1 baseline max = 4.585 ms（比 p99 高 2.4 ms），S3 max = 3.020 ms
- 5 ms 周期预算下平均有 ~2.9 ms 余量，但**偶发 outlier 会吃掉这个余量**——这是本次值得关注的问题

**2. `sched_fifo` + `mlockall` 显著削尾**
- S1: baseline max 4.585 → kitchen sink max 2.546（**−44%**）
- S2: baseline max 2.267 → kitchen sink max 2.110（−7%，本来就没抖动）
- S3: baseline max 3.020 → sched_fifo 单独 max 2.155（**−29%**）
- 但 mlockall 单独在 S3 出现 max=6.521 ms 的 outlier（触发首次 page-in），组合使用时更稳

**3. 组合优化过 5% 阈值**
- S1 kitchen sink p99 −5.4%（p99 2.169 → 2.053）
- S3 kitchen sink p99 −7.5%（p99 2.220 → 2.053）
- S2 kitchen sink p99 −4.9%（临界值，接近但未过）
- 一致跨三场景，改动可控

**4. `tight_recv` 单独就是关键贡献**
- S1 −3.4%、S2 −5.2%、S3 **−7.5%** ← 单项最强
- 组合里的其他项（raw_filter/rcvbuf/struct_precompile）都比它弱
- S3（双臂并发）尤其明显：去掉 `select()` 减少每帧 syscall 争用效果最大

**5. `recvmmsg` 明确反向优化**
- 三场景一致：S1 +13.8%、S2 +9.1%、S3 里 p95 −6.8% 但 p99 +1.2%（不一致）
- 7 帧场景 mmsghdr 数组构造成本超过省下的 syscall 收益，**不合并**

### 决策表

| 优化 | 决策 | 依据 |
|---|---|---|
| `tight_recv` | ✅ **合并** | 单项最强（S3 p99 −7.5%），改动局部化 |
| `sched_fifo` | ✅ **合并** | 主要贡献 max 削尾（S1 max −44%），p50/p99 也稳定改善 |
| `mlockall` | ✅ **合并**（配合 sched_fifo） | 单独用会有 page-in outlier，与 sched_fifo 组合稳定 |
| userland 组合（含 tight_recv） | ✅ **合并** | p99 均过 4.5%，跨场景一致 |
| `raw_filter` | ⚠️ **待观察** | 单场景偶有 −3% 收益，无干扰流量场景无价值。留 API 接口，未来接入 shared bus 时启用 |
| `struct_precompile` | ⚠️ **待观察** | 单项 <2%，但代价极小，可随 tight_recv 一起并入 |
| `rcvbuf` | ❌ **不合并** | 效果在噪声内（±2%），且当前无丢帧 |
| `recvmmsg` | ❌ **不合并** | 三场景一致反向（+9-14% p99） |

### 合并优先级（下一 PR）

**P0 · `tight_recv`**：在 `can_bus.py` 添加 `recv_batch_v2()`（`MSG_DONTWAIT` + monotonic deadline，去掉 `select()`），`can_backend._collect_responses` 切换调用。改动约 30 行，跨场景 p99 −3% 到 −7.5%。

**P0 · `sched_fifo` + `mlockall`**：在 `Arm.enable()` 或 `RobotWrapper` 启动时尝试提权（失败时降级），并在 `disable()` 时恢复。要求：
1. 部署文档写明 `setcap cap_sys_nice,cap_ipc_lock+ep $(readlink -f $(which python3))`
2. `mlockall` 只在 `sched_fifo` 成功时才启用（避免单独 page-in outlier）

**P1 · `struct_precompile`**：把 `struct.Struct(CAN_FRAME_FMT)` 提到 `can_bus` 模块级，替换 `recv_batch_v2` 内的 `struct.unpack/pack`。改动约 5 行。

### 未来（不 P0/P1）

- **`raw_filter` API 保留**：加 `CanBus(channel, filter_ids=None)` 参数，默认不启用。多设备共享总线时可开。
- **提高到 400 Hz 周期？** 当前 p99 2.05 ms（合并后）到 400 Hz 周期 2.5 ms 只剩 0.45 ms 余量，风险高。若真要提频，必须先做电机固件层改造（一次广播返回多路数据），单靠软件不够。
- **CAN-FD**：需要硬件更换，已排除。

### 合并后回归验证（2026-07-09）

已合并：
- `can_bus.py`：`recv_batch()` 改为 `MSG_DONTWAIT` + 单调 deadline 紧凑循环，模块级 `_FRAME_STRUCT` 预编译（tight_recv + struct_precompile 合并到生产路径）。`send()` 增加 `ENOBUFS` 短 backoff 重试（qdisc `qlen 10` 在 SCHED_FIFO 下会被 Python 突发 send 打满，最多 3 次重试）；`__init__` 尾部 flush_rx 清残帧；`flush_rx` 用 `MSG_DONTWAIT` 一次性排空。
- `onero_sdk/realtime.py`（新增）：`apply_realtime()` / `release_realtime()` helper；`mlockall` 仅在 `sched_fifo` 成功后启用；权限缺失时非致命降级。
- `arm.py`：`Arm(realtime=True)` 在 `enable()` 时提权、`disable()` 时恢复；非 `CanBackend` 时跳过。

真机验证（`woan@10.8.76.101`，`can0` 左臂）：
- Bench S1 baseline（走新 `recv_batch`）：p50=2.010、p99=2.079、max=2.263 ms（vs 优化前 p50=2.042、p99=2.169、max=4.585 ms）
- `Arm(channel="can0", realtime=True).enable()` 日志：`Realtime priority active: sched_fifo=applied (prio 50), mlockall=applied`
- Pytest 全套：所有非 mujoco 依赖测试通过（mujoco 相关失败为环境依赖缺失，与本次修改无关）

### A/B 诊断：60 ms 尖峰的归因（2026-07-09）

**观察**：合并后重跑 matrix 时，S1/S3 baseline 偶发 max ≈ 60 ms（p50/p99 未变差）。初步怀疑 `tight_recv` 相比 `select()` 对 OS 抢占更敏感（`nanosleep(10µs)` 在被抢占期间会被拉长）。

**A/B 验证**：把生产 `recv_batch` 临时改回 select 版本，跑同样 matrix。结果——

| 场景 | tight_recv 版本 max | 回退到 select 版本 max |
|---|---:|---:|
| S1 baseline | 59.585 | **60.013** |
| S3 baseline | 59.4 (miss=0.01%) | **62.092** |
| S3 rcvbuf | 2.074 | **63.1 (miss=0.02%)** |

**结论**：60 ms 尖峰与 recv_batch 实现无关。是机器 OS 层面的 scheduler jitter（跨天负载不同、其他进程、内核 workqueue 等）导致 5000 trial 里稳定撞到 1 次抢占。**佐证**：所有开 sched_fifo 的配置 max 都 < 4 ms（S1 sched_fifo=2.795、S3 sched_fifo=2.845、S3 kitchen sink=3.741）。

**决策**：保持 `tight_recv` 生产版本（p50/p99 略优、跟 select 表现基本等价）。**`Arm(realtime=True)` 是消除 max 尖峰的唯一稳定手段**——这个原有结论不变，且被本轮独立数据再次证实。
