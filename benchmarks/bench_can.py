"""
CAN communication benchmark — measures per-cycle latency and evaluates
candidate software optimizations.

The current control loop in `can_backend.py` uses a parallel send-then-recv
strategy (send all 7 command frames back-to-back, then batch recv). This
benchmark measures its real cost, then A/B tests candidate optimizations
that don't require hardware changes:

  raw_filter          — CAN_RAW_FILTER, kernel only delivers response IDs
  tight_recv          — non-blocking recv loop, no select() per frame
  recvmmsg            — libc recvmmsg via ctypes, one syscall for N frames
  rcvbuf              — larger SO_RCVBUF
  struct_precompile   — module-level struct.Struct instead of per-call
  sched_fifo          — SCHED_FIFO priority for the control thread (needs sudo)
  mlockall            — lock memory to avoid page fault jitter (needs sudo)
  cpu_affinity        — pin control thread to a CPU (recommend isolcpus'd core)

Scenarios:
  S1 idle_command        single arm, hold-position, isolates SW + bus RTT
  S2 sweep_command       single arm, small sinusoidal trajectory
  S3 dual_arm_parallel   both arms concurrent, tests bus isolation + GIL

Usage:
    # local dry-run: exercises CLI + report generation, no socket
    python -m benchmarks.bench_can --dry-run

    # real hardware smoke test (200-cycle quick run of S1)
    python -m benchmarks.bench_can --channel can0 --scenario S1 --quick

    # full matrix (baseline + each opt individually + combined) across all scenarios
    sudo -E python -m benchmarks.bench_can --matrix

Real hardware host: woan@10.8.76.101
    can0 = left arm (A1L)
    can1 = right arm (A1R)
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import os
import platform
import select
import socket
import struct
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onero_sdk.can_bus import CanBus, CanFrame, CAN_FRAME_FMT, CAN_FRAME_SIZE
from onero_sdk.model_registry import get_model, ArmModel
from onero_sdk.protocol import (
    MotorLimits, encode_mit, decode_feedback, mit_frame_id,
    ENABLE_DATA, DISABLE_DATA, CLEAR_ERROR_DATA, MotorError,
)


# ─── CAN socket constants (not exposed by python socket module) ─────────────
SOL_CAN_BASE = 100
SOL_CAN_RAW = SOL_CAN_BASE + 1
CAN_RAW_FILTER = 1

_CAN_FILTER_FMT = "=II"  # can_id, can_mask

# Module-level pre-compiled struct (used when struct_precompile is on)
_FRAME_STRUCT = struct.Struct(CAN_FRAME_FMT)


# ─── Trial + stats ──────────────────────────────────────────────────────────
@dataclass
class Trial:
    latency_ms: float
    send_ms: float
    recv_ms: float
    decode_ms: float
    misses: int  # 0..num_joints


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[k]


def summarize(name: str, trials: list[Trial], num_joints: int) -> dict:
    if not trials:
        return {"name": name, "n": 0, "skipped": True}
    lat = [t.latency_ms for t in trials]
    snd = [t.send_ms for t in trials]
    rcv = [t.recv_ms for t in trials]
    dec = [t.decode_ms for t in trials]
    total_frames = num_joints * len(trials)
    total_misses = sum(t.misses for t in trials)
    return {
        "name": name,
        "n": len(trials),
        "lat_p50": pct(lat, 0.50),
        "lat_p95": pct(lat, 0.95),
        "lat_p99": pct(lat, 0.99),
        "lat_max": max(lat),
        "send_p50": pct(snd, 0.50),
        "recv_p50": pct(rcv, 0.50),
        "recv_p95": pct(rcv, 0.95),
        "recv_p99": pct(rcv, 0.99),
        "decode_p50": pct(dec, 0.50),
        "miss_rate": total_misses / max(1, total_frames),
        "total_misses": total_misses,
        "total_frames": total_frames,
    }


# ─── Optimization mechanisms ────────────────────────────────────────────────

@dataclass
class OptConfig:
    """Which optimizations are enabled for a run."""
    raw_filter: bool = False
    tight_recv: bool = False
    recvmmsg: bool = False
    rcvbuf: bool = False
    struct_precompile: bool = False
    sched_fifo: bool = False
    mlockall: bool = False
    cpu_affinity: Optional[int] = None

    @property
    def label(self) -> str:
        parts = []
        for name in ("raw_filter", "tight_recv", "recvmmsg", "rcvbuf",
                     "struct_precompile", "sched_fifo", "mlockall"):
            if getattr(self, name):
                parts.append(name)
        if self.cpu_affinity is not None:
            parts.append(f"cpu{self.cpu_affinity}")
        return "+".join(parts) if parts else "baseline"


def apply_socket_opts(sock: socket.socket, response_ids: list[int],
                       cfg: OptConfig) -> None:
    """Apply socket-level optimizations."""
    if cfg.raw_filter:
        filters = b"".join(
            struct.pack(_CAN_FILTER_FMT, rid, 0x7FF) for rid in response_ids
        )
        sock.setsockopt(SOL_CAN_RAW, CAN_RAW_FILTER, filters)
    if cfg.rcvbuf:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)


# ── recvmmsg via ctypes ─────────────────────────────────────────────────────

class _iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


class _msghdr(ctypes.Structure):
    _fields_ = [
        ("msg_name", ctypes.c_void_p),
        ("msg_namelen", ctypes.c_uint),
        ("msg_iov", ctypes.POINTER(_iovec)),
        ("msg_iovlen", ctypes.c_size_t),
        ("msg_control", ctypes.c_void_p),
        ("msg_controllen", ctypes.c_size_t),
        ("msg_flags", ctypes.c_int),
    ]


class _mmsghdr(ctypes.Structure):
    _fields_ = [("msg_hdr", _msghdr), ("msg_len", ctypes.c_uint)]


class _timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


_LIBC = None
try:
    _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                        use_errno=True)
    _LIBC.recvmmsg.argtypes = [
        ctypes.c_int, ctypes.POINTER(_mmsghdr), ctypes.c_uint,
        ctypes.c_int, ctypes.POINTER(_timespec),
    ]
    _LIBC.recvmmsg.restype = ctypes.c_int
except Exception:  # pragma: no cover
    _LIBC = None

MSG_DONTWAIT = 0x40


def recvmmsg_batch(sock: socket.socket, max_frames: int) -> list[bytes]:
    """Non-blocking recvmmsg: drain up to max_frames from the socket.

    Returns list of raw frame bytes (16 bytes each). Empty list if none ready.
    """
    if _LIBC is None:
        raise RuntimeError("libc recvmmsg not available")
    bufs = [(ctypes.c_ubyte * CAN_FRAME_SIZE)() for _ in range(max_frames)]
    iovs = (_iovec * max_frames)()
    msgs = (_mmsghdr * max_frames)()
    for i in range(max_frames):
        iovs[i].iov_base = ctypes.cast(bufs[i], ctypes.c_void_p)
        iovs[i].iov_len = CAN_FRAME_SIZE
        msgs[i].msg_hdr.msg_iov = ctypes.pointer(iovs[i])
        msgs[i].msg_hdr.msg_iovlen = 1
        msgs[i].msg_hdr.msg_name = None
        msgs[i].msg_hdr.msg_namelen = 0
        msgs[i].msg_hdr.msg_control = None
        msgs[i].msg_hdr.msg_controllen = 0
        msgs[i].msg_hdr.msg_flags = 0
    n = _LIBC.recvmmsg(sock.fileno(), msgs, max_frames, MSG_DONTWAIT, None)
    if n < 0:
        err = ctypes.get_errno()
        if err in (errno.EAGAIN, errno.EWOULDBLOCK):
            return []
        raise OSError(err, os.strerror(err))
    return [bytes(bufs[i][:msgs[i].msg_len]) for i in range(n)]


# ── Process-level opts (sched_fifo, mlockall, cpu_affinity) ──────────────────

class _sched_param(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]


_SCHED_FIFO = 1
_MCL_CURRENT = 1
_MCL_FUTURE = 2


def try_apply_process_opts(cfg: OptConfig) -> dict:
    """Apply process-wide opts. Returns status dict with per-opt result.

    Non-fatal on failure (auto-degrade): logs and continues.
    """
    status = {}
    if cfg.cpu_affinity is not None:
        try:
            os.sched_setaffinity(0, {cfg.cpu_affinity})
            status["cpu_affinity"] = f"pinned to core {cfg.cpu_affinity}"
        except OSError as e:
            status["cpu_affinity"] = f"FAILED: {e}"
    if cfg.sched_fifo:
        if _LIBC is None:
            status["sched_fifo"] = "SKIPPED: libc not loaded"
        else:
            param = _sched_param(50)
            ret = _LIBC.sched_setscheduler(0, _SCHED_FIFO, ctypes.byref(param))
            if ret != 0:
                err = ctypes.get_errno()
                if err == errno.EPERM:
                    status["sched_fifo"] = "SKIPPED: EPERM (need root/CAP_SYS_NICE)"
                else:
                    status["sched_fifo"] = f"FAILED: {os.strerror(err)}"
            else:
                status["sched_fifo"] = "applied (prio 50)"
    if cfg.mlockall:
        if _LIBC is None:
            status["mlockall"] = "SKIPPED: libc not loaded"
        else:
            ret = _LIBC.mlockall(_MCL_CURRENT | _MCL_FUTURE)
            if ret != 0:
                err = ctypes.get_errno()
                if err == errno.EPERM:
                    status["mlockall"] = "SKIPPED: EPERM (need root)"
                else:
                    status["mlockall"] = f"FAILED: {os.strerror(err)}"
            else:
                status["mlockall"] = "applied"
    return status


# ─── recv strategies ────────────────────────────────────────────────────────

def recv_batch_baseline(bus: CanBus, target_ids: set[int],
                        timeout: float) -> dict[int, CanFrame]:
    """Current in-tree strategy: select() per frame, drain until all done."""
    return bus.recv_batch(target_ids, timeout=timeout)


def recv_batch_tight(bus: CanBus, target_ids: set[int], timeout: float,
                     use_precompile: bool = False) -> dict[int, CanFrame]:
    """Non-blocking recv loop, no select() per frame."""
    sock = bus._sock
    deadline = time.perf_counter() + timeout
    collected: dict[int, CanFrame] = {}
    remaining = set(target_ids)
    unpack = _FRAME_STRUCT.unpack if use_precompile else \
        (lambda raw: struct.unpack(CAN_FRAME_FMT, raw))
    while remaining and time.perf_counter() < deadline:
        try:
            raw = sock.recv(CAN_FRAME_SIZE, socket.MSG_DONTWAIT)
        except BlockingIOError:
            # No data available. Very brief spin — small sleep to avoid
            # burning CPU while still catching frames within 10s of µs.
            time.sleep(1e-5)
            continue
        can_id, dlc, data = unpack(raw)
        arb_id = can_id & 0x1FFFFFFF if (can_id & 0x80000000) else can_id & 0x7FF
        if arb_id in remaining:
            collected[arb_id] = CanFrame(can_id=can_id, data=data[:dlc], dlc=dlc)
            remaining.discard(arb_id)
    return collected


def recv_batch_mmsg(bus: CanBus, target_ids: set[int], timeout: float,
                    use_precompile: bool = False) -> dict[int, CanFrame]:
    """Use recvmmsg to batch-drain frames. Loops until all targets in or timeout."""
    sock = bus._sock
    deadline = time.perf_counter() + timeout
    collected: dict[int, CanFrame] = {}
    remaining = set(target_ids)
    unpack = _FRAME_STRUCT.unpack if use_precompile else \
        (lambda raw: struct.unpack(CAN_FRAME_FMT, raw))
    while remaining and time.perf_counter() < deadline:
        frames = recvmmsg_batch(sock, len(remaining) + 4)  # small headroom for stragglers
        if not frames:
            time.sleep(1e-5)
            continue
        for raw in frames:
            if len(raw) < CAN_FRAME_SIZE:
                continue
            can_id, dlc, data = unpack(raw)
            arb_id = can_id & 0x1FFFFFFF if (can_id & 0x80000000) else can_id & 0x7FF
            if arb_id in remaining:
                collected[arb_id] = CanFrame(can_id=can_id, data=data[:dlc], dlc=dlc)
                remaining.discard(arb_id)
    return collected


def get_recv_fn(cfg: OptConfig):
    """Pick the recv strategy based on config. Returns callable(bus, ids, timeout)."""
    if cfg.recvmmsg:
        return lambda bus, ids, to: recv_batch_mmsg(bus, ids, to, cfg.struct_precompile)
    if cfg.tight_recv or cfg.struct_precompile:
        return lambda bus, ids, to: recv_batch_tight(bus, ids, to, cfg.struct_precompile)
    return recv_batch_baseline


# ─── Arm session (enable + cycle runner) ─────────────────────────────────────

@dataclass
class ArmSession:
    channel: str
    model: ArmModel
    bus: CanBus
    limits: list[MotorLimits]
    can_ids: list[int]
    response_ids: list[int]
    home_positions: list[float]

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


def enable_arm(bus: CanBus, model: ArmModel) -> list[MotorLimits]:
    """Enable all motors, returning their limits for later encode calls."""
    limits = [MotorLimits(p_max=j.encoding_p_max, v_max=j.v_max, t_max=j.t_max)
              for j in model.joints]
    # Clear any residual errors first
    for spec in model.joints:
        bus.send(mit_frame_id(spec.can_id), CLEAR_ERROR_DATA)
    time.sleep(0.1)
    bus.flush_rx()
    # Enable
    for spec in model.joints:
        bus.send(mit_frame_id(spec.can_id), ENABLE_DATA)
    time.sleep(0.1)
    bus.flush_rx()
    return limits


def disable_arm(bus: CanBus, model: ArmModel) -> None:
    for spec in model.joints:
        try:
            bus.send(mit_frame_id(spec.can_id), DISABLE_DATA)
        except Exception:
            pass


def read_home_positions(bus: CanBus, model: ArmModel, limits: list[MotorLimits],
                        recv_timeout: float = 0.05) -> list[float]:
    """Send a zero-torque command and read back current joint positions."""
    positions = [0.0] * model.num_joints
    for i, spec in enumerate(model.joints):
        data = encode_mit(0.0, 0.0, 0.0, 0.0, 0.0, limits[i])
        bus.send(mit_frame_id(spec.can_id), data)
    time.sleep(0.005)
    for i, spec in enumerate(model.joints):
        frame = bus.recv_until(spec.response_id, timeout=recv_timeout)
        if frame is not None:
            state = decode_feedback(frame.data, limits[i])
            positions[i] = state.position
    return positions


@contextmanager
def open_arm_session(channel: str, model_name: str):
    """Context manager: enables arm on entry, disables on exit."""
    bus = CanBus(channel)
    model = get_model(model_name)
    try:
        limits = enable_arm(bus, model)
        home = read_home_positions(bus, model, limits)
        session = ArmSession(
            channel=channel, model=model, bus=bus, limits=limits,
            can_ids=[j.can_id for j in model.joints],
            response_ids=[j.response_id for j in model.joints],
            home_positions=home,
        )
        yield session
    finally:
        try:
            disable_arm(bus, model)
        except Exception:
            pass
        bus.close()


# ─── One-cycle command ──────────────────────────────────────────────────────

def run_cycle(session: ArmSession, positions: list[float],
              recv_fn: Callable, recv_timeout: float = 0.01) -> Trial:
    """Run one command cycle, timing send/recv/decode separately."""
    bus = session.bus
    limits = session.limits
    joints = session.model.joints

    t0 = time.perf_counter_ns()
    for i, spec in enumerate(joints):
        data = encode_mit(positions[i], 0.0, 0.0, 0.0, 0.0, limits[i])
        bus.send(mit_frame_id(spec.can_id), data)
    t1 = time.perf_counter_ns()

    target_ids = set(session.response_ids)
    received = recv_fn(bus, target_ids, recv_timeout)
    t2 = time.perf_counter_ns()

    misses = 0
    for i, spec in enumerate(joints):
        frame = received.get(spec.response_id)
        if frame is None:
            misses += 1
        else:
            _ = decode_feedback(frame.data, limits[i])  # exercise decode path
    t3 = time.perf_counter_ns()

    return Trial(
        latency_ms=(t3 - t0) / 1e6,
        send_ms=(t1 - t0) / 1e6,
        recv_ms=(t2 - t1) / 1e6,
        decode_ms=(t3 - t2) / 1e6,
        misses=misses,
    )


# ─── Scenarios ──────────────────────────────────────────────────────────────

def scenario_idle(session: ArmSession, recv_fn: Callable, n: int, hz: float) -> list[Trial]:
    """S1: hold current position at target Hz."""
    positions = list(session.home_positions)
    return _run_loop(session, recv_fn, n, hz, lambda t: positions)


def scenario_sweep(session: ArmSession, recv_fn: Callable, n: int, hz: float,
                    amplitude: float = 0.05) -> list[Trial]:
    """S2: small sinusoidal sweep on all joints."""
    home = list(session.home_positions)
    import math
    def targets_at(t: float) -> list[float]:
        return [home[i] + amplitude * math.sin(2 * math.pi * 0.5 * t)
                for i in range(len(home))]
    return _run_loop(session, recv_fn, n, hz, targets_at)


def _run_loop(session: ArmSession, recv_fn: Callable, n: int, hz: float,
              target_fn: Callable) -> list[Trial]:
    """Common trial loop with a target-rate ticker."""
    period = 1.0 / hz
    trials = []
    start = time.perf_counter()
    for i in range(n):
        deadline = start + i * period
        now = time.perf_counter()
        if now < deadline:
            time.sleep(deadline - now)
        elapsed = time.perf_counter() - start
        positions = target_fn(elapsed)
        trial = run_cycle(session, positions, recv_fn)
        trials.append(trial)
    return trials


def scenario_dual_arm(channel_left: str, channel_right: str, model_left: str,
                       model_right: str, cfg: OptConfig, n: int, hz: float
                       ) -> tuple[list[Trial], list[Trial]]:
    """S3: run S1 on both arms concurrently, one thread per bus."""
    trials_left: list[Trial] = []
    trials_right: list[Trial] = []
    err = [None]

    def worker(channel: str, model_name: str, sink: list[Trial]):
        try:
            with open_arm_session(channel, model_name) as session:
                apply_socket_opts(session.bus._sock, session.response_ids, cfg)
                recv_fn = get_recv_fn(cfg)
                # Warm-up
                for _ in range(50):
                    run_cycle(session, session.home_positions, recv_fn)
                sink.extend(scenario_idle(session, recv_fn, n, hz))
        except Exception as e:
            err[0] = e

    tl = threading.Thread(target=worker, args=(channel_left, model_left, trials_left))
    tr = threading.Thread(target=worker, args=(channel_right, model_right, trials_right))
    tl.start(); tr.start()
    tl.join(); tr.join()
    if err[0] is not None:
        raise err[0]
    return trials_left, trials_right


# ─── Config runner ──────────────────────────────────────────────────────────

def run_config(channel: str, model_name: str, scenario: str, cfg: OptConfig,
               n: int, hz: float) -> dict:
    """Run one (config, scenario) combination and return summary + status."""
    proc_status = try_apply_process_opts(cfg)
    if scenario == "S3":
        # Special-case: dual-arm needs two buses. Assume can0=A1L, can1=A1R.
        left, right = scenario_dual_arm("can0", "can1", "A1L", "A1R",
                                         cfg, n, hz)
        summary_l = summarize(f"S3-left · {cfg.label}", left, 7)
        summary_r = summarize(f"S3-right · {cfg.label}", right, 7)
        # Combine (report both for visibility)
        return {
            "config": cfg.label,
            "scenario": scenario,
            "process_opts": proc_status,
            "summary": summary_l,
            "summary_right": summary_r,
        }
    with open_arm_session(channel, model_name) as session:
        apply_socket_opts(session.bus._sock, session.response_ids, cfg)
        recv_fn = get_recv_fn(cfg)
        # Warm-up
        for _ in range(50):
            run_cycle(session, session.home_positions, recv_fn)
        if scenario == "S1":
            trials = scenario_idle(session, recv_fn, n, hz)
        elif scenario == "S2":
            trials = scenario_sweep(session, recv_fn, n, hz)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    summary = summarize(f"{scenario} · {cfg.label}", trials, 7)
    return {
        "config": cfg.label,
        "scenario": scenario,
        "process_opts": proc_status,
        "summary": summary,
    }


# ─── Matrix orchestration ───────────────────────────────────────────────────

def matrix_configs(include_privileged: bool = True) -> list[OptConfig]:
    """Return the set of configs to iterate for --matrix mode."""
    configs = [OptConfig()]  # baseline
    # Each opt individually
    configs.append(OptConfig(raw_filter=True))
    configs.append(OptConfig(tight_recv=True))
    if _LIBC is not None:
        configs.append(OptConfig(recvmmsg=True))
    configs.append(OptConfig(rcvbuf=True))
    configs.append(OptConfig(struct_precompile=True))
    if include_privileged:
        configs.append(OptConfig(sched_fifo=True))
        configs.append(OptConfig(mlockall=True))
    # Combined userland stack (all non-privileged)
    configs.append(OptConfig(
        raw_filter=True, tight_recv=True, rcvbuf=True, struct_precompile=True,
    ))
    if include_privileged:
        # Kitchen sink
        configs.append(OptConfig(
            raw_filter=True, tight_recv=True, rcvbuf=True, struct_precompile=True,
            sched_fifo=True, mlockall=True,
        ))
    return configs


# ─── Reporting ──────────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    s = result["summary"]
    if s.get("skipped"):
        print(f"  {result['config']:<40}  SKIPPED")
        return
    print(f"  {result['config']:<40}  n={s['n']:>4}  "
          f"lat p50={s['lat_p50']:.3f}  p95={s['lat_p95']:.3f}  "
          f"p99={s['lat_p99']:.3f}  max={s['lat_max']:.3f} ms  "
          f"miss={s['miss_rate']*100:.2f}%")


def write_markdown_report(results: list[dict], path: Path,
                          host_info: str) -> None:
    """Emit the full markdown report — matches bench_ik_report.md style."""
    lines = []
    lines.append("# CAN Benchmark Report — Onero A1 SDK")
    lines.append("")
    lines.append("> 硬件不能改，只测软件层优化的实际收益。判定标准：p99 改善 > 5% 值得合并，1-5% 待观察，< 1% 不合并。")
    lines.append("")
    lines.append("## 测试环境")
    lines.append("")
    lines.append(host_info)
    lines.append("")
    lines.append("## 场景")
    lines.append("")
    lines.append("| ID | 场景 | 说明 |")
    lines.append("|---|---|---|")
    lines.append("| S1 | idle_command | 单臂 hold-position，隔离软件 + 总线纯 RTT |")
    lines.append("| S2 | sweep_command | 单臂小幅正弦轨迹，模拟真实运动负载 |")
    lines.append("| S3 | dual_arm_parallel | 双臂并发（can0+can1），验证 GIL / CPU 竞争 |")
    lines.append("")
    lines.append("每 trial = 一次 `command()`：encode 7 → send 7 → recv batch → decode 7。N = 5000（或 --quick=500）。")
    lines.append("")

    # Group by scenario
    by_scen: dict[str, list[dict]] = {}
    for r in results:
        by_scen.setdefault(r["scenario"], []).append(r)

    # Table 1: baseline
    lines.append("## Table 1 — Baseline (无任何优化)")
    lines.append("")
    lines.append("| 场景 | n | p50 ms | p95 ms | p99 ms | max ms | send p50 | recv p50 | decode p50 | miss rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for scen, rs in sorted(by_scen.items()):
        baseline = next((r for r in rs if r["config"] == "baseline"), None)
        if baseline is None:
            continue
        s = baseline["summary"]
        if s.get("skipped"):
            continue
        lines.append(
            f"| {scen} | {s['n']} | {s['lat_p50']:.3f} | {s['lat_p95']:.3f} | "
            f"{s['lat_p99']:.3f} | {s['lat_max']:.3f} | {s['send_p50']:.3f} | "
            f"{s['recv_p50']:.3f} | {s['decode_p50']:.3f} | {s['miss_rate']*100:.2f}% |"
        )
    lines.append("")

    # Table 2: per-opt delta vs baseline
    lines.append("## Table 2 — 单项优化 A/B (delta vs baseline)")
    lines.append("")
    lines.append("| 场景 | 优化 | p50 ms | Δ p50 | p95 ms | Δ p95 | p99 ms | Δ p99 | miss |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for scen, rs in sorted(by_scen.items()):
        baseline = next((r for r in rs if r["config"] == "baseline"), None)
        if baseline is None or baseline["summary"].get("skipped"):
            continue
        bs = baseline["summary"]
        for r in rs:
            if r["config"] == "baseline":
                continue
            s = r["summary"]
            if s.get("skipped"):
                lines.append(f"| {scen} | {r['config']} | SKIPPED | | | | | | |")
                continue
            def d(new, old):
                if old == 0:
                    return "—"
                pct_ = (new - old) / old * 100
                return f"{pct_:+.1f}%"
            lines.append(
                f"| {scen} | {r['config']} | {s['lat_p50']:.3f} | "
                f"{d(s['lat_p50'], bs['lat_p50'])} | "
                f"{s['lat_p95']:.3f} | {d(s['lat_p95'], bs['lat_p95'])} | "
                f"{s['lat_p99']:.3f} | {d(s['lat_p99'], bs['lat_p99'])} | "
                f"{s['miss_rate']*100:.2f}% |"
            )
    lines.append("")

    # Table 3: process-opt status (which privileged ones actually applied?)
    lines.append("## Table 3 — 特权优化实际应用情况")
    lines.append("")
    lines.append("| 场景 | 优化组合 | 应用状态 |")
    lines.append("|---|---|---|")
    for r in results:
        if r["process_opts"]:
            for opt, status in r["process_opts"].items():
                lines.append(f"| {r['scenario']} | {r['config']} | {opt}: {status} |")
    lines.append("")

    # Conclusion placeholder — filled in manually after review
    lines.append("## 结论")
    lines.append("")
    lines.append("_（跑完 matrix 后手动填入：每个优化标记为「合并」「待观察」「不合并」并说明原因。）_")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def collect_host_info() -> str:
    """Grab OS / CPU / kernel info for the report header."""
    lines = []
    lines.append(f"- Host: {platform.node()}")
    lines.append(f"- Kernel: {platform.release()}")
    lines.append(f"- Python: {platform.python_version()}")
    try:
        import subprocess
        out = subprocess.check_output(["ip", "-details", "link", "show", "can0"],
                                       stderr=subprocess.DEVNULL, timeout=1).decode()
        can_line = next((l for l in out.splitlines() if "bitrate" in l), "")
        if can_line:
            lines.append(f"- can0: {can_line.strip()}")
    except Exception:
        pass
    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_opt_string(s: str) -> OptConfig:
    """Parse '--opt a,b,c' into an OptConfig."""
    cfg = OptConfig()
    if not s:
        return cfg
    for name in s.split(","):
        name = name.strip()
        if not name:
            continue
        if name.startswith("cpu"):
            cfg.cpu_affinity = int(name[3:])
        elif hasattr(cfg, name):
            setattr(cfg, name, True)
        else:
            raise ValueError(f"Unknown opt: {name}")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="CAN communication benchmark")
    ap.add_argument("--channel", default="can0",
                    help="CAN interface (default can0). S3 always uses can0+can1.")
    ap.add_argument("--model", default=None,
                    help="Arm model. Defaults to A1L for can0, A1R for can1.")
    ap.add_argument("--scenario", choices=["S1", "S2", "S3"], default="S1",
                    help="Scenario to run (default S1)")
    ap.add_argument("--opt", default="",
                    help="Comma-separated opt names, e.g. 'raw_filter,tight_recv'")
    ap.add_argument("--matrix", action="store_true",
                    help="Run baseline + each opt individually + combined, all scenarios")
    ap.add_argument("--quick", action="store_true",
                    help="500 trials/config (default 5000)")
    ap.add_argument("--n", type=int, default=None,
                    help="Explicit trial count (overrides --quick)")
    ap.add_argument("--hz", type=float, default=200.0,
                    help="Target control frequency (default 200)")
    ap.add_argument("--report-path", default="docs/bench_can_report.md",
                    help="Where to write the markdown report")
    ap.add_argument("--dry-run", action="store_true",
                    help="Exercise CLI + report generation without opening any socket")
    ap.add_argument("--no-privileged", action="store_true",
                    help="Skip sched_fifo/mlockall from --matrix")
    args = ap.parse_args()

    if args.n is not None:
        n = args.n
    elif args.quick:
        n = 500
    else:
        n = 5000

    if args.dry_run:
        print("# Dry-run — validating CLI + report generation")
        cfgs = matrix_configs(include_privileged=not args.no_privileged)
        print(f"# Would run {len(cfgs)} configs × 3 scenarios = {len(cfgs)*3} runs")
        for cfg in cfgs:
            print(f"  config: {cfg.label}")
        # Fake results for report generation
        results = []
        for scen in ("S1", "S2", "S3"):
            for cfg in cfgs:
                trials = [Trial(0.5, 0.1, 0.35, 0.05, 0) for _ in range(10)]
                results.append({
                    "config": cfg.label,
                    "scenario": scen,
                    "process_opts": {},
                    "summary": summarize(f"{scen} · {cfg.label}", trials, 7),
                })
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(results, path, "(dry-run — no host info)")
        print(f"# Report written to {path}")
        return 0

    model_name = args.model or ("A1L" if args.channel == "can0" else "A1R")

    print(f"# CAN benchmark")
    print(f"channel     : {args.channel}")
    print(f"model       : {model_name}")
    print(f"trials/cfg  : {n}")
    print(f"control Hz  : {args.hz}")
    print()

    results = []
    if args.matrix:
        configs = matrix_configs(include_privileged=not args.no_privileged)
        scenarios = ["S1", "S2", "S3"]
        for scen in scenarios:
            print(f"=== Scenario {scen} ===")
            for cfg in configs:
                try:
                    result = run_config(args.channel, model_name, scen, cfg, n, args.hz)
                except Exception as e:
                    print(f"  {cfg.label:<40}  ERROR: {e}")
                    result = {
                        "config": cfg.label, "scenario": scen,
                        "process_opts": {},
                        "summary": {"name": cfg.label, "n": 0, "skipped": True,
                                     "error": str(e)},
                    }
                results.append(result)
                print_summary(result)
    else:
        cfg = parse_opt_string(args.opt)
        result = run_config(args.channel, model_name, args.scenario, cfg, n, args.hz)
        results.append(result)
        print_summary(result)

    path = Path(args.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(results, path, collect_host_info())
    print(f"\n# Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
