"""
Real-time priority helpers for the control thread.

Elevating the caller thread to SCHED_FIFO and locking the process memory
significantly reduces CAN control-loop tail latency. Benchmarked (see
`docs/bench_can_report.md`): max cycle time on can0 idle drops from
4.6 ms to 2.5 ms (−44%), and p99 improves by 5-7% end-to-end.

Requires Linux capabilities `CAP_SYS_NICE` (for `sched_setscheduler`)
and `CAP_IPC_LOCK` (for `mlockall`). Grant them once per Python binary:

    sudo setcap cap_sys_nice,cap_ipc_lock+ep $(readlink -f $(which python3))

If the caps are missing, `apply_realtime()` silently degrades to a no-op
and returns a status dict so callers can log or surface the failure.

Scope: `sched_setscheduler(0, ...)` uses PID 0 = the calling thread only.
Callers running the control loop in a separate thread must call
`apply_realtime()` from that thread. `mlockall()` is process-wide.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


SCHED_FIFO = 1
MCL_CURRENT = 1
MCL_FUTURE = 2


class _sched_param(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]


_libc: Optional[ctypes.CDLL] = None


def _get_libc() -> Optional[ctypes.CDLL]:
    global _libc
    if _libc is None:
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                              use_errno=True)
            lib.sched_setscheduler.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(_sched_param),
            ]
            lib.sched_setscheduler.restype = ctypes.c_int
            lib.mlockall.argtypes = [ctypes.c_int]
            lib.mlockall.restype = ctypes.c_int
            lib.munlockall.argtypes = []
            lib.munlockall.restype = ctypes.c_int
            _libc = lib
        except (OSError, AttributeError) as e:
            logger.debug(f"libc not loadable for realtime helpers: {e}")
            _libc = None
    return _libc


@dataclass
class RealtimeStatus:
    """Result of an apply_realtime() call.

    `applied` is True only when every requested opt succeeded. Individual
    outcomes are in `results` — a dict of {opt_name: status_string}.
    """
    applied: bool
    results: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.results.items())


def apply_realtime(sched_fifo_priority: int = 50,
                    lock_memory: bool = True) -> RealtimeStatus:
    """
    Elevate the calling thread to SCHED_FIFO and lock process memory.

    Args:
        sched_fifo_priority: SCHED_FIFO priority (1-99). Default 50 keeps
            headroom above most system threads (which run at 0 = SCHED_OTHER)
            without competing with kernel IRQ threads (typically 50+).
        lock_memory: Call mlockall(MCL_CURRENT | MCL_FUTURE) to prevent
            future page faults. Only enabled when sched_fifo also succeeds
            (isolated mlockall was observed to cause page-in outliers on
            benchmark S3 without the priority boost). Set False to skip.

    Returns:
        RealtimeStatus with per-op result strings. Non-fatal on failure.
    """
    libc = _get_libc()
    results: dict[str, str] = {}

    if libc is None:
        results["sched_fifo"] = "SKIPPED: libc not loadable"
        results["mlockall"] = "SKIPPED: libc not loadable"
        return RealtimeStatus(applied=False, results=results)

    param = _sched_param(sched_fifo_priority)
    rc = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
    if rc != 0:
        err = ctypes.get_errno()
        if err == errno.EPERM:
            results["sched_fifo"] = (
                "SKIPPED: EPERM — grant with "
                "`setcap cap_sys_nice,cap_ipc_lock+ep $(readlink -f $(which python3))`"
            )
        else:
            results["sched_fifo"] = f"FAILED: {os.strerror(err)}"
        # Skip mlockall too — it's only worth it with sched_fifo.
        results["mlockall"] = "SKIPPED: sched_fifo failed, would add jitter without it"
        return RealtimeStatus(applied=False, results=results)

    results["sched_fifo"] = f"applied (prio {sched_fifo_priority})"

    if lock_memory:
        rc = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        if rc != 0:
            err = ctypes.get_errno()
            if err == errno.EPERM:
                results["mlockall"] = (
                    "SKIPPED: EPERM — need CAP_IPC_LOCK "
                    "(see setcap command above)"
                )
            else:
                results["mlockall"] = f"FAILED: {os.strerror(err)}"
            return RealtimeStatus(applied=False, results=results)
        results["mlockall"] = "applied"
    else:
        results["mlockall"] = "disabled by caller"

    return RealtimeStatus(applied=True, results=results)


def release_realtime() -> None:
    """
    Best-effort undo. Restores SCHED_OTHER on the calling thread and
    releases locked memory. Safe to call unconditionally.
    """
    libc = _get_libc()
    if libc is None:
        return
    SCHED_OTHER = 0
    param = _sched_param(0)
    libc.sched_setscheduler(0, SCHED_OTHER, ctypes.byref(param))
    libc.munlockall()
