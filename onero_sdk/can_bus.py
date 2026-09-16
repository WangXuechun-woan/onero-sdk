"""
SocketCAN transport layer using raw Linux sockets.

This module only handles communication over an already-active CAN interface.
For CAN interface setup (slcand, udev), see configs/ directory.
"""

import socket
import struct
import select
import time
import errno
from dataclasses import dataclass
from typing import Optional, Protocol

from .exceptions import OneroConnectionError


CAN_RAW = 1
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
CAN_EFF_MASK = 0x1FFFFFFF

CAN_FRAME_FMT = '=IB3x8s'
_FRAME_STRUCT = struct.Struct(CAN_FRAME_FMT)
CAN_FRAME_SIZE = _FRAME_STRUCT.size

# Tight recv poll interval when the socket has no data ready. Empirically
# 10 µs balances CPU load against latency: shorter values busy-spin without
# measurable gain; longer values add p99/max jitter.
_RECV_POLL_INTERVAL = 1e-5

# Send-side backoff when the CAN qdisc queue is full (ENOBUFS). Typical
# `pfifo_fast qlen 10` fills when a high-priority thread bursts frames faster
# than the driver drains at 1 Mbps (~130 µs / frame). Small sleep gives the
# driver time to advance the queue.
_SEND_BACKOFF_S = 100e-6
_SEND_MAX_RETRIES = 3


@dataclass
class CanFrame:
    can_id: int
    data: bytes
    dlc: int

    @property
    def is_extended(self) -> bool:
        return bool(self.can_id & CAN_EFF_FLAG)

    @property
    def arbitration_id(self) -> int:
        if self.is_extended:
            return self.can_id & CAN_EFF_MASK
        return self.can_id & CAN_SFF_MASK


class CanTransport(Protocol):
    """Structural interface accepted by CanBackend."""

    def send(self, can_id: int, data: bytes) -> None: ...
    def recv(self, timeout: float = 0.01) -> Optional[CanFrame]: ...
    def recv_until(self, target_id: int,
                   timeout: float = 0.05) -> Optional[CanFrame]: ...
    def recv_batch(self, target_ids: set[int],
                   timeout: float = 0.05) -> dict[int, CanFrame]: ...
    def flush_rx(self) -> None: ...
    def close(self) -> None: ...


class CanBus:
    def __init__(self, channel: str = "can0"):
        """
        Connect to an existing SocketCAN interface.

        Args:
            channel: SocketCAN interface name (e.g. "can0", "vcan0")

        Raises:
            OneroConnectionError: if the interface doesn't exist or isn't UP.
        """
        self._channel = channel
        self._closed = False

        self._sock = None
        try:
            self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, CAN_RAW)
            self._sock.bind((channel,))
            self._sock.setblocking(False)
        except OSError as e:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
            raise OneroConnectionError(
                channel,
                f"Interface not available. Ensure it is UP: "
                f"sudo ip link set up {channel}. "
                f"Run 'python -m onero_sdk.diag {channel}' for diagnostics."
            ) from e

        # Drain any pre-existing frames on the interface. Without this, an
        # earlier session's stale motor responses can survive into the new
        # socket's kernel rx queue and confuse the first recv_batch.
        self.flush_rx()

    @property
    def channel(self) -> str:
        return self._channel

    def send(self, can_id: int, data: bytes) -> None:
        dlc = len(data)
        data_padded = data.ljust(8, b'\x00')
        frame = _FRAME_STRUCT.pack(can_id, dlc, data_padded)
        sock = self._sock
        for attempt in range(_SEND_MAX_RETRIES):
            try:
                sock.send(frame)
                return
            except OSError as e:
                # ENOBUFS: qdisc tx queue full (default pfifo_fast qlen=10).
                # A brief backoff lets the driver drain a slot. Rare in
                # normal operation; observed under SCHED_FIFO when Python
                # bursts sends faster than the 1 Mbps wire.
                if e.errno == errno.ENOBUFS and attempt + 1 < _SEND_MAX_RETRIES:
                    time.sleep(_SEND_BACKOFF_S)
                    continue
                raise

    def recv(self, timeout: float = 0.01) -> Optional[CanFrame]:
        ready = select.select([self._sock], [], [], timeout)
        if not ready[0]:
            return None
        raw = self._sock.recv(CAN_FRAME_SIZE)
        can_id, dlc, data = _FRAME_STRUCT.unpack(raw)
        return CanFrame(can_id=can_id, data=data[:dlc], dlc=dlc)

    def recv_until(self, target_id: int, timeout: float = 0.05) -> Optional[CanFrame]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = self.recv(timeout=remaining)
            if frame is None:
                return None
            if frame.arbitration_id == target_id:
                return frame

    def recv_batch(self, target_ids: set[int], timeout: float = 0.05) -> dict[int, CanFrame]:
        """
        Collect response frames for multiple expected IDs.

        Reads frames from the bus until all target IDs are received or timeout.
        Ignores frames with unexpected IDs (stale frames from previous cycles).

        Uses a tight non-blocking recv loop rather than select() per frame —
        avoids one syscall per frame in the hot path. Benchmarked (bench_can.py):
        p99 −3.4% single-arm idle, −7.5% dual-arm parallel vs the previous
        select-per-frame implementation. See docs/bench_can_report.md.
        """
        sock = self._sock
        deadline = time.perf_counter() + timeout
        collected: dict[int, CanFrame] = {}
        remaining_ids = set(target_ids)
        unpack = _FRAME_STRUCT.unpack

        while remaining_ids and time.perf_counter() < deadline:
            try:
                raw = sock.recv(CAN_FRAME_SIZE, socket.MSG_DONTWAIT)
            except BlockingIOError:
                time.sleep(_RECV_POLL_INTERVAL)
                continue
            can_id, dlc, data = unpack(raw)
            arb_id = (can_id & CAN_EFF_MASK) if (can_id & CAN_EFF_FLAG) else (can_id & CAN_SFF_MASK)
            if arb_id in remaining_ids:
                collected[arb_id] = CanFrame(can_id=can_id, data=data[:dlc], dlc=dlc)
                remaining_ids.discard(arb_id)

        return collected

    def flush_rx(self) -> None:
        """Drain any pending frames on the rx queue non-blockingly."""
        sock = self._sock
        while True:
            try:
                sock.recv(CAN_FRAME_SIZE, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

