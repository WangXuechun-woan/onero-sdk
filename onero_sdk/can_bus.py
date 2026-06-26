"""
SocketCAN transport layer using raw Linux sockets.

This module only handles communication over an already-active CAN interface.
For CAN interface setup (slcand, udev), see configs/ directory.
"""

import socket
import struct
import select
import time
from dataclasses import dataclass
from typing import Optional

from .exceptions import OneroConnectionError


CAN_RAW = 1
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
CAN_EFF_MASK = 0x1FFFFFFF

CAN_FRAME_FMT = '=IB3x8s'
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)


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

    @property
    def channel(self) -> str:
        return self._channel

    def send(self, can_id: int, data: bytes) -> None:
        dlc = len(data)
        data_padded = data.ljust(8, b'\x00')
        frame = struct.pack(CAN_FRAME_FMT, can_id, dlc, data_padded)
        self._sock.send(frame)

    def recv(self, timeout: float = 0.01) -> Optional[CanFrame]:
        ready = select.select([self._sock], [], [], timeout)
        if not ready[0]:
            return None
        raw = self._sock.recv(CAN_FRAME_SIZE)
        can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
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

        Args:
            target_ids: Set of expected response CAN IDs.
            timeout: Maximum time to wait for all responses.

        Returns:
            Dict mapping received CAN ID → frame. Missing IDs are absent.
        """
        deadline = time.monotonic() + timeout
        collected: dict[int, CanFrame] = {}
        remaining_ids = set(target_ids)

        while remaining_ids:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                break
            frame = self.recv(timeout=remaining_time)
            if frame is None:
                break
            arb_id = frame.arbitration_id
            if arb_id in remaining_ids:
                collected[arb_id] = frame
                remaining_ids.discard(arb_id)

        return collected

    def flush_rx(self) -> None:
        while self.recv(timeout=0.001) is not None:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
