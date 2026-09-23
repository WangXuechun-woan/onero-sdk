"""Optional python-can transport for SocketCAN and direct SLCAN devices."""

from __future__ import annotations

import time
from typing import Optional

from .can_bus import (
    CAN_EFF_FLAG,
    CAN_EFF_MASK,
    CAN_SFF_MASK,
    CanFrame,
)


def _create_buffered_slcan_bus(can_module, options):
    """Use the same configuration resolution as can.Bus before specialization."""
    from ._buffered_slcan import BufferedSlcanBus

    @can_module.util.deprecated_args_alias(
        deprecation_start="4.2.0", deprecation_end="5.0.0",
        bustype="interface", context="config_context",
    )
    def create(channel=None, interface=None, config_context=None,
               ignore_config=False, **bus_options):
        if interface is not None:
            bus_options["interface"] = interface
        if channel is not None:
            bus_options["channel"] = channel
        if not ignore_config:
            bus_options = can_module.util.load_config(
                config=bus_options, context=config_context,
            )
        bus_options.pop("interface")
        channel = bus_options.pop("channel", channel)
        if channel is None:
            return BufferedSlcanBus(**bus_options)
        return BufferedSlcanBus(channel, **bus_options)

    return create(**options)


class PythonCanBus:
    """Adapt a :mod:`python-can` Bus to the SDK CAN transport interface.

    ``python-can`` is imported lazily so the core SDK keeps no mandatory
    dependency on it. The transport owns the created Bus and shuts it down in
    :meth:`close`.

    ``buffered_slcan=True`` uses bulk serial reads and arm-command writes
    with python-can's existing SLCAN receive parser. It is opt-in and leaves other python-can interfaces unchanged.
    """

    def __init__(self, channel: str, interface: str = "slcan",
                 bitrate: int = 1_000_000,
                 tty_baudrate: Optional[int] = None,
                 buffered_slcan: bool = False, **kwargs):
        if not isinstance(buffered_slcan, bool):
            raise ValueError("buffered_slcan must be a bool")
        if buffered_slcan and interface != "slcan":
            raise ValueError("buffered_slcan is only supported for interface='slcan'")
        try:
            import can
        except ImportError as exc:
            raise ImportError(
                "PythonCanBus requires python-can; install "
                "onero-sdk[python-can] or `pip install python-can`"
            ) from exc

        options = dict(kwargs)
        options.update(
            interface=interface,
            channel=channel,
            bitrate=bitrate,
        )
        if tty_baudrate is not None:
            options["tty_baudrate"] = tty_baudrate

        self._can = can
        if buffered_slcan:
            self._bus = _create_buffered_slcan_bus(can, options)
        else:
            self._bus = can.Bus(**options)
        self._channel = channel
        self._interface = interface
        self._closed = False

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def interface(self) -> str:
        return self._interface

    def _make_message(self, can_id: int, data: bytes):
        payload = bytes(data)
        if len(payload) > 8:
            raise ValueError("Classic CAN payload cannot exceed 8 bytes")

        is_extended = bool(can_id & CAN_EFF_FLAG) or can_id > CAN_SFF_MASK
        arbitration_id = (
            can_id & CAN_EFF_MASK if is_extended else can_id & CAN_SFF_MASK
        )
        return self._can.Message(
            arbitration_id=arbitration_id,
            data=payload,
            is_extended_id=is_extended,
        )

    def send(self, can_id: int, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("PythonCanBus is closed")
        self._bus.send(self._make_message(can_id, data))

    def send_batch(self, frames) -> None:
        """Validate all frames first; use bulk writes when the bus supports it."""
        if self._closed:
            raise RuntimeError("PythonCanBus is closed")
        messages = [self._make_message(can_id, data) for can_id, data in frames]
        batch_send = getattr(self._bus, "send_batch", None)
        if callable(batch_send):
            batch_send(messages)
        else:
            for message in messages:
                self._bus.send(message)

    def recv(self, timeout: float = 0.01) -> Optional[CanFrame]:
        if self._closed:
            return None
        message = self._bus.recv(timeout=max(0.0, timeout))
        if message is None:
            return None
        can_id = int(message.arbitration_id)
        if getattr(message, "is_extended_id", False):
            can_id |= CAN_EFF_FLAG
        data = bytes(message.data)
        return CanFrame(can_id=can_id, data=data, dlc=int(message.dlc))

    def recv_until(self, target_id: int,
                   timeout: float = 0.05) -> Optional[CanFrame]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            frame = self.recv(timeout=remaining)
            if frame is None:
                return None
            if frame.arbitration_id == target_id:
                return frame

    def recv_batch(self, target_ids: set[int],
                   timeout: float = 0.05) -> dict[int, CanFrame]:
        deadline = time.monotonic() + timeout
        remaining_ids = set(target_ids)
        collected: dict[int, CanFrame] = {}
        while remaining_ids:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            frame = self.recv(timeout=remaining)
            if frame is None:
                break
            arbitration_id = frame.arbitration_id
            if arbitration_id in remaining_ids:
                collected[arbitration_id] = frame
                remaining_ids.remove(arbitration_id)
        return collected

    def flush_rx(self) -> None:
        while self.recv(timeout=0.0) is not None:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
