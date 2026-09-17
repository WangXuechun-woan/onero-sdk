"""Concrete CAN gripper drivers used by Onero A1 arms.

The wire formats in this module are ported from the production H1 arm-control
implementation.  Visual tool selection, payload compensation, TCP selection,
and this hardware protocol remain independent concerns.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .can_bus import CanTransport
from .end_effector import Gripper


@dataclass(frozen=True)
class GripperState:
    """Latest gripper feedback in normalized SDK coordinates."""

    position: float = 0.0
    velocity: float = 0.0
    force: float = 0.0
    fault_code: int = 0
    motion_state: int = 0
    timestamp: float = 0.0
    raw_position: float = 0.0


class CanGripper(Gripper):
    """Base class for grippers attached to an SDK CAN transport.

    A gripper passed to :class:`onero_sdk.Arm` without a bus is bound to that
    arm's transport.  A standalone caller may provide a bus explicitly; the
    bus is only closed by the gripper when ``owns_bus=True``.
    """

    protocol_name = "can"

    def __init__(self, bus: Optional[CanTransport] = None, *,
                 owns_bus: bool = False) -> None:
        if owns_bus and bus is None:
            raise ValueError("owns_bus=True requires a bus")
        self._bus = bus
        self._owns_bus = owns_bus
        self._io_lock: threading.RLock | threading.Lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._state = GripperState()
        self._enabled = False

    def bind_transport(self, bus: CanTransport, io_lock=None) -> None:
        """Bind to an Arm-owned transport before the arm is used."""
        if self._bus is not None and self._bus is not bus:
            raise ValueError("gripper is already bound to a different CAN bus")
        self._bus = bus
        self._owns_bus = False
        if io_lock is not None:
            self._io_lock = io_lock

    @property
    def state(self) -> GripperState:
        with self._state_lock:
            return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _require_bus(self) -> CanTransport:
        if self._bus is None:
            raise RuntimeError(
                "gripper has no CAN transport; pass it to Arm(end_effector=...) "
                "or construct it with bus=..."
            )
        return self._bus

    @staticmethod
    def _normalized(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
        return value

    def open(self) -> None:
        self.set_position(1.0)

    def grasp(self) -> None:
        self.set_position(0.0)

    def shutdown(self) -> None:
        if self._owns_bus and self._bus is not None:
            self._bus.close()
        self._bus = None
        self._enabled = False


class OmniPickerGripper(CanGripper):
    """OmniPicker legacy 8-byte CAN driver.

    Command payload::

        D0 reserved, D1 position, D2 force, D3 velocity,
        D4 acceleration, D5 deceleration, D6-D7 reserved

    Position is ``0x00=closed`` and ``0xFF=open``.  This device has no
    separate enable/disable frame, so lifecycle methods intentionally do not
    move it or remove holding force.
    """

    protocol_name = "omnipicker"

    def __init__(self, bus: Optional[CanTransport] = None, *, can_id: int = 0x08,
                 force: int = 0x20, velocity: int = 0xFF,
                 acceleration: int = 0xFF, deceleration: int = 0xFF,
                 feedback_timeout: float = 0.05,
                 owns_bus: bool = False) -> None:
        super().__init__(bus, owns_bus=owns_bus)
        self.can_id = int(can_id)
        if not 0 <= self.can_id <= 0x7FF:
            raise ValueError("can_id must be an 11-bit CAN identifier")
        self._force_raw = self._byte("force", force)
        self._velocity_raw = self._byte("velocity", velocity)
        self._acceleration_raw = self._byte("acceleration", acceleration)
        self._deceleration_raw = self._byte("deceleration", deceleration)
        self._feedback_timeout = float(feedback_timeout)
        self._commanded_position = 0.0

    @staticmethod
    def _byte(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFF:
            raise ValueError(f"{name} must be an integer in [0, 255]")
        return value

    def enable(self) -> None:
        self._require_bus()
        self._enabled = True

    def disable(self) -> None:
        # The legacy protocol defines no disable command.  Do not move/open
        # the gripper as a side effect of Arm.disable().
        self._enabled = False

    def set_position(self, position: float) -> None:
        if not self._enabled:
            raise RuntimeError("OmniPicker gripper is not enabled")
        position = self._normalized("position", position)
        raw = int(round(position * 0xFF))
        payload = bytes([
            0x00, raw, self._force_raw, self._velocity_raw,
            self._acceleration_raw, self._deceleration_raw, 0x00, 0x00,
        ])
        bus = self._require_bus()
        with self._io_lock:
            bus.send(self.can_id, payload)
            frame = bus.recv_until(self.can_id, timeout=self._feedback_timeout)
        self._commanded_position = position
        if frame is not None:
            self._apply_feedback(frame.data)

    def _apply_feedback(self, data: bytes) -> None:
        if len(data) < 5:
            raise ValueError("OmniPicker feedback must contain at least 5 bytes")
        state = GripperState(
            position=data[2] / 255.0,
            velocity=data[3] / 255.0,
            force=data[4] / 255.0,
            fault_code=int(data[0]),
            motion_state=int(data[1]),
            timestamp=time.monotonic(),
            raw_position=float(data[2]),
        )
        with self._state_lock:
            self._state = state

    def get_position(self) -> float:
        state = self.state
        return state.position if state.timestamp else self._commanded_position

    def set_force(self, force: float) -> None:
        force = self._normalized("force", force)
        self._force_raw = int(round(force * 0xFF))

    def set_speed(self, velocity: float, *, acceleration: Optional[float] = None,
                  deceleration: Optional[float] = None) -> None:
        velocity = self._normalized("velocity", velocity)
        self._velocity_raw = int(round(velocity * 0xFF))
        if acceleration is not None:
            acceleration = self._normalized("acceleration", acceleration)
            self._acceleration_raw = int(round(acceleration * 0xFF))
        if deceleration is not None:
            deceleration = self._normalized("deceleration", deceleration)
            self._deceleration_raw = int(round(deceleration * 0xFF))


class OneroGripper(CanGripper):
    """Self-developed gripper using the DM-compatible MIT CAN protocol."""

    protocol_name = "onero"
    command_id = 0x08
    feedback_id = 0x18
    min_position = -10.0
    max_position = 55.0
    max_velocity = 250.0
    max_torque = 250.0

    _ENABLE = bytes.fromhex("fffffffffffffffc")
    _DISABLE = bytes.fromhex("fffffffffffffffd")

    # H1 hardware endpoint calibration measured from the published 0..255
    # feedback values. 257 converts each 8-bit endpoint exactly to the
    # equivalent 16-bit MIT position code (65535 / 255 = 257).
    _CALIBRATION_CODES = {
        "A1L": (37 * 257, 239 * 257),  # (fully open, fully closed)
        "A1R": (37 * 257, 238 * 257),
        "A1": (37 * 257, 238 * 257),   # legacy alias uses right calibration
    }

    def __init__(self, bus: Optional[CanTransport] = None, *,
                 model: Optional[str] = None,
                 kp: float = 20.0, kd: float = 1.0,
                 feedback_timeout: float = 0.05,
                 owns_bus: bool = False) -> None:
        super().__init__(bus, owns_bus=owns_bus)
        self.kp = float(kp)
        self.kd = float(kd)
        if not math.isfinite(self.kp) or not 0.0 <= self.kp <= 500.0:
            raise ValueError("kp must be finite and in [0, 500]")
        if not math.isfinite(self.kd) or not 0.0 <= self.kd <= 5.0:
            raise ValueError("kd must be finite and in [0, 5]")
        self._feedback_timeout = float(feedback_timeout)
        self._commanded_position = 0.0
        self._model: Optional[str] = None
        self._open_code: Optional[int] = None
        self._closed_code: Optional[int] = None
        if model is not None:
            self.configure_for_model(model)

    def configure_for_model(self, model: str) -> None:
        """Select measured left/right mechanical opening endpoints."""
        name = str(model).upper()
        try:
            open_code, closed_code = self._CALIBRATION_CODES[name]
        except KeyError as exc:
            raise ValueError(
                f"no Onero gripper endpoint calibration for model {model!r}"
            ) from exc
        if self._model is not None and self._model != name:
            raise ValueError(
                f"Onero gripper already calibrated for {self._model}, not {name}"
            )
        self._model = name
        self._open_code = open_code
        self._closed_code = closed_code

    def _require_calibration(self) -> tuple[int, int]:
        if self._open_code is None or self._closed_code is None:
            raise RuntimeError(
                "Onero gripper requires A1L/A1R calibration; pass it to Arm "
                "or construct OneroGripper(model='A1L'|'A1R')"
            )
        return self._open_code, self._closed_code

    @staticmethod
    def _encode_uint(value: float, minimum: float, maximum: float,
                     bits: int) -> int:
        value = min(maximum, max(minimum, value))
        return int((value - minimum) * ((1 << bits) - 1) /
                   (maximum - minimum))

    @staticmethod
    def _decode_uint(value: int, minimum: float, maximum: float,
                     bits: int) -> float:
        return value * (maximum - minimum) / ((1 << bits) - 1) + minimum

    def _transaction(self, payload: bytes):
        bus = self._require_bus()
        with self._io_lock:
            bus.send(self.command_id, payload)
            return bus.recv_until(self.feedback_id,
                                  timeout=self._feedback_timeout)

    def enable(self) -> None:
        frame = self._transaction(self._ENABLE)
        if frame is None:
            raise RuntimeError("Onero gripper did not respond to enable")
        self._apply_feedback(frame.data)
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        try:
            frame = self._transaction(self._DISABLE)
            if frame is not None:
                self._apply_feedback(frame.data)
        finally:
            self._enabled = False

    def set_position(self, position: float) -> None:
        if not self._enabled:
            raise RuntimeError("Onero gripper is not enabled")
        position = self._normalized("position", position)
        # Physical feedback increases toward closed. Use measured mechanical
        # endpoints instead of the wider [-10, 55] protocol codec limits.
        open_code, closed_code = self._require_calibration()
        p = int(round(closed_code + position * (open_code - closed_code)))
        v = self._encode_uint(0.0, -self.max_velocity, self.max_velocity, 12)
        kp = self._encode_uint(self.kp, 0.0, 500.0, 12)
        kd = self._encode_uint(self.kd, 0.0, 5.0, 12)
        tau = self._encode_uint(0.0, -self.max_torque, self.max_torque, 12)
        payload = bytes([
            (p >> 8) & 0xFF, p & 0xFF,
            (v >> 4) & 0xFF,
            ((v & 0xF) << 4) | ((kp >> 8) & 0xF), kp & 0xFF,
            (kd >> 4) & 0xFF,
            ((kd & 0xF) << 4) | ((tau >> 8) & 0xF), tau & 0xFF,
        ])
        frame = self._transaction(payload)
        self._commanded_position = position
        if frame is not None:
            self._apply_feedback(frame.data)

    def _apply_feedback(self, data: bytes) -> None:
        if len(data) < 6:
            raise ValueError("Onero gripper feedback must contain at least 6 bytes")
        status = (data[0] >> 4) & 0xF
        p = (data[1] << 8) | data[2]
        v = (data[3] << 4) | (data[4] >> 4)
        tau = ((data[4] & 0xF) << 8) | data[5]
        q = self._decode_uint(p, self.min_position, self.max_position, 16)
        open_code, closed_code = self._require_calibration()
        normalized = (p - closed_code) / (open_code - closed_code)
        velocity = self._decode_uint(v, -self.max_velocity, self.max_velocity, 12)
        force = self._decode_uint(tau, -self.max_torque, self.max_torque, 12)
        state = GripperState(
            position=min(1.0, max(0.0, normalized)),
            velocity=velocity / self.max_velocity,
            force=force / self.max_torque,
            # DM status 0/1 means disabled/enabled; only 8..14 are faults.
            fault_code=0 if status in (0, 1) else status,
            motion_state=status,
            timestamp=time.monotonic(),
            raw_position=q,
        )
        with self._state_lock:
            self._state = state
        # A fresh process may attach while the motor is already enabled. Keep
        # lifecycle state aligned with authoritative hardware feedback.
        self._enabled = status == 1

    def refresh(self) -> GripperState:
        """Read current Onero feedback without changing its motion target."""
        bus = self._require_bus()
        query = bytes([self.command_id, 0x00, 0xCC, 0, 0, 0, 0, 0])
        with self._io_lock:
            bus.send(0x7FF, query)
            frame = bus.recv_until(self.feedback_id,
                                   timeout=self._feedback_timeout)
        if frame is None:
            raise RuntimeError("Onero gripper did not respond to status query")
        self._apply_feedback(frame.data)
        return self.state

    def get_position(self) -> float:
        state = self.state
        return state.position if state.timestamp else self._commanded_position
