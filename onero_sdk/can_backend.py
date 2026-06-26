"""
CAN backend — real hardware implementation of ArmBackend.

Uses a parallel send-then-receive strategy: all command frames are sent
back-to-back before collecting responses. This minimizes bus idle time
and achieves higher control frequencies than sequential send-wait.
"""

import time
import logging
import threading
from typing import Optional

from .backend import ArmBackend, ArmState
from .can_bus import CanBus
from .motor import Motor
from .protocol import (
    MotorLimits, MotorState, MotorError,
    encode_mit, encode_pos_vel, encode_velocity, decode_feedback,
    mit_frame_id, pos_vel_frame_id, vel_frame_id,
    ENABLE_DATA, DISABLE_DATA, SET_ZERO_DATA, CLEAR_ERROR_DATA,
)
from .model_registry import JointSpec
from .exceptions import OneroConnectionError, OneroHardwareError

logger = logging.getLogger(__name__)

_ENABLE_RETRY_COUNT = 2
_ENABLE_CLEAR_DELAY = 0.1
_RECV_TIMEOUT = 0.01


class CanBackend(ArmBackend):
    """
    Real hardware backend using SocketCAN + parallel CAN I/O.

    Control cycle strategy (parallel):
      1. Send all command frames back-to-back (no waiting between sends)
      2. Collect all response frames with recv_batch (matched by response_id)
      3. For any motor that didn't respond: use last known state + increment miss counter

    This reduces per-cycle bus time from 7×RTT to 7×send + 1×batch_recv.
    """

    def __init__(self, channel: str, joint_specs: list[JointSpec],
                 recv_timeout: float = _RECV_TIMEOUT,
                 on_miss: Optional[callable] = None,
                 on_fault: Optional[callable] = None):
        """
        Args:
            channel: SocketCAN interface name (e.g. "can0"). Must be UP.
            joint_specs: Joint specifications from model registry.
            recv_timeout: Timeout for batch response collection (seconds).
            on_miss: Callback(joint_index, consecutive_misses) on frame miss.
            on_fault: Callback(joint_index, error_code) on motor hardware fault.
        """
        self._channel = channel
        self._joint_specs = joint_specs
        self._num_joints = len(joint_specs)
        self._recv_timeout = recv_timeout
        self._on_miss = on_miss
        self._on_fault = on_fault
        self._lock = threading.Lock()

        try:
            self._bus = CanBus(channel)
        except Exception as e:
            raise OneroConnectionError(channel, str(e))

        self._motors: list[Motor] = []
        self._response_ids: list[int] = []
        for spec in joint_specs:
            limits = MotorLimits(p_max=spec.p_max, v_max=spec.v_max, t_max=spec.t_max)
            motor = Motor(self._bus, spec.can_id, spec.response_id, limits)
            self._motors.append(motor)
            self._response_ids.append(spec.response_id)

        self._enabled = False

        # Last known state per motor (fallback on missed frames)
        self._last_states: list[MotorState] = [MotorState() for _ in range(self._num_joints)]
        # Consecutive miss counter per motor
        self._miss_count: list[int] = [0] * self._num_joints

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def enable(self) -> ArmState:
        with self._lock:
            return self._enable_impl()

    def _enable_impl(self) -> ArmState:
        errors = []

        for attempt in range(_ENABLE_RETRY_COUNT):
            errors.clear()

            # Clear any residual errors (sequential — only at enable time)
            for motor in self._motors:
                motor.clear_error()
            time.sleep(_ENABLE_CLEAR_DELAY)

            # Enable all motors (sequential — not performance critical)
            for i, motor in enumerate(self._motors):
                state = motor.enable()
                if state is None:
                    errors.append((i, "no response"))
                elif state.error not in (MotorError.DISABLED, MotorError.ENABLED):
                    errors.append((i, f"error 0x{state.error:X}"))

            if not errors:
                break

            if attempt < _ENABLE_RETRY_COUNT - 1:
                logger.warning(f"Enable attempt {attempt + 1} failed, retrying...")
                time.sleep(_ENABLE_CLEAR_DELAY)

        if errors:
            for motor in self._motors:
                try:
                    motor.disable()
                except Exception:
                    pass

            detail = "; ".join(f"joint {i}: {msg}" for i, msg in errors)
            raise OneroHardwareError(
                joint_index=errors[0][0],
                error_code=0,
                description=f"Failed to enable arm: {detail}"
            )

        self._enabled = True
        # Flush any stale frames in the buffer
        self._bus.flush_rx()
        return self._get_state_impl()

    def disable(self) -> None:
        with self._lock:
            for motor in self._motors:
                try:
                    motor.disable()
                except Exception:
                    pass
            self._enabled = False

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        with self._lock:
            ts = time.monotonic()

            for i, motor in enumerate(self._motors):
                data = encode_mit(
                    positions[i], velocities[i],
                    kps[i], kds[i], torques[i],
                    motor.limits,
                )
                self._bus.send(mit_frame_id(motor.can_id), data)

            return self._collect_responses(ts)

    def command_velocity(self, velocities: list[float]) -> ArmState:
        with self._lock:
            ts = time.monotonic()

            for i, motor in enumerate(self._motors):
                data = encode_velocity(velocities[i])
                self._bus.send(vel_frame_id(motor.can_id), data)

            return self._collect_responses(ts)

    def command_torque(self, torques: list[float]) -> ArmState:
        with self._lock:
            ts = time.monotonic()

            for i, motor in enumerate(self._motors):
                data = encode_mit(
                    0.0, 0.0, 0.0, 0.0, torques[i],
                    motor.limits,
                )
                self._bus.send(mit_frame_id(motor.can_id), data)

            return self._collect_responses(ts)

    def get_state(self) -> ArmState:
        """Actively query state (uses sequential mode for reliability)."""
        with self._lock:
            return self._get_state_impl()

    def _get_state_impl(self) -> ArmState:
        joints = []
        vels = []
        torqs = []
        ts = time.monotonic()

        for i, motor in enumerate(self._motors):
            state = motor.get_feedback()
            if state is not None:
                self._last_states[i] = state
            joints.append(self._last_states[i].position)
            vels.append(self._last_states[i].velocity)
            torqs.append(self._last_states[i].torque)

        return ArmState(joints=joints, velocities=vels, torques=torqs, timestamp=ts)

    def close(self) -> None:
        with self._lock:
            if self._enabled:
                for motor in self._motors:
                    try:
                        motor.disable()
                    except Exception:
                        pass
                self._enabled = False
            self._bus.close()

    # ─── Private ─────────────────────────────────────────────────────────────

    def _collect_responses(self, timestamp: float) -> ArmState:
        """
        Collect response frames for all motors after a batch send.

        Matches frames by response_id. On timeout, uses last known state
        and increments the miss counter for that motor. Logs a warning
        after 3 consecutive misses.
        """
        target_ids = set(self._response_ids)
        received = self._bus.recv_batch(target_ids, timeout=self._recv_timeout)

        joints = []
        vels = []
        torqs = []

        for i, motor in enumerate(self._motors):
            resp_id = self._response_ids[i]
            frame = received.get(resp_id)

            if frame is not None:
                state = decode_feedback(frame.data, motor.limits)
                self._last_states[i] = state
                self._miss_count[i] = 0
                # Check for hardware faults in motor feedback
                if state.error not in (MotorError.DISABLED, MotorError.ENABLED, 0):
                    if self._on_fault:
                        self._on_fault(i, state.error)
            else:
                # Missed frame — use last known state
                self._miss_count[i] += 1
                if self._miss_count[i] == 3:
                    logger.warning(f"Joint {i}: 3 consecutive missed frames")
                elif self._miss_count[i] >= 10:
                    logger.error(f"Joint {i}: {self._miss_count[i]} consecutive misses — possible disconnect")
                if self._on_miss:
                    self._on_miss(i, self._miss_count[i])

            joints.append(self._last_states[i].position)
            vels.append(self._last_states[i].velocity)
            torqs.append(self._last_states[i].torque)

        return ArmState(joints=joints, velocities=vels, torques=torqs, timestamp=timestamp)
