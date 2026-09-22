"""
CAN backend — real hardware implementation of ArmBackend.

Uses a parallel send-then-receive strategy: all command frames are sent
back-to-back before collecting responses. This minimizes bus idle time
and achieves higher control frequencies than sequential send-wait.
"""

import logging
import threading
import time
from typing import Optional

from .backend import ArmBackend, ArmState
from .can_bus import CanBus, CanTransport
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

    def __init__(self, channel: Optional[str], joint_specs: list[JointSpec],
                 recv_timeout: float = _RECV_TIMEOUT,
                 on_miss: Optional[callable] = None,
                 on_fault: Optional[callable] = None,
                 bus: Optional[CanTransport] = None,
                 on_response: Optional[callable] = None):
        """
        Args:
            channel: SocketCAN interface name (e.g. "can0"). Must be UP when
                     ``bus`` is omitted. May be None for an injected transport.
            joint_specs: Joint specifications from model registry.
            recv_timeout: Timeout for batch response collection (seconds).
            on_miss: Callback(joint_index, consecutive_misses) on frame miss.
            on_fault: Callback(joint_index, error_code) on motor hardware fault.
            bus: Optional transport implementing the CanBus interface. Ownership
                 transfers to this backend and it is closed by close().
            on_response: Callback(joint_index, received) for every joint in a
                         command response batch, including successful responses.
        """
        if bus is None and channel is None:
            raise ValueError("channel is required when bus is not provided")
        self._channel = channel or getattr(bus, "channel", None) or type(bus).__name__
        self._joint_specs = joint_specs
        self._num_joints = len(joint_specs)
        self._recv_timeout = recv_timeout
        self._on_miss = on_miss
        self._on_fault = on_fault
        self._on_response = on_response
        # A safety callback can synchronously disable this backend while a
        # command holds the transport lock. Preserve I/O serialization while
        # allowing that same-thread emergency-stop path to re-enter disable().
        self._lock = threading.RLock()

        if bus is not None:
            self._bus = bus
        else:
            try:
                self._bus = CanBus(channel)
            except Exception as e:
                raise OneroConnectionError(channel, str(e))

        self._motors: list[Motor] = []
        self._response_ids: list[int] = []
        for spec in joint_specs:
            # MIT codec ranges are motor-protocol constants.  Keep them
            # separate from URDF joint velocity/effort safety limits.
            limits = MotorLimits(
                p_max=spec.encoding_p_max,
                v_max=spec.codec_v_max,
                t_max=spec.codec_t_max,
            )
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

    def bind_can_device(self, device) -> None:
        """Bind an auxiliary CAN device to this backend's bus and I/O lock.

        End effectors share the Arm-owned transport so direct SLCAN devices
        are opened only once and arm/gripper receive operations cannot race.
        """
        bind = getattr(device, "bind_transport", None)
        if bind is None:
            raise TypeError("CAN device must provide bind_transport(bus, io_lock)")
        bind(self._bus, self._lock)

    def enable(self) -> ArmState:
        with self._lock:
            return self._enable_impl()

    def _enable_impl(self) -> ArmState:
        # An already-enabled arm may be holding a gravity-loaded pose.  Do
        # not run the normal clear-error/sequential-enable sequence in that
        # state: clear_error can remove the motor's holding torque long enough
        # for the arm to sag.  Reuse only an all-enabled state; mixed states
        # are rejected rather than silently completing a hazardous transition.
        startup_states = []
        for index, motor in enumerate(self._motors):
            state = motor.get_feedback(timeout=0.05)
            if state is None:
                raise OneroHardwareError(
                    joint_index=index,
                    error_code=0,
                    description="Cannot verify motor state before enable",
                )
            if state.error not in (MotorError.DISABLED, MotorError.ENABLED):
                raise OneroHardwareError(
                    joint_index=index,
                    error_code=int(state.error),
                    description=(
                        "Motor fault before enable: "
                        f"error=0x{int(state.error):X}"
                    ),
                )
            startup_states.append(state)

        startup_errors = {int(state.error) for state in startup_states}
        if startup_errors == {int(MotorError.ENABLED)}:
            self._enabled = True
            self._bus.flush_rx()
            logger.info(
                "All motors are already enabled; skipped clear-error/re-enable "
                "sequence to preserve holding torque"
            )
            return self._get_state_impl()
        if startup_errors != {int(MotorError.DISABLED)}:
            detail = ", ".join(
                f"J{i + 1}=0x{int(state.error):X}"
                for i, state in enumerate(startup_states)
            )
            raise OneroHardwareError(
                joint_index=next(
                    i for i, state in enumerate(startup_states)
                    if state.error == MotorError.ENABLED
                ),
                error_code=0,
                description=f"Mixed motor states before enable: {detail}",
            )

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

            # Encode every joint before the first bus write so an invalid
            # value cannot result in a partially sent arm command.
            frames = [
                (mit_frame_id(motor.can_id), encode_mit(
                    positions[i], velocities[i],
                    kps[i], kds[i], torques[i],
                    motor.limits,
                ))
                for i, motor in enumerate(self._motors)
            ]
            for frame_id, data in frames:
                self._bus.send(frame_id, data)

            return self._collect_responses(ts)

    def command_velocity(self, velocities: list[float]) -> ArmState:
        with self._lock:
            ts = time.monotonic()

            if len(velocities) != self._num_joints:
                raise ValueError(f"expected {self._num_joints} velocities")
            frames = [
                (vel_frame_id(motor.can_id), encode_velocity(velocities[i]))
                for i, motor in enumerate(self._motors)
            ]
            for frame_id, data in frames:
                self._bus.send(frame_id, data)

            return self._collect_responses(ts)

    def command_torque(self, torques: list[float]) -> ArmState:
        with self._lock:
            ts = time.monotonic()

            frames = [
                (mit_frame_id(motor.can_id), encode_mit(
                    0.0, 0.0, 0.0, 0.0, torques[i],
                    motor.limits,
                ))
                for i, motor in enumerate(self._motors)
            ]
            for frame_id, data in frames:
                self._bus.send(frame_id, data)

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

    def close(self, disable: bool = False) -> None:
        with self._lock:
            if disable and self._enabled:
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

            if self._on_response:
                self._on_response(i, frame is not None)

            joints.append(self._last_states[i].position)
            vels.append(self._last_states[i].velocity)
            torqs.append(self._last_states[i].torque)

        return ArmState(joints=joints, velocities=vels, torques=torqs, timestamp=timestamp)
