"""
Single motor controller — wraps protocol encoding and CAN I/O.
"""

from typing import Optional

from .can_bus import CanBus
from .protocol import (
    MotorLimits, MotorState,
    encode_mit, decode_feedback,
    encode_pos_vel, encode_velocity,
    encode_read_param, encode_write_param, decode_param_response,
    encode_save_params, encode_reset_params, encode_get_feedback,
    ENABLE_DATA, DISABLE_DATA, SET_ZERO_DATA, CLEAR_ERROR_DATA,
    PARAM_FRAME_ID,
    mit_frame_id, pos_vel_frame_id, vel_frame_id,
)


class Motor:
    def __init__(self, can_bus: CanBus, can_id: int, mst_id: int,
                 limits: Optional[MotorLimits] = None):
        self._bus = can_bus
        self._can_id = can_id
        self._mst_id = mst_id
        self._limits = limits or MotorLimits()
        self._state = MotorState()

    @property
    def can_id(self) -> int:
        return self._can_id

    @property
    def mst_id(self) -> int:
        return self._mst_id

    @property
    def limits(self) -> MotorLimits:
        return self._limits

    @property
    def state(self) -> MotorState:
        return self._state

    def _send_and_recv(self, frame_id: int, data: bytes,
                       timeout: float = 0.05) -> Optional[MotorState]:
        self._bus.send(frame_id, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        if frame is not None:
            self._state = decode_feedback(frame.data, self._limits)
            return self._state
        return None

    def enable(self, timeout: float = 0.05) -> Optional[MotorState]:
        return self._send_and_recv(mit_frame_id(self._can_id), ENABLE_DATA, timeout)

    def disable(self, timeout: float = 0.05) -> MotorState:
        return self._send_and_recv(mit_frame_id(self._can_id), DISABLE_DATA, timeout)

    def set_zero(self, timeout: float = 0.05) -> MotorState:
        return self._send_and_recv(mit_frame_id(self._can_id), SET_ZERO_DATA, timeout)

    def clear_error(self, timeout: float = 0.05) -> MotorState:
        return self._send_and_recv(mit_frame_id(self._can_id), CLEAR_ERROR_DATA, timeout)

    def set_mit(self, position: float = 0.0, velocity: float = 0.0,
                kp: float = 0.0, kd: float = 0.0, torque_ff: float = 0.0,
                timeout: float = 0.05) -> MotorState:
        data = encode_mit(position, velocity, kp, kd, torque_ff, self._limits)
        return self._send_and_recv(mit_frame_id(self._can_id), data, timeout)

    def set_position_velocity(self, position: float, velocity: float,
                              timeout: float = 0.05) -> MotorState:
        data = encode_pos_vel(position, velocity)
        return self._send_and_recv(pos_vel_frame_id(self._can_id), data, timeout)

    def set_velocity(self, velocity: float, timeout: float = 0.05) -> MotorState:
        data = encode_velocity(velocity)
        return self._send_and_recv(vel_frame_id(self._can_id), data, timeout)

    def get_feedback(self, timeout: float = 0.05) -> Optional[MotorState]:
        data = encode_get_feedback(self._can_id)
        self._bus.send(PARAM_FRAME_ID, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        if frame is not None:
            self._state = decode_feedback(frame.data, self._limits)
            return self._state
        return None

    def read_param(self, reg_id: int, as_float: bool = True, timeout: float = 0.1):
        data = encode_read_param(self._can_id, reg_id)
        self._bus.send(PARAM_FRAME_ID, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        if frame is None:
            return None
        return decode_param_response(frame.data, as_float=as_float)

    def write_param(self, reg_id: int, value, timeout: float = 0.1) -> bool:
        data = encode_write_param(self._can_id, reg_id, value)
        self._bus.send(PARAM_FRAME_ID, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        return frame is not None

    def save_params(self, timeout: float = 0.2) -> bool:
        data = encode_save_params(self._can_id)
        self._bus.send(PARAM_FRAME_ID, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        return frame is not None

    def reset_params(self, timeout: float = 0.2) -> bool:
        data = encode_reset_params(self._can_id)
        self._bus.send(PARAM_FRAME_ID, data)
        frame = self._bus.recv_until(self._mst_id, timeout=timeout)
        return frame is not None
