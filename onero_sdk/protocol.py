"""
CAN protocol codec for onero actuators (DM4310/DM4340 compatible).
"""

import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class MotorError(IntEnum):
    DISABLED = 0x0
    ENABLED = 0x1
    OVERVOLTAGE = 0x8
    UNDERVOLTAGE = 0x9
    OVERCURRENT = 0xA
    MOSFET_OVERTEMP = 0xB
    COIL_OVERTEMP = 0xC
    COMM_LOSS = 0xD
    OVERLOAD = 0xE


class RegisterID(IntEnum):
    UV_VALUE = 0
    OT_VALUE = 2
    OC_VALUE = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    CTRL_MODE = 10
    HW_VER = 13
    SW_VER = 14
    SN = 15
    NPP = 16
    RS = 17
    LS = 18
    FLUX = 19
    GR = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_VALUE = 29
    SUB_VER = 36


@dataclass
class MotorLimits:
    p_max: float = 12.5
    v_max: float = 45.0
    t_max: float = 30.0
    kp_max: float = 500.0
    kd_max: float = 5.0


@dataclass
class MotorState:
    motor_id: int = 0
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    t_mos: int = 0
    t_coil: int = 0
    error: int = 0


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    span = x_max - x_min
    offset = x_min
    x = max(x_min, min(x_max, x))
    return int((x - offset) * ((1 << bits) - 1) / span)


def _checked_float_to_uint(name: str, x: float, x_min: float,
                           x_max: float, bits: int) -> int:
    """Encode a motor command without silently saturating it."""
    if not math.isfinite(x):
        raise ValueError(f"MIT {name} must be finite, got {x!r}")
    # Gain/trajectory calculations can produce a value one or two ULPs past
    # an exact codec endpoint (for example 5.000000000000001 for kd_max=5).
    # Treat that as the mathematical endpoint, while still rejecting a real
    # out-of-range command.  Do not use the old broad silent saturation here.
    tolerance = max(1e-12, 1e-12 * max(1.0, abs(x_min), abs(x_max)))
    if x < x_min - tolerance or x > x_max + tolerance:
        raise ValueError(
            f"MIT {name}={x} is outside codec range [{x_min}, {x_max}]"
        )
    x = min(x_max, max(x_min, x))
    return float_to_uint(x, x_min, x_max, bits)


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    offset = x_min
    return float(x_int) * span / ((1 << bits) - 1) + offset


# --- MIT Mode ---

def encode_mit(p_des: float, v_des: float, kp: float, kd: float, t_ff: float,
               limits: MotorLimits) -> bytes:
    for field_name in ("p_max", "v_max", "t_max", "kp_max", "kd_max"):
        value = getattr(limits, field_name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid MotorLimits.{field_name}={value!r}")

    p_int = _checked_float_to_uint(
        "position", p_des, -limits.p_max, limits.p_max, 16)
    v_int = _checked_float_to_uint(
        "velocity", v_des, -limits.v_max, limits.v_max, 12)
    kp_int = _checked_float_to_uint(
        "kp", kp, 0.0, limits.kp_max, 12)
    kd_int = _checked_float_to_uint(
        "kd", kd, 0.0, limits.kd_max, 12)
    t_int = _checked_float_to_uint(
        "torque", t_ff, -limits.t_max, limits.t_max, 12)

    data = bytearray(8)
    data[0] = (p_int >> 8) & 0xFF
    data[1] = p_int & 0xFF
    data[2] = (v_int >> 4) & 0xFF
    data[3] = ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF)
    data[4] = kp_int & 0xFF
    data[5] = (kd_int >> 4) & 0xFF
    data[6] = ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF)
    data[7] = t_int & 0xFF
    return bytes(data)


def decode_mit(data: bytes, limits: MotorLimits) -> tuple[float, float, float, float, float]:
    p_int = (data[0] << 8) | data[1]
    v_int = (data[2] << 4) | (data[3] >> 4)
    kp_int = ((data[3] & 0xF) << 8) | data[4]
    kd_int = (data[5] << 4) | (data[6] >> 4)
    t_int = ((data[6] & 0xF) << 8) | data[7]

    p_des = uint_to_float(p_int, -limits.p_max, limits.p_max, 16)
    # A 12-bit symmetric range has no exact integer midpoint. Drives may use
    # either adjacent centre code for rest, so normalize both to exact zero.
    v_des = (0.0 if v_int in (2047, 2048) else
             uint_to_float(v_int, -limits.v_max, limits.v_max, 12))
    kp = uint_to_float(kp_int, 0.0, limits.kp_max, 12)
    kd = uint_to_float(kd_int, 0.0, limits.kd_max, 12)
    t_ff = uint_to_float(t_int, -limits.t_max, limits.t_max, 12)
    return p_des, v_des, kp, kd, t_ff


# --- Feedback Frame ---

def decode_feedback(data: bytes, limits: MotorLimits) -> MotorState:
    error = (data[0] >> 4) & 0xF
    motor_id = data[0] & 0xF
    pos_int = (data[1] << 8) | data[2]
    vel_int = (data[3] << 4) | ((data[4] >> 4) & 0xF)
    torque_int = ((data[4] & 0xF) << 8) | data[5]
    t_mos = struct.unpack('b', bytes([data[6]]))[0]
    t_coil = struct.unpack('b', bytes([data[7]]))[0]

    position = uint_to_float(pos_int, -limits.p_max, limits.p_max, 16)
    velocity = (0.0 if vel_int in (2047, 2048) else
                uint_to_float(vel_int, -limits.v_max, limits.v_max, 12))
    torque = uint_to_float(torque_int, -limits.t_max, limits.t_max, 12)

    return MotorState(
        motor_id=motor_id,
        position=position,
        velocity=velocity,
        torque=torque,
        t_mos=t_mos,
        t_coil=t_coil,
        error=error,
    )


# --- Position Velocity Mode ---
# CAN ID = CANId + 0x100, DLC = 8

def encode_pos_vel(p_des: float, v_des: float) -> bytes:
    return struct.pack('<ff', p_des, v_des)


def decode_pos_vel(data: bytes) -> tuple[float, float]:
    return struct.unpack('<ff', data[:8])


# --- Velocity Mode ---
# CAN ID = CANId + 0x200, DLC = 4

def encode_velocity(v_des: float) -> bytes:
    return struct.pack('<f', v_des)


def decode_velocity(data: bytes) -> float:
    return struct.unpack('<f', data[:4])[0]


# --- Special Frames ---

ENABLE_DATA = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
DISABLE_DATA = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
SET_ZERO_DATA = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])
CLEAR_ERROR_DATA = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB])


# --- Parameter Read/Write ---
# All via CAN ID 0x7FF

PARAM_FRAME_ID = 0x7FF


def encode_read_param(can_id: int, reg_id: int) -> bytes:
    can_id_l = can_id & 0xFF
    can_id_h = (can_id >> 8) & 0xFF
    data = bytearray(8)
    data[0] = can_id_l
    data[1] = can_id_h
    data[2] = 0x33
    data[3] = reg_id & 0xFF
    return bytes(data)


def encode_write_param(can_id: int, reg_id: int, value) -> bytes:
    can_id_l = can_id & 0xFF
    can_id_h = (can_id >> 8) & 0xFF
    data = bytearray(8)
    data[0] = can_id_l
    data[1] = can_id_h
    data[2] = 0x55
    data[3] = reg_id & 0xFF
    if isinstance(value, float):
        struct.pack_into('<f', data, 4, value)
    else:
        struct.pack_into('<I', data, 4, int(value))
    return bytes(data)


def decode_param_response(data: bytes, as_float: bool = True):
    if as_float:
        return struct.unpack('<f', data[4:8])[0]
    return struct.unpack('<I', data[4:8])[0]


def encode_save_params(can_id: int) -> bytes:
    can_id_l = can_id & 0xFF
    can_id_h = (can_id >> 8) & 0xFF
    return bytes([can_id_l, can_id_h, 0xAA, 0x01])


def encode_reset_params(can_id: int) -> bytes:
    can_id_l = can_id & 0xFF
    can_id_h = (can_id >> 8) & 0xFF
    return bytes([can_id_l, can_id_h, 0xAF, 0x01])


# --- ID Operations ---

GET_ID_DATA = bytes([0x55])


def encode_set_id(new_can_id: int, new_mst_id: int) -> bytes:
    # DLC=4: [0xAA, CANId(1B), MstId(1B), 0x55]
    return bytes([0xAA, new_can_id & 0xFF, new_mst_id & 0xFF, 0x55])


def decode_get_id(data: bytes) -> tuple[int, int]:
    # Response DLC=4: [0x55, CANId(1B), MstId(1B), 0xAA]
    can_id = data[1]
    mst_id = data[2]
    return can_id, mst_id


# --- Get Motor Feedback (request) ---

def encode_get_feedback(can_id: int) -> bytes:
    can_id_l = can_id & 0xFF
    can_id_h = (can_id >> 8) & 0xFF
    return bytes([can_id_l, can_id_h, 0xCC, 0x00])


# --- CAN ID helpers ---

def mit_frame_id(can_id: int) -> int:
    return can_id


def pos_vel_frame_id(can_id: int) -> int:
    return can_id + 0x100


def vel_frame_id(can_id: int) -> int:
    return can_id + 0x200
