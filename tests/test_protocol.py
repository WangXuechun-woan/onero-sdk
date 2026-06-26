"""
Unit tests for protocol encoding/decoding.
"""

import struct
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onero_sdk.protocol import (  # noqa: E402
    float_to_uint, uint_to_float,
    encode_mit, decode_mit,
    decode_feedback, encode_pos_vel, decode_pos_vel,
    encode_velocity, decode_velocity,
    encode_read_param, encode_write_param, decode_param_response,
    ENABLE_DATA, DISABLE_DATA, SET_ZERO_DATA, CLEAR_ERROR_DATA,
    MotorLimits,
)


def test_float_uint_roundtrip():
    limits = MotorLimits(p_max=12.5)
    for val in [-12.5, -5.0, 0.0, 5.0, 12.5]:
        encoded = float_to_uint(val, -12.5, 12.5, 16)
        decoded = uint_to_float(encoded, -12.5, 12.5, 16)
        assert abs(decoded - val) < 0.001, f"Roundtrip failed for {val}: got {decoded}"


def test_float_to_uint_example():
    # C code: (5.0 - (-12.5)) * 65535 / 25.0 = 45874 = 0xB332
    # (The doc's inline formula omits x_min subtraction; the C code is authoritative)
    result = float_to_uint(5.0, -12.5, 12.5, 16)
    assert result == 0xB332, f"Expected 0xB332, got {hex(result)}"


def test_mit_roundtrip():
    limits = MotorLimits(p_max=12.5, v_max=30.0, t_max=10.0)
    p, v, kp, kd, t = 2.0, 5.0, 100.0, 2.0, 1.5

    data = encode_mit(p, v, kp, kd, t, limits)
    assert len(data) == 8

    p2, v2, kp2, kd2, t2 = decode_mit(data, limits)
    assert abs(p2 - p) < 0.01, f"pos: {p2} vs {p}"
    assert abs(v2 - v) < 0.1, f"vel: {v2} vs {v}"
    assert abs(kp2 - kp) < 0.2, f"kp: {kp2} vs {kp}"
    assert abs(kd2 - kd) < 0.01, f"kd: {kd2} vs {kd}"
    assert abs(t2 - t) < 0.01, f"torque: {t2} vs {t}"


def test_feedback_decode():
    limits = MotorLimits(p_max=12.5, v_max=30.0, t_max=10.0)
    # Construct a synthetic feedback frame
    motor_id = 3
    error = 1  # enabled
    pos_int = float_to_uint(1.0, -12.5, 12.5, 16)
    vel_int = float_to_uint(2.0, -30.0, 30.0, 12)
    torque_int = float_to_uint(0.5, -10.0, 10.0, 12)
    t_mos = 45
    t_coil = 50

    data = bytearray(8)
    # onero format: byte0 = (ERR << 4) | (ID & 0xF)
    data[0] = (error << 4) | (motor_id & 0xF)
    data[1] = (pos_int >> 8) & 0xFF
    data[2] = pos_int & 0xFF
    data[3] = (vel_int >> 4) & 0xFF
    data[4] = ((vel_int & 0xF) << 4) | ((torque_int >> 8) & 0xF)
    data[5] = torque_int & 0xFF
    data[6] = struct.pack('b', t_mos)[0]
    data[7] = struct.pack('b', t_coil)[0]

    state = decode_feedback(bytes(data), limits)
    assert state.motor_id == 3
    assert state.error == 1
    assert abs(state.position - 1.0) < 0.01
    assert abs(state.velocity - 2.0) < 0.1
    assert abs(state.torque - 0.5) < 0.01
    assert state.t_mos == 45
    assert state.t_coil == 50


def test_pos_vel_roundtrip():
    p, v = 3.14, 2.71
    data = encode_pos_vel(p, v)
    p2, v2 = decode_pos_vel(data)
    assert abs(p2 - p) < 1e-6
    assert abs(v2 - v) < 1e-6


def test_velocity_roundtrip():
    v = -5.5
    data = encode_velocity(v)
    v2 = decode_velocity(data)
    assert abs(v2 - v) < 1e-6


def test_special_frames():
    assert len(ENABLE_DATA) == 8
    assert ENABLE_DATA[-1] == 0xFC
    assert DISABLE_DATA[-1] == 0xFD
    assert SET_ZERO_DATA[-1] == 0xFE
    assert CLEAR_ERROR_DATA[-1] == 0xFB


def test_param_read_write():
    data = encode_read_param(0x01, 21)
    assert data[0] == 0x01  # CAN_L
    assert data[1] == 0x00  # CAN_H
    assert data[2] == 0x33
    assert data[3] == 21

    data = encode_write_param(0x01, 21, 12.5)
    assert data[2] == 0x55
    assert data[3] == 21
    val = struct.unpack('<f', data[4:8])[0]
    assert abs(val - 12.5) < 1e-6

    # Decode param response
    resp = bytearray(8)
    struct.pack_into('<f', resp, 4, 12.5)
    result = decode_param_response(bytes(resp), as_float=True)
    assert abs(result - 12.5) < 1e-6


if __name__ == "__main__":
    test_float_uint_roundtrip()
    test_float_to_uint_example()
    test_mit_roundtrip()
    test_feedback_decode()
    test_pos_vel_roundtrip()
    test_velocity_roundtrip()
    test_special_frames()
    test_param_read_write()
    print("All tests passed!")
