"""
Unit tests for protocol encoding/decoding.
"""

import struct
import sys
import os
import math
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onero_sdk.protocol import (  # noqa: E402
    float_to_uint, uint_to_float,
    encode_mit, decode_mit,
    decode_feedback, encode_pos_vel, decode_pos_vel,
    encode_velocity, decode_velocity,
    encode_read_param, encode_write_param, decode_param_response,
    ENABLE_DATA, DISABLE_DATA, SET_ZERO_DATA, CLEAR_ERROR_DATA,
    MotorError, MotorLimits, MotorState,
)
from onero_sdk.exceptions import OneroHardwareError  # noqa: E402
from onero_sdk.model_registry import get_model  # noqa: E402


def test_float_uint_roundtrip():
    limits = MotorLimits(p_max=12.5)
    assert limits.v_max == 45.0
    assert limits.t_max == 30.0
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


def test_mit_zero_golden_vector_matches_drive_reference():
    limits = MotorLimits(p_max=12.5, v_max=45.0, t_max=30.0)
    assert encode_mit(0.0, 0.0, 0.0, 0.0, 0.0, limits) == bytes.fromhex(
        "7f ff 7f f0 00 00 07 ff"
    )


@pytest.mark.parametrize("field,value", [
    ("position", 12.5001),
    ("velocity", 45.0001),
    ("kp", 500.0001),
    ("kd", 5.0001),
    ("torque", 30.0001),
    ("position", math.nan),
    ("torque", math.inf),
])
def test_mit_command_out_of_range_is_rejected(field, value):
    limits = MotorLimits(p_max=12.5, v_max=45.0, t_max=30.0)
    command = {
        "position": 0.0,
        "velocity": 0.0,
        "kp": 0.0,
        "kd": 0.0,
        "torque": 0.0,
    }
    command[field] = value
    with pytest.raises(ValueError, match=field):
        encode_mit(
            command["position"], command["velocity"],
            command["kp"], command["kd"], command["torque"], limits,
        )


def test_mit_endpoint_rounding_is_clamped_but_not_silently_saturated():
    """Accept an ULP-sized endpoint overshoot from floating-point math."""
    limits = MotorLimits(p_max=12.5, v_max=45.0, t_max=30.0)
    data = encode_mit(0.0, 0.0, 0.0, 5.0 + 1e-13, 0.0, limits)
    assert len(data) == 8

    with pytest.raises(ValueError, match="kd"):
        encode_mit(0.0, 0.0, 0.0, 5.0001, 0.0, limits)


@pytest.mark.parametrize("center_code", [2047, 2048])
def test_both_mit_velocity_center_codes_decode_to_zero(center_code):
    limits = MotorLimits(p_max=12.5, v_max=45.0, t_max=30.0)
    command = bytearray(8)
    command[2] = (center_code >> 4) & 0xFF
    command[3] = (center_code & 0xF) << 4
    assert decode_mit(bytes(command), limits)[1] == 0.0

    feedback = bytearray(8)
    feedback[3] = (center_code >> 4) & 0xFF
    feedback[4] = (center_code & 0xF) << 4
    assert decode_feedback(bytes(feedback), limits).velocity == 0.0


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


@pytest.mark.parametrize("model", ["A1L", "A1R"])
def test_can_backend_uses_static_mit_codec_without_parameter_reads(
        monkeypatch, model):
    import onero_sdk.can_backend as module

    class FakeBus:
        def __init__(self, channel):
            self.sent = []

        def send(self, can_id, data):
            self.sent.append((can_id, bytes(data)))

        def recv_batch(self, target_ids, timeout):
            return {}

        def flush_rx(self):
            pass

        def close(self):
            pass

    class FakeMotor:
        def __init__(self, bus, can_id, mst_id, limits):
            self._can_id = can_id
            self._limits = limits
            self._error = MotorError.DISABLED
            self.parameter_accesses = 0
            self.clear_error_calls = 0
            self.enable_calls = 0

        @property
        def can_id(self):
            return self._can_id

        @property
        def limits(self):
            return self._limits

        def get_feedback(self, timeout=0.05):
            return MotorState(motor_id=self._can_id, error=self._error)

        def clear_error(self, timeout=0.05):
            self.clear_error_calls += 1
            return self.get_feedback(timeout)

        def enable(self, timeout=0.05):
            self.enable_calls += 1
            self._error = MotorError.ENABLED
            return self.get_feedback(timeout)

        def disable(self, timeout=0.05):
            self._error = MotorError.DISABLED
            return self.get_feedback(timeout)

        def read_param(self, *args, **kwargs):
            self.parameter_accesses += 1
            raise AssertionError("normal backend flow must not read parameters")

        def write_param(self, *args, **kwargs):
            self.parameter_accesses += 1
            raise AssertionError("normal backend flow must not write parameters")

    monkeypatch.setattr(module, "CanBus", FakeBus)
    monkeypatch.setattr(module, "Motor", FakeMotor)

    backend = module.CanBackend("can-test", get_model(model).joints)
    try:
        assert all(
            motor.limits == MotorLimits(12.5, 45.0, 30.0)
            for motor in backend._motors
        )
        backend.get_state()
        backend.enable()
        first_enable_counts = [motor.enable_calls for motor in backend._motors]
        first_clear_counts = [motor.clear_error_calls for motor in backend._motors]
        backend.enable()
        assert [motor.enable_calls for motor in backend._motors] == first_enable_counts
        assert [motor.clear_error_calls for motor in backend._motors] == first_clear_counts

        backend._motors[0]._error = MotorError.DISABLED
        with pytest.raises(OneroHardwareError, match="Mixed motor states"):
            backend.enable()
        assert all(motor.parameter_accesses == 0 for motor in backend._motors)
    finally:
        backend.close(disable=False)


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
