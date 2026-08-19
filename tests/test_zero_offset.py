"""
ZeroOffsetBackend + zero_calibration persistence tests.

Uses a mock backend to exercise coordinate translation and offset
management without touching real hardware.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from onero_sdk.backend import ArmBackend, ArmState
from onero_sdk.exceptions import OneroConfigError
from onero_sdk.zero_offset_backend import ZeroOffsetBackend
from onero_sdk.zero_calibration import (
    load_zero_offset, save_zero_offset, remove_zero_offset,
    validate_offsets, default_path, _config_dir,
)


class RecordingBackend(ArmBackend):
    """
    Mock backend that records the last raw command and returns raw
    feedback set by the test. Everything below the ZeroOffsetBackend
    sees raw motor coordinates.
    """

    def __init__(self, num_joints=7, raw_state=None):
        self._num_joints = num_joints
        self._raw_state = raw_state or ArmState(
            joints=[0.0] * num_joints,
            velocities=[0.0] * num_joints,
            torques=[0.0] * num_joints,
        )
        self.last_command = None
        self.disabled = False
        self.closed = False

    def set_raw_state(self, joints=None, velocities=None, torques=None):
        j = list(joints) if joints is not None else self._raw_state.joints
        v = list(velocities) if velocities is not None else self._raw_state.velocities
        t = list(torques) if torques is not None else self._raw_state.torques
        self._raw_state = ArmState(joints=j, velocities=v, torques=t)

    @property
    def num_joints(self):
        return self._num_joints

    def enable(self):
        return self._raw_state

    def disable(self):
        self.disabled = True

    def command(self, positions, velocities, kps, kds, torques):
        self.last_command = {
            "positions": list(positions),
            "velocities": list(velocities),
            "kps": list(kps),
            "kds": list(kds),
            "torques": list(torques),
        }
        return self._raw_state

    def command_hold(self, positions, velocities, kps, kds, torques):
        return self.command(positions, velocities, kps, kds, torques)

    def command_velocity(self, velocities):
        self.last_command = {"velocities": list(velocities)}
        return self._raw_state

    def command_torque(self, torques):
        self.last_command = {"torques": list(torques)}
        return self._raw_state

    def get_state(self):
        return self._raw_state

    def close(self):
        self.closed = True


@dataclass
class FakeJointSpec:
    p_min: float
    p_max: float
    encoding_p_max: float = 12.5


# ═══════════════════════════════════════════════════════════════════════════
# ZeroOffsetBackend — coordinate translation
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroOffsetBackend:

    def test_zero_offset_is_passthrough(self):
        """offset=[0]*n => user == raw on both directions."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner)
        inner.set_raw_state(joints=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        state = wrapped.get_state()
        assert state.joints == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        wrapped.command(
            positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        assert inner.last_command["positions"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_command_adds_offset(self):
        """user target 0 with offset 0.5 => raw target 0.5 on the wire."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)

        wrapped.command(
            positions=[0.0]*7,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        assert inner.last_command["positions"] == [0.5]*7

    def test_command_target_becomes_raw(self):
        """Various user targets get raw = user + offset."""
        inner = RecordingBackend()
        offsets = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        wrapped = ZeroOffsetBackend(inner, offsets=offsets)

        user = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        wrapped.command(
            positions=user,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        expected_raw = [u + o for u, o in zip(user, offsets)]
        for got, exp in zip(inner.last_command["positions"], expected_raw):
            assert abs(got - exp) < 1e-12

    def test_get_state_subtracts_offset(self):
        """raw feedback 1.0 with offset 0.3 => user reads 0.7."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.3]*7)
        inner.set_raw_state(joints=[1.0]*7)

        state = wrapped.get_state()
        for j in state.joints:
            assert abs(j - 0.7) < 1e-12

    def test_command_return_state_in_user_frame(self):
        """After command, returned ArmState.joints is user-space."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)
        inner.set_raw_state(joints=[0.5]*7)

        state = wrapped.command(
            positions=[0.0]*7,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        for j in state.joints:
            assert abs(j - 0.0) < 1e-12

    def test_enable_returns_user_frame(self):
        inner = RecordingBackend()
        inner.set_raw_state(joints=[0.5]*7)
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)
        state = wrapped.enable()
        for j in state.joints:
            assert abs(j - 0.0) < 1e-12

    def test_velocities_pass_through_unchanged(self):
        """velocities/kps/kds/torques are frame-invariant."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)
        wrapped.command(
            positions=[0.0]*7,
            velocities=[1.0]*7, kps=[20]*7, kds=[2]*7, torques=[0.5]*7,
        )
        assert inner.last_command["velocities"] == [1.0]*7
        assert inner.last_command["kps"] == [20]*7
        assert inner.last_command["kds"] == [2]*7
        assert inner.last_command["torques"] == [0.5]*7

    def test_command_velocity_unaffected_in_input(self):
        """command_velocity input is not offset-shifted (it's a velocity)."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)
        wrapped.command_velocity([1.0]*7)
        assert inner.last_command["velocities"] == [1.0]*7

    def test_command_torque_unaffected_in_input(self):
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.5]*7)
        wrapped.command_torque([2.0]*7)
        assert inner.last_command["torques"] == [2.0]*7

    def test_set_offsets_takes_effect_immediately(self):
        """Changing offset re-shifts subsequent commands."""
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner, offsets=[0.0]*7)

        wrapped.command(
            positions=[0.0]*7,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        assert inner.last_command["positions"] == [0.0]*7

        wrapped.set_offsets([0.5]*7)
        wrapped.command(
            positions=[0.0]*7,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        assert inner.last_command["positions"] == [0.5]*7

    def test_set_offsets_wrong_length_raises(self):
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner)
        with pytest.raises(ValueError):
            wrapped.set_offsets([0.0]*6)

    def test_init_wrong_length_raises(self):
        inner = RecordingBackend()
        with pytest.raises(ValueError):
            ZeroOffsetBackend(inner, offsets=[0.0]*8)

    def test_inner_and_num_joints_exposed(self):
        inner = RecordingBackend(num_joints=5)
        wrapped = ZeroOffsetBackend(inner)
        assert wrapped.num_joints == 5
        assert wrapped.inner is inner

    def test_disable_close_forwarded(self):
        inner = RecordingBackend()
        wrapped = ZeroOffsetBackend(inner)
        wrapped.disable()
        wrapped.close()
        assert inner.disabled is True
        assert inner.closed is True

    def test_round_trip_symmetry(self):
        """user -> raw -> user must be identity even with weird offsets."""
        inner = RecordingBackend()
        offsets = [0.017, -0.234, 1.0, -2.5, 0.001, -0.001, 0.5]
        wrapped = ZeroOffsetBackend(inner, offsets=offsets)

        user_target = [0.3, -0.4, 0.0, 1.2, -0.7, 0.1, 0.0]
        # Simulate the motor holding perfectly at target: raw = user + offset
        raw_hold = [u + o for u, o in zip(user_target, offsets)]
        inner.set_raw_state(joints=raw_hold)

        state = wrapped.command(
            positions=user_target,
            velocities=[0]*7, kps=[10]*7, kds=[1]*7, torques=[0]*7,
        )
        for got, exp in zip(state.joints, user_target):
            assert abs(got - exp) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# validate_offsets — URDF ↔ encoding safety
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateOffsets:

    def _make_specs(self, p_min=-1.0, p_max=1.0, encoding_p_max=12.5):
        return [FakeJointSpec(p_min, p_max, encoding_p_max) for _ in range(7)]

    def test_zero_offsets_pass(self):
        validate_offsets([0.0]*7, self._make_specs())

    def test_small_offsets_pass(self):
        validate_offsets([0.5]*7, self._make_specs(p_min=-2.0, p_max=2.0))

    def test_offset_pushing_urdf_limit_over_encoding_raises(self):
        # encoding_p_max=12.5, margin=0.9 => limit 11.25 rad
        # p_max=2, offset=10 => raw upper = 12 > 11.25 => must raise
        specs = self._make_specs(p_min=-2.0, p_max=2.0)
        with pytest.raises(OneroConfigError) as ei:
            validate_offsets([10.0]*7, specs)
        assert "encoding" in str(ei.value).lower()

    def test_boundary_offset_at_margin_passes(self):
        # p_max=1.0, encoding=12.5, margin=0.9 => 11.25
        # max offset that still fits: 11.25 - 1.0 = 10.25
        specs = self._make_specs(p_min=-1.0, p_max=1.0)
        validate_offsets([10.25]*7, specs)

    def test_offset_length_mismatch_raises(self):
        with pytest.raises(OneroConfigError):
            validate_offsets([0.0]*6, self._make_specs())

    def test_reports_first_violating_joint(self):
        specs = self._make_specs(p_min=-1.0, p_max=1.0)
        offsets = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]  # joint 2 out of range
        with pytest.raises(OneroConfigError) as ei:
            validate_offsets(offsets, specs)
        assert "Joint 2" in str(ei.value)


# ═══════════════════════════════════════════════════════════════════════════
# File persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistence:

    @pytest.fixture
    def tmp_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ONERO_CONFIG_DIR", str(tmp_path))
        return tmp_path

    def test_load_missing_returns_zeros(self, tmp_config):
        offsets = load_zero_offset("A1L", "can0", num_joints=7)
        assert offsets == [0.0]*7

    def test_save_then_load_round_trip(self, tmp_config):
        vals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        path = save_zero_offset("A1L", "can0", vals, note="unit test")
        assert path.exists()
        loaded = load_zero_offset("A1L", "can0", num_joints=7)
        for got, exp in zip(loaded, vals):
            assert abs(got - exp) < 1e-12

    def test_save_writes_json_v1_schema(self, tmp_config):
        vals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        path = save_zero_offset("A1L", "can0", vals, note="hi")
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert data["model"] == "A1L"
        assert data["channel"] == "can0"
        assert data["offsets_rad"] == vals
        assert data["note"] == "hi"
        assert "created_at" in data

    def test_file_key_includes_channel(self, tmp_config):
        p0 = default_path("A1L", "can0")
        p1 = default_path("A1L", "can1")
        pnone = default_path("A1L", None)
        assert p0 != p1
        assert p0 != pnone
        assert "can0" in p0.name
        assert "can1" in p1.name

    def test_wrong_length_file_falls_back_to_zeros(self, tmp_config):
        # Simulate a stale file from a different-DOF arm
        save_zero_offset("A1L", "can0", [0.1]*5)
        offsets = load_zero_offset("A1L", "can0", num_joints=7)
        assert offsets == [0.0]*7

    def test_corrupt_file_falls_back_to_zeros(self, tmp_config):
        p = default_path("A1L", "can0")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not valid json")
        offsets = load_zero_offset("A1L", "can0", num_joints=7)
        assert offsets == [0.0]*7

    def test_remove_absent_file_returns_false(self, tmp_config):
        assert remove_zero_offset("A1L", "can0") is False

    def test_remove_existing_file(self, tmp_config):
        save_zero_offset("A1L", "can0", [0.1]*7)
        assert remove_zero_offset("A1L", "can0") is True
        assert not default_path("A1L", "can0").exists()

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_config):
        save_zero_offset("A1L", "can0", [0.1]*7)
        # No .tmp file should remain after successful save
        tmp_files = list(tmp_config.rglob("*.tmp"))
        assert tmp_files == []

    def test_missing_version_falls_back_to_zeros(self, tmp_config):
        """A file without a version field (or wrong version) is rejected."""
        p = default_path("A1L", "can0")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write a v0-shaped record (no version key)
        p.write_text(json.dumps({
            "model": "A1L", "channel": "can0",
            "offsets_rad": [0.1]*7,
        }))
        offsets = load_zero_offset("A1L", "can0", num_joints=7)
        assert offsets == [0.0]*7

    def test_future_version_falls_back_to_zeros(self, tmp_config):
        """A newer schema version is rejected (semantics may differ)."""
        p = default_path("A1L", "can0")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "version": 99,
            "model": "A1L", "channel": "can0",
            "offsets_rad": [0.1]*7,
        }))
        offsets = load_zero_offset("A1L", "can0", num_joints=7)
        assert offsets == [0.0]*7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
