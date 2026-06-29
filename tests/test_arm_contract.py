"""
Contract tests — verify the Arm class satisfies the interface adapters depend on.

Uses real MujocoBackend instances (no mocks). If any method adapters call
doesn't exist or breaks, these tests fail immediately.
"""

import numpy as np
import pytest

from onero_sdk import Arm, ArmProtocol
from onero_sdk.backend import ArmState
from onero_sdk.mujoco_backend import MujocoBackend


# ─── API Surface Tests ──────────────────────────────────────────────────────

class TestArmSatisfiesProtocol:
    """Verify Arm has every method/property adapters rely on."""

    def test_arm_is_runtime_checkable_protocol(self, arm_sim):
        assert isinstance(arm_sim, ArmProtocol)

    @pytest.mark.parametrize("method", [
        "enable", "disable", "servoj", "servol", "movej",
    ])
    def test_adapter_methods_exist_and_callable(self, arm_sim, method):
        assert hasattr(arm_sim, method)
        assert callable(getattr(arm_sim, method))

    @pytest.mark.parametrize("prop", ["state", "enabled"])
    def test_adapter_properties_exist(self, arm_sim, prop):
        assert hasattr(arm_sim, prop)
        _ = getattr(arm_sim, prop)

    def test_servol_is_alias_for_servop(self):
        assert Arm.servol is Arm.servop


# ─── Property Contract Tests ────────────────────────────────────────────────

class TestPropertyContracts:
    """Verify property return types match adapter expectations."""

    def test_state_returns_armstate(self, arm_sim):
        state = arm_sim.state
        assert isinstance(state, ArmState)
        assert hasattr(state, 'joints')
        assert hasattr(state, 'velocities')
        assert hasattr(state, 'torques')
        assert len(state.joints) == 7

    def test_enabled_returns_bool(self, arm_sim):
        assert arm_sim.enabled is True


# ─── BEAVR Adapter Contract ─────────────────────────────────────────────────

class TestBeavrContract:
    """BEAVR adapter calls: enable, state, servol(6D), movej(joints, duration)."""

    def test_servol_accepts_6d_pose(self, arm_sim):
        """BEAVR calls arm.servol(pose_6d) with [x, y, z, rx, ry, rz]."""
        pos, rot = arm_sim.get_pose(frame="base")
        from scipy.spatial.transform import Rotation
        rvec = Rotation.from_matrix(rot).as_rotvec()
        pose_6d = list(pos) + rvec.tolist()
        result = arm_sim.servol(pose_6d)
        assert isinstance(result, ArmState)

    def test_movej_with_duration(self, arm_sim):
        """BEAVR calls arm.movej(joints, duration=3.0) for reset."""
        target = [0.0] * 7
        result = arm_sim.movej(target, duration=1.0)
        assert isinstance(result, ArmState)

    def test_state_has_joints_list(self, arm_sim):
        """BEAVR reads arm.state.joints for FK computation."""
        state = arm_sim.state
        assert isinstance(state.joints, list)
        assert all(isinstance(j, float) for j in state.joints)


# ─── LeRobot Adapter Contract ───────────────────────────────────────────────

class TestLeRobotContract:
    """LeRobot adapter calls: enable, state, enabled, servoj, disable."""

    def test_servoj_with_positions(self, arm_sim):
        """LeRobot calls arm.servoj(positions) — list of joint angles."""
        target = list(arm_sim.state.joints)
        result = arm_sim.servoj(target)
        assert isinstance(result, ArmState)

    def test_enabled_property_reflects_state(self, arm_sim):
        """LeRobot checks arm.enabled to determine connection status."""
        assert arm_sim.enabled is True
        arm_sim.disable()
        assert arm_sim.enabled is False
        arm_sim.enable()

    def test_state_joints_length_matches_model(self, arm_sim):
        """LeRobot uses len(state.joints) for observation shape."""
        state = arm_sim.state
        assert len(state.joints) == 7
        assert len(state.velocities) == 7
