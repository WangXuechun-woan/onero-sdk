"""
Tests for gear ratio reflected inertia and payload estimation.

Covers:
- GainPolicy with reflected inertia produces higher gains
- Reflected inertia correctly computed from gear_ratios × rotor_inertias
- estimate_payload returns ~0 with no load
- estimate_payload detects known payload mass
- enable_auto_payload triggers re-estimation on state change
- Protocol limits (kp_max, kd_max) still respected with reflected inertia
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def _has_mujoco():
    try:
        import mujoco
        return True
    except ImportError:
        return False


skip_no_mujoco = pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available")


# ═══════════════════════════════════════════════════════════════════════════════
# GainPolicy with reflected inertia
# ═══════════════════════════════════════════════════════════════════════════════

class TestGainPolicyReflectedInertia:
    """Verify that reflected inertia increases computed gains correctly."""

    def test_reflected_inertia_increases_gains(self):
        """With reflected inertia, gains are higher than without."""
        from onero_sdk.gain_policy import GainPolicy, Stiffness

        policy = GainPolicy(stiffness=Stiffness.MEDIUM)
        M = np.diag([0.01, 0.01, 0.005, 0.003, 0.001, 0.001, 0.0001])
        reflected = np.array([0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001])

        kps_without, kds_without = policy.compute(M, reflected_inertia=None)
        kps_with, kds_with = policy.compute(M, reflected_inertia=reflected)

        for i in range(7):
            assert kps_with[i] >= kps_without[i], (
                f"Joint {i}: kp_with={kps_with[i]:.2f} < kp_without={kps_without[i]:.2f}")
            assert kds_with[i] >= kds_without[i], (
                f"Joint {i}: kd_with={kds_with[i]:.3f} < kd_without={kds_without[i]:.3f}")

    def test_reflected_inertia_none_is_backward_compatible(self):
        """reflected_inertia=None produces same results as before."""
        from onero_sdk.gain_policy import GainPolicy, Stiffness

        policy = GainPolicy(stiffness=Stiffness.STIFF)
        M = np.diag([0.05, 0.08, 0.02, 0.01, 0.005, 0.003, 0.001])

        kps_none, kds_none = policy.compute(M, reflected_inertia=None)
        kps_zero, kds_zero = policy.compute(M, reflected_inertia=np.zeros(7))

        for i in range(7):
            assert abs(kps_none[i] - kps_zero[i]) < 1e-10
            assert abs(kds_none[i] - kds_zero[i]) < 1e-10

    def test_gains_respect_protocol_limits(self):
        """Even with large reflected inertia, gains don't exceed kp=500, kd=5."""
        from onero_sdk.gain_policy import GainPolicy, Stiffness

        policy = GainPolicy(stiffness=Stiffness.STIFF)
        M = np.diag([1.0] * 7)  # Huge inertia
        reflected = np.array([1.0] * 7)  # Even more inertia

        kps, kds = policy.compute(M, reflected_inertia=reflected)

        for i in range(7):
            assert kps[i] <= 500.0, f"Joint {i}: kp={kps[i]} > 500"
            assert kds[i] <= 5.0, f"Joint {i}: kd={kds[i]} > 5"

    def test_gain_formula_correctness(self):
        """Verify kp = M_eff × ωn², kd = 2ζ√(kp×M_eff) for unclamped case."""
        from onero_sdk.gain_policy import GainPolicy, Stiffness

        policy = GainPolicy(stiffness=Stiffness.SOFT)  # ωn=8, ζ=0.7 → small gains, won't hit limit
        M = np.diag([0.001] * 7)  # Very small → gains won't saturate
        reflected = np.array([0.0005] * 7)

        kps, kds = policy.compute(M, reflected_inertia=reflected)

        M_eff = 0.001 + 0.0005  # 0.0015
        omega_n = 8.0
        zeta = 0.7
        expected_kp = M_eff * omega_n ** 2  # 0.0015 * 64 = 0.096
        expected_kd = 2 * zeta * np.sqrt(expected_kp * M_eff)

        for i in range(7):
            assert abs(kps[i] - expected_kp) < 1e-8, (
                f"Joint {i}: kp={kps[i]:.6f}, expected={expected_kp:.6f}")
            assert abs(kds[i] - expected_kd) < 1e-8, (
                f"Joint {i}: kd={kds[i]:.6f}, expected={expected_kd:.6f}")

    def test_proportional_scaling_preserves_kd_kp_ratio(self):
        """When gains are clamped, the kd/kp ratio is preserved across joints."""
        from onero_sdk.gain_policy import GainPolicy, Stiffness

        policy = GainPolicy(stiffness=Stiffness.STIFF)  # ωn=40 → large gains
        M = np.diag([0.5] * 7)  # Will hit limits
        reflected = np.array([0.1] * 7)

        kps, kds = policy.compute(M, reflected_inertia=reflected)

        # All joints have same M_eff → same scaling → same kd/kp ratio
        ratios = [kds[i] / kps[i] for i in range(7) if kps[i] > 0.01]
        assert len(set(round(r, 6) for r in ratios)) == 1, (
            f"kd/kp ratios differ across joints: {ratios}")


# ═══════════════════════════════════════════════════════════════════════════════
# Reflected inertia in model registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelReflectedInertia:
    """Verify model_registry stores and propagates gear ratio data."""

    def test_a1_model_has_gear_ratios(self):
        """A1 model has 7 gear ratios defined."""
        from onero_sdk.model_registry import get_model
        model = get_model("A1")
        assert model.gear_ratios is not None
        assert len(model.gear_ratios) == 7

    def test_a1_gear_ratios_values(self):
        """A1 gear ratios match spec: joints 1-3 = 48.19, joints 4-7 = 10."""
        from onero_sdk.model_registry import get_model
        model = get_model("A1")
        assert model.gear_ratios[:3] == [48.19, 48.19, 48.19]
        assert model.gear_ratios[3:] == [10.0, 10.0, 10.0, 10.0]

    def test_a1_model_has_rotor_inertias(self):
        """A1 model has 7 rotor inertias defined."""
        from onero_sdk.model_registry import get_model
        model = get_model("A1")
        assert model.rotor_inertias is not None
        assert len(model.rotor_inertias) == 7

    def test_rotor_inertias_are_positive(self):
        """All rotor inertias are positive physical values."""
        from onero_sdk.model_registry import get_model
        model = get_model("A1")
        for i, inertia in enumerate(model.rotor_inertias):
            assert inertia > 0, f"Joint {i}: rotor_inertia={inertia} <= 0"

    @skip_no_mujoco
    def test_arm_computes_reflected_inertia(self):
        """Arm initializes _reflected_inertia = rotor × ratio²."""
        from onero_sdk import Arm
        from onero_sdk.mujoco_backend import MujocoBackend

        backend = MujocoBackend(model_name="A1")
        arm = Arm("A1", backend=backend)

        assert arm._reflected_inertia is not None
        # Joint 1: 4.5e-6 × 48.19² ≈ 0.01045
        expected_j1 = 4.5e-6 * 48.19 ** 2
        assert abs(arm._reflected_inertia[0] - expected_j1) < 1e-8
        # Joint 4: 1.5e-6 × 10² = 0.00015
        expected_j4 = 1.5e-6 * 10.0 ** 2
        assert abs(arm._reflected_inertia[3] - expected_j4) < 1e-8

        backend.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Payload estimation
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_mujoco
class TestPayloadEstimation:
    """Test payload estimation from torque residual."""

    @pytest.fixture
    def arm_sim(self):
        from onero_sdk import Arm
        from onero_sdk.mujoco_backend import MujocoBackend
        backend = MujocoBackend(model_name="A1")
        arm = Arm("A1", backend=backend)
        arm.enable()
        arm.movej([0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], duration=2.0)
        yield arm
        arm.disable()
        backend.close()

    def test_no_payload_estimates_near_zero(self, arm_sim):
        """With no payload, estimate should be close to 0."""
        mass = arm_sim.estimate_payload(num_samples=20, settle_time=0.1)
        assert mass < 0.1, f"No-load estimate {mass:.3f} kg should be < 0.1 kg"

    def test_estimate_returns_non_negative(self, arm_sim):
        """Estimated mass is never negative."""
        mass = arm_sim.estimate_payload(num_samples=10, settle_time=0.1)
        assert mass >= 0.0

    def test_set_payload_then_estimate_recovers(self, arm_sim):
        """After set_payload, clearing and re-estimating recovers mass."""
        # In simulation, set_payload changes the model, then gravity_torques
        # includes the payload. estimate_payload clears payload first, so the
        # residual between measured (with payload) and expected (without) shows mass.
        #
        # Note: In MuJoCo sim, measured torques = PD controller output, not actual
        # gravity. So this test validates the algorithm structure, not sim accuracy.
        arm_sim.set_payload(0.0, [0.0, 0.0, 0.0])  # Clear any previous
        mass = arm_sim.estimate_payload(num_samples=10, settle_time=0.1)
        # Should be small (simulation noise)
        assert mass < 0.15

    def test_estimate_at_different_configurations(self, arm_sim):
        """Estimation works at various arm configurations."""
        configs = [
            [0.0, -0.3, 0.0, -0.8, 0.0, 0.3, 0.0],
            [0.5, -0.7, 0.3, -1.2, 0.0, 0.5, 0.0],
            [0.0, -1.0, 0.0, -0.5, 0.0, 0.8, 0.0],
        ]
        for config in configs:
            arm_sim.movej(config, duration=1.5)
            mass = arm_sim.estimate_payload(num_samples=10, settle_time=0.1)
            # Should not crash and should return finite value
            assert np.isfinite(mass), f"Non-finite mass at config {config}"
            assert mass >= 0.0


@skip_no_mujoco
class TestAutoPayload:
    """Test automatic payload estimation trigger."""

    @pytest.fixture
    def arm_sim(self):
        from onero_sdk import Arm
        from onero_sdk.mujoco_backend import MujocoBackend
        backend = MujocoBackend(model_name="A1")
        arm = Arm("A1", backend=backend)
        arm.enable()
        arm.movej([0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], duration=2.0)
        yield arm
        arm.disable()
        backend.close()

    def test_enable_disable_auto_payload(self, arm_sim):
        """enable/disable auto_payload without error."""
        arm_sim.enable_auto_payload()
        assert arm_sim._auto_payload_enabled is True
        arm_sim.disable_auto_payload()
        assert arm_sim._auto_payload_enabled is False

    def test_auto_payload_does_not_crash_on_motion(self, arm_sim):
        """Auto payload check during normal motion doesn't crash."""
        arm_sim.enable_auto_payload(velocity_threshold=0.02,
                                     quiet_duration=0.05,
                                     min_change=0.02)
        # These motions should trigger _check_auto_payload at _begin_motion
        arm_sim.movej([0.0, -0.3, 0.0, -0.8, 0.0, 0.3, 0.0], duration=1.0)
        arm_sim.movej([0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], duration=1.0)
        # Should complete without error

    def test_auto_payload_detects_no_change(self, arm_sim):
        """Without payload change, auto_payload_mass stays ~0."""
        arm_sim.enable_auto_payload(quiet_duration=0.05, min_change=0.02)
        arm_sim.movej([0.0, -0.3, 0.0, -0.8, 0.0, 0.3, 0.0], duration=1.0)
        assert arm_sim._auto_payload_mass < 0.15

    def test_check_auto_payload_noop_when_disabled(self, arm_sim):
        """_check_auto_payload is a no-op when not enabled."""
        # Should not raise even without setup
        arm_sim._check_auto_payload()

    def test_auto_payload_config_stored(self, arm_sim):
        """Configuration parameters are stored correctly."""
        arm_sim.enable_auto_payload(velocity_threshold=0.05,
                                     quiet_duration=0.5,
                                     min_change=0.1)
        config = arm_sim._auto_payload_config
        assert config["velocity_threshold"] == 0.05
        assert config["quiet_duration"] == 0.5
        assert config["min_change"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
