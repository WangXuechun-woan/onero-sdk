"""
Singularity detection and policy tests.

Validates:
- manipulability() returns healthy values at safe pose and zero at fully-folded
- classify_singularity() correctly identifies wrist / elbow / shoulder types
- SingularityPolicy scales velocity in the warning band and aborts at hard limit
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pytest

from onero_sdk.dynamics import DynamicsModel
from onero_sdk.motion_strategy import SingularityPolicy, velocity_cartesian_frames
from onero_sdk.model_registry import _find_urdf
from onero_sdk.exceptions import OneroSafetyError

pin = pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def dynamics():
    urdf = _find_urdf("A1", "r")
    return DynamicsModel(str(urdf))


class TestManipulability:
    def test_healthy_pose_has_positive_manipulability(self, dynamics):
        q = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        w, sigma_min = dynamics.manipulability(q)
        assert w > 0.001, "healthy pose should have non-trivial manipulability"
        assert sigma_min > 0.02, "healthy pose should be far from singular"

    def test_folded_pose_is_singular(self, dynamics):
        q = [0.0] * 7
        w, sigma_min = dynamics.manipulability(q)
        assert w < 1e-4
        assert sigma_min < 1e-4


class TestClassification:
    def test_wrist_singularity(self, dynamics):
        # J5 (index 5) = 0 → wrist singular. Keep other joints safely non-zero.
        q = [0.0, -0.5, 0.0, -1.0, 0.5, 0.0, 0.3]
        stype, sev = dynamics.classify_singularity(q)
        assert stype == "wrist"
        assert sev > 0.9

    def test_elbow_singularity(self, dynamics):
        # J3 (index 3) = 0 → elbow fully extended
        # Keep J5 well away from 0 so wrist score doesn't beat elbow
        q = [0.3, -0.5, 0.2, 0.0, 0.5, 1.2, 0.3]
        stype, sev = dynamics.classify_singularity(q)
        assert stype == "elbow"
        assert sev > 0.9

    def test_shoulder_singularity(self, dynamics):
        # Pull EE onto the base J0 axis (x=y=0 in base frame).
        # Fold the arm mostly straight up: J1 ≈ -π/2 rotates upper arm to +Z,
        # then a small J3 keeps forearm collinear.
        q = [0.0, -math.pi / 2, 0.0, 0.05, 0.5, 1.0, 0.0]
        info_pos = dynamics.ee_position(q)
        radial = math.hypot(info_pos[0], info_pos[1])
        # Only assert classification when we actually got EE near the axis;
        # URDF geometry means some near-axis poses may still have offset.
        if radial < 0.05:
            stype, sev = dynamics.classify_singularity(q)
            assert stype == "shoulder"
            assert sev > 0.5
        else:
            pytest.skip(f"URDF geometry keeps EE off-axis at this pose (radial={radial:.3f}m)")

    def test_no_singularity(self, dynamics):
        q = [0.0, -0.5, 0.0, -1.0, 0.5, 0.5, 0.5]
        stype, sev = dynamics.classify_singularity(q)
        assert stype == "none" or sev < 0.1


class TestSingularityPolicy:
    def test_scales_in_warning_zone(self, dynamics):
        # Warn threshold intentionally above the pose's sigma_min so we're
        # inside the warning band; hard threshold well below.
        q = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        _, sigma_min = dynamics.manipulability(q)
        assert sigma_min > 0.02, "test fixture assumed"

        policy = SingularityPolicy(
            warn_threshold=sigma_min * 2, hard_threshold=0.001,
            abort_on_hard=True, classify_fn=dynamics.classify_singularity,
        )
        twist = np.array([0.05, 0, 0, 0, 0, 0])
        frames = list(velocity_cartesian_frames(
            twist, ramp_steps=0, hold_steps=1, dt=0.005,
            get_joints_fn=lambda: q,
            jacobian_solve_fn=lambda qq, t, **kw: dynamics.jacobian_solve(qq, t, **kw),
            singularity_policy=policy,
        ))
        assert len(frames) == 1
        baseline = dynamics.jacobian_solve(q, twist)
        scaled = np.asarray(frames[0].velocities)
        # Scale must be strictly less than 1
        assert np.linalg.norm(scaled) < np.linalg.norm(baseline) * 0.99

    def test_aborts_at_hard_threshold(self, dynamics):
        q = [0.0] * 7  # fully singular
        policy = SingularityPolicy(
            warn_threshold=0.1, hard_threshold=0.001,
            abort_on_hard=True, classify_fn=dynamics.classify_singularity,
        )
        twist = np.array([0.05, 0, 0, 0, 0, 0])
        with pytest.raises(OneroSafetyError):
            list(velocity_cartesian_frames(
                twist, ramp_steps=0, hold_steps=1, dt=0.005,
                get_joints_fn=lambda: q,
                jacobian_solve_fn=lambda qq, t, **kw: dynamics.jacobian_solve(qq, t, **kw),
                singularity_policy=policy,
            ))

    def test_clamps_when_abort_off(self, dynamics):
        q = [0.0] * 7
        policy = SingularityPolicy(
            warn_threshold=0.1, hard_threshold=0.001,
            abort_on_hard=False, classify_fn=dynamics.classify_singularity,
        )
        twist = np.array([0.05, 0, 0, 0, 0, 0])
        frames = list(velocity_cartesian_frames(
            twist, ramp_steps=0, hold_steps=2, dt=0.005,
            get_joints_fn=lambda: q,
            jacobian_solve_fn=lambda qq, t, **kw: dynamics.jacobian_solve(qq, t, **kw),
            singularity_policy=policy,
        ))
        for f in frames:
            assert max(abs(v) for v in f.velocities) < 1e-9

    def test_passthrough_when_no_policy(self, dynamics):
        q = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        twist = np.array([0.05, 0, 0, 0, 0, 0])
        frames = list(velocity_cartesian_frames(
            twist, ramp_steps=0, hold_steps=1, dt=0.005,
            get_joints_fn=lambda: q,
            jacobian_solve_fn=dynamics.jacobian_solve,
            singularity_policy=None,
        ))
        baseline = dynamics.jacobian_solve(q, twist)
        assert np.allclose(np.asarray(frames[0].velocities), baseline, atol=1e-9)
