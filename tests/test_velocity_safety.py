"""Safety contracts for blocking speedj/speedl motion."""

import math

from onero_sdk import Arm
from onero_sdk.backend import ArmBackend, ArmState
from onero_sdk.safety import SafetyConfig


class RecordingVelocityBackend(ArmBackend):
    """Minimal hardware boundary that records every impedance command."""

    def __init__(self):
        self.q = [0.0] * 7
        self.v = [0.0] * 7
        self.commands = []
        self.samples = []
        self.fail_on_command = None
        self.command_count = 0
        self.lock_motion = False

    @property
    def num_joints(self):
        return 7

    def enable(self):
        return self.get_state()

    def disable(self):
        pass

    def command(self, positions, velocities, kps, kds, torques):
        self.command_count += 1
        if self.command_count == self.fail_on_command:
            self.fail_on_command = None
            raise RuntimeError("injected transport failure")
        self.commands.append((list(positions), list(velocities), list(kps), list(kds)))
        self.samples.append({
            "q_before": list(self.q), "positions": list(positions),
            "velocities": list(velocities), "kps": list(kps), "kds": list(kds),
        })
        self.v = list(velocities)
        if not self.lock_motion:
            self.q = [q + v / 200.0 for q, v in zip(self.q, self.v)]
        return self.get_state()

    def command_velocity(self, velocities):
        self.v = list(velocities)
        self.q = [q + v / 200.0 for q, v in zip(self.q, self.v)]
        return self.get_state()

    def command_torque(self, torques):
        return self.get_state()

    def get_state(self):
        return ArmState(list(self.q), list(self.v), [0.0] * 7)

    def close(self, disable=False):
        pass


def test_default_a1_velocity_limits_are_3_for_shoulder_and_6_for_wrist():
    shoulder_backend = RecordingVelocityBackend()
    shoulder = Arm("A1R", backend=shoulder_backend)
    shoulder.enable()
    shoulder.speedj([10.0, 0, 0, 0, 0, 0, 0], acceleration=1000.0, duration=0.02)

    wrist_backend = RecordingVelocityBackend()
    wrist = Arm("A1R", backend=wrist_backend)
    wrist.enable()
    wrist.speedj([0, 0, 0, 0, 10.0, 0, 0], acceleration=1000.0, duration=0.02)

    shoulder_peak = max(abs(command[1][0]) for command in shoulder_backend.commands)
    wrist_peak = max(abs(command[1][4]) for command in wrist_backend.commands)
    assert shoulder_peak <= 3.0
    assert wrist_peak <= 6.0
    assert wrist_peak > 5.0


def test_speedj_never_forwards_velocity_above_safety_limit():
    backend = RecordingVelocityBackend()
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=1.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    arm.speedj([100.0, 0, 0, 0, 0, 0, 0], acceleration=1000.0, duration=0.02)

    assert backend.commands
    assert max(abs(v) for _, velocities, _, _ in backend.commands for v in velocities) <= 1.0
    assert all(math.isfinite(v) for _, velocities, _, _ in backend.commands for v in velocities)


def test_speedj_does_not_drive_outward_through_joint_limit():
    backend = RecordingVelocityBackend()
    backend.q[0] = 3.139  # A1R joint 1 upper limit is 3.14 rad.
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    arm.speedj([1.0, 0, 0, 0, 0, 0, 0], acceleration=100.0, duration=0.05)

    assert backend.q[0] <= 3.14


def test_speedj_uses_integrated_position_reference_with_moderate_gains():
    backend = RecordingVelocityBackend()
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    arm.speedj([0.5, 0, 0, 0, 0, 0, 0], acceleration=10.0, duration=0.1)

    moving = [s for s in backend.samples if s["velocities"][0] > 0]
    assert moving
    assert all(s["positions"][0] > s["q_before"][0] for s in moving)
    assert all(0 < s["kps"][0] <= 10.0 for s in moving)
    assert all(0 < s["kds"][0] < 5.0 for s in moving)


def test_speedj_accelerates_from_measured_velocity_without_step_change():
    backend = RecordingVelocityBackend()
    backend.v[0] = 0.4
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    arm.speedj([-0.4, 0, 0, 0, 0, 0, 0], acceleration=1.0, duration=2.0)

    first_velocity = backend.commands[0][1][0]
    assert abs(first_velocity - 0.4) <= 1.0 / 200.0 + 1e-9


def test_speedj_transport_failure_attempts_zero_velocity_position_hold():
    import pytest

    backend = RecordingVelocityBackend()
    backend.fail_on_command = 3
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    with pytest.raises(RuntimeError, match="injected transport failure"):
        arm.speedj([0.5, 0, 0, 0, 0, 0, 0], acceleration=10.0, duration=0.1)

    _, velocities, kps, _ = backend.commands[-1]
    assert velocities == [0.0] * 7
    assert all(kp > 0 for kp in kps)


def test_speedl_enforces_joint_acceleration_as_jacobian_changes():
    backend = RecordingVelocityBackend()
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()
    acceleration = 0.5

    arm.speedl([0.05, 0, 0, 0, 0, 0], acceleration=acceleration, duration=0.5)

    velocity_samples = [velocities for _, velocities, _, _ in backend.commands]
    previous = [0.0] * 7
    max_delta = acceleration / 200.0 + 1e-9
    for current in velocity_samples:
        assert max(abs(a - b) for a, b in zip(current, previous)) <= max_delta
        previous = current


def test_stop_rejects_nonpositive_deceleration():
    import pytest
    from onero_sdk.exceptions import OneroConfigError

    backend = RecordingVelocityBackend()
    backend.v[0] = 0.2
    arm = Arm("A1R", backend=backend)
    arm.enable()

    with pytest.raises(OneroConfigError, match="greater than zero"):
        arm.stop(deceleration=0.0)


def test_speedj_position_reference_lead_is_bounded_when_joint_is_stuck():
    backend = RecordingVelocityBackend()
    backend.lock_motion = True
    arm = Arm(
        "A1R",
        backend=backend,
        safety=SafetyConfig(max_velocity=5.0, tracking_error_hard_threshold=0,
                            tracking_error_soft_threshold=0),
    )
    arm.enable()

    arm.speedj([0.5, 0, 0, 0, 0, 0, 0], acceleration=2.0, duration=1.0)

    assert max(command[0][0] for command in backend.commands) <= 0.1 + 1e-9
