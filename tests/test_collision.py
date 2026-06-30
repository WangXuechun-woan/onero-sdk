"""
Tests for multi-strategy collision detection.

Tests CollisionDetector algorithms in isolation using synthetic torque sequences.
No hardware or simulation needed.
"""

import time
import numpy as np
import pytest

from onero_sdk.collision import (
    CollisionConfig,
    CollisionDetector,
    CollisionEvent,
    CollisionResponse,
)


NUM_JOINTS = 7
DT = 1.0 / 200.0  # 200Hz


def make_detector(config=None, on_collision=None, residual_fn=None):
    if config is None:
        config = CollisionConfig()
    return CollisionDetector(NUM_JOINTS, config, on_collision, residual_fn)


def warm_up(detector, frames=25, base_torque=5.0, velocity=0.5):
    """Run warmup frames with gentle steady motion to arm the detector."""
    for i in range(frames):
        tau = [base_torque] * NUM_JOINTS
        vel = [velocity] * NUM_JOINTS
        pos = [0.1 * i * DT] * NUM_JOINTS
        detector.process_frame(pos, vel, tau)


class TestImpactChannel:
    def test_no_trigger_during_warmup(self):
        detector = make_detector()
        # Huge spike during warmup — should NOT trigger
        detector.process_frame([0.0] * 7, [1.0] * 7, [0.0] * 7)
        event = detector.process_frame([0.0] * 7, [1.0] * 7, [100.0] * 7)
        assert event is None
        assert not detector.armed

    def test_arms_after_warmup(self):
        detector = make_detector()
        warm_up(detector)
        assert detector.armed

    def test_no_trigger_below_threshold(self):
        detector = make_detector(CollisionConfig(impact_threshold=50.0))
        warm_up(detector)
        # Build a longer baseline at the exact same level
        for _ in range(100):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)
        # Very small torque change: 0.01 Nm per frame = 2 Nm/s — well below threshold
        for i in range(10):
            tau = [5.0 + 0.01 * i] * NUM_JOINTS
            event = detector.process_frame([0.0] * 7, [0.5] * 7, tau)
            assert event is None

    def test_trigger_on_spike(self):
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Steady state — build baseline
        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        # Sudden collision: 20 Nm jump in one frame = 4000 Nm/s
        event = detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert event is not None
        assert event.channel == "impact"
        assert event.magnitude > 50.0

    def test_slow_drift_no_trigger(self):
        """Payload gradually increasing (garment scenario) should NOT trigger."""
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Slow drift: 0.01 Nm per frame = 2 Nm/s at 200Hz
        for i in range(200):
            tau = [5.0 + 0.01 * i] * NUM_JOINTS
            event = detector.process_frame([0.0] * 7, [0.5] * 7, tau)
            assert event is None, f"Falsely triggered at frame {i}"

    def test_holdoff_prevents_double_trigger(self):
        config = CollisionConfig(
            impact_threshold=50.0, warmup_frames=5, impact_holdoff_frames=10,
            energy_enabled=False,
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Build steady baseline
        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        # First spike triggers on all joints
        event1 = detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert event1 is not None

        # All joints are now in holdoff — repeat same spike should be suppressed
        event2 = detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert event2 is None

    def test_no_trigger_when_static(self):
        """Arm at rest (low velocity) — detection disabled."""
        config = CollisionConfig(min_velocity=0.05, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Nearly zero velocity — min_velocity gate should block detection
        detector.process_frame([0.0] * 7, [0.01] * 7, [5.0] * 7)
        event = detector.process_frame([0.0] * 7, [0.01] * 7, [50.0] * 7)
        assert event is None

    def test_reports_strongest_joint(self):
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        # Joint 3 gets the biggest spike
        tau_spike = [5.0] * 7
        tau_spike[3] = 30.0  # 25 Nm jump = 5000 Nm/s on joint 3
        event = detector.process_frame([0.0] * 7, [0.5] * 7, tau_spike)
        assert event is not None
        assert event.joint_index == 3


class TestEnergyChannel:
    def test_no_trigger_below_threshold(self):
        config = CollisionConfig(
            impact_enabled=False, energy_threshold=15.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Steady low power
        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [2.0] * 7)

        # Small increase (power = 2.5 * 0.5 = 1.25W, baseline ~1.0W, anomaly ~0.25W)
        event = detector.process_frame([0.0] * 7, [0.5] * 7, [2.5] * 7)
        assert event is None

    def test_trigger_on_power_spike(self):
        config = CollisionConfig(
            impact_enabled=False, energy_threshold=15.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Build steady baseline (power = |2.0 * 0.5| = 1.0 W per joint)
        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [2.0] * 7)

        # Sudden power injection: torque jumps to 50 Nm at 0.5 rad/s = 25W
        # Anomaly = 25 - baseline(~1.0) ≈ 24 W >> 15W threshold
        event = detector.process_frame([0.0] * 7, [0.5] * 7, [50.0] * 7)
        assert event is not None
        assert event.channel == "energy"
        assert event.magnitude > 15.0

    def test_no_trigger_static_torque(self):
        """Holding payload at rest — high torque but zero velocity means zero power."""
        config = CollisionConfig(
            impact_enabled=False, energy_threshold=15.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # High torque (holding weight) but zero velocity → zero power
        # min_velocity gate filters this case
        for _ in range(50):
            event = detector.process_frame([0.0] * 7, [0.01] * 7, [20.0] * 7)
            assert event is None


class TestResidualChannel:
    def test_disabled_by_default(self):
        config = CollisionConfig()
        assert not config.residual_enabled

    def test_no_trigger_when_disabled(self):
        def bad_residual(q, v, tau):
            return [100.0] * 7  # Would trigger if enabled

        config = CollisionConfig(residual_enabled=False)
        detector = make_detector(config, residual_fn=bad_residual)
        warm_up(detector, frames=25)

        for _ in range(10):
            event = detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)
            assert event is None

    def test_trigger_when_enabled(self):
        def high_residual(q, v, tau):
            return [10.0] * 7  # Above threshold of 8.0

        config = CollisionConfig(
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=8.0,
            warmup_frames=5,
        )
        detector = make_detector(config, residual_fn=high_residual)
        warm_up(detector, frames=10)

        event = detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)
        assert event is not None
        assert event.channel == "residual"
        assert event.magnitude == pytest.approx(10.0)

    def test_no_trigger_below_threshold(self):
        def low_residual(q, v, tau):
            return [3.0] * 7  # Below threshold of 8.0

        config = CollisionConfig(
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=8.0,
            warmup_frames=5,
        )
        detector = make_detector(config, residual_fn=low_residual)
        warm_up(detector, frames=10)

        event = detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)
        assert event is None


class TestIntegration:
    def test_impact_fires_before_energy(self):
        """When both would trigger, impact (checked first) wins."""
        config = CollisionConfig(
            impact_threshold=50.0, energy_threshold=5.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        # Steady baseline
        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        # Both channels would trigger: big torque spike + power spike
        event = detector.process_frame([0.0] * 7, [0.5] * 7, [30.0] * 7)
        assert event is not None
        assert event.channel == "impact"

    def test_reset_clears_state(self):
        config = CollisionConfig(warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)
        assert detector.armed

        detector.reset()
        assert not detector.armed
        assert detector.last_event is None

    def test_callback_invoked(self):
        events = []

        def on_collision(event):
            events.append(event)

        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config, on_collision=on_collision)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert len(events) == 1
        assert events[0].channel == "impact"

    def test_event_contains_snapshot(self):
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        vel = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        tau = [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
        event = detector.process_frame(pos, vel, tau)
        assert event is not None
        assert event.joints == pos
        assert event.velocities == vel
        assert event.torques == tau
        assert event.timestamp > 0

    def test_severity_property(self):
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        event = detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert event is not None
        assert event.severity > 1.0

    def test_last_event_stored(self):
        config = CollisionConfig(impact_threshold=50.0, warmup_frames=5)
        detector = make_detector(config)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        assert detector.last_event is None
        detector.process_frame([0.0] * 7, [0.5] * 7, [25.0] * 7)
        assert detector.last_event is not None
        assert detector.last_event.channel == "impact"


class TestCollisionConfig:
    def test_defaults(self):
        config = CollisionConfig()
        assert config.impact_enabled is True
        assert config.energy_enabled is True
        assert config.residual_enabled is False
        assert config.response == CollisionResponse.STOP
        assert config.warmup_frames == 20
        assert config.control_freq == 200.0

    def test_custom_config(self):
        config = CollisionConfig(
            impact_threshold=25.0,
            energy_threshold=8.0,
            response=CollisionResponse.CALLBACK,
        )
        assert config.impact_threshold == 25.0
        assert config.energy_threshold == 8.0
        assert config.response == CollisionResponse.CALLBACK


class TestCollisionResponse:
    def test_stop_response(self):
        assert CollisionResponse.STOP.value == "stop"

    def test_callback_response(self):
        assert CollisionResponse.CALLBACK.value == "callback"

    def test_both_response(self):
        assert CollisionResponse.BOTH.value == "both"


class TestExecutorIntegration:
    """Test that MotionExecutor correctly calls the collision detector."""

    def test_executor_without_detector_unchanged(self):
        """Executor without collision_detector works exactly as before."""
        from unittest.mock import MagicMock, patch
        from onero_sdk.motion_executor import MotionExecutor, MotionFrame

        backend = MagicMock()
        safety = MagicMock()
        safety.emergency_stopped = False
        recorder = MagicMock()
        clock = MagicMock()
        state = MagicMock()
        state.joints = [0.0] * 7
        state.velocities = [0.0] * 7
        state.torques = [0.0] * 7

        backend.command.return_value = state
        get_state = MagicMock(return_value=state)
        set_state = MagicMock()

        executor = MotionExecutor(
            backend=backend, safety=safety, recorder=recorder,
            clock=clock, get_state=get_state, set_state=set_state,
            collision_detector=None,
        )

        frames = [
            MotionFrame([0.1] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005)
        ]
        result = executor.run(iter(frames))
        assert backend.command.called
        assert set_state.called

    def test_executor_with_detector_stops_on_collision(self):
        """Executor stops motion when detector returns an event with STOP response."""
        from unittest.mock import MagicMock
        from onero_sdk.motion_executor import MotionExecutor, MotionFrame

        backend = MagicMock()
        safety = MagicMock()
        safety.emergency_stopped = False
        recorder = MagicMock()
        clock = MagicMock()
        state = MagicMock()
        state.joints = [0.0] * 7
        state.velocities = [0.5] * 7
        state.torques = [5.0] * 7

        backend.command.return_value = state
        get_state = MagicMock(return_value=state)
        set_state = MagicMock()

        # Detector that always triggers
        config = CollisionConfig(
            impact_threshold=50.0,
            warmup_frames=0,
            response=CollisionResponse.STOP,
        )
        detector = CollisionDetector(7, config)
        # Force arm detector
        detector._armed = True
        detector._prev_torques = np.zeros(7)
        detector._dtau_baseline = np.zeros(7)

        executor = MotionExecutor(
            backend=backend, safety=safety, recorder=recorder,
            clock=clock, get_state=get_state, set_state=set_state,
            collision_detector=detector,
        )

        # Multiple frames but should stop after first collision
        frames = [
            MotionFrame([0.1] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005),
            MotionFrame([0.2] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005),
            MotionFrame([0.3] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005),
        ]

        # Set up torques that will spike (measured returns 50 Nm)
        spike_state = MagicMock()
        spike_state.joints = [0.0] * 7
        spike_state.velocities = [0.5] * 7
        spike_state.torques = [50.0] * 7
        backend.command.return_value = spike_state

        executor.run(iter(frames))
        # Should have called trigger_emergency_stop due to collision
        safety.trigger_emergency_stop.assert_called()

    def test_executor_callback_only_continues(self):
        """With CALLBACK response, executor continues motion after detection."""
        from unittest.mock import MagicMock
        from onero_sdk.motion_executor import MotionExecutor, MotionFrame

        backend = MagicMock()
        safety = MagicMock()
        safety.emergency_stopped = False
        recorder = MagicMock()
        clock = MagicMock()

        state = MagicMock()
        state.joints = [0.0] * 7
        state.velocities = [0.5] * 7
        state.torques = [5.0] * 7
        backend.command.return_value = state
        get_state = MagicMock(return_value=state)
        set_state = MagicMock()

        config = CollisionConfig(
            impact_threshold=50.0,
            warmup_frames=0,
            response=CollisionResponse.CALLBACK,
        )
        detector = CollisionDetector(7, config)

        executor = MotionExecutor(
            backend=backend, safety=safety, recorder=recorder,
            clock=clock, get_state=get_state, set_state=set_state,
            collision_detector=detector,
        )

        frames = [
            MotionFrame([0.1] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005),
            MotionFrame([0.2] * 7, [0.0] * 7, [50.0] * 7, [2.0] * 7, [0.0] * 7, 0.005),
        ]
        executor.run(iter(frames))
        # Should NOT stop — CALLBACK only
        safety.set_emergency_stop.assert_not_called()
        assert backend.command.call_count == 2


class TestBackwardCompat:
    """Ensure existing collision detection API is unaffected."""

    def test_safety_check_collision_unchanged(self):
        """SafetyMonitor.check_collision still works with residuals."""
        from onero_sdk.safety import SafetyConfig, SafetyMonitor

        config = SafetyConfig(collision_threshold=5.0)
        safety = SafetyMonitor(
            num_joints=7,
            joint_limits=[(-3.14, 3.14)] * 7,
            config=config,
        )

        # Below threshold
        assert not safety.check_collision([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        # Above threshold
        assert safety.check_collision([1.0, 2.0, 3.0, 4.0, 5.1, 2.0, 1.0])


class TestHoldoffAllJoints:
    """After a collision, ALL joints should be in holdoff to prevent cascade."""

    def test_collision_cascade_suppressed(self):
        """Collision on one joint doesn't re-trigger on adjacent joints next frame."""
        config = CollisionConfig(
            impact_threshold=50.0, warmup_frames=5,
            impact_holdoff_frames=10, energy_enabled=False,
        )
        detector = make_detector(config)
        warm_up(detector, frames=10)

        for _ in range(50):
            detector.process_frame([0.0] * 7, [0.5] * 7, [5.0] * 7)

        # Spike on joint 0 only
        tau_spike = [5.0] * 7
        tau_spike[0] = 25.0
        event1 = detector.process_frame([0.0] * 7, [0.5] * 7, tau_spike)
        assert event1 is not None
        assert event1.joint_index == 0

        # Next frame: different joint spikes (propagation) — should be suppressed
        tau_spike2 = [5.0] * 7
        tau_spike2[3] = 30.0
        event2 = detector.process_frame([0.0] * 7, [0.5] * 7, tau_spike2)
        assert event2 is None


class TestEnergyChannelRobustness:
    """Energy channel should not trigger during normal acceleration."""

    def test_gradual_acceleration_no_trigger(self):
        """Arm accelerating from rest: both torque and velocity increase smoothly."""
        config = CollisionConfig(
            impact_enabled=False, energy_threshold=15.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10, base_torque=1.0, velocity=0.1)

        # Simulate smooth acceleration over 50 frames
        # Torque ramps linearly, velocity ramps linearly
        for i in range(50):
            t = i / 50.0
            tau = [1.0 + 4.0 * t] * NUM_JOINTS  # 1 → 5 Nm over 50 frames
            vel = [0.1 + 0.9 * t] * NUM_JOINTS  # 0.1 → 1.0 rad/s over 50 frames
            event = detector.process_frame([0.0] * 7, vel, tau)
            assert event is None, f"False trigger at frame {i}, vel={vel[0]:.2f}, tau={tau[0]:.2f}"

    def test_sudden_deceleration_may_trigger(self):
        """Hard stop (hitting something) causes power reversal — should trigger."""
        config = CollisionConfig(
            impact_enabled=False, energy_threshold=15.0, warmup_frames=5
        )
        detector = make_detector(config)
        warm_up(detector, frames=10, base_torque=3.0, velocity=0.5)

        # Steady state
        for _ in range(100):
            detector.process_frame([0.0] * 7, [0.5] * 7, [3.0] * 7)

        # Power baseline ≈ |3.0 * 0.5| = 1.5W per joint
        # Collision: torque spikes to 40 Nm while velocity is still 0.5
        # Power = 40 * 0.5 = 20W, anomaly = 20 - 1.5 = 18.5W > 15W threshold
        event = detector.process_frame([0.0] * 7, [0.5] * 7, [40.0] * 7)
        assert event is not None
        assert event.channel == "energy"


class TestVelocityModeCollision:
    """Test collision detection during velocity-mode control (VelocityFrame)."""

    def test_velocity_frame_collision_detected(self):
        from unittest.mock import MagicMock
        from onero_sdk.motion_executor import MotionExecutor, VelocityFrame

        backend = MagicMock()
        safety = MagicMock()
        safety.emergency_stopped = False
        recorder = MagicMock()
        clock = MagicMock()

        # First call: normal. Second call: collision torque spike.
        normal_state = MagicMock()
        normal_state.joints = [0.0] * 7
        normal_state.velocities = [0.5] * 7
        normal_state.torques = [5.0] * 7

        spike_state = MagicMock()
        spike_state.joints = [0.0] * 7
        spike_state.velocities = [0.5] * 7
        spike_state.torques = [50.0] * 7

        backend.command.side_effect = [normal_state, spike_state, normal_state]
        get_state = MagicMock(return_value=normal_state)
        set_state = MagicMock()

        config = CollisionConfig(
            impact_threshold=50.0, warmup_frames=0, response=CollisionResponse.STOP,
        )
        detector = CollisionDetector(7, config)
        detector._armed = True
        detector._prev_torques = np.array([5.0] * 7)
        detector._dtau_baseline = np.array([10.0] * 7)

        def gravity(q):
            return [0.0] * 7

        executor = MotionExecutor(
            backend=backend, safety=safety, recorder=recorder,
            clock=clock, get_state=get_state, set_state=set_state,
            gravity_torques_fn=gravity,
            collision_detector=detector,
        )

        frames = [
            VelocityFrame([0.5] * 7, 0.005),
            VelocityFrame([0.5] * 7, 0.005),
            VelocityFrame([0.5] * 7, 0.005),
        ]
        executor.run(iter(frames))
        # Second frame has spike — should trigger stop
        safety.trigger_emergency_stop.assert_called()
        # Should have run at most 2 backend commands (stop on 2nd)
        assert backend.command.call_count <= 2


class TestNaNGuard:
    """Verify NaN/Inf inputs are rejected without corrupting baselines."""

    def test_nan_torques_ignored(self):
        config = CollisionConfig(warmup_frames=2, control_freq=200.0)
        detector = CollisionDetector(7, config)
        base = [5.0] * 7
        vel = [1.0] * 7
        pos = [0.0] * 7

        # Warmup
        for _ in range(3):
            detector.process_frame(pos, vel, base)

        # Inject NaN — should return None and not corrupt state
        nan_torques = [float('nan')] * 7
        result = detector.process_frame(pos, vel, nan_torques)
        assert result is None

        # Normal frame after NaN — detector still works
        result = detector.process_frame(pos, vel, base)
        assert result is None  # no spike

    def test_inf_velocities_ignored(self):
        config = CollisionConfig(warmup_frames=2, control_freq=200.0)
        detector = CollisionDetector(7, config)
        base = [5.0] * 7
        vel = [1.0] * 7
        pos = [0.0] * 7

        for _ in range(3):
            detector.process_frame(pos, vel, base)

        inf_vel = [float('inf')] * 7
        result = detector.process_frame(pos, inf_vel, base)
        assert result is None

    def test_baseline_not_corrupted_by_nan(self):
        """After NaN frame, baseline should be unchanged."""
        config = CollisionConfig(warmup_frames=2, control_freq=200.0)
        detector = CollisionDetector(7, config)
        base = [5.0] * 7
        vel = [1.0] * 7
        pos = [0.0] * 7

        for _ in range(3):
            detector.process_frame(pos, vel, base)

        baseline_before = detector._dtau_baseline.copy()
        detector.process_frame(pos, vel, [float('nan')] * 7)
        assert np.array_equal(detector._dtau_baseline, baseline_before)


class TestCallbackIsolation:
    """Verify that a failing callback doesn't block the detector."""

    def test_exception_in_callback_does_not_crash(self):
        events = []

        def bad_callback(event):
            events.append(event)
            raise RuntimeError("callback exploded")

        config = CollisionConfig(
            warmup_frames=2, impact_threshold=50.0, control_freq=200.0
        )
        detector = CollisionDetector(7, config, on_collision=bad_callback)
        vel = [1.0] * 7
        pos = [0.0] * 7

        # Warmup
        for _ in range(3):
            detector.process_frame(pos, vel, [5.0] * 7)

        # Spike — callback will throw, but event should still be returned
        spike = [5.0] * 7
        spike[2] = 500.0
        event = detector.process_frame(pos, vel, spike)
        assert event is not None
        assert event.channel == "impact"
        assert len(events) == 1  # callback was called before throwing

    def test_callback_exception_does_not_prevent_next_detection(self):
        call_count = [0]

        def bad_callback(event):
            call_count[0] += 1
            raise ValueError("oops")

        config = CollisionConfig(
            warmup_frames=2, impact_threshold=50.0,
            impact_holdoff_frames=0, control_freq=200.0,
        )
        detector = CollisionDetector(7, config, on_collision=bad_callback)
        vel = [1.0] * 7
        pos = [0.0] * 7

        for _ in range(3):
            detector.process_frame(pos, vel, [5.0] * 7)

        # Two spikes in succession — callback fires each time despite exceptions
        spike = [5.0] * 7
        spike[0] = 500.0
        detector.process_frame(pos, vel, spike)
        detector.process_frame(pos, vel, [5.0] * 7)
        spike = [5.0] * 7
        spike[0] = 500.0
        detector.process_frame(pos, vel, spike)
        assert call_count[0] >= 2  # proves callback keeps working after throw


class TestResidualAtLowVelocity:
    """Verify residual channel fires even when arm is nearly static."""

    def test_residual_triggers_below_min_velocity(self):
        def fake_residual(pos, vel, tau):
            return [20.0] * 7  # large residual

        config = CollisionConfig(
            warmup_frames=2,
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=8.0,
            min_velocity=0.05,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config, residual_fn=fake_residual)
        pos = [0.0] * 7
        vel = [0.001] * 7  # well below min_velocity
        tau = [5.0] * 7

        # Warmup
        for _ in range(3):
            detector.process_frame(pos, vel, tau)

        # Should still trigger via residual despite low velocity
        event = detector.process_frame(pos, vel, tau)
        assert event is not None
        assert event.channel == "residual"

    def test_impact_energy_gated_at_low_velocity(self):
        """Impact/Energy should NOT fire at low velocity, only residual."""
        def fake_residual(pos, vel, tau):
            return [0.0] * 7  # no residual

        config = CollisionConfig(
            warmup_frames=2,
            impact_enabled=True,
            energy_enabled=True,
            residual_enabled=True,
            residual_threshold=8.0,
            min_velocity=0.05,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config, residual_fn=fake_residual)
        pos = [0.0] * 7
        vel = [0.001] * 7  # below min_velocity
        tau = [5.0] * 7

        for _ in range(3):
            detector.process_frame(pos, vel, tau)

        # Large torque spike — but velocity too low for impact/energy
        spike = [5.0] * 7
        spike[0] = 500.0
        event = detector.process_frame(pos, vel, spike)
        assert event is None  # impact gated, residual says 0


class TestParameterValidation:
    """Verify constructor rejects invalid parameters."""

    def test_zero_joints_raises(self):
        with pytest.raises(ValueError, match="num_joints must be positive"):
            CollisionDetector(0, CollisionConfig())

    def test_negative_joints_raises(self):
        with pytest.raises(ValueError, match="num_joints must be positive"):
            CollisionDetector(-1, CollisionConfig())

    def test_zero_control_freq_raises(self):
        with pytest.raises(ValueError, match="control_freq must be finite and positive"):
            CollisionConfig(control_freq=0.0)

    def test_negative_control_freq_raises(self):
        with pytest.raises(ValueError, match="control_freq must be finite and positive"):
            CollisionConfig(control_freq=-100.0)

    def test_string_response_raises(self):
        with pytest.raises(TypeError, match="response must be a CollisionResponse enum"):
            CollisionConfig(response="stop")

    def test_nan_threshold_raises(self):
        with pytest.raises(ValueError, match="impact_threshold must be finite"):
            CollisionConfig(impact_threshold=float('nan'))

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="impact_baseline_alpha must be in"):
            CollisionConfig(impact_baseline_alpha=1.5)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup_frames must be non-negative"):
            CollisionConfig(warmup_frames=-1)

    def test_negative_holdoff_raises(self):
        with pytest.raises(ValueError, match="impact_holdoff_frames must be non-negative"):
            CollisionConfig(impact_holdoff_frames=-5)


class TestWarmupZero:
    """Verify warmup_frames=0 fires on the very first frame."""

    def test_energy_fires_on_first_frame(self):
        config = CollisionConfig(
            warmup_frames=0,
            impact_enabled=False,
            energy_enabled=True,
            energy_threshold=10.0,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config)
        pos = [0.0] * 7
        vel = [1.0] * 7
        # Large power = |tau * vel| = 50.0 >> threshold 10.0
        tau = [50.0] * 7

        event = detector.process_frame(pos, vel, tau)
        assert event is not None
        assert event.channel == "energy"

    def test_residual_fires_on_first_frame(self):
        def fake_residual(pos, vel, tau):
            return [20.0] * 7

        config = CollisionConfig(
            warmup_frames=0,
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=8.0,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config, residual_fn=fake_residual)
        pos = [0.0] * 7
        vel = [0.0] * 7
        tau = [5.0] * 7

        event = detector.process_frame(pos, vel, tau)
        assert event is not None
        assert event.channel == "residual"


class TestShapeValidation:
    """Verify wrong array shapes are silently rejected."""

    def test_wrong_torque_length_ignored(self):
        config = CollisionConfig(warmup_frames=0, control_freq=200.0)
        detector = CollisionDetector(7, config)
        pos = [0.0] * 7
        vel = [1.0] * 7
        tau_wrong = [50.0] * 5  # only 5 instead of 7

        result = detector.process_frame(pos, vel, tau_wrong)
        assert result is None

    def test_wrong_position_length_ignored(self):
        config = CollisionConfig(warmup_frames=0, control_freq=200.0)
        detector = CollisionDetector(7, config)
        pos = [0.0] * 3  # only 3 instead of 7
        vel = [1.0] * 7
        tau = [5.0] * 7

        result = detector.process_frame(pos, vel, tau)
        assert result is None

    def test_residual_nan_output_ignored(self):
        def bad_residual(pos, vel, tau):
            return [float('nan')] * 7

        config = CollisionConfig(
            warmup_frames=0,
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=1.0,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config, residual_fn=bad_residual)
        pos = [0.0] * 7
        vel = [0.0] * 7
        tau = [5.0] * 7

        result = detector.process_frame(pos, vel, tau)
        assert result is None

    def test_residual_exception_ignored(self):
        def exploding_residual(pos, vel, tau):
            raise RuntimeError("model crashed")

        config = CollisionConfig(
            warmup_frames=0,
            impact_enabled=False,
            energy_enabled=False,
            residual_enabled=True,
            residual_threshold=1.0,
            control_freq=200.0,
        )
        detector = CollisionDetector(7, config, residual_fn=exploding_residual)
        pos = [0.0] * 7
        vel = [0.0] * 7
        tau = [5.0] * 7

        result = detector.process_frame(pos, vel, tau)
        assert result is None


class TestArmIntegration:
    """Test enable/disable collision detection on Arm with MujocoBackend."""

    def test_enable_disable_with_mujoco(self):
        from onero_sdk import Arm
        try:
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("mujoco not installed")

        backend = MujocoBackend(model_name="A1R")
        arm = Arm("A1R", backend=backend)
        arm.enable()
        try:
            assert arm.last_collision is None
            assert arm._collision_detector is None

            arm.enable_collision_detection()
            assert arm._collision_detector is not None
            assert arm._collision_detector._config.impact_enabled is True
            assert arm._collision_detector._config.control_freq == arm.control_freq

            arm.disable_collision_detection()
            assert arm._collision_detector is None
            assert arm.last_collision is None
        finally:
            arm.disable()

    def test_enable_with_custom_config(self):
        from onero_sdk import Arm
        try:
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("mujoco not installed")

        backend = MujocoBackend(model_name="A1R")
        arm = Arm("A1R", backend=backend)
        arm.enable()
        try:
            config = CollisionConfig(
                impact_threshold=30.0, energy_enabled=False,
                response=CollisionResponse.CALLBACK,
            )
            events = []
            arm.enable_collision_detection(
                config=config, on_collision=lambda e: events.append(e)
            )
            assert arm._collision_detector._config.impact_threshold == 30.0
            assert arm._collision_detector._config.energy_enabled is False
            assert arm._collision_detector._config.control_freq == arm.control_freq
        finally:
            arm.disable()

    def test_motion_with_collision_detection_enabled(self):
        """A normal motion should complete without false collision triggers."""
        from onero_sdk import Arm
        try:
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("mujoco not installed")

        backend = MujocoBackend(model_name="A1R")
        arm = Arm("A1R", backend=backend)
        arm.enable()
        try:
            arm.enable_collision_detection()
            # Simple joint move — should NOT trigger collision
            arm.movej([0.1, -0.2, 0.1, -0.3, 0.0, 0.1, 0.0], duration=1.0)
            assert arm.last_collision is None
            assert not arm._safety.emergency_stopped
        finally:
            arm.disable()
