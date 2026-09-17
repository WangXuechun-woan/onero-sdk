"""
Motion execution engine — consumes frame generators, applies timing, and sends.

The MotionExecutor is the single place where the control loop runs:
check e-stop → compute gravity comp → send command → record → tick clock.
Safety clamping is handled transparently by SafeBackend at the backend seam.
"""

from dataclasses import dataclass
from typing import Callable, Generator, Optional

from .backend import ArmBackend, ArmState
from .clock import ControlClock
from .safety import SafetyMonitor
from .trajectory import Recorder


@dataclass
class MotionFrame:
    """A single frame of commanded motion."""
    positions: list[float]
    velocities: list[float]
    kps: list[float]
    kds: list[float]
    torques: list[float]
    dt: float


@dataclass
class VelocityFrame:
    """A single frame of velocity-mode motion."""
    velocities: list[float]
    dt: float


FrameGenerator = Generator[MotionFrame | VelocityFrame, None, None]


class MotionExecutor:
    """
    Executes a stream of motion frames with recording and timing.

    All planned motions feed frames into this executor. The executor owns:
    - Emergency stop checking
    - Gravity compensation (added to all frames as feedforward)
    - Backend command dispatch
    - State recording
    - Deadline-based timing (ControlClock)

    Safety clamping is handled by SafeBackend (transparent at the backend seam).
    """

    def __init__(self, backend: ArmBackend, safety: SafetyMonitor,
                 recorder: Recorder, clock: ControlClock,
                 get_state: Callable[[], ArmState],
                 set_state: Callable[[ArmState], None],
                 gravity_torques_fn: Optional[Callable[[list[float]], list[float]]] = None,
                 collision_detector=None,
                 velocity_gains_fn: Optional[
                     Callable[[list[float]], tuple[list[float], list[float]]]
                 ] = None,
                 velocity_kp: float = 5.0,
                 velocity_kd: float = 0.5,
                 velocity_position_lead: float = 0.1):
        self._backend = backend
        self._safety = safety
        self._recorder = recorder
        self._clock = clock
        self._get_state = get_state
        self._set_state = set_state
        self._gravity_torques_fn = gravity_torques_fn
        self._collision_detector = collision_detector
        self._velocity_gains_fn = velocity_gains_fn
        self._velocity_kp = velocity_kp
        self._velocity_kd = velocity_kd
        self._velocity_position_lead = velocity_position_lead
        if collision_detector is not None:
            from .collision import CollisionResponse
            self._collision_response_stop = (
                CollisionResponse.STOP, CollisionResponse.BOTH,
            )
        else:
            self._collision_response_stop = ()

    def run(self, frames: FrameGenerator) -> ArmState:
        """
        Execute a motion defined by a frame generator.

        Consumes frames until the generator is exhausted or emergency stop fires.
        Gravity compensation torques are computed each frame from current joint
        positions and added to the commanded torques.
        Returns the final ArmState.
        """
        self._clock.reset()
        state = self._get_state()
        # Velocity-mode commands use a bounded integrated position reference.
        # This avoids the noisy kp=0/kd=max pure-velocity loop while retaining
        # gravity feed-forward and a small, continuously moving position lead.
        velocity_q_ref = list(state.joints)

        if self._collision_detector is not None:
            self._collision_detector.reset()

        for frame in frames:
            if self._safety.emergency_stopped:
                break

            if isinstance(frame, VelocityFrame):
                safe_vel = self._safety.clamp_velocity(
                    frame.velocities, positions=list(state.joints))
                velocity_q_ref = [
                    q + v * frame.dt for q, v in zip(velocity_q_ref, safe_vel)
                ]
                # Never let the integrated reference run far ahead of measured
                # feedback if a joint is blocked or tracking poorly.
                lead = self._velocity_position_lead
                velocity_q_ref = [
                    max(measured - lead, min(measured + lead, reference))
                    for measured, reference in zip(state.joints, velocity_q_ref)
                ]
                velocity_q_ref = self._safety.clamp_joint_limits(velocity_q_ref)
                if self._gravity_torques_fn:
                    g = self._gravity_torques_fn(list(state.joints))
                    n = len(state.joints)
                    if self._velocity_gains_fn is not None:
                        kps, kds = self._velocity_gains_fn(list(state.joints))
                    else:
                        kps = [self._velocity_kp] * n
                        kds = [self._velocity_kd] * n
                    state = self._backend.command(
                        positions=velocity_q_ref,
                        velocities=safe_vel,
                        kps=kps,
                        kds=kds,
                        torques=g,
                    )
                else:
                    state = self._backend.command_velocity(safe_vel)
            else:
                if self._gravity_torques_fn:
                    g = self._gravity_torques_fn(list(state.joints))
                    torques = [t + gt for t, gt in zip(frame.torques, g)]
                else:
                    torques = frame.torques
                state = self._backend.command(
                    positions=frame.positions,
                    velocities=frame.velocities,
                    kps=frame.kps,
                    kds=frame.kds,
                    torques=torques,
                )

            if self._collision_detector is not None:
                event = self._collision_detector.process_frame(
                    state.joints, state.velocities, state.torques)
                if event is not None and self._collision_detector.response in self._collision_response_stop:
                    self._safety.trigger_emergency_stop(
                        event.joint_index,
                        f"Collision detected ({event.channel}): joint {event.joint_index}, "
                        f"magnitude {event.magnitude:.1f} > threshold {event.threshold:.1f}",
                    )
                    self._set_state(state)
                    break

            self._set_state(state)
            self._recorder.record_frame(
                state.joints, state.velocities, state.torques)
            self._clock.tick(frame.dt)

        return state
