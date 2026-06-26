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
                 gravity_torques_fn: Optional[Callable[[list[float]], list[float]]] = None):
        self._backend = backend
        self._safety = safety
        self._recorder = recorder
        self._clock = clock
        self._get_state = get_state
        self._set_state = set_state
        self._gravity_torques_fn = gravity_torques_fn

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

        for frame in frames:
            if self._safety.emergency_stopped:
                break

            if isinstance(frame, VelocityFrame):
                if self._gravity_torques_fn:
                    g = self._gravity_torques_fn(list(state.joints))
                    n = len(state.joints)
                    state = self._backend.command(
                        positions=list(state.joints),
                        velocities=frame.velocities,
                        kps=[0.0] * n,
                        kds=[5.0] * n,
                        torques=g,
                    )
                else:
                    state = self._backend.command_velocity(frame.velocities)
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

            self._set_state(state)
            self._recorder.record_frame(
                state.joints, state.velocities, state.torques)
            self._clock.tick(frame.dt)

        return state
