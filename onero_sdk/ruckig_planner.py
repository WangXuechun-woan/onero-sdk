"""
Ruckig integration — online jerk-limited trajectory generation.

Ruckig generates time-optimal trajectories with jerk constraints in real-time.
Unlike pre-computed profiles, Ruckig can change targets mid-motion and still
produce smooth, jerk-limited transitions. Used by Franka Robotics.

Key features:
- Time-optimal: fastest trajectory within velocity/acceleration/jerk limits
- Online: can update target at every control cycle
- Multi-DOF: handles coupled joint constraints

Install: pip install onero-sdk[trajectory]  (includes ruckig)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ruckig import InputParameter, OutputParameter, Ruckig, Result
except ImportError:
    Ruckig = None


@dataclass
class RuckigState:
    """Current trajectory state from Ruckig."""
    position: list[float]
    velocity: list[float]
    acceleration: list[float]
    duration: float
    finished: bool


class RuckigPlanner:
    """
    Online jerk-limited trajectory generation using Ruckig.

    Generates time-optimal trajectories that respect per-joint
    velocity, acceleration, and jerk limits. Can update targets
    mid-motion for reactive control.

    Usage:
        planner = RuckigPlanner(num_joints=7)
        planner.set_limits(v_max=[2.0]*7, a_max=[5.0]*7, j_max=[20.0]*7)

        # One-shot: compute full trajectory
        frames = planner.plan(start, target, dt=0.01)

        # Online: step-by-step (can change target anytime)
        planner.set_state(current_pos, current_vel, current_acc)
        planner.set_target(target_pos)
        while not planner.step(dt):
            state = planner.get_state()
            arm.servoj(state.position)
    """

    def __init__(self, num_joints: int):
        if Ruckig is None:
            raise ImportError(
                "RuckigPlanner requires ruckig. "
                "Install with: pip install ruckig"
            )
        self._dof = num_joints
        self._otg = Ruckig(num_joints)
        self._input = InputParameter(num_joints)
        self._output = OutputParameter(num_joints)
        self._initialized = False

        # Default limits
        self._input.max_velocity = [2.0] * num_joints
        self._input.max_acceleration = [5.0] * num_joints
        self._input.max_jerk = [20.0] * num_joints

    @property
    def num_joints(self) -> int:
        return self._dof

    def set_limits(self, v_max: list[float],
                   a_max: list[float],
                   j_max: list[float]) -> None:
        """Set per-joint velocity, acceleration, and jerk limits."""
        self._input.max_velocity = list(v_max)
        self._input.max_acceleration = list(a_max)
        self._input.max_jerk = list(j_max)

    def set_state(self, position: list[float],
                  velocity: Optional[list[float]] = None,
                  acceleration: Optional[list[float]] = None) -> None:
        """Set current robot state (call before planning)."""
        self._input.current_position = list(position)
        self._input.current_velocity = list(velocity) if velocity else [0.0] * self._dof
        self._input.current_acceleration = list(acceleration) if acceleration else [0.0] * self._dof
        self._initialized = True

    def set_target(self, position: list[float],
                   velocity: Optional[list[float]] = None,
                   acceleration: Optional[list[float]] = None) -> None:
        """Set target state."""
        self._input.target_position = list(position)
        self._input.target_velocity = list(velocity) if velocity else [0.0] * self._dof
        self._input.target_acceleration = list(acceleration) if acceleration else [0.0] * self._dof

    def plan(self, start: list[float], target: list[float],
             dt: float = 0.01,
             start_vel: Optional[list[float]] = None,
             start_acc: Optional[list[float]] = None) -> list[RuckigState]:
        """
        Compute a complete trajectory from start to target.

        Args:
            start: Start joint positions.
            target: Target joint positions.
            dt: Sample interval.
            start_vel: Initial velocities (default zeros).
            start_acc: Initial accelerations (default zeros).

        Returns:
            List of RuckigState at each dt interval.
        """
        self.set_state(start, start_vel, start_acc)
        self.set_target(target)

        # Create a fresh Ruckig instance for offline computation
        otg = Ruckig(self._dof, dt)
        inp = InputParameter(self._dof)
        out = OutputParameter(self._dof)

        inp.current_position = list(start)
        inp.current_velocity = list(start_vel) if start_vel else [0.0] * self._dof
        inp.current_acceleration = list(start_acc) if start_acc else [0.0] * self._dof
        inp.target_position = list(target)
        inp.target_velocity = [0.0] * self._dof
        inp.target_acceleration = [0.0] * self._dof
        inp.max_velocity = list(self._input.max_velocity)
        inp.max_acceleration = list(self._input.max_acceleration)
        inp.max_jerk = list(self._input.max_jerk)

        frames = []
        total_time = 0.0
        first_result = otg.calculate(inp, out)
        if first_result == Result.ErrorInvalidInput:
            raise ValueError("Ruckig: invalid input parameters")

        while True:
            result = otg.update(inp, out)
            frames.append(RuckigState(
                position=list(out.new_position),
                velocity=list(out.new_velocity),
                acceleration=list(out.new_acceleration),
                duration=total_time,
                finished=(result == Result.Finished),
            ))
            if result == Result.Finished:
                break
            out.pass_to_input(inp)
            total_time += dt

            if total_time > 60.0:
                break

        return frames

    def step(self, dt: float) -> RuckigState:
        """
        Compute one step of the online trajectory.

        Call this at control frequency. Returns the next state.
        Can call set_target() between steps to change the goal.

        Returns:
            RuckigState with next position/velocity/acceleration.
        """
        if not self._initialized:
            raise RuntimeError("Call set_state() before stepping")

        otg = Ruckig(self._dof, dt)
        result = otg.update(self._input, self._output)
        self._output.pass_to_input(self._input)

        return RuckigState(
            position=list(self._output.new_position),
            velocity=list(self._output.new_velocity),
            acceleration=list(self._output.new_acceleration),
            duration=0.0,
            finished=(result == Result.Finished),
        )

    def to_trajectory_frames(self, states: list[RuckigState],
                              dt: float) -> list:
        """Convert RuckigState list to TrajectoryFrame list for execute_trajectory()."""
        from .trajectory import TrajectoryFrame
        return [
            TrajectoryFrame(
                joints=s.position,
                velocities=s.velocity,
                dt=dt,
            )
            for s in states
        ]
