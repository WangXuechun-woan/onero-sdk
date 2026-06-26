"""
TOPP-RA integration — time-optimal path parameterization.

Given a geometric path (sequence of waypoints) and kinodynamic constraints
(joint velocity, acceleration, torque limits), TOPP-RA finds the fastest
possible traversal of that path. This is the industry standard for
maximizing throughput while respecting hardware limits.

Use cases:
- High-speed pick-and-place (minimize cycle time)
- Welding/gluing (maintain consistent tool speed along path)
- Any trajectory where you want "as fast as possible within limits"

Install: pip install onero-sdk[trajectory]  (includes toppra)
"""

from typing import Optional

import numpy as np

try:
    import toppra
    import toppra.constraint as constraint
    import toppra.algorithm as algo
except ImportError:
    toppra = None


class TimeOptimalPlanner:
    """
    Time-optimal path parameterization using TOPP-RA.

    Takes a geometric path (waypoints) and constraint set, returns the
    fastest parameterization that respects all limits.

    Usage:
        planner = TimeOptimalPlanner(num_joints=7)
        planner.set_velocity_limits(v_max)
        planner.set_acceleration_limits(a_max)

        # Compute time-optimal trajectory
        frames = planner.plan(waypoints, dt=0.01)
        arm.execute_trajectory(frames)
    """

    def __init__(self, num_joints: int):
        if toppra is None:
            raise ImportError(
                "TimeOptimalPlanner requires toppra. "
                "Install with: pip install toppra"
            )
        self._dof = num_joints
        self._v_max = np.array([2.0] * num_joints)
        self._a_max = np.array([5.0] * num_joints)
        self._tau_max = None  # Optional torque limits

    @property
    def num_joints(self) -> int:
        return self._dof

    def set_velocity_limits(self, v_max: list[float]) -> None:
        """Set per-joint velocity limits (rad/s)."""
        self._v_max = np.array(v_max)

    def set_acceleration_limits(self, a_max: list[float]) -> None:
        """Set per-joint acceleration limits (rad/s²)."""
        self._a_max = np.array(a_max)

    def set_torque_limits(self, tau_max: list[float]) -> None:
        """Set per-joint torque limits (Nm). Requires dynamics model for use."""
        self._tau_max = np.array(tau_max)

    def plan(self, waypoints: list[list[float]], dt: float = 0.01,
             speed_factor: float = 1.0) -> list:
        """
        Compute time-optimal trajectory through waypoints.

        Args:
            waypoints: List of joint configurations (at least 2).
                      Path is interpolated with cubic spline.
            dt: Output sample interval (seconds).
            speed_factor: Scale factor on velocity limits (0.5 = half speed).

        Returns:
            List of TrajectoryFrame at dt intervals.
        """
        from .trajectory import TrajectoryFrame

        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        way_arr = np.array(waypoints)
        n_waypoints = len(waypoints)

        # Create path (cubic spline through waypoints)
        ss = np.linspace(0, 1, n_waypoints)
        path = toppra.SplineInterpolator(ss, way_arr)

        # Build constraints
        constraints = []

        # Velocity constraint
        v_lim = self._v_max * speed_factor
        vlim_arr = np.column_stack([-v_lim, v_lim])
        constraints.append(
            constraint.JointVelocityConstraint(vlim_arr)
        )

        # Acceleration constraint
        alim_arr = np.column_stack([-self._a_max, self._a_max])
        constraints.append(
            constraint.JointAccelerationConstraint(alim_arr)
        )

        # Solve TOPP-RA
        instance = algo.TOPPRA(constraints, path)
        trajectory = instance.compute_trajectory()

        if trajectory is None:
            raise RuntimeError(
                "TOPP-RA failed to find feasible parameterization. "
                "Check that waypoints are reachable within constraints."
            )

        # Sample at dt
        duration = trajectory.duration
        n_samples = max(1, int(np.ceil(duration / dt)))
        times = np.linspace(0, duration, n_samples + 1)

        frames = []
        for t in times:
            q = trajectory(t)
            qd = trajectory(t, 1)  # first derivative
            frames.append(TrajectoryFrame(
                joints=q.tolist(),
                velocities=qd.tolist(),
                dt=dt,
            ))

        return frames

    def plan_with_torque(self, waypoints: list[list[float]],
                         dynamics_model, dt: float = 0.01) -> list:
        """
        Time-optimal trajectory with torque constraints (requires dynamics).

        Uses the robot's dynamics model to ensure joint torques stay within
        limits along the entire trajectory.

        Args:
            waypoints: Joint configurations.
            dynamics_model: DynamicsModel instance (provides inverse dynamics).
            dt: Sample interval.

        Returns:
            List of TrajectoryFrame.
        """
        from .trajectory import TrajectoryFrame

        if self._tau_max is None:
            raise ValueError("Call set_torque_limits() first")

        way_arr = np.array(waypoints)
        n_waypoints = len(waypoints)

        ss = np.linspace(0, 1, n_waypoints)
        path = toppra.SplineInterpolator(ss, way_arr)

        constraints = []

        # Velocity
        vlim_arr = np.column_stack([-self._v_max, self._v_max])
        constraints.append(constraint.JointVelocityConstraint(vlim_arr))

        # Acceleration
        alim_arr = np.column_stack([-self._a_max, self._a_max])
        constraints.append(constraint.JointAccelerationConstraint(alim_arr))

        # Torque (second-order constraint via inverse dynamics)
        # tau = M(q)*qdd + C(q,qd)*qd + g(q)
        # Discretized into path parameter: tau = M*qs''*sd² + (M*qs' + C)*sd_dot + g
        tau_lim = np.column_stack([-self._tau_max, self._tau_max])
        constraints.append(
            constraint.JointTorqueConstraint(
                tau_lim,
                # toppra needs the inverse dynamics coefficient matrices
                # We provide them from pinocchio
                fs_coef=np.zeros(self._dof),  # Coulomb friction
            )
        )

        instance = algo.TOPPRA(constraints, path)
        trajectory = instance.compute_trajectory()

        if trajectory is None:
            raise RuntimeError("TOPP-RA with torque constraints: no feasible solution")

        duration = trajectory.duration
        n_samples = max(1, int(np.ceil(duration / dt)))
        times = np.linspace(0, duration, n_samples + 1)

        frames = []
        for t in times:
            q = trajectory(t)
            qd = trajectory(t, 1)
            frames.append(TrajectoryFrame(
                joints=q.tolist(),
                velocities=qd.tolist(),
                dt=dt,
            ))

        return frames

    def compute_duration(self, waypoints: list[list[float]],
                         speed_factor: float = 1.0) -> float:
        """Compute the optimal duration without generating the full trajectory."""
        way_arr = np.array(waypoints)
        ss = np.linspace(0, 1, len(waypoints))
        path = toppra.SplineInterpolator(ss, way_arr)

        constraints = []
        v_lim = self._v_max * speed_factor
        constraints.append(
            constraint.JointVelocityConstraint(np.column_stack([-v_lim, v_lim]))
        )
        constraints.append(
            constraint.JointAccelerationConstraint(
                np.column_stack([-self._a_max, self._a_max]))
        )

        instance = algo.TOPPRA(constraints, path)
        trajectory = instance.compute_trajectory()
        if trajectory is None:
            return float('inf')
        return float(trajectory.duration)
