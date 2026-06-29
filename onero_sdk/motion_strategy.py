"""
Motion strategies — frame generators extracted from arm.py.

Each strategy encapsulates the interpolation logic for one motion type,
yielding MotionFrame or VelocityFrame objects for the MotionExecutor.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .motion_executor import MotionFrame, VelocityFrame


@dataclass
class MotionContext:
    """Shared parameters for all motion strategies."""
    start_joints: list
    num_joints: int
    dt: float
    steps: int
    compute_gains: Optional[Callable] = None
    fixed_kp: Optional[float] = None
    fixed_kd: Optional[float] = None

    def gains_for(self, q):
        if self.compute_gains is not None:
            return self.compute_gains(q)
        return [self.fixed_kp] * self.num_joints, [self.fixed_kd] * self.num_joints


def cosine_joint_frames(ctx: MotionContext, target_joints: list, duration: float):
    n = ctx.num_joints
    start = ctx.start_joints
    for step in range(ctx.steps + 1):
        t = step / ctx.steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        vel_alpha = 0.5 * math.pi * math.sin(math.pi * t) / duration
        positions = [start[i] + alpha * (target_joints[i] - start[i]) for i in range(n)]
        velocities = [vel_alpha * (target_joints[i] - start[i]) for i in range(n)]
        kps, kds = ctx.gains_for(positions)
        yield MotionFrame(
            positions=positions, velocities=velocities,
            kps=kps, kds=kds, torques=[0.0] * n, dt=ctx.dt,
        )


def trapezoidal_joint_frames(ctx: MotionContext, joint_profiles: list):
    n = ctx.num_joints
    num_frames = len(joint_profiles[0]) if joint_profiles else ctx.steps + 1
    for step in range(num_frames):
        positions = [joint_profiles[i][min(step, len(joint_profiles[i])-1)].position for i in range(n)]
        velocities = [joint_profiles[i][min(step, len(joint_profiles[i])-1)].velocity for i in range(n)]
        kps, kds = ctx.gains_for(positions)
        yield MotionFrame(
            positions=positions, velocities=velocities,
            kps=kps, kds=kds, torques=[0.0] * n, dt=ctx.dt,
        )


def scurve_joint_frames(ctx: MotionContext, profiles: list, actual_steps: int):
    n = ctx.num_joints
    start = ctx.start_joints
    dt = ctx.dt
    for step in range(actual_steps + 1):
        t = step * dt
        positions = []
        velocities = []
        for i in range(n):
            pt = profiles[i].evaluate(min(t, profiles[i].duration))
            positions.append(start[i] + pt.position)
            velocities.append(pt.velocity)
        kps, kds = ctx.gains_for(positions)
        yield MotionFrame(
            positions=positions, velocities=velocities,
            kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
        )


def cartesian_linear_frames(ctx: MotionContext, start_pos, start_rot, target_pos, target_rot,
                            ik_solver, slerp_fn):
    n = ctx.num_joints
    dt = ctx.dt
    pos_delta = target_pos - start_pos
    prev_joints = None
    for step in range(ctx.steps + 1):
        t = step / ctx.steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        interp_pos = start_pos + alpha * pos_delta
        interp_rot = slerp_fn(start_rot, target_rot, alpha)

        q = ik_solver.solve(interp_pos, interp_rot, dt=dt)
        joints = q.tolist() if hasattr(q, 'tolist') else list(q)

        if step == 0 or step == ctx.steps:
            velocities = [0.0] * n
        elif prev_joints is not None:
            velocities = [(joints[i] - prev_joints[i]) / dt for i in range(n)]
        else:
            velocities = [0.0] * n
        prev_joints = list(joints)

        kps, kds = ctx.gains_for(joints)
        yield MotionFrame(
            positions=joints, velocities=velocities,
            kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
        )


def circular_arc_frames(ctx: MotionContext, center, v0, normal, total_angle,
                        r0, r1, r2, ik_solver, slerp_fn, rotate_vector_fn):
    n = ctx.num_joints
    dt = ctx.dt
    prev_joints = None
    for step in range(ctx.steps + 1):
        t = step / ctx.steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        angle = alpha * total_angle

        interp_pos = center + rotate_vector_fn(v0, normal, angle)

        R_01 = slerp_fn(r0, r1, alpha)
        R_12 = slerp_fn(r1, r2, alpha)
        interp_rot = slerp_fn(R_01, R_12, alpha)

        q = ik_solver.solve(interp_pos, interp_rot, dt=dt)
        joints = q.tolist() if hasattr(q, 'tolist') else list(q)

        if prev_joints is not None:
            velocities = [(joints[i] - prev_joints[i]) / dt for i in range(n)]
        else:
            velocities = [0.0] * n
        prev_joints = list(joints)

        kps, kds = ctx.gains_for(joints)
        yield MotionFrame(
            positions=joints, velocities=velocities,
            kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
        )


def velocity_joint_frames(target_vel: list, ramp_steps: int, hold_steps: int, dt: float):
    for step in range(ramp_steps):
        alpha = (step + 1) / max(ramp_steps, 1)
        yield VelocityFrame(velocities=[alpha * v for v in target_vel], dt=dt)
    for step in range(hold_steps):
        yield VelocityFrame(velocities=target_vel, dt=dt)
    for step in range(ramp_steps):
        alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
        yield VelocityFrame(velocities=[alpha * v for v in target_vel], dt=dt)


def velocity_cartesian_frames(twist, ramp_steps: int, hold_steps: int, dt: float,
                              get_joints_fn, jacobian_solve_fn,
                              twist_fn=None):
    """twist_fn: if provided, called each tick to get current base-frame twist (for dynamic frames)."""
    for step in range(ramp_steps):
        alpha = (step + 1) / max(ramp_steps, 1)
        current_twist = twist_fn() if twist_fn else twist
        joint_vel = jacobian_solve_fn(get_joints_fn(), alpha * current_twist)
        yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)
    for step in range(hold_steps):
        current_twist = twist_fn() if twist_fn else twist
        joint_vel = jacobian_solve_fn(get_joints_fn(), current_twist)
        yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)
    for step in range(ramp_steps):
        alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
        current_twist = twist_fn() if twist_fn else twist
        joint_vel = jacobian_solve_fn(get_joints_fn(), alpha * current_twist)
        yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)
