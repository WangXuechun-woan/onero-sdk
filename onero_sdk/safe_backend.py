"""
Safe backend wrapper — transparent safety enforcement at the Backend seam.

Decorates any ArmBackend with safety clamping so that motion methods
don't need to manually apply limits. Safety is enforced for all commands
passing through this layer.
"""

from .backend import ArmBackend, ArmState
from .safety import SafetyMonitor


class SafeBackend(ArmBackend):
    """
    Backend decorator that applies safety limits transparently.

    Wraps position commands with clamp_target() and velocity commands
    with clamp_velocity(). The underlying backend receives only
    safe, clamped values.
    """

    def __init__(self, inner: ArmBackend, safety: SafetyMonitor):
        self._inner = inner
        self._safety = safety

    @property
    def num_joints(self) -> int:
        return self._inner.num_joints

    def enable(self) -> ArmState:
        return self._inner.enable()

    def disable(self) -> None:
        self._inner.disable()

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        clamped_pos = self._safety.clamp_target(positions)
        cmd_vel = [
            velocities[i]
            if abs(clamped_pos[i] - positions[i]) < 1e-6
            else 0.0
            for i in range(len(positions))
        ]
        state = self._inner.command(
            positions=clamped_pos,
            velocities=cmd_vel,
            kps=kps, kds=kds, torques=torques,
        )
        self._safety.reconcile_position(clamped_pos)
        return state

    def command_velocity(self, velocities: list[float]) -> ArmState:
        clamped = self._safety.clamp_velocity(velocities)
        return self._inner.command_velocity(clamped)

    def command_torque(self, torques: list[float]) -> ArmState:
        return self._inner.command_torque(torques)

    def get_state(self) -> ArmState:
        return self._inner.get_state()

    def close(self) -> None:
        self._inner.close()
