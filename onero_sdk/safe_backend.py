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

    command() enforces position clamping + rate limiting + torque clamping.
    Pass hold=True to skip rate limiting (joint-limit + torque clamping only) —
    use for position-hold after velocity motion, force_mode, or freedrive exit.

    All paths enforce torque limits unconditionally.
    All paths refuse to forward commands when emergency stop is active.
    """

    def __init__(self, inner: ArmBackend, safety: SafetyMonitor,
                 torque_limits: list[float] | None = None):
        self._inner = inner
        self._safety = safety
        self._torque_limits = torque_limits

    def _check_emergency_stop(self) -> None:
        if self._safety.emergency_stopped:
            from .exceptions import OneroHardwareError
            raise OneroHardwareError(
                joint_index=-1, error_code=0,
                description="Emergency stop active — call reset/re-enable before commanding",
            )

    @property
    def num_joints(self) -> int:
        return self._inner.num_joints

    def enable(self) -> ArmState:
        state = self._inner.enable()
        self._safety.reset(state.joints)
        return state

    def disable(self) -> None:
        self._inner.disable()

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float], hold: bool = False) -> ArmState:
        self._check_emergency_stop()
        # Velocity is a first-class safety boundary even for MIT impedance
        # commands.  speedj/speedl use command() so limiting only the native
        # command_velocity() path would leave their feed-forward velocity
        # completely unchecked.
        safe_vel = self._safety.clamp_velocity(velocities, positions=positions)
        if hold:
            clamped_pos = self._safety.clamp_joint_limits(positions)
            cmd_vel = safe_vel
        else:
            clamped_pos = self._safety.clamp_target(positions)
            cmd_vel = safe_vel
        clamped_tau = self._clamp_torques(torques)
        state = self._inner.command(
            positions=clamped_pos,
            velocities=cmd_vel,
            kps=kps, kds=kds, torques=clamped_tau,
        )
        self._safety.check_tracking_error(state.joints, clamped_pos)
        return state

    def command_hold(self, positions: list[float], velocities: list[float],
                     kps: list[float], kds: list[float],
                     torques: list[float]) -> ArmState:
        """
        Command with joint-limit + torque clamping but NO rate limiting.

        Use when holding current position after velocity motion, or in
        force_mode/freedrive where the arm is already at the target.
        """
        return self.command(positions, velocities, kps, kds, torques, hold=True)

    def command_velocity(self, velocities: list[float]) -> ArmState:
        self._check_emergency_stop()
        clamped = self._safety.clamp_velocity(velocities)
        state = self._inner.command_velocity(clamped)
        self._safety.reconcile_position(state.joints)
        return state

    def command_torque(self, torques: list[float]) -> ArmState:
        self._check_emergency_stop()
        clamped = self._clamp_torques(torques)
        state = self._inner.command_torque(clamped)
        self._safety.reconcile_position(state.joints)
        return state

    def _clamp_torques(self, torques: list[float]) -> list[float]:
        if not self._safety.enabled or self._torque_limits is None:
            return torques
        import math
        return [
            max(-lim, min(lim, t)) if math.isfinite(t) else 0.0
            for t, lim in zip(torques, self._torque_limits)
        ]

    def get_state(self) -> ArmState:
        return self._inner.get_state()

    def close(self, disable: bool = False) -> None:
        self._inner.close(disable=disable)
