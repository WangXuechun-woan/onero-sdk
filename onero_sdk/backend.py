"""
Backend abstraction — decouples arm control from transport.

Implementations:
- CanBackend: real hardware via SocketCAN
- (future) MujocoBackend: simulation via onero_sdk[sim]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArmState:
    """Snapshot of arm state at a point in time."""
    joints: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    torques: list[float] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def num_joints(self) -> int:
        return len(self.joints)


class ArmBackend(ABC):
    """
    Abstract interface for arm control backends.

    Operates at the arm level (all joints simultaneously).
    """

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of joints this backend controls."""
        ...

    @abstractmethod
    def enable(self) -> ArmState:
        """
        Enable all joints. Auto-clears errors and retries.
        Returns initial state on success.
        Raises OneroHardwareError on failure (all-or-nothing).
        """
        ...

    @abstractmethod
    def disable(self) -> None:
        """Disable all joints."""
        ...

    @abstractmethod
    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        """
        Send MIT impedance command to all joints and return feedback.

        Args:
            positions: target position per joint (rad)
            velocities: target velocity per joint (rad/s)
            kps: position gain per joint
            kds: damping gain per joint
            torques: feedforward torque per joint (Nm)

        Returns:
            ArmState from motor feedback
        """
        ...

    @abstractmethod
    def get_state(self) -> ArmState:
        """Actively query current state from all joints."""
        ...

    @abstractmethod
    def close(self, disable: bool = True) -> None:
        """Release resources (close sockets, etc.).

        ``disable=True`` preserves the normal safe shutdown behavior.  A
        caller that has an explicit reason to leave the drives enabled while
        detaching the transport may pass ``disable=False``.
        """
        ...

    @abstractmethod
    def command_velocity(self, velocities: list[float]) -> ArmState:
        """
        Send native velocity commands to all joints.

        Args:
            velocities: target velocity per joint (rad/s)

        Returns:
            ArmState from motor feedback
        """
        ...

    @abstractmethod
    def command_torque(self, torques: list[float]) -> ArmState:
        """
        Send pure torque commands to all joints (MIT with kp=0, kd=0).

        Args:
            torques: feedforward torque per joint (Nm)

        Returns:
            ArmState from motor feedback
        """
        ...

    def command_hold(self, positions: list[float], velocities: list[float],
                     kps: list[float], kds: list[float],
                     torques: list[float]) -> ArmState:
        """
        Command with joint-limit clamping but NO rate limiting.

        Use when holding current position after velocity motion, or in
        force_mode/freedrive where the arm is already at the target.
        Default implementation delegates to command().
        """
        return self.command(positions, velocities, kps, kds, torques)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
