"""
End-effector abstraction for grippers and dexterous hands.

The SDK defines this interface; concrete CAN drivers for known grippers live
in :mod:`onero_sdk.grippers`. Third-party or custom end effectors can inherit
from EndEffector or Gripper.

Example — built-in gripper::

    arm = Arm("A1R", channel="can0",
              end_effector=OmniPickerGripper(can_id=0x08))
    arm.enable()          # also enables the gripper
    arm.gripper.close()   # close = grasp

Example — custom end effector::

    class MyHand(EndEffector):
        def __init__(self, bus, can_id):
            self._bus = bus
            self._can_id = can_id

        def enable(self):
            self._bus.send(...)   # your enable frame

        def disable(self):
            self._bus.send(...)   # your disable frame

        def shutdown(self):
            self._bus.close()   # release owned resources

    arm = Arm("A1R", channel="can0", end_effector=MyHand(bus, 0x08))
"""

from abc import ABC, abstractmethod
from typing import Optional


class EndEffector(ABC):
    """
    Base class for all end effectors (grippers, dexterous hands, etc.).

    Subclasses implement the CAN protocol specific to their hardware.
    The Arm class calls enable()/disable() automatically during its own
    lifecycle if an end effector is attached.
    """

    @abstractmethod
    def enable(self) -> None:
        """Power on / activate the end effector."""
        ...

    @abstractmethod
    def disable(self) -> None:
        """Power off / deactivate the end effector."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources (called on Arm context exit)."""
        ...


class Gripper(EndEffector):
    """
    ABC for simple 1-DOF grippers.

    Provides a standardized interface (open/close/set_position) that
    works regardless of the underlying CAN protocol.
    """

    @abstractmethod
    def open(self) -> None:
        """Fully open the gripper."""
        ...

    @abstractmethod
    def grasp(self) -> None:
        """Fully close the gripper (grasp). Same as close()."""
        ...

    def close(self) -> None:
        """Fully close the gripper. Alias for grasp()."""
        self.grasp()

    @abstractmethod
    def set_position(self, position: float) -> None:
        """
        Set gripper opening.

        Args:
            position: 0.0 = fully closed, 1.0 = fully open.
        """
        ...

    @abstractmethod
    def get_position(self) -> float:
        """
        Get current gripper opening.

        Returns:
            0.0 = fully closed, 1.0 = fully open.
        """
        ...

    def set_force(self, force: float) -> None:
        """
        Set grasping force (optional, not all grippers support this).

        Args:
            force: 0.0 = minimum, 1.0 = maximum.
        """
        raise NotImplementedError("This gripper does not support force control")


class DexterousHand(EndEffector):
    """
    ABC for multi-DOF dexterous hands.

    Each hand has its own DOF count and control interface.
    This base class defines the shared structure; concrete
    implementations add finger-level control.
    """

    @property
    @abstractmethod
    def num_fingers(self) -> int:
        """Number of independently controllable fingers."""
        ...

    @property
    @abstractmethod
    def num_dof(self) -> int:
        """Total degrees of freedom."""
        ...

    @abstractmethod
    def set_positions(self, positions: list[float]) -> None:
        """
        Set all joint positions.

        Args:
            positions: Target positions for each DOF. Units are normalized
                       0.0 (fully closed/retracted) to 1.0 (fully open/extended).
                       Implementations that use raw radians should document this
                       in their subclass docstring.
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[float]:
        """Get current joint positions for all DOFs."""
        ...
