"""
Example: Arm + third-party CAN gripper pick-and-place.

Demonstrates how to implement a Gripper subclass for a third-party
gripper and use it with the arm in a pick-and-place workflow.

Gripper CAN protocol (8 bytes):
    D0: Reserved (0x00)
    D1: Position   (0x00=closed, 0xFF=fully open)
    D2: Velocity   (0x00-0xFF, 0xFF=max)
    D3: Force      (0x00-0xFF, 0xFF=max)
    D4: Acceleration (0x00-0xFF, 0xFF=max)
    D5: Deceleration (0x00-0xFF, 0xFF=max)
    D6: Reserved (0x00)
    D7: Reserved (0x00)

Hardware:
    - A1 arm on can0
    - Gripper on the same CAN bus, node ID 0x08
"""

import time
from onero_sdk import Arm, Gripper
from onero_sdk.can_bus import CanBus


# ─── Gripper driver (implement the protocol) ─────────────────────────────────

class CANGripper(Gripper):
    """Third-party CAN gripper driver."""

    def __init__(self, bus: CanBus, can_id: int = 0x08,
                 default_velocity: int = 0xFF,
                 default_force: int = 0x80,
                 default_acc: int = 0xFF,
                 default_dec: int = 0xFF):
        self._bus = bus
        self._can_id = can_id
        self._velocity = default_velocity
        self._force = default_force
        self._acc = default_acc
        self._dec = default_dec
        self._position = 0xFF

    def enable(self) -> None:
        self.open()

    def disable(self) -> None:
        self.open()

    def close(self) -> None:
        self.grasp()

    def shutdown(self) -> None:
        # The CAN bus is owned by the caller and may also carry arm traffic.
        return None

    def open(self) -> None:
        """Fully open."""
        self._send(0xFF)

    def grasp(self) -> None:
        """Fully close (grasp)."""
        self._send(0x00)

    def set_position(self, position: float) -> None:
        """Set opening: 0.0=closed, 1.0=open."""
        pos_byte = int(round(position * 0xFF))
        pos_byte = max(0, min(0xFF, pos_byte))
        self._send(pos_byte)

    def get_position(self) -> float:
        return self._position / 0xFF

    def set_params(self, velocity: int = None, force: int = None,
                   acc: int = None, dec: int = None) -> None:
        """Update motion parameters (0-255 each)."""
        if velocity is not None:
            self._velocity = max(0, min(0xFF, velocity))
        if force is not None:
            self._force = max(0, min(0xFF, force))
        if acc is not None:
            self._acc = max(0, min(0xFF, acc))
        if dec is not None:
            self._dec = max(0, min(0xFF, dec))

    def _send(self, pos: int) -> None:
        data = bytes([0x00, pos, self._velocity, self._force,
                      self._acc, self._dec, 0x00, 0x00])
        self._bus.send(self._can_id, data)
        self._position = pos


# ─── Pick-and-place workflow ──────────────────────────────────────────────────

# Shared CAN bus
bus = CanBus("can0")

# Create gripper
gripper = CANGripper(bus, can_id=0x08, default_force=0x80)

# Create arm with gripper attached
arm = Arm("A1R", channel="can0", end_effector=gripper)

# Waypoints
PRE_GRASP = [0.0, 0.8, 0.0, -1.0, 0.0, 0.5, 0.0]
GRASP_POS = [0.0, 1.0, 0.0, -1.2, 0.0, 0.5, 0.0]
PLACE_POS = [0.5, 0.8, 0.0, -1.0, 0.0, 0.5, 0.0]

with arm:
    # 1. Open gripper, move to pre-grasp
    arm.gripper.open()
    time.sleep(0.5)
    arm.movej(PRE_GRASP, duration=2.0)

    # 2. Lower to object
    arm.movej(GRASP_POS, duration=1.5)

    # 3. Grasp
    arm.gripper.grasp()
    time.sleep(0.5)

    # 4. Lift
    arm.movej(PRE_GRASP, duration=1.5)

    # 5. Move to place
    arm.movej(PLACE_POS, duration=2.0)

    # 6. Release
    arm.gripper.open()
    time.sleep(0.5)

    # 7. Home
    arm.home(duration=3.0)

print("Pick-and-place complete.")
