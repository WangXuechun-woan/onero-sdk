"""
DualArm — convenience wrapper for two Arm instances (left + right).

Not the core primitive. Single-arm deployment is first-class.
This is a thin orchestration layer with no coordination logic.
"""

import copy
import threading
from typing import Optional, Callable

from .arm import Arm
from .backend import ArmState
from .safety import SafetyConfig
from .exceptions import OneroError


class DualArm:
    """
    Coordinates two Arms (left + right) on separate CAN buses.

    Provides joint enable/disable/move for both arms together.
    No collision avoidance or coordination — just convenience.

    Only basic symmetric operations (enable, disable, movej, servoj, home)
    are wrapped here. For advanced control (speedj, speedl, movel, movec,
    freedrive, force_mode, recording), access the individual arms directly
    via the `.left` and `.right` properties.
    """

    def __init__(self, left: Arm, right: Arm):
        """
        Args:
            left: Left arm instance.
            right: Right arm instance.
        """
        self._left = left
        self._right = right

    @classmethod
    def create(cls, model: str = "A1R",
               left_channel: Optional[str] = None,
               right_channel: Optional[str] = None,
               safety: Optional[SafetyConfig] = None,
               on_error: Optional[Callable] = None) -> "DualArm":
        """
        Create a DualArm from model name and channel names.

        Args:
            model: Arm model name (same for both arms).
            left_channel: CAN channel for left arm (e.g. "can0").
            right_channel: CAN channel for right arm (e.g. "can1").
            safety: Safety config (shared for both arms).
            on_error: Error callback (shared for both arms).
        """
        left = Arm(model=model, channel=left_channel,
                   safety=copy.copy(safety) if safety else None, on_error=on_error)
        right = Arm(model=model, channel=right_channel,
                    safety=copy.copy(safety) if safety else None, on_error=on_error)
        return cls(left, right)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "DualArm":
        """
        Create a DualArm from a config file.

        Config format:
            left:
              model: "A1L"
              channel: "can0"
            right:
              model: "A1R"
              channel: "can1"
        """
        from .config import load_dual_arm_config
        raw = load_dual_arm_config(path)

        left_cfg = raw["left"]
        right_cfg = raw["right"]

        left = Arm(
            model=left_cfg.get("model", "A1L"),
            channel=left_cfg.get("channel"),
            **kwargs,
        )
        right = Arm(
            model=right_cfg.get("model", "A1R"),
            channel=right_cfg.get("channel"),
            **kwargs,
        )
        return cls(left, right)

    @property
    def left(self) -> Arm:
        return self._left

    @property
    def right(self) -> Arm:
        return self._right

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def enable(self) -> tuple[ArmState, ArmState]:
        """Enable both arms. Disables the first arm if the second fails."""
        left_state = self._left.enable()
        try:
            right_state = self._right.enable()
        except Exception:
            self._left.disable()
            raise
        return left_state, right_state

    def disable(self) -> None:
        """Disable both arms."""
        self._left.disable()
        self._right.disable()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self._left.close()
        self._right.close()

    # ─── Planned Motion ───────────────────────────────────────────────────────

    def movej(self, left_joints: list[float], right_joints: list[float],
              duration: Optional[float] = None) -> tuple[ArmState, ArmState]:
        """Move both arms to target joint angles in parallel."""
        results = [None, None]
        errors = [None, None]

        def move_left():
            try:
                results[0] = self._left.movej(left_joints, duration=duration)
            except Exception as e:
                errors[0] = e

        def move_right():
            try:
                results[1] = self._right.movej(right_joints, duration=duration)
            except Exception as e:
                errors[1] = e

        t_left = threading.Thread(target=move_left)
        t_right = threading.Thread(target=move_right)
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        if errors[0] and errors[1]:
            raise OneroError(
                f"Both arms failed: left={errors[0]}, right={errors[1]}"
            ) from errors[0]
        if errors[0]:
            raise errors[0]
        if errors[1]:
            raise errors[1]

        return results[0], results[1]

    def home(self, duration: Optional[float] = None) -> tuple[ArmState, ArmState]:
        """Home both arms to zero (parallel)."""
        return self.movej(
            left_joints=[0.0] * self._left.num_joints,
            right_joints=[0.0] * self._right.num_joints,
            duration=duration or 3.0,
        )

    # ─── Real-time Control ────────────────────────────────────────────────────

    def servoj(self, left_joints: list[float],
               right_joints: list[float]) -> tuple[ArmState, ArmState]:
        """Send joint targets to both arms in parallel (single frame)."""
        results = [None, None]
        errors = [None, None]

        def cmd_left():
            try:
                results[0] = self._left.servoj(left_joints)
            except Exception as e:
                errors[0] = e

        def cmd_right():
            try:
                results[1] = self._right.servoj(right_joints)
            except Exception as e:
                errors[1] = e

        t_left = threading.Thread(target=cmd_left)
        t_right = threading.Thread(target=cmd_right)
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        if errors[0] and errors[1]:
            raise OneroError(
                f"Both arms failed: left={errors[0]}, right={errors[1]}"
            ) from errors[0]
        if errors[0]:
            raise errors[0]
        if errors[1]:
            raise errors[1]

        return results[0], results[1]
