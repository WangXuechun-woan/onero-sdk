"""
DualArm — coordinated dual-arm controller.

Primary usage scenario for bi-manual manipulation. Ensures left/right arms
share a common world frame and have correct per-arm mounting/model defaults.
"""

import copy
import threading
from typing import Optional, Callable

import numpy as np

from .arm import Arm
from .backend import ArmBackend, ArmState
from .safety import SafetyConfig
from .exceptions import OneroError, OneroConfigError


class DualArm:
    """
    Coordinates two Arms (left + right) sharing a common world frame.

    Both arms operate in the same world coordinate system (X=forward, Y=left, Z=up).
    Each arm's mounting orientation and base_position are configured so that
    Cartesian commands (movel, servop, etc.) are expressed in a unified frame.

    Quick start (simulation):
        dual = DualArm.create()  # A1L + A1R with default side mounts
        with dual:
            dual.left.movel([0.3, 0.2, 0.4], frame="world")
            dual.right.movel([0.3, -0.2, 0.4], frame="world")

    Quick start (hardware):
        dual = DualArm.create(left_channel="can0", right_channel="can1")

    For advanced control (speedl, force_mode, freedrive, recording), access
    the individual arms via `.left` and `.right` properties.
    """

    def __init__(self, left: Arm, right: Arm):
        """
        Args:
            left: Left arm instance (should have correct mounting/base_position).
            right: Right arm instance (should have correct mounting/base_position).
        """
        self._left = left
        self._right = right

    @classmethod
    def create(cls, *,
               left_model: str = "A1L",
               right_model: str = "A1R",
               left_channel: Optional[str] = None,
               right_channel: Optional[str] = None,
               left_backend: Optional[ArmBackend] = None,
               right_backend: Optional[ArmBackend] = None,
               left_mounting: str = "humanoid_left",
               right_mounting: str = "humanoid_right",
               left_with_gripper: bool = False,
               right_with_gripper: bool = False,
               base_position_left: Optional[list] = None,
               base_position_right: Optional[list] = None,
               arm_spacing: Optional[float] = None,
               safety: Optional[SafetyConfig] = None,
               on_error: Optional[Callable] = None) -> "DualArm":
        """
        Create a DualArm with correct per-arm defaults.

        By default, both arms share a world frame (world_frame=True) with
        appropriate mounting orientations for left/right side mounts.

        Args:
            left_model: Left arm model name. Default "A1L".
            right_model: Right arm model name. Default "A1R".
            left_channel: CAN channel for left arm (e.g. "can0").
            right_channel: CAN channel for right arm (e.g. "can1").
            left_backend: Custom backend for left arm (alternative to channel).
            right_backend: Custom backend for right arm (alternative to channel).
            left_mounting: Left arm mounting preset. Default "humanoid_left".
            right_mounting: Right arm mounting preset. Default "humanoid_right".
            left_with_gripper: Select the calibrated gripper gravity profile
                               for the left arm.
            right_with_gripper: Select the calibrated gripper gravity profile
                                for the right arm.
            base_position_left: [x, y, z] left arm base in world frame (meters).
                                Default: [0, +spacing/2, 0] if arm_spacing given.
            base_position_right: [x, y, z] right arm base in world frame (meters).
                                 Default: [0, -spacing/2, 0] if arm_spacing given.
            arm_spacing: Distance between left and right arm bases along Y axis
                         (meters). Convenience alternative to specifying base_position
                         individually. Ignored if base_position_left/right are set.
            safety: Safety config (copied separately for each arm).
            on_error: Error callback (shared for both arms).
        """
        # Resolve base positions
        bp_left = base_position_left
        bp_right = base_position_right
        if arm_spacing is not None:
            if bp_left is None:
                bp_left = [0.0, arm_spacing / 2, 0.0]
            if bp_right is None:
                bp_right = [0.0, -arm_spacing / 2, 0.0]
        if bp_left is None or bp_right is None:
            raise OneroConfigError(
                "DualArm always uses a shared world frame, so arm positioning is required. "
                "Provide arm_spacing, or both base_position_left and base_position_right, "
                "to define where each arm is mounted in world coordinates."
            )

        left = Arm(
            model=left_model,
            channel=left_channel,
            backend=left_backend,
            mounting=left_mounting,
            world_frame=True,
            base_position=bp_left,
            safety=copy.copy(safety) if safety else None,
            on_error=on_error,
            with_gripper=left_with_gripper,
        )
        right = Arm(
            model=right_model,
            channel=right_channel,
            backend=right_backend,
            mounting=right_mounting,
            world_frame=True,
            base_position=bp_right,
            safety=copy.copy(safety) if safety else None,
            on_error=on_error,
            with_gripper=right_with_gripper,
        )
        return cls(left, right)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "DualArm":
        """
        Create a DualArm from a YAML config file.

        Config format:
            left:
              model: "A1L"
              channel: "can0"
              mounting: "humanoid_left"
              base_position: [0, 0.15, 0]
            right:
              model: "A1R"
              channel: "can1"
              mounting: "humanoid_right"
              base_position: [0, -0.15, 0]
            # Optional top-level defaults:
            world_frame: true
            arm_spacing: 0.3  # alternative to per-arm base_position
        """
        from .config import load_dual_arm_config
        raw = load_dual_arm_config(path)

        left_cfg = raw["left"]
        right_cfg = raw["right"]
        world_frame = raw.get("world_frame", True)

        # Resolve base positions
        arm_spacing = raw.get("arm_spacing")
        bp_left = left_cfg.get("base_position")
        bp_right = right_cfg.get("base_position")
        if arm_spacing is not None:
            if bp_left is None:
                bp_left = [0.0, arm_spacing / 2, 0.0]
            if bp_right is None:
                bp_right = [0.0, -arm_spacing / 2, 0.0]
        if world_frame and (bp_left is None or bp_right is None):
            raise OneroConfigError(
                "DualArm.from_config(): world_frame=True requires arm positioning. "
                "Set top-level arm_spacing or both left/right base_position values."
            )

        left = Arm(
            model=left_cfg.get("model", "A1L"),
            channel=left_cfg.get("channel"),
            mounting=left_cfg.get("mounting", "humanoid_left"),
            world_frame=world_frame,
            base_position=bp_left,
            with_gripper=left_cfg.get("with_gripper", False),
            safety=copy.copy(kwargs.get("safety")) if kwargs.get("safety") else None,
            on_error=kwargs.get("on_error"),
        )
        right = Arm(
            model=right_cfg.get("model", "A1R"),
            channel=right_cfg.get("channel"),
            mounting=right_cfg.get("mounting", "humanoid_right"),
            world_frame=world_frame,
            base_position=bp_right,
            with_gripper=right_cfg.get("with_gripper", False),
            safety=copy.copy(kwargs.get("safety")) if kwargs.get("safety") else None,
            on_error=kwargs.get("on_error"),
        )
        return cls(left, right)

    @property
    def left(self) -> Arm:
        """Left arm instance."""
        return self._left

    @property
    def right(self) -> Arm:
        """Right arm instance."""
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

    # ─── Convenience: Pose Query ─────────────────────────────────────────────

    def get_poses(self, frame: Optional[str] = None
                  ) -> tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]:
        """Get both EE poses in the same frame. Default: world frame."""
        return (self._left.get_pose(frame=frame),
                self._right.get_pose(frame=frame))

    # ─── Planned Motion ───────────────────────────────────────────────────────

    def movej(self, left_joints: list[float], right_joints: list[float],
              duration: Optional[float] = None) -> tuple[ArmState, ArmState]:
        """Move both arms to target joint angles in parallel."""
        return self._parallel(
            lambda: self._left.movej(left_joints, duration=duration),
            lambda: self._right.movej(right_joints, duration=duration),
        )

    def movel(self, left_position, right_position,
              left_rotation=None, right_rotation=None,
              duration: Optional[float] = None,
              frame: Optional[str] = None) -> tuple[ArmState, ArmState]:
        """Move both arms along straight Cartesian lines in parallel."""
        return self._parallel(
            lambda: self._left.movel(left_position, left_rotation,
                                     duration=duration, frame=frame),
            lambda: self._right.movel(right_position, right_rotation,
                                      duration=duration, frame=frame),
        )

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
        return self._parallel(
            lambda: self._left.servoj(left_joints),
            lambda: self._right.servoj(right_joints),
        )

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _parallel(self, left_fn, right_fn) -> tuple[ArmState, ArmState]:
        """Run two operations in parallel, raising if either fails."""
        results = [None, None]
        errors = [None, None]

        def run_left():
            try:
                results[0] = left_fn()
            except Exception as e:
                errors[0] = e

        def run_right():
            try:
                results[1] = right_fn()
            except Exception as e:
                errors[1] = e

        t_left = threading.Thread(target=run_left)
        t_right = threading.Thread(target=run_right)
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
