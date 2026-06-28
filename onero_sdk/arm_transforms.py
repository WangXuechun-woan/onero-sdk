"""
Coordinate transform mixin for Arm — mounting, world-frame, and TCP transforms.

This module separates the coordinate transform layer from the main Arm class,
keeping the pose/twist/wrench conversion logic in one focused file.
"""

import logging

import numpy as np

from . import geometry
from .mounting import (
    MountingPreset, MountingConfig, resolve_mounting,
    get_mounting_rotation, gravity_in_base_frame,
)

logger = logging.getLogger(__name__)


class ArmTransformsMixin:
    """
    Mixin providing coordinate transform methods for the Arm class.

    Manages:
    - Mounting orientation (R_world_base, gravity direction)
    - World-frame mode (base↔world transform for all Cartesian APIs)
    - TCP offset (flange↔tool transform)

    Attributes set by __init__ (must be initialized before using mixin methods):
        _mounting_config: MountingConfig
        _R_world_base: np.ndarray (3x3)
        _gravity_in_base: np.ndarray (3,)
        _world_frame: bool
        _tcp_offset_pos: np.ndarray (3,)
        _tcp_offset_rot: np.ndarray (3x3)
    """

    def _init_transforms(self, mounting, world_frame: bool):
        """Initialize transform state. Called from Arm.__init__."""
        self._mounting_config = resolve_mounting(mounting)
        self._R_world_base = get_mounting_rotation(self._mounting_config)
        self._gravity_in_base = gravity_in_base_frame(self._mounting_config)
        self._world_frame = world_frame
        self._tcp_offset_pos = np.zeros(3)
        self._tcp_offset_rot = np.eye(3)

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def mounting(self) -> MountingConfig:
        """Current mounting configuration."""
        return self._mounting_config

    @property
    def gravity_direction(self) -> np.ndarray:
        """Gravity vector in base frame coordinates (m/s²)."""
        return self._gravity_in_base.copy()

    @property
    def R_world_base(self) -> np.ndarray:
        """Rotation matrix from base frame to world frame (3x3)."""
        return self._R_world_base.copy()

    def set_world_frame(self, enabled: bool = True) -> None:
        """
        Enable/disable world-frame mode for Cartesian APIs.

        When enabled, get_pose/movel/movep/movec/servop/speedl/force_mode/
        get_external_wrench operate in world coordinates (X=forward, Y=left, Z=up).
        """
        self._world_frame = enabled

    def set_tcp(self, position: list[float], rotation=None) -> None:
        """
        Set TCP (Tool Center Point) offset from flange to tool tip.

        Affects all Cartesian APIs: get_pose returns TCP pose,
        movel/servop targets are interpreted as TCP targets.

        Args:
            position: [x, y, z] offset from flange to tool tip (meters),
                      expressed in the flange frame.
            rotation: Rotation from flange to tool frame. Accepts 3x3 matrix,
                      quaternion [x,y,z,w], or rotation vector [rx,ry,rz].
                      None = no rotation (tool aligned with flange).
        """
        self._tcp_offset_pos = np.asarray(position, dtype=float)
        if rotation is not None:
            self._tcp_offset_rot = geometry.ensure_rotation_matrix(rotation)
        else:
            self._tcp_offset_rot = np.eye(3)
        logger.info(f"TCP set: pos={self._tcp_offset_pos.tolist()}, "
                    f"has_rotation={rotation is not None}")

    def clear_tcp(self) -> None:
        """Remove TCP offset (revert to flange frame)."""
        self._tcp_offset_pos = np.zeros(3)
        self._tcp_offset_rot = np.eye(3)

    # ─── Internal Transform Methods ─────────────────────────────────────────

    def _apply_tcp_forward(self, pos: np.ndarray, rot: np.ndarray):
        """Apply TCP offset to FK output: flange → TCP."""
        tcp_pos = pos + rot @ self._tcp_offset_pos
        tcp_rot = rot @ self._tcp_offset_rot
        return tcp_pos, tcp_rot

    def _apply_tcp_inverse(self, pos: np.ndarray, rot: np.ndarray):
        """Remove TCP offset from target: TCP target → flange target for IK."""
        flange_rot = rot @ self._tcp_offset_rot.T
        flange_pos = pos - flange_rot @ self._tcp_offset_pos
        return flange_pos, flange_rot

    def _input_pose_transform(self, pos: np.ndarray, rot: np.ndarray):
        """Transform input pose: world→base (if world_frame), then TCP→flange."""
        if self._world_frame:
            R_base_world = self._R_world_base.T
            pos = R_base_world @ pos
            rot = R_base_world @ rot
        pos, rot = self._apply_tcp_inverse(pos, rot)
        return pos, rot

    def _output_pose_transform(self, pos: np.ndarray, rot: np.ndarray):
        """Transform output pose: flange→TCP, then base→world (if world_frame)."""
        pos, rot = self._apply_tcp_forward(pos, rot)
        if self._world_frame:
            pos = self._R_world_base @ pos
            rot = self._R_world_base @ rot
        return pos, rot

    def _input_twist_transform(self, twist: np.ndarray) -> np.ndarray:
        """Transform input twist (6D velocity) from world to base frame."""
        if not self._world_frame:
            return twist
        R_base_world = self._R_world_base.T
        result = np.empty(6)
        result[:3] = R_base_world @ twist[:3]
        result[3:] = R_base_world @ twist[3:]
        return result

    def _output_wrench_transform(self, wrench: np.ndarray) -> np.ndarray:
        """Transform output wrench from base to world frame."""
        if not self._world_frame:
            return wrench
        result = np.empty(6)
        result[:3] = self._R_world_base @ wrench[:3]
        result[3:] = self._R_world_base @ wrench[3:]
        return result

    def _input_wrench_transform(self, wrench: np.ndarray) -> np.ndarray:
        """Transform input wrench from world to base frame."""
        if not self._world_frame:
            return wrench
        R_base_world = self._R_world_base.T
        result = np.empty(6)
        result[:3] = R_base_world @ wrench[:3]
        result[3:] = R_base_world @ wrench[3:]
        return result
