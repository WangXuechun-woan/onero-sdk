"""
Mounting orientation configuration for Onero robotic arms.

Onero A1 is typically mounted sideways on a desktop bracket or robot
shoulder (base Z-axis pointing left/right), NOT upright (Z-up).

This module provides:
  - MountingPreset: common installation orientations
  - MountingConfig: configuration with fine-tuning offsets
  - get_mounting_rotation(): compute R_world_base rotation matrix
  - gravity_in_base_frame(): compute gravity vector in base coordinates

Coordinate conventions:
  - World frame: X=forward (away from operator), Y=left, Z=up
  - Base frame: as defined by URDF, varies with mounting
  - R_world_base: rotates a vector FROM base frame TO world frame
    i.e. p_world = R_world_base @ p_base
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .exceptions import OneroConfigError

GRAVITY_MAGNITUDE = 9.81
_WORLD_GRAVITY = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])


class MountingPreset(Enum):
    """Installation orientation presets."""

    UPRIGHT = "upright"
    SIDE_MOUNT_LEFT = "side_mount_left"
    SIDE_MOUNT_RIGHT = "side_mount_right"
    HUMANOID_LEFT = "humanoid_left"
    HUMANOID_RIGHT = "humanoid_right"
    CEILING = "ceiling"
    CUSTOM = "custom"


@dataclass
class MountingConfig:
    """Mounting configuration with optional fine-tuning."""
    preset: MountingPreset = MountingPreset.SIDE_MOUNT_LEFT
    custom_rotation: Optional[np.ndarray] = None
    yaw_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0


def _validate_rotation_matrix(R: np.ndarray) -> None:
    """Validate that R is a proper rotation matrix (SO(3))."""
    if R.shape != (3, 3):
        raise OneroConfigError(
            f"Rotation matrix must be 3x3, got shape {R.shape}")
    RTR = R.T @ R
    if not np.allclose(RTR, np.eye(3), atol=1e-6):
        raise OneroConfigError(
            "Rotation matrix is not orthogonal (R^T @ R != I). "
            f"Max deviation: {np.max(np.abs(RTR - np.eye(3))):.6f}")
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise OneroConfigError(
            f"Rotation matrix determinant is {det:.6f}, expected 1.0 (proper rotation)")


def get_mounting_rotation(config: MountingConfig) -> np.ndarray:
    """
    Compute R_world_base (3x3) for the given mounting configuration.

    Converts a vector from base frame to world frame:
        p_world = R_world_base @ p_base

    World frame convention: X=forward, Y=left, Z=up.
    """
    if config.preset == MountingPreset.UPRIGHT:
        R = np.eye(3)

    elif config.preset == MountingPreset.SIDE_MOUNT_LEFT:
        # Base Z points toward world +Y (left)
        # Base X points toward world +X (forward)
        # Base Y points toward world -Z (down) — right-hand rule
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])

    elif config.preset == MountingPreset.SIDE_MOUNT_RIGHT:
        # Base Z points toward world -Y (right)
        # Base X points toward world +X (forward)
        # Base Y points toward world +Z (up)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

    elif config.preset == MountingPreset.HUMANOID_LEFT:
        # Same as SIDE_MOUNT_LEFT for now (shoulder mount, Z outward-left)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])

    elif config.preset == MountingPreset.HUMANOID_RIGHT:
        # Same as SIDE_MOUNT_RIGHT for now
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

    elif config.preset == MountingPreset.CEILING:
        # Base Z points toward world -Z (down)
        # Base X points toward world +X (forward)
        # Base Y points toward world -Y (right)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

    elif config.preset == MountingPreset.CUSTOM:
        if config.custom_rotation is None:
            raise OneroConfigError(
                "CUSTOM mounting requires custom_rotation (3x3 numpy array)")
        R = np.asarray(config.custom_rotation, dtype=float)
        _validate_rotation_matrix(R)

    else:
        R = np.eye(3)

    # Apply yaw/pitch fine-tuning offsets
    if abs(config.yaw_offset_deg) > 0.01 or abs(config.pitch_offset_deg) > 0.01:
        yaw = math.radians(config.yaw_offset_deg)
        pitch = math.radians(config.pitch_offset_deg)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        Ry = np.array([
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ])
        R = Rz @ Ry @ R

    return R


def gravity_in_base_frame(config: MountingConfig) -> np.ndarray:
    """
    Compute gravity vector in the base frame for a given mounting.

    Returns:
        3D gravity acceleration vector in base coordinates (m/s²).
        For UPRIGHT: [0, 0, -9.81]
        For SIDE_MOUNT_LEFT: [0, 9.81, 0] (gravity pulls along +Y in base)
    """
    R_world_base = get_mounting_rotation(config)
    # g_base = R_base_world @ g_world = R_world_base.T @ [0, 0, -9.81]
    return R_world_base.T @ _WORLD_GRAVITY


def resolve_mounting(mounting) -> MountingConfig:
    """
    Normalize a mounting argument into a MountingConfig.

    Accepts:
      - None → UPRIGHT (backward compatible)
      - str → converted to MountingPreset
      - MountingPreset → wrapped in MountingConfig
      - MountingConfig → returned as-is
    """
    if mounting is None:
        return MountingConfig(preset=MountingPreset.UPRIGHT)
    if isinstance(mounting, str):
        try:
            preset = MountingPreset(mounting)
        except ValueError:
            valid = [m.value for m in MountingPreset]
            raise OneroConfigError(
                f"Unknown mounting preset '{mounting}'. Valid: {valid}")
        return MountingConfig(preset=preset)
    if isinstance(mounting, MountingPreset):
        return MountingConfig(preset=mounting)
    if isinstance(mounting, MountingConfig):
        return mounting
    raise OneroConfigError(
        f"mounting must be None, str, MountingPreset, or MountingConfig, "
        f"got {type(mounting).__name__}")


def transform_pose_to_world(pose_base: np.ndarray,
                            R_world_base: np.ndarray) -> np.ndarray:
    """
    Transform a 6D pose from base frame to world frame.

    Args:
        pose_base: [x, y, z, rx, ry, rz] in base coordinates.
        R_world_base: 3x3 rotation matrix.

    Returns:
        [x, y, z, rx, ry, rz] in world coordinates.
    """
    pos_base = np.asarray(pose_base[:3], dtype=float)
    rvec_base = np.asarray(pose_base[3:6], dtype=float)

    pos_world = R_world_base @ pos_base

    angle = np.linalg.norm(rvec_base)
    if angle < 1e-10:
        R_base_rot = np.eye(3)
    else:
        axis = rvec_base / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R_base_rot = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K

    R_world_rot = R_world_base @ R_base_rot

    angle_w = math.acos(max(-1.0, min(1.0, (np.trace(R_world_rot) - 1) / 2)))
    if angle_w < 1e-10:
        rvec_world = np.zeros(3)
    else:
        axis_w = np.array([
            R_world_rot[2, 1] - R_world_rot[1, 2],
            R_world_rot[0, 2] - R_world_rot[2, 0],
            R_world_rot[1, 0] - R_world_rot[0, 1],
        ]) / (2 * math.sin(angle_w))
        rvec_world = axis_w * angle_w

    return np.concatenate([pos_world, rvec_world])


def transform_pose_to_base(pose_world: np.ndarray,
                           R_world_base: np.ndarray) -> np.ndarray:
    """
    Transform a 6D pose from world frame to base frame.

    Args:
        pose_world: [x, y, z, rx, ry, rz] in world coordinates.
        R_world_base: 3x3 rotation matrix.

    Returns:
        [x, y, z, rx, ry, rz] in base coordinates.
    """
    return transform_pose_to_world(pose_world, R_world_base.T)
