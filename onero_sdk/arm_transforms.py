"""
Coordinate transform mixin for Arm — mounting, world-frame, and TCP transforms.

This module separates the coordinate transform layer from the main Arm class,
keeping the pose/twist/wrench conversion logic in one focused file.
"""

import logging
from typing import Optional

import numpy as np

from . import geometry
from .mounting import (
    MountingPreset, MountingConfig, resolve_mounting,
    get_mounting_rotation, gravity_in_base_frame,
)

logger = logging.getLogger(__name__)

BUILTIN_FRAMES = frozenset({"base", "world", "tool"})


class ArmTransformsMixin:
    """
    Mixin providing coordinate transform methods for the Arm class.

    Manages:
    - Mounting orientation (R_world_base, gravity direction)
    - Frame resolution (base/world/tool/custom user frames)
    - TCP offset (flange↔tool transform)

    Attributes set by __init__ (must be initialized before using mixin methods):
        _mounting_config: MountingConfig
        _R_world_base: np.ndarray (3x3)
        _gravity_in_base: np.ndarray (3,)
        _default_frame: str
        _custom_frames: dict
        _tcp_offset_pos: np.ndarray (3,)
        _tcp_offset_rot: np.ndarray (3x3)
    """

    def _init_transforms(self, mounting, world_frame: bool, base_position=None):
        """Initialize transform state. Called from Arm.__init__."""
        self._mounting_config = resolve_mounting(mounting)
        self._R_world_base = get_mounting_rotation(self._mounting_config)
        self._gravity_in_base = gravity_in_base_frame(self._mounting_config)
        self._default_frame = "world" if world_frame else "base"
        self._custom_frames: dict[str, dict] = {}
        if base_position is not None:
            self._base_position_in_world = self._validate_vec3(
                base_position, "base_position")
        else:
            self._base_position_in_world = np.zeros(3)
        self._tcp_offset_pos = np.zeros(3)
        self._tcp_offset_rot = np.eye(3)
        self._tcp_configured = False
        self._ee_pose_cache = None

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

    @property
    def default_frame(self) -> str:
        """Current default coordinate frame for Cartesian APIs."""
        return self._default_frame

    @property
    def base_position_in_world(self) -> np.ndarray:
        """Position of arm base origin in world coordinates (meters)."""
        return self._base_position_in_world.copy()

    def set_base_position(self, position: list[float]) -> None:
        """
        Set the arm base origin position in world coordinates.

        Required for frame="world" to correctly transform positions
        (especially in multi-arm setups where each arm has a different base location).

        Args:
            position: [x, y, z] base origin in world frame (meters).
        """
        self._base_position_in_world = self._validate_vec3(position, "base_position")
        logger.info(f"Base position set: {self._base_position_in_world.tolist()}")

    def set_world_frame(self, enabled: bool = True) -> None:
        """
        Enable/disable world-frame mode for Cartesian APIs.

        When enabled, get_pose/movel/movep/movec/servop/speedl/force_mode/
        get_external_wrench operate in world coordinates (X=forward, Y=left, Z=up).

        Equivalent to setting default_frame to "world" or "base".
        """
        self._default_frame = "world" if enabled else "base"

    def register_frame(self, name: str, rotation, position=None) -> None:
        """
        Register a custom coordinate frame.

        The rotation defines the orientation of this frame relative to the world frame.
        Specifically, it is R_world_custom: the matrix that transforms a vector
        expressed in this custom frame into world-frame coordinates.

        Example:
            # Camera tilted 30° down around Y axis relative to world
            R_cam = rotvec_to_matrix([0, -0.524, 0])  # -30° around Y
            arm.register_frame("camera", rotation=R_cam, position=[0.5, 0, 0.3])
            # Now arm.get_pose(frame="camera") returns EE pose in camera coords

        Args:
            name: Frame name (must not collide with built-in frames "base"/"world"/"tool").
            rotation: Rotation from this frame to world frame (R_world_custom).
                      Accepts 3x3 matrix, quaternion [x,y,z,w], or rotation vector [rx,ry,rz].
            position: [x, y, z] origin of this frame in world coordinates.
                      Defaults to [0, 0, 0].
        """
        if name in BUILTIN_FRAMES:
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"Cannot register frame '{name}': conflicts with built-in frame.")
        R = geometry.ensure_rotation_matrix(rotation)
        self._validate_rotation_matrix(R, f"register_frame('{name}', rotation=...)")
        p = self._validate_vec3(position, f"register_frame('{name}', position=...)") if position is not None else np.zeros(3)
        self._custom_frames[name] = {"rotation": R, "position": p}
        logger.info(f"Registered frame '{name}'")

    def unregister_frame(self, name: str) -> None:
        """Remove a previously registered custom frame."""
        self._custom_frames.pop(name, None)

    def set_tcp(self, position: list[float], rotation=None) -> None:
        """
        Set TCP (Tool Center Point) offset from flange to tool tip.

        Required before using frame="tool". Defines the tool coordinate frame
        origin and orientation relative to the flange.

        Affects all Cartesian APIs: get_pose returns TCP pose,
        movel/servop targets are interpreted as TCP targets.

        Args:
            position: [x, y, z] offset from flange to tool tip (meters),
                      expressed in the flange frame.
            rotation: Rotation from flange to tool frame. Accepts 3x3 matrix,
                      quaternion [x,y,z,w], or rotation vector [rx,ry,rz].
                      None = no rotation (tool aligned with flange).
        """
        self._tcp_offset_pos = self._validate_vec3(position, "set_tcp(position)")
        if rotation is not None:
            self._tcp_offset_rot = geometry.ensure_rotation_matrix(rotation)
            self._validate_rotation_matrix(self._tcp_offset_rot, "set_tcp(rotation)")
        else:
            self._tcp_offset_rot = np.eye(3)
        self._tcp_configured = True
        self._ee_pose_cache = None
        logger.info(f"TCP set: pos={self._tcp_offset_pos.tolist()}, "
                    f"has_rotation={rotation is not None}")

    def clear_tcp(self) -> None:
        """Remove TCP offset (revert to flange frame). Disables frame='tool'."""
        self._tcp_offset_pos = np.zeros(3)
        self._tcp_offset_rot = np.eye(3)
        self._tcp_configured = False
        self._ee_pose_cache = None

    # ─── Internal Transform Methods ─────────────────────────────────────────

    @staticmethod
    def _validate_vec3(value, name: str) -> np.ndarray:
        """Validate and convert a value to a (3,) numpy array."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"{name} must be a 3-element vector [x, y, z], got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"{name} contains non-finite values (NaN/Inf): {arr.tolist()}")
        return arr

    @staticmethod
    def _validate_vec6(value, name: str) -> np.ndarray:
        """Validate and convert a value to a (6,) numpy array."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != (6,):
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"{name} must be a 6-element vector, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"{name} contains non-finite values (NaN/Inf)")
        return arr

    @staticmethod
    def _validate_rotation_matrix(R: np.ndarray, name: str) -> None:
        """Check that a 3x3 matrix is a valid rotation (orthogonal, det=+1)."""
        from .exceptions import OneroConfigError
        err = np.linalg.norm(R @ R.T - np.eye(3))
        det = np.linalg.det(R)
        if err > 1e-4 or abs(det - 1.0) > 1e-4:
            raise OneroConfigError(
                f"{name}: not a valid rotation matrix "
                f"(orthogonality error={err:.2e}, det={det:.6f}). "
                f"A rotation matrix must satisfy R @ R.T = I and det(R) = +1.")

    def _resolve_frame(self, frame: Optional[str]) -> str:
        """Resolve frame=None to the current default frame."""
        return frame if frame is not None else self._default_frame

    def _resolve_frame_rotation(self, frame: Optional[str]) -> Optional[np.ndarray]:
        """
        Resolve a frame name to R_base_frame (rotation from frame to base).

        Returns None for base frame (no rotation needed).
        Used by force_mode and wrench transforms for consistent frame resolution.
        """
        resolved = self._resolve_frame(frame)
        self._check_frame_preconditions(resolved)
        if resolved == "base":
            return None
        if resolved == "world":
            return self._R_world_base.T
        if resolved == "tool":
            _, ee_rot = self._get_current_ee_pose_base()
            return ee_rot
        # Custom frame
        if resolved not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"Unknown frame '{resolved}'. "
                f"Available: base, world, tool"
                + (f", {', '.join(self._custom_frames.keys())}" if self._custom_frames else ""))
        R_frame_to_world = self._custom_frames[resolved]["rotation"]
        return self._R_world_base.T @ R_frame_to_world

    def _apply_tcp_forward(self, pos: np.ndarray, rot: np.ndarray):
        """Apply TCP offset to FK output: flange → TCP."""
        if not self._tcp_configured:
            return pos, rot
        tcp_pos = pos + rot @ self._tcp_offset_pos
        tcp_rot = rot @ self._tcp_offset_rot
        return tcp_pos, tcp_rot

    def _apply_tcp_inverse(self, pos: np.ndarray, rot: np.ndarray):
        """Remove TCP offset from target: TCP target → flange target for IK."""
        if not self._tcp_configured:
            return pos, rot
        flange_rot = rot @ self._tcp_offset_rot.T
        flange_pos = pos - flange_rot @ self._tcp_offset_pos
        return flange_pos, flange_rot

    def _get_current_ee_pose_base(self):
        """Get current EE pose in base frame (after TCP forward). Used for tool-frame."""
        joints = tuple(self._state.joints)
        cache = self._ee_pose_cache
        if cache is not None and cache[0] == joints:
            return cache[1], cache[2]
        fk = self._get_fk()
        pos, rot = fk.get_ee_pose(list(joints))
        pos, rot = self._apply_tcp_forward(pos, rot)
        self._ee_pose_cache = (joints, pos, rot)
        return pos, rot

    def _check_frame_preconditions(self, frame: str):
        """Raise if frame prerequisites are not met."""
        from .exceptions import OneroConfigError
        if frame == "tool" and not self._tcp_configured:
            raise OneroConfigError(
                "frame='tool' requires set_tcp() first. "
                "Call arm.set_tcp(position, rotation) to define the tool frame.")
        if frame == "world":
            pass

    def _input_pose_transform(self, pos: np.ndarray, rot: np.ndarray,
                              frame: Optional[str] = None):
        """Transform input pose from the specified frame to base-flange for IK."""
        frame = self._resolve_frame(frame)
        self._check_frame_preconditions(frame)

        if frame == "tool":
            ee_pos, ee_rot = self._get_current_ee_pose_base()
            target_pos = ee_pos + ee_rot @ pos
            target_rot = ee_rot @ rot
            return self._apply_tcp_inverse(target_pos, target_rot)

        if frame == "base":
            return self._apply_tcp_inverse(pos, rot)

        if frame == "world":
            R_base_world = self._R_world_base.T
            pos_base = R_base_world @ (pos - self._base_position_in_world)
            rot = R_base_world @ rot
            return self._apply_tcp_inverse(pos_base, rot)

        # Custom frame: transform custom→world→base
        if frame not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"Unknown frame '{frame}'. "
                f"Available: base, world, tool"
                + (f", {', '.join(self._custom_frames.keys())}" if self._custom_frames else ""))
        fdef = self._custom_frames[frame]
        R_frame_to_world = fdef["rotation"]
        p_frame_origin = fdef["position"]
        pos_world = R_frame_to_world @ pos + p_frame_origin
        rot_world = R_frame_to_world @ rot
        R_base_world = self._R_world_base.T
        pos_base = R_base_world @ (pos_world - self._base_position_in_world)
        rot_base = R_base_world @ rot_world
        return self._apply_tcp_inverse(pos_base, rot_base)

    def _output_pose_transform(self, pos: np.ndarray, rot: np.ndarray,
                               frame: Optional[str] = None):
        """Transform output pose from base-flange to the specified frame."""
        frame = self._resolve_frame(frame)
        self._check_frame_preconditions(frame)
        pos, rot = self._apply_tcp_forward(pos, rot)

        if frame == "tool":
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                "get_pose(frame='tool') is not supported. "
                "The tool frame is for relative motion (movel/servop/speedl), "
                "not for reading pose. Use frame='base' or frame='world' instead.")

        if frame == "base":
            return pos, rot

        if frame == "world":
            pos_world = self._R_world_base @ pos + self._base_position_in_world
            rot_world = self._R_world_base @ rot
            return pos_world, rot_world

        # Custom frame: base→world→custom
        if frame not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(
                f"Unknown frame '{frame}'. "
                f"Available: base, world, tool"
                + (f", {', '.join(self._custom_frames.keys())}" if self._custom_frames else ""))
        fdef = self._custom_frames[frame]
        R_frame_to_world = fdef["rotation"]
        p_frame_origin = fdef["position"]
        R_world_to_frame = R_frame_to_world.T
        pos_world = self._R_world_base @ pos + self._base_position_in_world
        rot_world = self._R_world_base @ rot
        pos_frame = R_world_to_frame @ (pos_world - p_frame_origin)
        rot_frame = R_world_to_frame @ rot_world
        return pos_frame, rot_frame

    def _input_twist_transform(self, twist: np.ndarray,
                               frame: Optional[str] = None) -> np.ndarray:
        """Transform input twist (6D velocity) from the specified frame to base frame."""
        twist = self._validate_vec6(twist, "twist")
        frame = self._resolve_frame(frame)
        self._check_frame_preconditions(frame)

        if frame == "base":
            return twist

        if frame == "tool":
            ee_pos, ee_rot = self._get_current_ee_pose_base()
            result = np.empty(6)
            v_tool = twist[:3]
            w_tool = twist[3:]
            # ee_rot = R_base_tcp. Jacobian is at flange, so we need v_flange.
            # v_tcp = v_flange + ω × r_flange_to_tcp
            # => v_flange = v_tcp - ω × r_flange_to_tcp
            w_base = ee_rot @ w_tool
            v_flange_base = ee_rot @ v_tool
            if self._tcp_configured:
                # r_flange_to_tcp in base = R_base_flange @ tcp_offset_pos
                R_base_flange = ee_rot @ self._tcp_offset_rot.T
                r_tcp_base = R_base_flange @ self._tcp_offset_pos
                v_flange_base -= np.cross(w_base, r_tcp_base)
            result[:3] = v_flange_base
            result[3:] = w_base
            return result

        if frame == "world":
            R_base_world = self._R_world_base.T
            result = np.empty(6)
            result[:3] = R_base_world @ twist[:3]
            result[3:] = R_base_world @ twist[3:]
            return result

        # Custom frame → world → base
        if frame not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(f"Unknown frame '{frame}'.")
        R_frame_to_world = self._custom_frames[frame]["rotation"]
        R_base_world = self._R_world_base.T
        R_frame_to_base = R_base_world @ R_frame_to_world
        result = np.empty(6)
        result[:3] = R_frame_to_base @ twist[:3]
        result[3:] = R_frame_to_base @ twist[3:]
        return result

    def _output_wrench_transform(self, wrench: np.ndarray,
                                 frame: Optional[str] = None) -> np.ndarray:
        """Transform output wrench from base to the specified frame."""
        wrench = self._validate_vec6(wrench, "wrench")
        frame = self._resolve_frame(frame)
        self._check_frame_preconditions(frame)

        if frame == "base":
            return wrench

        if frame == "tool":
            ee_pos, ee_rot = self._get_current_ee_pose_base()
            R_base_to_tool = ee_rot.T
            f_base = wrench[:3]
            tau_base = wrench[3:]
            result = np.empty(6)
            result[:3] = R_base_to_tool @ f_base
            # Wrench at flange → wrench at TCP: tau_tcp = tau_flange - r × f
            # In base frame, then rotate to tool frame
            if self._tcp_configured:
                R_base_flange = ee_rot @ self._tcp_offset_rot.T
                r_tcp_base = R_base_flange @ self._tcp_offset_pos
                result[3:] = R_base_to_tool @ (tau_base - np.cross(r_tcp_base, f_base))
            else:
                result[3:] = R_base_to_tool @ tau_base
            return result

        if frame == "world":
            result = np.empty(6)
            result[:3] = self._R_world_base @ wrench[:3]
            result[3:] = self._R_world_base @ wrench[3:]
            return result

        # Custom: base → world → custom
        if frame not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(f"Unknown frame '{frame}'.")
        R_frame_to_world = self._custom_frames[frame]["rotation"]
        R_world_to_frame = R_frame_to_world.T
        R_base_to_frame = R_world_to_frame @ self._R_world_base
        result = np.empty(6)
        result[:3] = R_base_to_frame @ wrench[:3]
        result[3:] = R_base_to_frame @ wrench[3:]
        return result

    def _input_wrench_transform(self, wrench: np.ndarray,
                                frame: Optional[str] = None) -> np.ndarray:
        """Transform input wrench from the specified frame to base frame."""
        wrench = self._validate_vec6(wrench, "wrench")
        frame = self._resolve_frame(frame)
        self._check_frame_preconditions(frame)

        if frame == "base":
            return wrench

        if frame == "tool":
            ee_pos, ee_rot = self._get_current_ee_pose_base()
            f_tool = wrench[:3]
            tau_tool = wrench[3:]
            result = np.empty(6)
            f_base = ee_rot @ f_tool
            result[:3] = f_base
            # Wrench at TCP → wrench at flange: tau_flange = tau_tcp + r × f
            if self._tcp_configured:
                R_base_flange = ee_rot @ self._tcp_offset_rot.T
                r_tcp_base = R_base_flange @ self._tcp_offset_pos
                result[3:] = ee_rot @ tau_tool + np.cross(r_tcp_base, f_base)
            else:
                result[3:] = ee_rot @ tau_tool
            return result

        if frame == "world":
            R_base_world = self._R_world_base.T
            result = np.empty(6)
            result[:3] = R_base_world @ wrench[:3]
            result[3:] = R_base_world @ wrench[3:]
            return result

        # Custom: custom → world → base
        if frame not in self._custom_frames:
            from .exceptions import OneroConfigError
            raise OneroConfigError(f"Unknown frame '{frame}'.")
        R_frame_to_world = self._custom_frames[frame]["rotation"]
        R_base_world = self._R_world_base.T
        R_frame_to_base = R_base_world @ R_frame_to_world
        result = np.empty(6)
        result[:3] = R_frame_to_base @ wrench[:3]
        result[3:] = R_frame_to_base @ wrench[3:]
        return result
