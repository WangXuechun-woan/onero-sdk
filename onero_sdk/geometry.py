"""
Rotation and spatial geometry utilities.

Pure functions — no state, no SDK dependencies. Used by Arm for
Cartesian motion interpolation (SLERP, circular arc fitting).
"""

import math

import numpy as np


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Convert rotation vector (axis-angle compact form) to 3x3 rotation matrix."""
    rotvec = np.asarray(rotvec, dtype=float)
    if not np.all(np.isfinite(rotvec)):
        from .exceptions import OneroConfigError
        raise OneroConfigError(
            f"Rotation vector contains non-finite values: {rotvec.tolist()}")
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.eye(3)
    axis = rotvec / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def ensure_rotation_matrix(rotation) -> np.ndarray:
    """
    Convert rotation input to 3x3 matrix.

    Accepts:
        - 3x3 rotation matrix (passed through)
        - 4-element quaternion [x, y, z, w]
        - 3-element rotation vector (axis-angle)
    """
    from .exceptions import OneroConfigError

    rot = np.asarray(rotation, dtype=float)
    if rot.shape == (3, 3):
        return rot
    elif rot.shape == (4,):
        norm = np.linalg.norm(rot)
        if norm < 1e-10:
            raise OneroConfigError("Quaternion has near-zero norm")
        rot = rot / norm
        x, y, z, w = rot
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
    elif rot.shape == (3,):
        return rotvec_to_matrix(rot)
    else:
        raise OneroConfigError(
            f"Rotation must be 3x3 matrix, quaternion [x,y,z,w], or rotation vector [rx,ry,rz], got shape {rot.shape}"
        )


def slerp_matrix(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two rotation matrices."""
    if not np.isfinite(t):
        return R0.copy()
    if not np.all(np.isfinite(R0)) or not np.all(np.isfinite(R1)):
        return R0.copy() if np.all(np.isfinite(R0)) else np.eye(3)

    R_delta = R0.T @ R1
    cos_angle = (np.trace(R_delta) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    if abs(angle) < 1e-8:
        return R0.copy()

    sin_angle = np.sin(angle)
    if abs(sin_angle) < 1e-6:
        # Near 0 or pi: extract axis from R_delta + I (robust for 180° case)
        M = R_delta + np.eye(3)
        col_norms = [np.linalg.norm(M[:, i]) for i in range(3)]
        idx = int(np.argmax(col_norms))
        if col_norms[idx] < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = M[:, idx]
            axis = axis / np.linalg.norm(axis)
    else:
        skew = (R_delta - R_delta.T) / (2 * sin_angle)
        axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    interp_angle = t * angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_interp = np.eye(3) + math.sin(interp_angle) * K + (1 - math.cos(interp_angle)) * (K @ K)
    return R0 @ R_interp


def fit_circle_3points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """
    Fit a circle through 3 points in 3D space.

    Returns (center, radius, normal) of the circle's plane.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        center = np.zeros(3)
        radius = 0.0
        normal = np.array([0.0, 0.0, 1.0])
        return center, radius, normal

    v01 = p1 - p0
    v02 = p2 - p0
    normal = np.cross(v01, v02)
    norm_len = np.linalg.norm(normal)

    if norm_len < 1e-10:
        center = (p0 + p2) / 2
        radius = np.linalg.norm(p2 - p0) / 2
        if np.linalg.norm(v02) < 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            arbitrary = np.array([1, 0, 0]) if abs(v02[0]) < 0.9 else np.array([0, 1, 0])
            normal = np.cross(v02, arbitrary)
            n_len = np.linalg.norm(normal)
            normal = normal / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])
        return center, radius, normal

    normal = normal / norm_len

    d01 = np.dot(v01, v01)
    d02 = np.dot(v02, v02)
    d01_02 = np.dot(v01, v02)

    denom = 2.0 * (d01 * d02 - d01_02 * d01_02)
    if abs(denom) < 1e-12:
        center = (p0 + p2) / 2
        radius = np.linalg.norm(p2 - p0) / 2
        return center, radius, normal

    alpha = d02 * (d01 - d01_02) / denom
    beta = d01 * (d02 - d01_02) / denom

    center = p0 + alpha * v01 + beta * v02
    radius = np.linalg.norm(center - p0)
    return center, radius, normal


def signed_arc_angle(v0: np.ndarray, v2: np.ndarray,
                     normal: np.ndarray, v1: np.ndarray) -> float:
    """
    Compute signed angle from v0 to v2 around normal, ensuring v1 is traversed.

    All vectors are from circle center to points on the circle.
    """
    v0 = np.asarray(v0, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    normal = np.asarray(normal, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v2)) and
            np.all(np.isfinite(normal)) and np.all(np.isfinite(v1))):
        return 0.0

    cos_a = np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2) + 1e-12)
    cos_a = np.clip(cos_a, -1, 1)
    angle = np.arccos(cos_a)

    cross = np.cross(v0, v2)
    if np.dot(cross, normal) < 0:
        angle = -angle

    # Handle full-circle case: v0 ≈ v2 but via point is elsewhere
    cos_via = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-12)
    cos_via = np.clip(cos_via, -1, 1)
    angle_via = np.arccos(cos_via)
    cross_via = np.cross(v0, v1)
    if np.dot(cross_via, normal) < 0:
        angle_via = -angle_via

    if abs(angle) < 1e-8 and abs(angle_via) > 1e-4:
        # v0 ≈ v2 but via is different → full circle
        if np.dot(cross_via, normal) >= 0:
            return 2 * math.pi
        else:
            return -2 * math.pi

    if angle > 0:
        if angle_via < 0 or angle_via > angle:
            angle = angle - 2 * math.pi
    else:
        if angle_via > 0 or angle_via < angle:
            angle = angle + 2 * math.pi

    return angle


def arc_length_3points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                       center: np.ndarray, radius: float,
                       normal: np.ndarray) -> float:
    """Compute arc length for the circular arc from p0 through p1 to p2."""
    v0 = p0 - center
    v2 = p2 - center
    v1 = p1 - center
    total_angle = signed_arc_angle(v0, v2, normal, v1)
    return abs(total_angle) * radius


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around axis by angle (Rodrigues' formula)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
