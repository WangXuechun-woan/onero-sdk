"""
Lightweight forward kinematics — URDF-based, zero external dependencies.

Parses a URDF file and computes end-effector pose via chain multiplication.
Only requires numpy (already a core dependency).
"""

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np


def _rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    """Roll-pitch-yaw (XYZ extrinsic) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues)."""
    ax = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = ax

    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _make_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """Create 4x4 homogeneous transform from xyz translation and rpy rotation."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


class URDFChain:
    """
    A kinematic chain parsed from URDF.

    Only extracts revolute/continuous joints in order from base to tip.
    """

    def __init__(self, urdf_path: str, ee_link: str = "Link7"):
        self._joints = []  # list of (origin_T, axis)
        self._ee_fixed_transform = np.eye(4)  # fixed transform after last joint
        self._parse(urdf_path, ee_link)

    @property
    def num_joints(self) -> int:
        return len(self._joints)

    def _parse(self, urdf_path: str, ee_link: str) -> None:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Build link→joint and joint→child maps
        joints = []
        parent_to_joint = {}  # parent_link → joint element
        for joint_elem in root.findall("joint"):
            jtype = joint_elem.get("type")
            if jtype not in ("revolute", "continuous", "fixed"):
                continue

            parent_link = joint_elem.find("parent").get("link")
            child_link = joint_elem.find("child").get("link")

            origin = joint_elem.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(v) for v in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(v) for v in origin.get("rpy").split()]

            axis_elem = joint_elem.find("axis")
            axis = np.array([0.0, 0.0, 1.0])
            if axis_elem is not None and axis_elem.get("xyz"):
                axis = np.array([float(v) for v in axis_elem.get("xyz").split()])

            joints.append({
                "name": joint_elem.get("name"),
                "type": jtype,
                "parent": parent_link,
                "child": child_link,
                "origin": _make_transform(xyz, rpy),
                "axis": axis,
            })
            parent_to_joint[parent_link] = parent_to_joint.get(parent_link, [])
            parent_to_joint[parent_link].append(joints[-1])

        # Walk chain from base to ee_link
        # Find the root link (a link that is never a child)
        all_children = {j["child"] for j in joints}
        all_parents = {j["parent"] for j in joints}
        root_links = all_parents - all_children

        # BFS/walk from root to ee_link
        chain = self._find_chain(joints, root_links, ee_link)

        # Extract revolute joints and fixed transforms
        accumulated_fixed = np.eye(4)
        for j in chain:
            if j["type"] in ("revolute", "continuous"):
                # Combine any accumulated fixed transforms with this joint's origin
                T = accumulated_fixed @ j["origin"]
                self._joints.append((T, j["axis"]))
                accumulated_fixed = np.eye(4)
            else:  # fixed
                accumulated_fixed = accumulated_fixed @ j["origin"]

        self._ee_fixed_transform = accumulated_fixed

    def _find_chain(self, joints: list, root_links: set, target: str) -> list:
        """Find joint chain from any root link to target link."""
        child_map = {}  # parent_link → list of joints
        for j in joints:
            child_map.setdefault(j["parent"], []).append(j)

        # DFS
        for root in root_links:
            path = self._dfs(child_map, root, target, [])
            if path is not None:
                return path
        return []

    def _dfs(self, child_map: dict, current: str, target: str, path: list) -> Optional[list]:
        if current == target:
            return path
        for j in child_map.get(current, []):
            result = self._dfs(child_map, j["child"], target, path + [j])
            if result is not None:
                return result
        return None

    def forward_kinematics(self, q: list[float]) -> np.ndarray:
        """
        Compute end-effector pose given joint angles.

        Args:
            q: joint angles in radians (length must equal num_joints)

        Returns:
            4x4 homogeneous transformation matrix (base frame to EE)
        """
        assert len(q) == self.num_joints, \
            f"Expected {self.num_joints} joint angles, got {len(q)}"

        T = np.eye(4)
        for i, (origin_T, axis) in enumerate(self._joints):
            T = T @ origin_T
            R_joint = np.eye(4)
            R_joint[:3, :3] = _axis_angle_matrix(axis, q[i])
            T = T @ R_joint

        T = T @ self._ee_fixed_transform
        return T

    def get_ee_pose(self, q: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector position and rotation.

        Returns:
            (position [3], rotation_matrix [3x3])
        """
        T = self.forward_kinematics(q)
        return T[:3, 3].copy(), T[:3, :3].copy()
