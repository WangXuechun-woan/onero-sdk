"""
Rotation math & circle geometry — algorithmic correctness tests.

Validates SLERP interpolation, circle fitting, and arc angle
computation against known mathematical invariants. Pure math — no
simulation or hardware dependency.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.geometry import (
    slerp_matrix,
    fit_circle_3points,
    signed_arc_angle,
    rotvec_to_matrix,
)


def _angle_between_rotations(R0, R1):
    """Geodesic angle between two rotation matrices."""
    R_delta = R0.T @ R1
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def _is_valid_rotation(R, tol=1e-8):
    """Check if R is a valid SO(3) matrix."""
    orth_err = np.linalg.norm(R.T @ R - np.eye(3))
    det_err = abs(np.linalg.det(R) - 1.0)
    return orth_err < tol and det_err < tol


class TestSLERP:
    """SLERP 球面线性插值正确性."""

    def test_slerp_identity_at_endpoints(self):
        """slerp(R0, R1, 0) = R0, slerp(R0, R1, 1) = R1."""
        R0 = rotvec_to_matrix([0.3, -0.5, 0.7])
        R1 = rotvec_to_matrix([-0.4, 0.8, 0.2])

        R_at_0 = slerp_matrix(R0, R1, 0.0)
        R_at_1 = slerp_matrix(R0, R1, 1.0)

        assert np.allclose(R_at_0, R0, atol=1e-10), "slerp(t=0) != R0"
        assert np.allclose(R_at_1, R1, atol=1e-10), "slerp(t=1) != R1"

    def test_slerp_midpoint_is_geodesic_midpoint(self):
        """Midpoint rotation angle = total angle / 2."""
        R0 = rotvec_to_matrix([0.5, 0.0, 0.0])
        R1 = rotvec_to_matrix([0.5, 1.2, 0.0])

        total_angle = _angle_between_rotations(R0, R1)
        R_mid = slerp_matrix(R0, R1, 0.5)

        angle_0_mid = _angle_between_rotations(R0, R_mid)
        angle_mid_1 = _angle_between_rotations(R_mid, R1)

        assert abs(angle_0_mid - total_angle / 2) < 1e-6, (
            f"Midpoint not at geodesic half: {angle_0_mid:.6f} vs {total_angle/2:.6f}"
        )
        assert abs(angle_0_mid - angle_mid_1) < 1e-6, (
            "Midpoint is not equidistant from both endpoints"
        )

    def test_slerp_continuity_across_half(self):
        """No discontinuity at alpha=0.5 (adjacent samples have similar step size)."""
        R0 = rotvec_to_matrix([0.1, 0.2, 0.3])
        R1 = rotvec_to_matrix([1.0, -0.5, 0.8])

        total_angle = _angle_between_rotations(R0, R1)
        N = 100
        angles = []
        prev = R0
        for i in range(1, N + 1):
            t = i / N
            R = slerp_matrix(R0, R1, t)
            angles.append(_angle_between_rotations(prev, R))
            prev = R

        avg_step = total_angle / N
        # No single step should be more than 1.5x the average (uniform geodesic)
        for i, step in enumerate(angles):
            assert step < 1.5 * avg_step, (
                f"Discontinuity at step {i}: {step:.6f} vs avg {avg_step:.6f}"
            )

    def test_slerp_near_180_degrees(self):
        """Near-π rotation: result is valid SO(3), no NaN."""
        R0 = np.eye(3)
        # 179° around Z axis
        angle = math.radians(179)
        R1 = rotvec_to_matrix([0, 0, angle])

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            R = slerp_matrix(R0, R1, t)
            assert not np.any(np.isnan(R)), f"NaN in SLERP at t={t} (near 180°)"
            assert _is_valid_rotation(R, tol=1e-6), (
                f"Invalid SO(3) at t={t}: det={np.linalg.det(R):.6f}"
            )

    def test_slerp_exactly_180_degrees(self):
        """Exactly π rotation: result is valid SO(3), deterministic."""
        R0 = np.eye(3)
        # Exactly 180° around Z axis
        R1 = rotvec_to_matrix([0, 0, math.pi])

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            R = slerp_matrix(R0, R1, t)
            assert not np.any(np.isnan(R)), f"NaN at t={t} (exactly 180°)"
            assert _is_valid_rotation(R, tol=1e-6), (
                f"Invalid SO(3) at t={t}"
            )

        # Midpoint should be at 90° from R0
        R_mid = slerp_matrix(R0, R1, 0.5)
        angle_mid = _angle_between_rotations(R0, R_mid)
        assert abs(angle_mid - math.pi / 2) < 0.01, (
            f"180° SLERP midpoint not at 90°: {math.degrees(angle_mid):.1f}°"
        )

    def test_slerp_zero_rotation(self):
        """Interpolating between identical matrices returns the same matrix."""
        R = rotvec_to_matrix([0.4, -0.3, 0.6])
        for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
            R_interp = slerp_matrix(R, R, t)
            assert np.allclose(R_interp, R, atol=1e-10), (
                f"slerp(R, R, {t}) != R"
            )


class TestCircleFitting:
    """Circle fitting and arc angle correctness."""

    def test_fit_circle_3points_all_on_circle(self):
        """All three input points lie on the fitted circle (dist = radius)."""
        # Points on a known circle: center=(1,0,0), radius=2, in XY plane
        theta = [0, math.pi / 3, 2 * math.pi / 3]
        center_true = np.array([1.0, 0.0, 0.0])
        radius_true = 2.0
        points = [
            center_true + radius_true * np.array([math.cos(t), math.sin(t), 0.0])
            for t in theta
        ]

        center, radius, normal = fit_circle_3points(*points)

        assert abs(radius - radius_true) < 1e-8, (
            f"Radius mismatch: {radius:.6f} vs {radius_true}"
        )
        assert np.allclose(center, center_true, atol=1e-8), (
            f"Center mismatch: {center} vs {center_true}"
        )
        # All points at distance radius from center
        for i, p in enumerate(points):
            dist = np.linalg.norm(p - center)
            assert abs(dist - radius) < 1e-8, (
                f"Point {i} not on circle: dist={dist:.8f}, radius={radius:.8f}"
            )

    def test_fit_circle_collinear_degeneracy(self):
        """Collinear points: does not crash, returns finite values."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])

        center, radius, normal = fit_circle_3points(p0, p1, p2)

        assert np.all(np.isfinite(center)), "center has non-finite values"
        assert np.isfinite(radius), "radius is non-finite"
        assert np.all(np.isfinite(normal)), "normal has non-finite values"

    def test_signed_arc_angle_traverses_via(self):
        """Arc angle direction ensures the path goes through the via point."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0
        normal = np.array([0.0, 0.0, 1.0])

        # p0 at 0°, via at 90°, p2 at 180° → should be positive ~π
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])

        angle = signed_arc_angle(v0, v2, normal, v1)
        assert angle > 0, f"Expected positive arc angle, got {angle}"
        assert abs(angle - math.pi) < 1e-5, (
            f"Expected π, got {angle:.6f}"
        )

        # Reverse: via at 270° (-90°) → path goes the other way → angle ~ -π
        v1_rev = np.array([0.0, -1.0, 0.0])
        angle_rev = signed_arc_angle(v0, v2, normal, v1_rev)
        assert angle_rev < 0, f"Expected negative arc angle, got {angle_rev}"
        assert abs(angle_rev + math.pi) < 1e-5, (
            f"Expected -π, got {angle_rev:.6f}"
        )


class TestCircleFitting3D:
    """Circle fitting in arbitrary 3D orientations."""

    def test_fit_circle_non_planar_points_on_result(self):
        """Points on a tilted 3D circle recover the correct geometry."""
        # Circle in a tilted plane: center at (1,1,1), radius=0.5
        # Normal along (1,1,1)/sqrt(3)
        normal_true = np.array([1, 1, 1]) / math.sqrt(3)
        center_true = np.array([1.0, 1.0, 1.0])
        radius_true = 0.5

        # Build two orthogonal vectors in the circle's plane
        arb = np.array([1, 0, 0])
        u = np.cross(normal_true, arb)
        u = u / np.linalg.norm(u)
        v = np.cross(normal_true, u)

        theta = [0, 1.2, 2.5]
        points = [
            center_true + radius_true * (math.cos(t) * u + math.sin(t) * v)
            for t in theta
        ]

        center, radius, normal = fit_circle_3points(*points)

        assert abs(radius - radius_true) < 1e-6
        assert np.allclose(center, center_true, atol=1e-6)
        # Normal should be parallel to true normal (same or opposite direction)
        dot = abs(np.dot(normal, normal_true))
        assert dot > 0.999, f"Normal direction mismatch: dot={dot}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
