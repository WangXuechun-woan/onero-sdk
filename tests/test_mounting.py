"""
Unit tests for mounting configuration — rotation matrices, gravity vectors, validation.

Run: pytest tests/test_mounting.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.mounting import (
    MountingPreset, MountingConfig, get_mounting_rotation,
    gravity_in_base_frame, resolve_mounting, _validate_rotation_matrix,
    GRAVITY_MAGNITUDE,
)
from onero_sdk.exceptions import OneroConfigError


class TestRotationMatrices:
    """Validate that all preset rotation matrices are proper SO(3)."""

    @pytest.fixture(params=[p for p in MountingPreset if p != MountingPreset.CUSTOM])
    def preset(self, request):
        return request.param

    def test_orthogonality(self, preset):
        config = MountingConfig(preset=preset)
        R = get_mounting_rotation(config)
        RTR = R.T @ R
        assert np.allclose(RTR, np.eye(3), atol=1e-10), \
            f"{preset.value}: R^T R != I, max dev={np.max(np.abs(RTR - np.eye(3)))}"

    def test_determinant_is_one(self, preset):
        config = MountingConfig(preset=preset)
        R = get_mounting_rotation(config)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"{preset.value}: det(R)={det}"

    def test_shape(self, preset):
        config = MountingConfig(preset=preset)
        R = get_mounting_rotation(config)
        assert R.shape == (3, 3)


class TestGravityInBaseFrame:
    """Validate gravity vector direction and magnitude for each preset."""

    @pytest.fixture(params=[p for p in MountingPreset if p != MountingPreset.CUSTOM])
    def preset(self, request):
        return request.param

    def test_magnitude(self, preset):
        config = MountingConfig(preset=preset)
        g = gravity_in_base_frame(config)
        assert abs(np.linalg.norm(g) - GRAVITY_MAGNITUDE) < 1e-10, \
            f"{preset.value}: |g|={np.linalg.norm(g)}, expected {GRAVITY_MAGNITUDE}"

    def test_upright_gravity_is_negative_z(self):
        config = MountingConfig(preset=MountingPreset.UPRIGHT)
        g = gravity_in_base_frame(config)
        expected = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
        assert np.allclose(g, expected, atol=1e-10), \
            f"UPRIGHT gravity: {g}, expected {expected}"

    def test_side_mount_left_gravity_direction(self):
        config = MountingConfig(preset=MountingPreset.SIDE_MOUNT_LEFT)
        g = gravity_in_base_frame(config)
        # Base Y points toward world -Z (down), so gravity should have
        # component along base +Y direction
        assert abs(g[0]) < 1e-10, f"X component should be 0, got {g[0]}"
        assert g[1] > 0, f"Y component should be positive, got {g[1]}"
        assert abs(g[2]) < 1e-10, f"Z component should be 0, got {g[2]}"

    def test_side_mount_right_gravity_direction(self):
        config = MountingConfig(preset=MountingPreset.SIDE_MOUNT_RIGHT)
        g = gravity_in_base_frame(config)
        # Base Y points toward world +Z (up), so gravity should have
        # component along base -Y direction
        assert abs(g[0]) < 1e-10, f"X component should be 0, got {g[0]}"
        assert g[1] < 0, f"Y component should be negative, got {g[1]}"
        assert abs(g[2]) < 1e-10, f"Z component should be 0, got {g[2]}"

    def test_ceiling_gravity_is_positive_z(self):
        config = MountingConfig(preset=MountingPreset.CEILING)
        g = gravity_in_base_frame(config)
        expected = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        assert np.allclose(g, expected, atol=1e-10), \
            f"CEILING gravity: {g}, expected {expected}"

    def test_side_mount_left_right_symmetric(self):
        g_left = gravity_in_base_frame(MountingConfig(preset=MountingPreset.SIDE_MOUNT_LEFT))
        g_right = gravity_in_base_frame(MountingConfig(preset=MountingPreset.SIDE_MOUNT_RIGHT))
        # Y components should be opposite sign
        assert abs(g_left[1] + g_right[1]) < 1e-10
        assert abs(g_left[0] - g_right[0]) < 1e-10
        assert abs(g_left[2] - g_right[2]) < 1e-10


class TestCustomRotationValidation:
    """Validate that CUSTOM mounting rejects invalid matrices."""

    def test_rejects_non_3x3(self):
        config = MountingConfig(
            preset=MountingPreset.CUSTOM,
            custom_rotation=np.eye(4)
        )
        with pytest.raises(OneroConfigError, match="3x3"):
            get_mounting_rotation(config)

    def test_rejects_non_orthogonal(self):
        bad_R = np.array([[1, 0, 0], [0, 1, 0.5], [0, 0, 1]], dtype=float)
        config = MountingConfig(
            preset=MountingPreset.CUSTOM,
            custom_rotation=bad_R
        )
        with pytest.raises(OneroConfigError, match="orthogonal"):
            get_mounting_rotation(config)

    def test_rejects_improper_rotation(self):
        # Reflection (det = -1)
        reflection = np.diag([1.0, 1.0, -1.0])
        config = MountingConfig(
            preset=MountingPreset.CUSTOM,
            custom_rotation=reflection
        )
        with pytest.raises(OneroConfigError, match="determinant"):
            get_mounting_rotation(config)

    def test_accepts_valid_rotation(self):
        # 45 degree rotation around Z
        angle = math.pi / 4
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ])
        config = MountingConfig(preset=MountingPreset.CUSTOM, custom_rotation=R)
        result = get_mounting_rotation(config)
        assert np.allclose(result, R, atol=1e-10)

    def test_rejects_missing_custom_rotation(self):
        config = MountingConfig(preset=MountingPreset.CUSTOM)
        with pytest.raises(OneroConfigError, match="custom_rotation"):
            get_mounting_rotation(config)


class TestYawPitchOffset:
    """Validate yaw/pitch fine-tuning offset composition."""

    def test_zero_offset_is_identity(self):
        config = MountingConfig(preset=MountingPreset.UPRIGHT,
                                yaw_offset_deg=0.0, pitch_offset_deg=0.0)
        R = get_mounting_rotation(config)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_yaw_90_swaps_xy(self):
        config = MountingConfig(preset=MountingPreset.UPRIGHT,
                                yaw_offset_deg=90.0)
        R = get_mounting_rotation(config)
        # Yaw 90°: world_x from base_y (R[0,:] = [0, -1, 0] from Rz convention)
        # Actually Rz(90) = [[0,-1,0],[1,0,0],[0,0,1]]
        # R = Rz @ I so R = Rz(90)
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert np.allclose(R, expected, atol=1e-10), f"Got:\n{R}"

    def test_pitch_90(self):
        config = MountingConfig(preset=MountingPreset.UPRIGHT,
                                pitch_offset_deg=90.0)
        R = get_mounting_rotation(config)
        # Ry(90) = [[0,0,1],[0,1,0],[-1,0,0]]
        expected = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        assert np.allclose(R, expected, atol=1e-10), f"Got:\n{R}"

    def test_offset_preserves_orthogonality(self):
        config = MountingConfig(preset=MountingPreset.SIDE_MOUNT_LEFT,
                                yaw_offset_deg=15.0, pitch_offset_deg=-5.0)
        R = get_mounting_rotation(config)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestResolveMounting:
    """Test the flexible mounting argument resolution."""

    def test_none_gives_upright(self):
        config = resolve_mounting(None)
        assert config.preset == MountingPreset.UPRIGHT

    def test_string(self):
        config = resolve_mounting("side_mount_left")
        assert config.preset == MountingPreset.SIDE_MOUNT_LEFT

    def test_preset_enum(self):
        config = resolve_mounting(MountingPreset.CEILING)
        assert config.preset == MountingPreset.CEILING

    def test_config_passthrough(self):
        original = MountingConfig(preset=MountingPreset.HUMANOID_LEFT, yaw_offset_deg=5.0)
        result = resolve_mounting(original)
        assert result is original

    def test_invalid_string_raises(self):
        with pytest.raises(OneroConfigError, match="Unknown mounting"):
            resolve_mounting("invalid_preset")

    def test_invalid_type_raises(self):
        with pytest.raises(OneroConfigError, match="mounting must be"):
            resolve_mounting(42)


class TestTransformRoundtrip:
    """Verify that base→world→base is identity (roundtrip)."""

    @pytest.fixture(params=[p for p in MountingPreset if p != MountingPreset.CUSTOM])
    def R_world_base(self, request):
        return get_mounting_rotation(MountingConfig(preset=request.param))

    def test_position_roundtrip(self, R_world_base):
        from onero_sdk.mounting import transform_pose_to_world, transform_pose_to_base
        pose = np.array([0.3, -0.1, 0.5, 0.1, -0.2, 0.3])
        world = transform_pose_to_world(pose, R_world_base)
        back = transform_pose_to_base(world, R_world_base)
        assert np.allclose(back[:3], pose[:3], atol=1e-8), \
            f"Position roundtrip failed: {pose[:3]} → {world[:3]} → {back[:3]}"

    def test_rotation_roundtrip(self, R_world_base):
        from onero_sdk.mounting import transform_pose_to_world, transform_pose_to_base
        pose = np.array([0.0, 0.0, 0.0, 0.5, -0.3, 0.1])
        world = transform_pose_to_world(pose, R_world_base)
        back = transform_pose_to_base(world, R_world_base)
        assert np.allclose(back[3:], pose[3:], atol=1e-6), \
            f"Rotation roundtrip failed: {pose[3:]} → {world[3:]} → {back[3:]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
