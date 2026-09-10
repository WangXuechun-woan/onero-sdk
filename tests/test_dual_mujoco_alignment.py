"""Coordinate-alignment regression tests for the dual-arm MuJoCo scene."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from onero_sdk.dual_mujoco_backend import DualMujocoBackend


def _body_pose(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def test_vertical_mount_points_both_base_x_axes_up():
    backend = DualMujocoBackend(realtime=False)
    backend.reset(right_pos=[0.0] * 7, left_pos=[0.0] * 7)

    for side in ("R", "L"):
        _, rotation = _body_pose(
            backend._model, backend._data, f"{side}_base_link"
        )
        base_x_in_world = rotation[:, 0]
        assert base_x_in_world == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)

        gravity_in_base = rotation.T @ np.array([0.0, 0.0, -9.81])
        assert gravity_in_base == pytest.approx([-9.81, 0.0, 0.0], abs=1e-8)


def test_zero_pose_hangs_below_both_shoulders():
    backend = DualMujocoBackend(realtime=False)
    backend.reset(right_pos=[0.0] * 7, left_pos=[0.0] * 7)

    for side in ("R", "L"):
        base_pos, _ = _body_pose(
            backend._model, backend._data, f"{side}_base_link"
        )
        tool_pos, _ = _body_pose(
            backend._model, backend._data, f"{side}_Link7"
        )
        assert tool_pos[2] < base_pos[2] - 0.5


def test_vertical_mount_has_outward_base_z_axes():
    backend = DualMujocoBackend(realtime=False)
    backend.reset(right_pos=[0.0] * 7, left_pos=[0.0] * 7)

    _, right_rotation = _body_pose(
        backend._model, backend._data, "R_base_link"
    )
    _, left_rotation = _body_pose(
        backend._model, backend._data, "L_base_link"
    )

    assert right_rotation[:, 2] == pytest.approx([0.0, -1.0, 0.0], abs=1e-8)
    assert left_rotation[:, 2] == pytest.approx([0.0, 1.0, 0.0], abs=1e-8)
