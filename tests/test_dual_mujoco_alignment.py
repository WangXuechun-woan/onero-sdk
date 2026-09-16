"""Coordinate-alignment regression tests for the dual-arm MuJoCo scene."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from onero_sdk import DualArm
from onero_sdk.dual_mujoco_backend import DualMujocoBackend


def _body_pose(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def _has_table_contact(backend):
    table_id = mujoco.mj_name2id(
        backend._model, mujoco.mjtObj.mjOBJ_BODY, "table")
    for index in range(backend._data.ncon):
        contact = backend._data.contact[index]
        body1 = backend._model.geom_bodyid[int(contact.geom1)]
        body2 = backend._model.geom_bodyid[int(contact.geom2)]
        if table_id in (body1, body2):
            return True
    return False


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


def test_public_dual_arm_movej_produces_mirrored_world_motion():
    backend = DualMujocoBackend(realtime=False)
    left_backend, right_backend = backend.arm_backends()
    dual = DualArm.create(
        left_backend=left_backend,
        right_backend=right_backend,
        arm_spacing=0.24,
    )
    dual.enable()

    table_contacts = []
    backend.on_step = lambda shared: table_contacts.append(
        _has_table_contact(shared))
    right_target = np.array([0.25, -0.10, 0.10, 0.30, 0.05, 0.20, 0.0])
    left_target = -right_target
    dual.movej(left_target.tolist(), right_target.tolist(), duration=1.0)

    right_pos, _ = _body_pose(backend._model, backend._data, "R_Link7")
    left_pos, _ = _body_pose(backend._model, backend._data, "L_Link7")
    mirrored_right = right_pos * np.array([1.0, -1.0, 1.0])
    assert left_pos == pytest.approx(mirrored_right, abs=0.002)
    assert backend.control_cycles > 0
    assert not any(table_contacts)


def test_demo_mirrored_joint_path_stays_clear_of_table():
    backend = DualMujocoBackend(realtime=False)
    right_poses = [
        np.array([0.25, -0.10, 0.10, 0.30, 0.05, 0.20, 0.00]),
        np.array([-0.45, -0.20, 0.20, 0.55, 0.10, 0.30, 0.05]),
        np.array([0.00, 0.00, -0.30, 0.30, -0.20, 0.20, 0.00]),
        np.zeros(7),
    ]

    start = np.zeros(7)
    for target in right_poses:
        for alpha in np.linspace(0.0, 1.0, 101):
            right = start + alpha * (target - start)
            backend._data.qpos[:14] = np.concatenate([right, -right])
            mujoco.mj_forward(backend._model, backend._data)
            assert not _has_table_contact(backend)
        start = target


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
