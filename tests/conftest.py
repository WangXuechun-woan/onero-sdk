"""Shared fixtures for the test suite."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def arm_sim():
    """MuJoCo simulated arm fixture."""
    from onero_sdk import Arm
    from onero_sdk.mujoco_backend import MujocoBackend

    backend = MujocoBackend(model_name="A1")
    arm = Arm("A1", backend=backend)
    arm.enable()
    yield arm
    arm.disable()
    backend.close()


@pytest.fixture
def arm_side_mount():
    """MuJoCo arm with side-mount configuration."""
    from onero_sdk import Arm
    from onero_sdk.mujoco_backend import MujocoBackend

    backend = MujocoBackend(model_name="A1")
    arm = Arm("A1", backend=backend, mounting="side_mount_left")
    arm.enable()
    yield arm
    arm.disable()
    backend.close()
