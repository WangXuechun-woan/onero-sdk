"""Debug UI behavior tests through the public ArmAdapter API."""

import json
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np

from onero_sdk.backend import ArmState
from onero_sdk.debugui.server import ArmAdapter, LabServer


class RecordingGripper:
    protocol_name = "omnipicker"
    enabled = True
    state = SimpleNamespace(
        position=0.25, velocity=0.0, force=0.1,
        fault_code=0, motion_state=0, timestamp=1.0,
    )

    def __init__(self):
        self.positions = []
        self.lifecycle = []

    def enable(self):
        self.lifecycle.append("enable")

    def disable(self):
        self.lifecycle.append("disable")

    def set_position(self, position):
        self.positions.append(position)

    def get_position(self):
        return self.state.position


class RecordingArm:
    """Minimal arm contract used to observe commands issued by Debug UI."""

    num_joints = 7
    enabled = True
    is_freedrive = False
    is_moving = False
    model = SimpleNamespace(name="A1L")
    default_frame = "world"
    R_world_base = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    base_position_in_world = np.array([0.1, 0.2, 0.3])

    def __init__(self):
        self.calls = []
        self.gripper = None
        self.command_received = threading.Event()
        self.motion_release = threading.Event()
        self.motion_release.set()
        self.fail_movep = False

    def stop(self):
        self.calls.append(("stop",))
        return self.get_state()

    def get_state(self):
        return ArmState(
            joints=[0.0] * 7,
            velocities=[0.0] * 7,
            torques=[0.0] * 7,
            timestamp=0.0,
        )

    def get_pose(self):
        return np.array([0.4, 0.1, 0.2]), np.eye(3)

    def get_joint_limits(self):
        return [(-3.14, 3.14)] * 7

    def movej(self, joints, speed=None, duration=None):
        self.calls.append(("movej", list(joints), speed, duration))
        self.command_received.set()
        self.motion_release.wait(timeout=1.0)
        return self.get_state()

    def movep(self, position, rotation=None, speed=None, duration=None):
        self.calls.append(("movep", list(position), rotation, speed, duration))
        self.command_received.set()
        if self.fail_movep:
            raise RuntimeError("IK target unreachable")
        return self.get_state()

    def servoj(self, joints, kp=None, kd=None, velocities=None):
        self.calls.append(("servoj", list(joints)))
        return self.get_state()

    def freedrive_disable(self):
        self.is_freedrive = False


def test_state_exposes_cartesian_frame_metadata():
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    try:
        state = adapter.get_state_dict()
        assert state["pose_frame"] == "world"
        assert state["R_world_base"] == arm.R_world_base.tolist()
        assert state["base_position"] == [0.1, 0.2, 0.3]
    finally:
        adapter.close()


def test_state_exposes_selected_visual_tool_model():
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left", tool_model="omnipicker")
    try:
        tool = adapter.get_state_dict()["tool_model"]
        assert tool["name"] == "omnipicker"
        assert tool["urdf"] == "omnipicker.urdf"
        assert tool["mount_rpy"] == [0.0, 0.0, -1.5708]
        assert tool["mass"] == 0.41689308
        assert tool["tcp"] is None
        assert tool["animated"] is True
        assert tool["opening_default"] == 0.5
    finally:
        adapter.close()

    woan = ArmAdapter(arm, side="right", tool_model="woan_picker")
    try:
        tool = woan.get_state_dict()["tool_model"]
        assert tool["name"] == "woan_picker"
        assert tool["urdf"] == "woan_picker.urdf"
        assert tool["mount_xyz"] == [0.0, 0.0, 0.07]
        assert tool["mount_rpy"] == [-1.5708, 0.0, -1.5708]
        assert tool["opening_default"] == 0.0
        assert len(tool["opening_joints"]) == 8
    finally:
        woan.close()

    woan_left = ArmAdapter(arm, side="left", tool_model="woan_picker")
    try:
        assert woan_left.get_state_dict()["tool_model"]["mount_rpy"] == [
            -1.5708, 0.0, -1.5708]
    finally:
        woan_left.close()


def test_debugui_commands_real_gripper_and_reports_feedback():
    arm = RecordingArm()
    arm.gripper = RecordingGripper()
    adapter = ArmAdapter(arm, side="left", tool_model="woan_picker")
    try:
        assert adapter.enable_gripper() == {"ok": True, "msg": "gripper enabled"}
        result = adapter.set_gripper_position(0.75)
        assert result == {"ok": True, "msg": "gripper position=75%"}
        assert adapter.disable_gripper() == {"ok": True, "msg": "gripper disabled"}
        assert arm.gripper.lifecycle == ["enable", "disable"]
        assert arm.gripper.positions == [0.75]
        state = adapter.get_state_dict()["gripper"]
        assert state["protocol"] == "omnipicker"
        assert state["position"] == 0.25
        assert state["hardware_control"] is True
    finally:
        adapter.close()


def test_http_gripper_position_endpoint_controls_hardware_driver():
    arm = RecordingArm()
    arm.gripper = RecordingGripper()
    adapter = ArmAdapter(arm, side="left", tool_model="woan_picker")
    server = LabServer(
        adapters={"left": adapter}, dual_mode=False,
        host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/gripper/position",
            data=json.dumps({"position": 0.6}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1.0) as response:
            result = json.load(response)
        assert result == {"ok": True, "msg": "gripper position=60%"}
        assert arm.gripper.positions == [0.6]
    finally:
        server.shutdown()
        server.server_close()
        adapter.close()


def test_movep_is_available_and_preserves_current_orientation():
    """A position-only MoveP request reaches Arm.movep with current rotation."""
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    try:
        result = adapter.movep([0.42, 0.10, 0.20], speed=0.2)

        assert result == {"ok": True, "msg": "movep queued"}
        assert arm.command_received.wait(timeout=1.0)
        movep_call = next(call for call in arm.calls if call[0] == "movep")
        assert movep_call[1] == [0.42, 0.10, 0.20]
        np.testing.assert_allclose(movep_call[2], np.eye(3))
        assert movep_call[3:] == (0.2, None)
    finally:
        adapter.close()


def test_http_movep_endpoint_queues_cartesian_point_motion():
    """The browser-facing REST API exposes MoveP for a configured arm."""
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    server = LabServer(
        adapters={"left": adapter}, dual_mode=False,
        host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/movep",
            data=json.dumps({
                "position": [0.42, 0.10, 0.20],
                "speed": 0.25,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1.0) as response:
            result = json.load(response)

        assert result == {"ok": True, "msg": "movep queued"}
        assert arm.command_received.wait(timeout=1.0)
        assert any(call[0] == "movep" for call in arm.calls)
    finally:
        server.shutdown()
        server.server_close()
        adapter.close()


def test_web_ui_exposes_drag_ball_for_movep():
    """The served browser UI offers target-ball control for MoveP."""
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    server = LabServer(
        adapters={"left": adapter}, dual_mode=False,
        host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/static/app.js", timeout=1.0) as response:
            script = response.read().decode()

        assert "TransformControls" in script
        assert "control.setSpace('world')" in script
        assert "pose_frame" in script
        assert "R_world_base" in script
        assert "base_position" in script
        assert "SCENE_R_BASE" not in script
        assert "ROOT_ROT" not in script
        assert "loadTool(side)" in script
        assert "robot.links && robot.links.Link7" in script
        assert "/assets/tools/${spec.urdf}" in script
        assert "applyToolOpening" in script
        assert "commandGripperOpening" in script
        assert "gripper/position" in script
        assert "释放后发送一次命令" in script
        assert "jointToolGhosts" in script
        assert "cartesianToolGhosts" in script
        assert "pointercancel" in script
        assert "movejPending" in script
        assert "state.end_effector || state.pose" in script
        assert "previewLocked[side] ? preview[side]" in script
        assert "Rz(yaw) * Ry(pitch) * Rx(roll)" in script
        assert "'ZYX'" in script
        assert "CARTESIAN_MAX_STEP" not in script
        assert "hasNewError" in script
        assert '${side}_pv_act' in script
        assert "activateCartesianBall(side, 'movep', true)" in script
        assert "sendCartesian(s, mode)" in script
    finally:
        server.shutdown()
        server.server_close()
        adapter.close()


def test_tool_urdf_and_mesh_assets_are_served():
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left", tool_model="omnipicker")
    server = LabServer(
        adapters={"left": adapter}, dual_mode=False,
        host="127.0.0.1", port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        for model, base_mesh in (
            ("omnipicker", "gripper_base_link.STL"),
            ("woan_picker", "base_link.STL"),
        ):
            with urlopen(
                f"http://{host}:{port}/assets/tools/{model}.urdf",
                timeout=1.0,
            ) as response:
                urdf = response.read().decode()
            assert f'name="{model}"' in urdf
            assert f'/assets/meshes/grippers/{model}/{base_mesh}' in urdf

            with urlopen(
                f"http://{host}:{port}/assets/meshes/grippers/{model}/"
                f"{base_mesh}",
                timeout=1.0,
            ) as response:
                mesh = response.read()
                assert response.status == 200
                assert len(mesh) > 100_000
    finally:
        server.shutdown()
        server.server_close()
        adapter.close()


def test_second_blocking_motion_is_rejected_while_first_is_running():
    """A new Cartesian command cannot cancel/race an active trajectory."""
    arm = RecordingArm()
    arm.motion_release.clear()
    adapter = ArmAdapter(arm, side="left")
    try:
        assert adapter.movej([0.05] + [0.0] * 6)["ok"]
        assert arm.command_received.wait(timeout=1.0)

        result = adapter.movep([0.42, 0.10, 0.20])
        assert result == {"ok": False, "msg": "motion already in progress"}
    finally:
        arm.motion_release.set()
        adapter.close()


def test_ik_failure_recovers_worker_and_reports_new_error_sequence():
    """An unreachable target is held safely and does not poison later commands."""
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    try:
        arm.fail_movep = True
        assert adapter.movep([1.0, 1.0, 1.0])["ok"]
        assert arm.command_received.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        state = adapter.get_state_dict()
        while not state["fault"] and time.monotonic() < deadline:
            time.sleep(0.01)
            state = adapter.get_state_dict()

        assert state["fault"] == "IK target unreachable"
        assert state["error_seq"] == 1
        assert sum(call[0] == "stop" for call in arm.calls) >= 1

        arm.fail_movep = False
        arm.command_received.clear()
        assert adapter.movep([0.42, 0.10, 0.20])["ok"]
        assert arm.command_received.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while adapter.get_state_dict()["fault"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert adapter.get_state_dict()["fault"] == ""
    finally:
        adapter.close()


def test_blocking_motion_stops_active_servo_before_dispatch():
    """MoveJ and other blocking motions cannot race the 100 Hz servo loop."""
    arm = RecordingArm()
    adapter = ArmAdapter(arm, side="left")
    try:
        assert adapter.servo_start(kp=20, kd=1)["ok"]
        assert adapter.movej([0.05] + [0.0] * 6)["ok"]

        result = adapter.servo_target([0.0] * 7)
        assert result == {"ok": False, "msg": "servo not active"}
    finally:
        adapter.close()
