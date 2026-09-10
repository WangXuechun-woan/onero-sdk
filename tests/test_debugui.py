"""Debug UI behavior tests through the public ArmAdapter API."""

import json
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np

from onero_sdk.backend import ArmState
from onero_sdk.debugui.server import ArmAdapter, LabServer


class RecordingArm:
    """Minimal arm contract used to observe commands issued by Debug UI."""

    num_joints = 7
    enabled = True
    is_freedrive = False
    is_moving = False
    model = SimpleNamespace(name="A1L")

    def __init__(self):
        self.calls = []
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
        assert "ROOT_ROT" not in script
        assert "CARTESIAN_MAX_STEP" not in script
        assert "hasNewError" in script
        assert '${side}_pv_act' in script
        assert "activateCartesianBall(side, 'movep', true)" in script
        assert "sendCartesian(s, mode)" in script
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
