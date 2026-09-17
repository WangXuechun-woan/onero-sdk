"""Hardware gripper behavior through the public SDK API."""

from onero_sdk import Arm, OmniPickerGripper, OneroGripper
from onero_sdk.can_bus import CanFrame


class FakeBus:
    channel = "vcan-test"

    def __init__(self, responses=None):
        self.sent = []
        self.responses = list(responses or [])
        self.closed = False

    def send(self, can_id, data):
        self.sent.append((can_id, bytes(data)))

    def recv(self, timeout=0.01):
        return self.responses.pop(0) if self.responses else None

    def recv_until(self, target_id, timeout=0.05):
        while self.responses:
            frame = self.responses.pop(0)
            if frame.arbitration_id == target_id:
                return frame
        return None

    def recv_batch(self, target_ids, timeout=0.05):
        return {}

    def flush_rx(self):
        return None

    def close(self):
        self.closed = True


def test_omnipicker_sends_authoritative_can_frame_and_uses_feedback():
    bus = FakeBus([
        CanFrame(0x08, bytes([0x00, 0x01, 0x7F, 0x22, 0x33, 0, 0, 0]), 8)
    ])
    gripper = OmniPickerGripper(
        bus=bus, can_id=0x08, force=0x20, velocity=0x30,
        acceleration=0x40, deceleration=0x50,
    )

    gripper.enable()
    gripper.set_position(0.5)

    assert bus.sent == [
        (0x08, bytes([0x00, 0x80, 0x20, 0x30, 0x40, 0x50, 0x00, 0x00]))
    ]
    assert gripper.get_position() == 0x7F / 0xFF
    state = gripper.state
    assert state.fault_code == 0
    assert state.motion_state == 1
    assert state.velocity == 0x22 / 0xFF
    assert state.force == 0x33 / 0xFF


def test_onero_gripper_uses_mit_protocol_and_physical_feedback_range():
    bus = FakeBus([
        CanFrame(0x18, bytes.fromhex("1800000000001415"), 8),
        CanFrame(0x18, bytes.fromhex("188a8a7ff7ff1415"), 8),
    ])
    gripper = OneroGripper(bus=bus, model="A1L")

    gripper.enable()
    gripper.set_position(0.5)

    assert bus.sent == [
        (0x08, bytes.fromhex("fffffffffffffffc")),
        (0x08, bytes.fromhex("8a8a7ff0a33337ff")),
    ]
    assert abs(gripper.get_position() - 0.5) < 2e-5
    assert gripper.state.motion_state == 1  # enabled
    assert gripper.state.fault_code == 0


def test_onero_refresh_recovers_enabled_state_from_hardware_feedback():
    bus = FakeBus([
        CanFrame(0x18, bytes.fromhex("188a8a7ff7ff1415"), 8),
    ])
    gripper = OneroGripper(bus=bus, model="A1L")

    assert gripper.enabled is False
    gripper.refresh()

    assert bus.sent == [
        (0x7FF, bytes.fromhex("0800cc0000000000")),
    ]
    assert gripper.enabled is True
    assert gripper.state.motion_state == 1
    assert gripper.state.fault_code == 0


def test_onero_public_opening_direction_is_opposite_motor_position():
    bus = FakeBus([
        CanFrame(0x18, bytes.fromhex("1800000000001415"), 8),
        CanFrame(0x18, bytes.fromhex("1800000000001415"), 8),
    ])
    gripper = OneroGripper(bus=bus, model="A1L")
    gripper.enable()

    gripper.open()

    # Public 1.0=open maps to the measured H1 left endpoint raw=37/255.
    assert bus.sent[-1][0] == 0x08
    assert bus.sent[-1][1][:2] == bytes.fromhex("2525")
    assert abs(gripper.get_position() - 1.0) < 2e-5


def test_invalid_position_is_rejected_before_any_motion_frame():
    bus = FakeBus()
    gripper = OmniPickerGripper(bus=bus)
    gripper.enable()

    try:
        gripper.set_position(1.1)
        raise AssertionError("out-of-range command was accepted")
    except ValueError:
        pass

    assert bus.sent == []


def test_arm_applies_side_specific_onero_endpoint_calibration():
    bus = FakeBus([
        CanFrame(0x18, bytes.fromhex("1800000000001415"), 8),
        CanFrame(0x18, bytes.fromhex("18eeee7ff7ff1415"), 8),
    ])
    gripper = OneroGripper()
    arm = Arm("A1R", can_bus=bus, end_effector=gripper)
    try:
        gripper.enable()
        gripper.grasp()
        assert bus.sent[-1][1][:2] == bytes.fromhex("eeee")
        assert gripper.get_position() == 0.0
    finally:
        arm.close()


def test_arm_binds_gripper_to_its_single_owned_transport():
    bus = FakeBus([
        CanFrame(0x08, bytes([0, 0, 255, 0, 0, 0, 0, 0]), 8)
    ])
    gripper = OmniPickerGripper()
    arm = Arm("A1R", can_bus=bus, end_effector=gripper)
    try:
        gripper.enable()
        gripper.open()
        assert bus.sent[-1] == (
            0x08, bytes([0, 255, 0x20, 0xFF, 0xFF, 0xFF, 0, 0]))
    finally:
        arm.close()
    assert bus.closed is True
