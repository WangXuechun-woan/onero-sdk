"""Tests for the optional python-can transport adapter."""

import sys
from types import SimpleNamespace

import pytest


class FakeMessage:
    def __init__(self, arbitration_id, data=b"", is_extended_id=False, dlc=None):
        self.arbitration_id = arbitration_id
        self.data = bytearray(data)
        self.is_extended_id = is_extended_id
        self.dlc = len(data) if dlc is None else dlc


class FakeBus:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.sent = []
        self.shutdown_calls = 0

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        return self.messages.pop(0) if self.messages else None

    def shutdown(self):
        self.shutdown_calls += 1


def test_python_can_bus_sends_and_batches_matching_responses(monkeypatch):
    created = []
    backend = FakeBus([
        FakeMessage(0x777, b"stale"),
        FakeMessage(0x11, b"left"),
        FakeMessage(0x12, b"right"),
    ])

    def make_bus(**kwargs):
        created.append(kwargs)
        return backend

    fake_can = SimpleNamespace(Bus=make_bus, Message=FakeMessage)
    monkeypatch.setitem(sys.modules, "can", fake_can)

    from onero_sdk import PythonCanBus

    bus = PythonCanBus(
        interface="slcan",
        channel="/dev/ttyACM0@1000000",
        bitrate=1_000_000,
    )
    bus.send(0x21, b"\x01\x02")
    bus.send(0x80000000 | 0x12345, b"\x03")
    responses = bus.recv_batch({0x11, 0x12}, timeout=0.05)
    bus.close()
    bus.close()

    assert created == [{
        "interface": "slcan",
        "channel": "/dev/ttyACM0@1000000",
        "bitrate": 1_000_000,
    }]
    assert backend.sent[0].arbitration_id == 0x21
    assert bytes(backend.sent[0].data) == b"\x01\x02"
    assert not backend.sent[0].is_extended_id
    assert backend.sent[1].arbitration_id == 0x12345
    assert backend.sent[1].is_extended_id
    assert set(responses) == {0x11, 0x12}
    assert responses[0x11].data == b"left"
    assert responses[0x12].data == b"right"
    assert backend.shutdown_calls == 1


def test_arm_accepts_one_injected_can_transport():
    from onero_sdk import Arm

    class Transport:
        channel = "/dev/serial/by-id/test-arm"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    transport = Transport()
    arm = Arm("A1R", can_bus=transport)
    arm.close()

    assert transport.closed


def test_arm_rejects_ambiguous_can_transport_selection():
    from onero_sdk import Arm, OneroConfigError

    with pytest.raises(OneroConfigError, match="exactly one"):
        Arm("A1R", channel="can0", can_bus=object())
