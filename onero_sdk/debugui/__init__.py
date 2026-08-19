"""
onero SDK Debug UI
==================
A lightweight web-based debug tool for the onero SDK.

Quick start::

    # CLI — real hardware, single arm
    python -m onero_sdk.debugui --arm right --channel slcan0

    # CLI — dual arm
    python -m onero_sdk.debugui --arm both --left-channel slcan1 --right-channel slcan0

    # CLI — simulation (no hardware needed)
    python -m onero_sdk.debugui --arm both --sim

    # Embedded — pass an existing Arm instance
    from onero_sdk.debugui import start_lab
    arm = Arm("A1R", channel="slcan0")
    arm.enable()
    start_lab(arm, port=8090, block=False)

    # Embedded — dual arm
    from onero_sdk.debugui import start_lab_dual
    dual = DualArm.create(left_channel="slcan1", right_channel="slcan0",
                          base_position_left=[0, 0.18, 0],
                          base_position_right=[0, -0.18, 0])
    start_lab_dual(dual, port=8090, block=True)
"""

from .server import start_lab, start_lab_dual, LabServer

__all__ = ["start_lab", "start_lab_dual", "LabServer"]
