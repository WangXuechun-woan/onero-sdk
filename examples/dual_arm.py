"""
Dual arm example — control two A1 arms simultaneously.
"""

from onero_sdk import DualArm

# Create dual arm from channels
dual = DualArm.create(
    model="A1",
    left_channel="can0",
    right_channel="can1",
)

with dual:
    # Move both arms to different poses
    dual.movej(
        left_joints=[0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0],
        right_joints=[0.0, -0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
        duration=3.0,
    )

    # Home both
    dual.home(duration=3.0)
