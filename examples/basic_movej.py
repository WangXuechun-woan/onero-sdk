"""
Basic movej example — move to target joint angles.

Demonstrates the simplest use of the SDK high-level API.
"""

from onero_sdk import Arm

# Create arm (requires CAN interface "can0" to be UP)
with Arm("A1", channel="can0") as arm:
    # Move to target joint angles over 3 seconds
    arm.movej(
        joints=[0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0],
        duration=3.0,
    )

    # Read current state
    state = arm.get_state()
    print(f"Final joints: {state.joints}")
    print(f"Final torques: {state.torques}")

    # Home (back to zero)
    arm.home(duration=3.0)
