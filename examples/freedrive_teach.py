"""
Example: Freedrive (teach mode) + trajectory recording and replay.

Demonstrates:
1. Enter freedrive mode — user physically guides the arm
2. Record the trajectory during freedrive
3. Replay the recorded trajectory at different speeds
4. Save/load recordings to disk
"""

import time
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # Start recording before entering freedrive
    arm.start_recording()

    # Enter freedrive — arm can be guided by hand
    # (requires pinocchio: pip install onero-sdk[dynamics])
    arm.freedrive_enable(damping=0.3)
    print("Freedrive active — guide the arm. Press Ctrl+C to stop.")

    # Let the user guide for 10 seconds
    time.sleep(10.0)

    # Exit freedrive
    arm.freedrive_disable()

    # Get the recorded trajectory
    recording = arm.stop_recording()
    print(f"Recorded {recording.num_frames} frames, {recording.duration:.1f}s")

    # Save to disk
    recording.save("my_trajectory.json")
    # Or as compact numpy format:
    # recording.save_npz("my_trajectory.npz")

    # Replay at original speed
    print("Replaying at 1x speed...")
    arm.replay(recording, speed_factor=1.0)

    # Replay faster
    print("Replaying at 2x speed...")
    arm.replay(recording, speed_factor=2.0)

    # Return home
    arm.home(duration=3.0)

finally:
    arm.disable()
