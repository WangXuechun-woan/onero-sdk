"""
Real-time servo control example — user manages the control loop.

Demonstrates servoj at 100Hz with a simple sine wave trajectory.
"""

import time
import math
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

freq = 100.0
dt = 1.0 / freq
duration = 5.0
steps = int(duration * freq)

try:
    for step in range(steps):
        t = step * dt

        # Generate sine wave targets for joints 1 and 3
        joints = [0.0] * 7
        joints[1] = 0.5 * math.sin(2 * math.pi * 0.5 * t)
        joints[3] = -0.3 * math.sin(2 * math.pi * 0.5 * t)

        # Send single-frame command
        state = arm.servoj(joints)

        # Optional: print state periodically
        if step % 100 == 0:
            print(f"t={t:.1f}s pos[1]={state.joints[1]:.3f} torque[1]={state.torques[1]:.3f}")

        time.sleep(dt)
finally:
    arm.disable()
