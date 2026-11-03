"""
Example: Velocity mode control — speedj / speedl.

Demonstrates:
- speedj: move joints at constant velocity
- stop: emergency deceleration
"""

from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # Move joint 1 at 0.5 rad/s for 3 seconds
    velocities = [0.0, 0.0, 0.0, -0.3, 0.0, 0.1, 0.0]
    arm.speedj(velocities, acceleration=1.0, duration=5.0)

    # Move multiple joints simultaneously
    velocities = [0.2, 0.0, 0.0, 0.4, 0.0, -0.2, 0.0]
    arm.speedj(velocities, acceleration=2.0, duration=5.0)

    # Return home
    arm.home(duration=5.0)

finally:
    arm.disable()
