"""
Basic movel example — move end-effector along a straight line.

Requires:
    pip install -e ".[ik]"
"""

import numpy as np
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()
try:
    # Leave the fully extended zero pose before solving Cartesian motion.
    ready_joints = [0.3, -0.8, 0.5, -0.5, 0.2, 0.3, 0.0]
    arm.movej(ready_joints, duration=3.0)

    # Keep the current tool orientation and move 15 cm along base +X.
    start_pos, start_rot = arm.get_pose()
    target_pos = start_pos + np.array([0.15, 0.0, 0.0])

    arm.movel(
        position=target_pos,
        rotation=start_rot,
        duration=3.0,
    )

    # Get EE pose after motion
    pos, rot = arm.get_pose()
    error_mm = np.linalg.norm(pos - target_pos) * 1000.0
    print(f"EE position: {pos}")
    print(f"Target error: {error_mm:.2f} mm")
    print(f"EE rotation:\n{rot}")
finally:
    arm.disable()
