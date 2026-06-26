"""
Basic movel example — move end-effector along a straight line.

Requires: pip install onero-sdk[ik]
"""

import numpy as np
from onero_sdk import Arm

with Arm("A1", channel="can0") as arm:
    # Move to a Cartesian pose in a straight line
    target_pos = [0.3, 0.0, 0.2]
    target_rot = np.eye(3)  # identity rotation (or use quaternion [x,y,z,w])

    arm.movel(
        position=target_pos,
        rotation=target_rot,
        duration=3.0,
    )

    # Get EE pose after motion
    pos, rot = arm.get_pose()
    print(f"EE position: {pos}")
    print(f"EE rotation:\n{rot}")
