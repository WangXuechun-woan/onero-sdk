"""Arm + physical CAN gripper pick-and-place example.

The gripper shares the Arm-owned CAN transport. Select the installed protocol
explicitly: ``OmniPickerGripper`` for the legacy byte protocol, or
``OneroGripper`` for the self-developed MIT protocol. Visual model, TCP,
payload, and gravity compensation are configured separately.

Hardware example:
    - A1R arm and gripper on can0
    - Gripper command ID 0x08
"""

import time

from onero_sdk import Arm, OmniPickerGripper


# Conservative legacy closing force (0x20/255). For self-developed hardware,
# explicitly replace this with: gripper = OneroGripper()
gripper = OmniPickerGripper(can_id=0x08, force=0x20, velocity=0xFF)
arm = Arm(
    "A1R",
    channel="can0",
    end_effector=gripper,
    with_gripper=True,  # independent empirical gravity profile
)

PRE_GRASP = [0.0, 0.8, 0.0, -1.0, 0.0, 0.5, 0.0]
GRASP_POS = [0.0, 1.0, 0.0, -1.2, 0.0, 0.5, 0.0]
PLACE_POS = [0.5, 0.8, 0.0, -1.0, 0.0, 0.5, 0.0]

with arm:
    arm.gripper.open()
    time.sleep(0.5)
    arm.movej(PRE_GRASP, duration=2.0)
    arm.movej(GRASP_POS, duration=1.5)

    arm.gripper.grasp()
    time.sleep(0.5)
    arm.movej(PRE_GRASP, duration=1.5)
    arm.movej(PLACE_POS, duration=2.0)

    arm.gripper.open()
    time.sleep(0.5)
    arm.home(duration=3.0)

# Context exit detaches without disabling a gravity-loaded arm. If torque must
# be removed, physically support the arm and call arm.disable() *inside* the
# context before it exits (or avoid the context and use close(disable=True)).
print("Pick-and-place complete; transport detached without disabling motors.")
