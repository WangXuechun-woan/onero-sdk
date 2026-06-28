"""
Example: Multi-waypoint trajectory with blending.

Demonstrates using TrajectoryPlanner to create smooth motion
through multiple waypoints without stopping at each one.
"""

from onero_sdk import Arm, TrajectoryPlanner, Waypoint

arm = Arm("A1R", channel="can0")

# Define waypoints
waypoints = [
    Waypoint(joints=[0.0, 0.5, 0.0, -0.3, 0.0, 0.0, 0.0], duration=2.0, blend=0.3),
    Waypoint(joints=[0.3, 0.8, 0.2, -0.5, 0.0, 0.1, 0.0], duration=2.0, blend=0.3),
    Waypoint(joints=[-0.2, 0.3, -0.1, -0.8, 0.0, 0.2, 0.0], duration=2.0, blend=0.3),
    Waypoint(joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0, blend=0.0),  # final: no blend
]

# Plan the trajectory
planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
for wp in waypoints:
    planner.add_waypoint(wp)

with arm:
    # Compute blended trajectory from current position
    start = arm.state.joints
    frames = planner.compute(start)
    print(f"Trajectory: {len(frames)} frames")

    # Execute
    arm.execute_trajectory(frames)

    print("Done.")
