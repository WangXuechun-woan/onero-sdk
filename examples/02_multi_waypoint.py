"""
连续多点运动 — 工厂巡检 / 码垛等场景。

演示如何让机械臂依次经过一系列位置（waypoints），
以及使用 blending 实现不停顿的平滑过渡。

运行:
    python examples/02_multi_waypoint.py
"""

from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # === 方式 1: 逐点移动（每个点都停顿） ===
    waypoints = [
        [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0],
        [0.3, -0.8, 0.5, -0.5, 0.2, 0.3, 0.0],
        [-0.2, -0.3, 0.1, -1.2, -0.1, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 回零
    ]

    print("逐点移动 (有停顿):")
    for i, wp in enumerate(waypoints):
        arm.movej(wp, duration=2.0)
        print(f"  到达第 {i+1} 个点")

    # === 方式 2: 轨迹融合（平滑不停顿） ===
    from onero_sdk import TrajectoryPlanner, Waypoint

    print("\n轨迹融合 (无停顿):")
    planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
    planner.add_waypoint(Waypoint(joints=[0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0, blend=0.3))
    planner.add_waypoint(Waypoint(joints=[0.3, -0.8, 0.5, -0.5, 0.2, 0.3, 0.0], duration=2.0, blend=0.3))
    planner.add_waypoint(Waypoint(joints=[-0.2, -0.3, 0.1, -1.2, -0.1, 0.8, 0.0], duration=2.0, blend=0.3))
    planner.add_waypoint(Waypoint(joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0, blend=0.0))

    start_joints = list(arm.state.joints)
    trajectory = planner.compute(start_joints)
    arm.execute_trajectory(trajectory)
    print("  轨迹完成!")

finally:
    arm.disable()
