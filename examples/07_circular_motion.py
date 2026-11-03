"""
圆弧运动 (movec) — 弧形擦拭、弧形焊接等场景。

movec 通过 3 点定义圆弧: 起点(当前位置) → 经过点 → 终点。
本示例不假设机械臂当前位于某个固定关节姿态，而是从当前反馈位姿
生成与 docs/assets/movec.gif 类似的明显 X-Z 平面圆弧，弦长约 14cm。
末端姿态保持不变，movec 会在真正发送指令前预检所有路径点的 IK。

运行:
    pip install onero-sdk[ik]
    python3 examples/07_circular_motion.py

真机风险提示:
    运行前确认机械臂周围无障碍物，并确认工具、安装方式和安全限位配置正确。
    示例结束时调用 disable() 只是失能，不是普通停止或紧急停止；失能前请确认
    机械臂已经有可靠支撑，避免因去除保持力矩而下落。
"""

import numpy as np

from onero_sdk import Arm

DEMO_START_JOINTS = [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]


def _report_error(stage: str, error: Exception) -> None:
    """打印错误类型和 SDK 提供的详细原因。"""
    print(f"   ✗ {stage}失败: {type(error).__name__}: {error}")


def _rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """将 get_pose() 的旋转矩阵转换为 movec 使用的旋转向量。"""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])

    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _check_ik_feedback(arm: Arm, tolerance: float = 0.02) -> bool:
    """在调用 movec 前确认反馈关节与 IK 模型使用同一限位范围。"""
    joints = arm.joints
    limits = arm.get_joint_limits()
    invalid = [
        (index + 1, joint, lower, upper)
        for index, (joint, (lower, upper)) in enumerate(zip(joints, limits))
        if joint < lower - tolerance or joint > upper + tolerance
    ]
    if not invalid:
        return True

    print("   ✗ 未发送圆弧：当前反馈超出 IK 模型限位，无法安全初始化 IK")
    for joint_index, joint, lower, upper in invalid:
        print(
            f"     J{joint_index}: feedback={joint:.6f} rad, "
            f"URDF=[{lower:.6f}, {upper:.6f}] rad, "
            f"超限容差={tolerance:.3f} rad"
        )
    print("   请检查 A1R 型号/左右臂配置、编码器零点和 URDF 限位；")
    print("   不要通过放宽 IK 容差或直接裁剪关节值来绕过此检查。")
    return False


def _run_base_arc(arm: Arm, name: str, start_pos: np.ndarray,
                  start_rot: np.ndarray, via_xyz: tuple[float, float, float],
                  target_xyz: tuple[float, float, float]) -> bool:
    """在基座 X-Z 方向执行圆弧；IK 失败时缩小圆弧并重试。"""
    last_error = None
    start_rotvec = _rotation_matrix_to_rotvec(start_rot)
    for scale in (1.0, 0.75, 0.5, 0.25):
        via_pos = start_pos + np.asarray(via_xyz, dtype=float) * scale
        target_pos = start_pos + np.asarray(target_xyz, dtype=float) * scale
        via_pose = via_pos.tolist() + start_rotvec.tolist()
        target_pose = target_pos.tolist() + start_rotvec.tolist()
        try:
            arm.movec(
                via_pose=via_pose,
                target_pose=target_pose,
                duration=5.0,
                frame="base",
            )
            print(
                f"   ✓ {name} 的 SDK 调用返回成功 (局部缩放: {scale:g})；"
                "请现场确认实际轨迹"
            )
            return True
        except Exception as error:
            last_error = error
            if isinstance(error, ValueError) and "feedback exceeds URDF limits" in str(error):
                print("   - 当前反馈仍超出 URDF 限位，停止缩放重试")
                break
            print(f"   - 局部缩放 {scale:g} 的基座 X-Z 圆弧 IK/运动检查未通过")

    _report_error(name, last_error)
    return False


arm = None
try:
    print("真机示例：请先确认工作空间、工具和安全支撑条件。")
    arm = Arm("A1R", channel="can0")
    arm.enable()

    # 先用关节空间回到软件定义的零位；这不是零点标定，也不会执行 IK。
    print("正在从当前位置回到关节零位...")
    arm.home(duration=3.0)
    print("已回到关节零位。")

    # 全关节零位接近奇异位形；先从零位移动到已知的非奇异关节姿态，
    # 再执行保持姿态的三维圆弧，避免 IK 在奇异点附近无法收敛。
    print("正在从零位移动到 IK 演示姿态...")
    arm.movej(DEMO_START_JOINTS, duration=2.0)
    print("已到达 IK 演示姿态，开始读取圆弧起点。")

    # 使用零偏移 TCP 让示例的 TCP 与法兰重合；圆弧本身使用 base frame，
    # 因此轨迹平面固定跟随机械臂基座方向，而不会随末端姿态旋转。
    arm.set_tcp(position=[0.0, 0.0, 0.0])
    start_pos, start_rot = arm.get_pose(frame="base")
    print(
        "当前起点(base): "
        f"[{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]"
    )

    if _check_ik_feedback(arm):
        print("\n1. 逆时针基座 X-Z 圆弧（参考 movec.gif）")
        print("   经过点偏移: [0.06, 0.00, 0.10] m；终点偏移: [0.14, 0.00, 0.01] m")
        _run_base_arc(
            arm, "逆时针基座 X-Z 圆弧", start_pos, start_rot,
            (0.06, 0.00, 0.10), (0.14, 0.00, 0.01),
        )

        print("\n正在回到相同的圆弧起点...")
        arm.movej(DEMO_START_JOINTS, duration=2.0)
        start_pos, start_rot = arm.get_pose(frame="base")
        print("\n2. 顺时针基座 X-Z 圆弧")
        print("   经过点偏移: [0.06, 0.00, -0.10] m；终点偏移: [0.14, 0.00, 0.01] m")
        _run_base_arc(
            arm, "顺时针基座 X-Z 圆弧", start_pos, start_rot,
            (0.06, 0.00, -0.10), (0.14, 0.00, 0.01),
        )

except Exception as error:
    _report_error("示例初始化或状态读取", error)
finally:
    if arm is not None:
        try:
            arm.clear_tcp()
        except Exception as error:
            _report_error("清理 TCP", error)
        try:
            if arm.enabled:
                arm.disable()
        except Exception as error:
            _report_error("失能机械臂", error)
