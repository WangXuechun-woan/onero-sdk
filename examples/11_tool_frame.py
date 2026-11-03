"""
工具坐标系 (TCP) — 夹爪/灵巧手的坐标变换。

本示例先回到关节零位，再进入非奇异演示姿态。设置 15cm TCP 后，
沿工具自身 Z 轴移动 2cm，验证 TCP 偏移和工具坐标系的相对运动。

运行:
    pip install onero-sdk[ik]
    python3 examples/11_tool_frame.py

真机风险提示:
    运行前确认机械臂周围无障碍物，15cm 工具长度与实际安装一致，并确认
    机械臂已经有可靠支撑。示例结束时的 disable() 是失能，不是普通停止或紧急停止。
"""

import math

from onero_sdk import Arm


DEMO_START_JOINTS = [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]


def _report_error(stage: str, error: Exception) -> None:
    """打印错误类型和 SDK 提供的详细原因。"""
    print(f"   ✗ {stage}失败: {type(error).__name__}: {error}")


def _check_ik_feedback(arm: Arm, tolerance: float = 0.02) -> bool:
    """确认反馈关节在 IK 模型允许范围内。"""
    invalid = []
    for index, (joint, (lower, upper)) in enumerate(
            zip(arm.joints, arm.get_joint_limits())):
        if joint < lower - tolerance or joint > upper + tolerance:
            invalid.append((index + 1, joint, lower, upper))

    if not invalid:
        return True

    print("   ✗ 未发送 TCP 运动：当前反馈超出 IK 模型限位")
    for index, joint, lower, upper in invalid:
        print(
            f"     J{index}: feedback={joint:.6f} rad, "
            f"URDF=[{lower:.6f}, {upper:.6f}] rad, "
            f"超限容差={tolerance:.3f} rad"
        )
    print("   请检查型号、左右臂配置、编码器零点和 URDF 限位。")
    return False


def _run_tool_z_movel(arm: Arm) -> bool:
    """沿当前工具自身 Z 轴移动 2cm。"""
    try:
        # frame="tool" 的平移量是 TCP 坐标系下的相对位移。
        arm.movel([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], duration=2.0, frame="tool")
        print("   ✓ TCP 已沿工具自身 Z 轴移动 2cm；请现场确认实际轨迹")
        return True
    except Exception as error:
        _report_error("沿工具 Z 轴移动 2cm", error)
        return False


arm = None
try:
    print("真机示例：请确认工具实际尺寸、安装方向和机械臂支撑条件。")
    arm = Arm("A1R", channel="can0")
    arm.enable()

    print("正在从当前位置回到关节零位...")
    arm.home(duration=3.0)
    print("已回到关节零位。")

    print("正在从零位移动到 IK 演示姿态...")
    arm.movej(DEMO_START_JOINTS, duration=2.0)
    print("已到达 IK 演示姿态。")

    # === 1. 设置 TCP 偏移 ===
    print("\n1. 设置 TCP (工具坐标系偏移)")
    print("   假设末端安装了一个 15cm 的平行夹爪:")
    print("   夹爪尖端在法兰面 Z 轴正方向 0.15m 处")
    arm.set_tcp(position=[0.0, 0.0, 0.15])
    print("   TCP 设置完成: position=[0, 0, 0.15]")
    print("   get_pose() 返回夹爪尖端位姿，笛卡尔目标也以 TCP 为参考")

    # === 2. 使用 TCP 后沿工具自身 Z 轴移动 ===
    print("\n2. 沿工具自身 Z 轴移动 2cm")
    tcp_pos, _ = arm.get_pose(frame="base")
    print(
        "   当前 TCP 位姿(base): "
        f"[{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]"
    )
    print("   目标：TCP 坐标系相对位移 [0.00, 0.00, 0.02] m")
    if _check_ik_feedback(arm) and _run_tool_z_movel(arm):
        pos_after, _ = arm.get_pose(frame="base")
        print(
            "   运动后 TCP 反馈位姿(base): "
            f"[{pos_after[0]:.3f}, {pos_after[1]:.3f}, {pos_after[2]:.3f}]"
        )

    # === 3. 不同工具的 TCP 设置举例 ===
    print("\n3. 常见工具的 TCP 偏移示例:")
    print("   (a) 平行夹爪 (直装, 长 15cm): arm.set_tcp(position=[0, 0, 0.15])")
    print("   (b) 灵巧手 (长 20cm, 指尖作为 TCP): arm.set_tcp(position=[0, 0, 0.20])")
    print(
        "   (c) 吸盘 (45° 安装角, 长 10cm): "
        "arm.set_tcp(position=[0, 0, 0.10], rotation=[0, 0.785, 0])"
    )
    print(
        "   (d) 焊枪 (偏心安装 + 倾斜): "
        "arm.set_tcp(position=[0.05, 0, 0.12], rotation=[0, 0.52, 0])"
    )

    # === 4. 多工具切换 ===
    print("\n4. 多工具快速切换")
    tools = {
        "gripper": {"position": [0.0, 0.0, 0.15]},
        "camera": {"position": [0.0, 0.05, 0.08]},
        "suction": {
            "position": [0.0, 0.0, 0.10],
            "rotation": [0.0, math.pi / 4, 0.0],
        },
    }
    for name, tcp_args in tools.items():
        try:
            arm.set_tcp(**tcp_args)
            print(f"   切换到 '{name}': {tcp_args}")
        except Exception as error:
            _report_error(f"切换工具 '{name}'", error)

    # === 5. 重置 TCP (回到法兰面) ===
    print("\n5. 重置 TCP (法兰面为参考)")
    arm.clear_tcp()
    print("   TCP 已重置，运动指令以法兰面为参考")

    # === 6. TCP 的坐标变换公式 ===
    print("\n6. 坐标变换关系:")
    print("   T_base_tcp = T_base_flange × T_flange_tcp")
    print("   T_base_flange = FK(joints)        ← 正运动学计算")
    print("   T_flange_tcp  = set_tcp() 设定    ← 用户配置")
    print("   T_base_tcp    = arm.get_pose()    ← SDK 返回 (含 TCP 偏移)")
    print("   T_flange_tcp  = T_base_flange⁻¹ × T_base_tcp")
    print("   T_base_flange = T_base_tcp × T_flange_tcp⁻¹")

except Exception as error:
    _report_error("示例初始化或运动", error)
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
