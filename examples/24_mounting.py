"""
Example 15: 安装方式 (Mounting) — 侧装、倒装、肩装配置

演示如何配置不同安装方式，使 SDK 自动处理:
  1. 重力补偿方向 (freedrive/力控)
  2. 坐标系变换 (世界坐标系模式)
  3. TCP 工具偏移

实际场景:
  - 桌面支架侧装 (最常见): Z 朝左/右
  - 人形肩装: Z 朝外
  - 天花板/龙门架倒装: Z 朝下

Run: python examples/24_mounting.py
"""

import numpy as np
from onero_sdk import Arm, MountingPreset, MountingConfig


def example_basic_mounting():
    """基本安装方式配置。"""
    print("=" * 50)
    print("1. 基本安装方式")
    print("=" * 50)

    # 方式一: 字符串 (最简洁)
    arm = Arm("A1R", channel="can0", mounting="side_mount_left")

    # 方式二: 枚举 (有 IDE 补全)
    arm = Arm("A1R", channel="can0", mounting=MountingPreset.SIDE_MOUNT_LEFT)

    # 方式三: 完整配置 (可微调)
    config = MountingConfig(
        preset=MountingPreset.SIDE_MOUNT_LEFT,
        yaw_offset_deg=2.0,    # 支架安装稍有偏斜，补偿 2°
        pitch_offset_deg=-1.0,  # 支架稍微前倾
    )
    arm = Arm("A1R", channel="can0", mounting=config)

    # 查看配置
    print(f"  安装方式: {arm.mounting.preset.value}")
    print(f"  基座系重力方向: {arm.gravity_direction}")
    print(f"  R_world_base:\n{arm.R_world_base}")


def example_side_mount_freedrive():
    """侧装下的 freedrive — 重力补偿自动修正。"""
    print("\n" + "=" * 50)
    print("2. 侧装 freedrive (重力补偿自动修正)")
    print("=" * 50)

    # 侧装左臂: 重力方向在基座系下为 [0, +9.81, 0]
    # SDK 自动设置 pinocchio 模型的重力方向
    arm = Arm("A1R", channel="can0", mounting="side_mount_left")
    arm.enable()

    # freedrive 内部会调用 gravity_torques()
    # 由于重力方向已正确设置，补偿力矩正确 → 松手后关节不会下垂
    arm.freedrive_enable(damping=0.5)

    print("  零力拖动已启用，重力方向自动修正")
    print("  按 Ctrl+C 退出...")

    try:
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    arm.freedrive_disable()
    arm.disable()


def example_world_frame():
    """世界坐标系模式 — 用直觉方向控制。"""
    print("\n" + "=" * 50)
    print("3. 世界坐标系模式")
    print("=" * 50)

    # world_frame=True: 所有坐标 API 自动转换到世界坐标系
    # 世界坐标系: X=前(远离操作员), Y=左, Z=上
    arm = Arm("A1R", channel="can0",
              mounting="side_mount_left",
              world_frame=True)
    arm.enable()

    # 获取当前位姿 — 直接是世界坐标系
    pos, rot = arm.get_pose()
    print(f"  末端位置 (世界坐标系): x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    # 运动指令也在世界坐标系下
    # "向前移动 10cm" → x += 0.1
    target = pos.copy()
    target[0] += 0.1
    arm.movel(target, rot, duration=2.0)
    print("  向前移动 10cm 完成")

    # "向下移动 5cm" → z -= 0.05
    target[2] -= 0.05
    arm.movel(target, rot, duration=1.5)
    print("  向下移动 5cm 完成")

    # 速度控制: "以 0.05 m/s 向左移动"
    # Y 轴 = 左方向
    arm.speedl([0, 0.05, 0, 0, 0, 0], duration=2.0)
    print("  向左匀速移动完成")

    arm.disable()


def example_tcp_offset():
    """TCP 工具偏移 — 夹爪/灵巧手。"""
    print("\n" + "=" * 50)
    print("4. TCP 工具偏移")
    print("=" * 50)

    arm = Arm("A1R", channel="can0", mounting="side_mount_left")
    arm.enable()

    # 无 TCP: get_pose 返回法兰坐标
    pos_flange, rot_flange = arm.get_pose()
    print(f"  法兰位置: {pos_flange}")

    # 设置 TCP: 夹爪尖端在法兰 Z 方向 15cm 处
    arm.set_tcp([0.0, 0.0, 0.15])

    # 现在 get_pose 返回夹爪尖端坐标
    pos_tcp, rot_tcp = arm.get_pose()
    print(f"  TCP 位置: {pos_tcp}")
    print(f"  偏移量:   {np.linalg.norm(pos_tcp - pos_flange):.3f} m")

    # movel 目标也是 TCP 目标 — "夹爪尖端移到这个位置"
    arm.movel([0.3, 0.0, 0.2, 0, 0, 0], duration=2.0)

    # 清除 TCP
    arm.clear_tcp()
    arm.disable()


def example_force_mode_side_mount():
    """侧装下的力控 — 世界坐标系下 "向下推"。"""
    print("\n" + "=" * 50)
    print("5. 侧装力控 (世界坐标系)")
    print("=" * 50)

    arm = Arm("A1R", channel="can0",
              mounting="side_mount_left",
              world_frame=True)
    arm.enable()

    # 移动到工作位置
    arm.movel([0.3, 0.0, 0.2, 0, 0, 0], duration=2.0)

    # "Z 方向施加 5N 向下的力" — 直觉操作
    # selection: Z 方向力控, 其他方向位置控制
    import time
    for _ in range(200):
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],   # 只 Z 轴力控
            wrench=[0, 0, -5.0, 0, 0, 0],   # Z 轴 -5N (向下)
            limits=[0, 0, 0.1, 0, 0, 0],    # Z 方向速度限制 0.1 m/s
        )
        time.sleep(1.0 / 200)

    arm.force_mode_disable()
    arm.disable()


def example_all_presets():
    """打印所有安装方式的重力方向。"""
    print("\n" + "=" * 50)
    print("6. 所有安装方式预设")
    print("=" * 50)

    from onero_sdk.mounting import gravity_in_base_frame

    for preset in MountingPreset:
        if preset == MountingPreset.CUSTOM:
            continue
        config = MountingConfig(preset=preset)
        g = gravity_in_base_frame(config)
        print(f"  {preset.value:20s} → gravity_base = "
              f"[{g[0]:+.2f}, {g[1]:+.2f}, {g[2]:+.2f}]")


if __name__ == "__main__":
    # 只运行不需要硬件的示例
    example_all_presets()

    # 以下需要连接真实硬件:
    # example_basic_mounting()
    # example_side_mount_freedrive()
    # example_world_frame()
    # example_tcp_offset()
    # example_force_mode_side_mount()
