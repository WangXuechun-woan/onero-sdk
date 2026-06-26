"""
运动学标定示例 — 单点约束法提高绝对定位精度。

原理:
    让末端从不同构型触碰同一个固定参考点（如桌角），
    由于 URDF 参数不完全精确，FK 计算出的位置会有分散。
    优化关节原点的微小修正量，使所有构型的 FK 输出收敛到同一点。

运行方式:
    python examples/kinematic_calibration_demo.py

该示例使用 MuJoCo 仿真演示流程（无需真实硬件）。
实际使用时，替换为真实 Arm 的关节角度读取即可。
"""
import sys
import numpy as np

sys.path.insert(0, '.')

from onero_sdk.fk import URDFChain
from onero_sdk.kinematic_calibration import KinematicCalibrator


def simulate_noisy_touches(fk: URDFChain, n_touches: int = 20, seed: int = 42):
    """
    Simulate touching a fixed point from multiple configurations.

    Strategy: pick a base config, then perturb joints while keeping the EE
    roughly at the same point (simulate the "true" robot having slightly
    different kinematics from the URDF model).
    """
    from onero_sdk.kinematic_calibration import _axis_angle_matrix
    rng = np.random.default_rng(seed)
    n_joints = fk.num_joints

    # Simulate "true" kinematics: URDF + small unknown offsets
    true_offsets = rng.uniform(-0.002, 0.002, size=(n_joints, 3))  # ±2mm

    def true_fk(q):
        T = np.eye(4)
        for i, (origin_T, axis) in enumerate(fk._joints):
            T_real = origin_T.copy()
            T_real[:3, 3] += true_offsets[i]
            T = T @ T_real
            R_joint = np.eye(4)
            R_joint[:3, :3] = _axis_angle_matrix(axis, q[i])
            T = T @ R_joint
        T = T @ fk._ee_fixed_transform
        return T[:3, 3]

    # Generate configs that all reach the same physical point
    # Start with a base config, compute the "true" EE position
    q_base = np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.3, -0.1])
    target_point = true_fk(q_base)

    # Perturb joints slightly — in real life these are different arm poses
    # touching the same point. Here we just use diverse configs.
    configs = [q_base.tolist()]
    for _ in range(n_touches - 1):
        q = q_base + rng.uniform(-0.3, 0.3, size=n_joints)
        configs.append(q.tolist())

    return configs, true_offsets


def main():
    import os
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sdk_dir = os.path.join(pkg_dir, '..', 'onero_sdk')

    # Find URDF
    urdf_path = os.path.join(sdk_dir, 'urdf', 'A1', 'a1_r.urdf')
    if not os.path.exists(urdf_path):
        print(f"URDF not found at {urdf_path}")
        print("Please adjust the path to your URDF file.")
        return

    print("=" * 60)
    print("  运动学标定 — 单点约束法")
    print("=" * 60)
    print()

    # Load FK chain
    fk = URDFChain(urdf_path)
    print(f"Loaded URDF: {fk.num_joints} joints")
    print()

    # Create calibrator
    calibrator = KinematicCalibrator(fk)

    # Simulate: generate configs (in real use, these come from physical touching)
    print("Recording touch configurations...")
    configs, true_offsets = simulate_noisy_touches(fk, n_touches=20)

    for i, q in enumerate(configs):
        pos = calibrator.record_touch(q=q)
        if i < 5 or i == len(configs) - 1:
            print(f"  Touch {i+1:2d}: pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif i == 5:
            print(f"  ...")

    print(f"\nTotal touches: {calibrator.num_touches}")
    print(f"Position spread (before): {calibrator.get_spread_mm():.2f} mm")
    print()

    # Solve
    print("Solving for kinematic corrections...")
    result = calibrator.solve(max_correction_mm=5.0)

    print(f"\nResults:")
    print(f"  Residual (after):  {result.residual_mm:.3f} mm")
    print(f"  Touches used:      {result.num_touches}")
    print(f"  Centroid:          [{result.centroid[0]:.4f}, {result.centroid[1]:.4f}, {result.centroid[2]:.4f}]")
    print()
    print("  Per-joint corrections (mm):")
    for i, c in enumerate(result.corrections):
        print(f"    Joint {i+1}: dx={c[0]*1000:+.3f}, dy={c[1]*1000:+.3f}, dz={c[2]*1000:+.3f}")

    print()
    print("  True offsets (mm) — for comparison:")
    for i, c in enumerate(true_offsets):
        print(f"    Joint {i+1}: dx={c[0]*1000:+.3f}, dy={c[1]*1000:+.3f}, dz={c[2]*1000:+.3f}")

    # Save
    cal_path = "kinematic_cal.json"
    result.save(cal_path)
    print(f"\nCalibration saved to: {cal_path}")

    # Show usage with Arm
    print()
    print("-" * 60)
    print("Usage with real hardware:")
    print("-" * 60)
    print("""
from onero_sdk import Arm, KinematicCalibrator

arm = Arm("A1", channel="can0")
arm.enable()

# Step 1: Record touches
calibrator = KinematicCalibrator.from_arm(arm)
for i in range(20):
    input(f"Touch point (config {i+1}/20), press Enter")
    calibrator.record_touch(arm=arm)

# Step 2: Solve
result = calibrator.solve()
print(f"Residual: {result.residual_mm:.2f} mm")
result.save("kinematic_cal.json")

# Step 3: Apply (all subsequent FK/IK use corrected params)
arm.apply_kinematic_calibration(result)

# Or load from file next time:
# arm.apply_kinematic_calibration("kinematic_cal.json")
""")


if __name__ == "__main__":
    main()
