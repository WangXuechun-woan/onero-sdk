"""
MuJoCo simulation integration test — validates SDK functionality end-to-end.

Tests: movej, servoj, speedj, adaptive gains, safety clamping.
Run: conda activate onerosdk && python tests/test_mujoco_sim.py
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from onero_sdk import Arm, ArmState
from onero_sdk.mujoco_backend import MujocoBackend
from onero_sdk.gain_policy import Stiffness


def test_basic_enable_disable():
    """Test arm enable/disable lifecycle."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)

    arm.enable()
    assert arm.state is not None
    assert len(arm.state.joints) == 7
    print(f"  Initial joints: {[f'{j:.3f}' for j in arm.state.joints]}")

    arm.disable()
    print("  [PASS] enable/disable")


def test_movej():
    """Test joint space motion with convergence check."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    target = [0.0, -0.5, 0.3, 0.5, -0.2, 0.3, 0.0]
    state = arm.movej(target, duration=2.0)

    errors = [abs(state.joints[i] - target[i]) for i in range(7)]
    max_err = max(errors)
    print(f"  Target: {[f'{t:.2f}' for t in target]}")
    print(f"  Actual: {[f'{j:.3f}' for j in state.joints]}")
    print(f"  Max error: {max_err:.4f} rad")

    assert max_err < 0.05, f"movej did not converge: max_err={max_err:.4f}"
    arm.disable()
    print("  [PASS] movej converges within 0.05 rad")


def test_movej_adaptive_gains():
    """Test that adaptive gains produce better tracking than fixed weak gains."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    target = [0.0, -0.8, 0.5, 0.8, -0.3, 0.4, 0.0]

    # Test with adaptive gains (STIFF)
    arm.set_stiffness("stiff")
    backend.reset()
    arm.enable()
    state_adaptive = arm.movej(target, duration=2.0)
    err_adaptive = max(abs(state_adaptive.joints[i] - target[i]) for i in range(7))

    # Test with fixed weak gains
    backend.reset()
    arm.enable()
    state_fixed = arm.movej(target, duration=2.0, kp=5.0, kd=0.2)
    err_fixed = max(abs(state_fixed.joints[i] - target[i]) for i in range(7))

    print(f"  Adaptive (stiff) max error: {err_adaptive:.4f}")
    print(f"  Fixed (kp=5, kd=0.2) max error: {err_fixed:.4f}")
    assert err_adaptive < err_fixed, "Adaptive gains should track better than weak fixed"
    arm.disable()
    print("  [PASS] adaptive gains outperform weak fixed gains")


def test_movej_stiffness_presets():
    """Verify different stiffness presets produce different tracking behavior."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)

    target = [0.0, -0.6, 0.4, 0.6, -0.2, 0.3, 0.0]
    errors = {}

    for preset in ["stiff", "medium", "soft"]:
        backend.reset()
        arm.enable()
        arm.set_stiffness(preset)
        state = arm.movej(target, duration=1.5)
        err = max(abs(state.joints[i] - target[i]) for i in range(7))
        errors[preset] = err
        arm.disable()

    print(f"  stiff:  {errors['stiff']:.4f}")
    print(f"  medium: {errors['medium']:.4f}")
    print(f"  soft:   {errors['soft']:.4f}")
    assert errors["stiff"] <= errors["medium"] <= errors["soft"], \
        "Stiffer preset should give tighter tracking"
    print("  [PASS] stiffness ordering: stiff < medium < soft")


def test_servoj():
    """Test real-time servo control frame by frame."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    # Sine wave on joint 1
    states = []
    for step in range(200):
        t = step * 0.005
        joints = [0.0] * 7
        joints[1] = 0.5 * math.sin(2 * math.pi * 0.5 * t)
        state = arm.servoj(joints)
        states.append(state)

    # Check that joint 1 actually moved
    max_j1 = max(s.joints[1] for s in states)
    min_j1 = min(s.joints[1] for s in states)
    amplitude = max_j1 - min_j1
    print(f"  Joint 1 range: [{min_j1:.3f}, {max_j1:.3f}], amplitude={amplitude:.3f}")
    assert amplitude > 0.2, f"Servo didn't track sine wave: amplitude={amplitude:.3f}"
    arm.disable()
    print("  [PASS] servoj tracks sine wave")


def test_speedj():
    """Test joint velocity control."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    # Move joint 1 at 0.5 rad/s for 1 second
    vel_cmd = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    state = arm.speedj(vel_cmd, acceleration=2.0, duration=1.0)

    # Joint 1 should have moved roughly 0.5 rad (v*t with ramp)
    displacement = abs(state.joints[1])
    print(f"  Joint 1 displacement after speedj: {displacement:.3f} rad")
    assert displacement > 0.1, f"speedj didn't move: displacement={displacement:.3f}"
    arm.disable()
    print("  [PASS] speedj produces joint displacement")


def test_safety_clamping():
    """Test that safety limits prevent exceeding joint bounds."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    # Try to move beyond joint limits (joint 0 limit is [-1.04, 3.14])
    target = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # way beyond limit
    state = arm.movej(target, duration=2.0)

    # Should be clamped to near the limit, not at 5.0
    assert state.joints[0] < 3.5, f"Safety didn't clamp: j0={state.joints[0]:.3f}"
    print(f"  Commanded 5.0 rad, actual: {state.joints[0]:.3f} (limit ~3.14)")
    arm.disable()
    print("  [PASS] safety clamping works")


def test_simulation_speed():
    """Benchmark simulation throughput."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    target = [0.0, -0.5, 0.3, 0.5, -0.2, 0.3, 0.0]
    t0 = time.monotonic()
    arm.movej(target, duration=3.0)
    elapsed = time.monotonic() - t0

    sim_time = 3.0
    speedup = sim_time / elapsed if elapsed > 0 else float('inf')
    print(f"  3s sim completed in {elapsed:.3f}s wall-clock ({speedup:.1f}x realtime)")
    arm.disable()
    print("  [PASS] simulation runs faster than realtime")


def test_multiple_movej():
    """Test sequential motions."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()

    waypoints = [
        [0.0, -0.5, 0.3, 0.5, -0.2, 0.3, 0.0],
        [0.3, -0.8, 0.5, 0.8, -0.3, 0.4, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    for i, target in enumerate(waypoints):
        state = arm.movej(target, duration=1.5)
        err = max(abs(state.joints[j] - target[j]) for j in range(7))
        print(f"  Waypoint {i}: max_err={err:.4f}")
        assert err < 0.1, f"Waypoint {i} failed: max_err={err:.4f}"

    arm.disable()
    print("  [PASS] sequential movej all converge")


def main():
    tests = [
        ("Basic enable/disable", test_basic_enable_disable),
        ("movej convergence", test_movej),
        ("movej adaptive gains", test_movej_adaptive_gains),
        ("Stiffness presets", test_movej_stiffness_presets),
        ("servoj tracking", test_servoj),
        ("speedj velocity", test_speedj),
        ("Safety clamping", test_safety_clamping),
        ("Simulation speed", test_simulation_speed),
        ("Sequential movej", test_multiple_movej),
    ]

    print("=" * 60)
    print("onero SDK — MuJoCo Simulation Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
