"""
Hardware Basic Tests — onero A1 Arm (Real Machine)
===================================================
Preconditions:
  - /dev/ttyACM0 → right arm (A1R)
  - /dev/ttyACM1 → left  arm (A1L)
  - CAN interfaces slcan0 / slcan1 must be UP before running
    (use: sudo bash configs/onero-can-up.sh slcan0 ttyACM0)
  - Arms are in a known SAFE STARTING POSTURE: all joints near zero.
  - Front space is BLOCKED — all test motions move into rear/clear space only.

Run:
  python tests/test_hardware_basic.py [--arm right|left|both] [--dry-run]

Safety rules baked into this script:
  - Max motion amplitude: 0.15 rad (≈8.6°) per joint
  - Only joint 0 (shoulder-rotate, sweeps horizontally) is moved
  - Motion direction: negative j0 = toward rear (clear side)
  - Every test wrapped in try/finally to guarantee disable()
"""

import argparse
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RIGHT_ARM_CHANNEL = "slcan0"   # ttyACM0
LEFT_ARM_CHANNEL  = "slcan1"   # ttyACM1

MOVE_AMPLITUDE_RAD = 0.10      # max displacement per test motion (≈5.7°)
MOVE_DURATION_S    = 2.0       # seconds for each test motion
SETTLE_S           = 0.5       # settle time after motion
MAX_POSITION_ERROR = 0.05      # rad; PASS threshold

JOINT_MOVE_IDX = 0             # shoulder rotate — sweeps horizontally


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt(arr):
    return "[" + ", ".join(f"{v:+.4f}" for v in arr) + "]"


def heading(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def result(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{detail_str}")
    return passed


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def test_enable_disable(arm, name):
    heading(f"[{name}] 1. Enable / Disable")
    arm.enable()
    time.sleep(0.3)
    state = arm.get_state()
    positions = np.array(state.joints)
    print(f"  Joint positions after enable: {fmt(positions)}")
    max_drift = float(np.max(np.abs(positions)))
    p1 = result("Enable succeeds and joints read near zero",
                max_drift < 0.30,
                f"max |q| = {max_drift:.4f} rad")
    arm.disable()
    time.sleep(0.2)
    return p1


def test_state_readback(arm, name):
    heading(f"[{name}] 2. State Readback")
    arm.enable()
    time.sleep(0.2)
    samples = []
    for _ in range(20):
        state = arm.get_state()
        samples.append(np.array(state.joints))
        time.sleep(0.01)
    arm.disable()
    samples = np.array(samples)
    drift = np.max(np.abs(samples[-1] - samples[0]))
    print(f"  20 samples collected, enabled-idle drift: {drift:.5f} rad")
    p1 = result("20 state reads succeed (no exception)", True)
    p2 = result("Idle drift < 0.02 rad (gravity held)",
                drift < 0.02, f"drift = {drift:.5f} rad")
    return p1 and p2


def test_movej_small(arm, name):
    heading(f"[{name}] 3. Small movej (rear direction, joint {JOINT_MOVE_IDX})")
    arm.enable()
    time.sleep(0.3)

    q0 = np.array(arm.get_state().joints)
    print(f"  Start: {fmt(q0)}")

    # target: move joint 0 slightly toward rear (negative = clear side)
    target = q0.copy()
    target[JOINT_MOVE_IDX] -= MOVE_AMPLITUDE_RAD
    print(f"  Target: {fmt(target)}")
    print(f"  Moving {MOVE_AMPLITUDE_RAD:.3f} rad on joint {JOINT_MOVE_IDX} (rear direction)…")

    arm.movej(target.tolist(), duration=MOVE_DURATION_S)
    time.sleep(SETTLE_S)

    q_after = np.array(arm.get_state().joints)
    print(f"  Actual: {fmt(q_after)}")
    err = float(np.max(np.abs(q_after - target)))
    p1 = result(f"movej converges within {MAX_POSITION_ERROR} rad",
                err < MAX_POSITION_ERROR, f"max_err = {err:.4f} rad")

    # Return to start
    print(f"  Returning to start…")
    arm.movej(q0.tolist(), duration=MOVE_DURATION_S)
    time.sleep(SETTLE_S)
    q_back = np.array(arm.get_state().joints)
    err_back = float(np.max(np.abs(q_back - q0)))
    p2 = result("Return to start converges",
                err_back < MAX_POSITION_ERROR, f"max_err = {err_back:.4f} rad")

    arm.disable()
    return p1 and p2


def test_joint_velocity(arm, name):
    heading(f"[{name}] 4. speedj — brief velocity command")
    arm.enable()
    time.sleep(0.3)

    q0 = np.array(arm.get_state().joints)
    vel = np.zeros(len(q0))
    vel[JOINT_MOVE_IDX] = -0.3   # rad/s toward rear

    print(f"  Applying v[{JOINT_MOVE_IDX}] = {vel[JOINT_MOVE_IDX]} rad/s for 0.5 s")
    arm.speedj(vel.tolist(), duration=0.5)
    time.sleep(0.2)

    q_after = np.array(arm.get_state().joints)
    displacement = abs(q_after[JOINT_MOVE_IDX] - q0[JOINT_MOVE_IDX])
    print(f"  Displacement on joint {JOINT_MOVE_IDX}: {displacement:.4f} rad")
    p1 = result("speedj produces positive displacement",
                displacement > 0.02, f"disp = {displacement:.4f} rad")

    # Return
    arm.movej(q0.tolist(), duration=2.0)
    time.sleep(SETTLE_S)
    arm.disable()
    return p1


def test_torque_readback(arm, name):
    heading(f"[{name}] 5. Torque / current readback")
    arm.enable()
    time.sleep(0.3)
    state = arm.get_state()
    arm.disable()

    tau = np.array(state.torques) if state.torques else None
    has_tau = tau is not None and len(tau) > 0
    if has_tau:  # type: ignore
        print(f"  Torques: {fmt(tau)}")
        p1 = result("Torque field present and non-NaN",
                    not np.any(np.isnan(tau)))
    else:
        print("  Torque field not available in state (skipped)")
        p1 = result("Torque field present", False, "torques empty")
    return p1


# ---------------------------------------------------------------------------
# Per-arm runner
# ---------------------------------------------------------------------------
def run_arm_tests(model, channel, label, dry_run=False):
    from onero_sdk import Arm

    passed = []
    failed = []

    print(f"\n{'#'*60}")
    print(f"#  Testing {label} ({model} on {channel})")
    print(f"{'#'*60}")

    if dry_run:
        print("  [DRY RUN] skipping hardware — would connect to", channel)
        return True

    try:
        arm = Arm(model, channel=channel)
    except Exception as e:
        print(f"  [FATAL] Cannot create Arm({model}, channel={channel}): {e}")
        traceback.print_exc()
        return False

    tests = [
        ("enable_disable",    test_enable_disable),
        ("state_readback",    test_state_readback),
        ("movej_small",       test_movej_small),
        ("speedj",            test_joint_velocity),
        ("torque_readback",   test_torque_readback),
    ]

    for tname, tfunc in tests:
        try:
            ok = tfunc(arm, label)
            (passed if ok else failed).append(tname)
        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] disabling arm and aborting.")
            try:
                arm.disable()
            except Exception:
                pass
            sys.exit(1)
        except Exception as e:
            print(f"  [EXCEPTION] {tname}: {e}")
            traceback.print_exc()
            failed.append(tname)
            try:
                arm.disable()
            except Exception:
                pass

    # Final summary
    print(f"\n  --- {label} Results ---")
    for t in passed:
        print(f"    PASS  {t}")
    for t in failed:
        print(f"    FAIL  {t}")
    total = len(passed) + len(failed)
    print(f"  {len(passed)}/{total} passed")
    return len(failed) == 0


# ---------------------------------------------------------------------------
# CAN setup helper (prints instructions, does not auto-sudo)
# ---------------------------------------------------------------------------
def check_can_interfaces(channels):
    import subprocess
    all_up = True
    for ch in channels:
        r = subprocess.run(["ip", "link", "show", ch],
                           capture_output=True, text=True)
        if "UP" in r.stdout:
            print(f"  [OK] {ch} is UP")
        else:
            print(f"  [MISSING] {ch} not found or not UP")
            print(f"    Fix: sudo bash configs/onero-can-up.sh {ch} ttyACM{'0' if ch.endswith('0') else '1'}")
            all_up = False
    return all_up


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="onero A1 hardware basic tests")
    parser.add_argument("--arm", choices=["right", "left", "both"], default="right",
                        help="Which arm(s) to test (default: right)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without connecting to hardware")
    args = parser.parse_args()

    arms_to_test = []
    if args.arm in ("right", "both"):
        arms_to_test.append(("A1R", RIGHT_ARM_CHANNEL, "RIGHT"))
    if args.arm in ("left", "both"):
        arms_to_test.append(("A1L", LEFT_ARM_CHANNEL, "LEFT"))

    channels = [ch for _, ch, _ in arms_to_test]

    print("\nonero A1 — Hardware Basic Tests")
    print("=" * 60)
    print(f"  Arms     : {args.arm}")
    print(f"  Channels : {', '.join(channels)}")
    print(f"  Move amp : {MOVE_AMPLITUDE_RAD} rad on joint {JOINT_MOVE_IDX} (rear, clear)")
    print(f"  Dry run  : {args.dry_run}")
    print("=" * 60)

    if not args.dry_run:
        print("\nChecking CAN interfaces…")
        if not check_can_interfaces(channels):
            print("\n[ABORT] CAN interface(s) not ready. Bring them up first.")
            sys.exit(1)

    all_passed = True
    for model, channel, label in arms_to_test:
        ok = run_arm_tests(model, channel, label, dry_run=args.dry_run)
        all_passed = all_passed and ok

    print(f"\n{'='*60}")
    if all_passed:
        print("  OVERALL: ALL PASSED")
    else:
        print("  OVERALL: SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
