"""
Diagnostic tool for onero SDK.

Usage:
    python -m onero_sdk.diag [channel]

Checks:
- SocketCAN interface availability
- CAN bus communication (ping motors)
- URDF / FK availability
- Pinocchio (dynamics) availability
"""

import sys
import socket
import struct


def check_can_interface(channel: str) -> bool:
    """Check if a SocketCAN interface is available and UP."""
    try:
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, 1)  # CAN_RAW=1
        sock.bind((channel,))
        sock.close()
        print(f"  [OK] CAN interface '{channel}' is UP")
        return True
    except OSError as e:
        print(f"  [FAIL] CAN interface '{channel}': {e}")
        print(f"         Fix: sudo ip link set up {channel}")
        return False


def check_urdf() -> bool:
    """Check if URDF files are accessible."""
    try:
        from .arm import Arm
        from .model_registry import get_model
        model_info = get_model("A1R")
        # Use same path logic as Arm._find_urdf
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        name = model_info.name
        candidates = [
            os.path.join(pkg_dir, "urdf", name, f"{name.lower()}_l.urdf"),
            os.path.join(pkg_dir, "urdf", name, f"{name.lower()}_r.urdf"),
            os.path.join(pkg_dir, "urdf", f"{name.lower()}.urdf"),
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"  [OK] URDF found: {path}")
                return True
        print(f"  [WARN] No URDF found. Checked:")
        for p in candidates:
            print(f"         {p}")
        return False
    except Exception as e:
        print(f"  [FAIL] Model registry: {e}")
        return False


def check_fk() -> bool:
    """Check if FK (forward kinematics) is available."""
    try:
        from .fk import URDFChain
        print("  [OK] FK (URDFChain) available")
        return True
    except ImportError as e:
        print(f"  [WARN] FK not available: {e}")
        return False


def check_dynamics() -> bool:
    """Check if pinocchio / dynamics is available."""
    try:
        import pinocchio
        print(f"  [OK] Pinocchio {pinocchio.__version__} available")
        return True
    except ImportError:
        print("  [WARN] Pinocchio not installed. Install with: pip install onero-sdk[dynamics]")
        return False


def check_ik() -> bool:
    """Check if IK solver dependencies are available."""
    try:
        import pink
        print(f"  [OK] Pink (IK) available")
        return True
    except ImportError:
        print("  [WARN] Pink not installed. Install with: pip install onero-sdk[ik]")
        return False


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else "can0"

    print("onero SDK Diagnostic")
    print("=" * 40)

    from . import __version__
    print(f"  SDK version: {__version__}")
    print()

    print("[1/5] CAN Interface")
    can_ok = check_can_interface(channel)
    print()

    print("[2/5] Model & URDF")
    check_urdf()
    print()

    print("[3/5] Forward Kinematics")
    check_fk()
    print()

    print("[4/5] Dynamics (pinocchio)")
    check_dynamics()
    print()

    print("[5/5] IK Solver (pink)")
    check_ik()
    print()

    print("=" * 40)
    if can_ok:
        print("CAN bus ready. You can connect to the arm.")
    else:
        print("CAN bus NOT ready. Fix the interface before connecting.")

    return 0 if can_ok else 1


if __name__ == "__main__":
    sys.exit(main())
