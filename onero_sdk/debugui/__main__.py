"""Command-line entry point: python -m onero_sdk.debugui"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m onero_sdk.debugui",
        description="onero SDK Debug UI — web-based arm control panel",
    )
    ap.add_argument("--arm", choices=["right", "left", "both"], default="right",
                    help="Which arm(s) to control (default: right)")
    ap.add_argument("--channel", default=None,
                    help="CAN channel for single-arm mode (e.g. slcan0)")
    ap.add_argument("--left-channel", default="slcan1",
                    help="CAN channel for left arm (dual mode, default: slcan1)")
    ap.add_argument("--right-channel", default="slcan0",
                    help="CAN channel for right arm (dual mode, default: slcan0)")
    ap.add_argument("--arm-spacing", type=float, default=0.36,
                    help="Distance between arm bases in dual mode (meters, default: 0.36)")
    ap.add_argument("--sim", action="store_true",
                    help="Use MuJoCo simulation backend instead of real hardware")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8090, help="TCP port (default: 8090)")
    args = ap.parse_args()

    try:
        from onero_sdk import Arm
        from onero_sdk.debugui import start_lab, start_lab_dual
    except ImportError as exc:
        print(f"[debugui] import error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.sim:
        # Simulation backend
        from onero_sdk.mujoco_backend import MujocoBackend
        if args.arm == "both":
            from onero_sdk import DualArm
            dual = DualArm.create(
                left_backend=MujocoBackend(model_name="A1L"),
                right_backend=MujocoBackend(model_name="A1R"),
                arm_spacing=args.arm_spacing,
            )
            dual.enable()
            start_lab_dual(dual, host=args.host, port=args.port, block=True)
        else:
            model = "A1R" if args.arm == "right" else "A1L"
            arm = Arm(model, backend=MujocoBackend(model_name=model))
            arm.enable()
            start_lab(arm, side=args.arm, host=args.host, port=args.port, block=True)
    else:
        # Real hardware
        if args.arm == "both":
            from onero_sdk import DualArm
            dual = DualArm.create(
                left_channel=args.left_channel,
                right_channel=args.right_channel,
                arm_spacing=args.arm_spacing,
            )
            start_lab_dual(dual, host=args.host, port=args.port, block=True)
        else:
            model = "A1R" if args.arm == "right" else "A1L"
            channel = args.channel or ("slcan0" if args.arm == "right" else "slcan1")
            arm = Arm(model, channel=channel)
            start_lab(arm, side=args.arm, host=args.host, port=args.port, block=True)


if __name__ == "__main__":
    main()
