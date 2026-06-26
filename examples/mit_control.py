"""
Example: MIT impedance control on a single motor.

Usage:
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set up can0
    python mit_control.py
"""

import time
import math
from onero_sdk import CanBus, Motor, MotorLimits


def main():
    limits = MotorLimits(p_max=12.5, v_max=30.0, t_max=15.0)

    with CanBus("can0") as bus:
        motor = Motor(bus, can_id=0x01, mst_id=0x11, limits=limits)

        print("Enabling motor...")
        state = motor.enable()
        print(f"  pos={state.position:.3f} rad, error={state.error}")

        print("Setting zero position...")
        motor.set_zero()
        time.sleep(0.1)

        print("Running sine wave trajectory (MIT mode)...")
        kp, kd = 30.0, 1.0
        freq = 0.5  # Hz
        amplitude = 1.0  # rad
        duration = 5.0  # seconds

        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t > duration:
                break

            target_pos = amplitude * math.sin(2 * math.pi * freq * t)
            state = motor.set_mit(position=target_pos, kp=kp, kd=kd)
            print(f"\r  t={t:.2f}s pos={state.position:.3f} target={target_pos:.3f}", end="")
            time.sleep(0.005)  # 200 Hz

        print("\n\nDisabling motor...")
        motor.disable()
        print("Done.")


if __name__ == "__main__":
    main()
