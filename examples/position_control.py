"""
Example: Position-velocity mode control.

Usage:
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set up can0
    python position_control.py
"""

import time
from onero_sdk import CanBus, Motor, MotorLimits


def main():
    limits = MotorLimits(p_max=12.5, v_max=30.0, t_max=15.0)

    with CanBus("can0") as bus:
        motor = Motor(bus, can_id=0x01, mst_id=0x11, limits=limits)

        print("Enabling motor...")
        motor.enable()
        time.sleep(0.1)

        waypoints = [1.0, 2.0, 0.0, -1.0, 0.0]
        velocity = 3.0  # rad/s max speed

        for wp in waypoints:
            print(f"Moving to {wp:.2f} rad at {velocity:.1f} rad/s...")
            motor.set_position_velocity(wp, velocity)
            time.sleep(1.5)
            state = motor.get_feedback()
            print(f"  Arrived: pos={state.position:.3f} rad")

        print("Disabling motor...")
        motor.disable()
        print("Done.")


if __name__ == "__main__":
    main()
