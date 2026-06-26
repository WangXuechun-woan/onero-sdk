"""
Deadline-based control loop timing.

Provides a ControlClock that maintains a fixed-rate loop by sleeping
only the residual time after command processing. Handles drift gracefully
by resetting when the loop falls too far behind.
"""

import time


class ControlClock:
    """
    Fixed-rate loop timer with drift compensation.

    Usage:
        clock = ControlClock()
        for frame in trajectory:
            do_work(frame)
            clock.tick(dt)
    """

    def __init__(self):
        self._next_tick: float = 0.0

    def reset(self) -> None:
        """Reset the clock (call at start of a new motion)."""
        self._next_tick = 0.0

    def tick(self, dt: float) -> None:
        """
        Sleep until the next deadline, advancing by dt.

        If the loop has fallen behind by more than 3 cycles,
        resets the deadline to avoid a burst of catch-up frames.
        """
        now = time.monotonic()
        if self._next_tick == 0.0:
            self._next_tick = now + dt
        else:
            if now - self._next_tick > 3 * dt:
                self._next_tick = now + dt
            else:
                sleep_time = self._next_tick - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._next_tick += dt
