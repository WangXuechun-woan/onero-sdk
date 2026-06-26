"""
Trajectory profile generators — industry-standard motion curves.

Provides:
- TrapezoidalProfile: trapezoidal velocity (constant acceleration)
- SCurveProfile: 7-segment S-curve (jerk-limited, smooth acceleration)
- QuinticBlend: 5th-order polynomial transition (C² continuous at boundaries)
- compute_synchronized_duration: multi-joint time synchronization

These replace cosine interpolation when precise speed/acceleration control
is needed. All profiles guarantee zero velocity at start and end.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProfilePoint:
    """A single point on a trajectory profile."""
    position: float
    velocity: float
    acceleration: float
    time: float


class TrapezoidalProfile:
    """
    Trapezoidal velocity profile (constant acceleration/deceleration).

    Three phases: accelerate → cruise → decelerate.
    If the distance is too short to reach max speed, it becomes a triangular
    profile (accelerate → decelerate, no cruise phase).

    This is the standard UR/KUKA profile for movej.
    """

    def __init__(self, distance: float, v_max: float, a_max: float):
        """
        Args:
            distance: Total displacement (absolute value used).
            v_max: Maximum velocity (positive).
            a_max: Maximum acceleration (positive).
        """
        self._sign = 1.0 if distance >= 0 else -1.0
        d = abs(distance)
        v_max = abs(v_max)
        a_max = abs(a_max)

        # Time to accelerate to v_max
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel ** 2

        if 2 * d_accel >= d:
            # Triangular profile (can't reach v_max)
            t_accel = math.sqrt(d / a_max)
            self._v_peak = a_max * t_accel
            self._t_accel = t_accel
            self._t_cruise = 0.0
            self._t_decel = t_accel
        else:
            # Full trapezoidal
            self._v_peak = v_max
            self._t_accel = t_accel
            d_cruise = d - 2 * d_accel
            self._t_cruise = d_cruise / v_max
            self._t_decel = t_accel

        self._a_max = a_max
        self._distance = d
        self._duration = self._t_accel + self._t_cruise + self._t_decel

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def v_peak(self) -> float:
        return self._v_peak * self._sign

    def evaluate(self, t: float) -> ProfilePoint:
        """Evaluate position, velocity, acceleration at time t."""
        t = max(0.0, min(t, self._duration))
        s = self._sign

        if t <= self._t_accel:
            # Acceleration phase
            a = self._a_max
            v = a * t
            p = 0.5 * a * t ** 2
        elif t <= self._t_accel + self._t_cruise:
            # Cruise phase
            dt = t - self._t_accel
            a = 0.0
            v = self._v_peak
            p = 0.5 * self._a_max * self._t_accel ** 2 + self._v_peak * dt
        else:
            # Deceleration phase
            dt = t - self._t_accel - self._t_cruise
            a = -self._a_max
            v = self._v_peak - self._a_max * dt
            p = (0.5 * self._a_max * self._t_accel ** 2 +
                 self._v_peak * self._t_cruise +
                 self._v_peak * dt - 0.5 * self._a_max * dt ** 2)

        return ProfilePoint(
            position=p * s,
            velocity=v * s,
            acceleration=a * s,
            time=t,
        )

    def sample(self, dt: float) -> list[ProfilePoint]:
        """Sample the profile at fixed intervals."""
        n = max(1, int(math.ceil(self._duration / dt)))
        points = []
        for i in range(n + 1):
            t = min(i * dt, self._duration)
            points.append(self.evaluate(t))
        return points


class SCurveProfile:
    """
    7-segment S-curve velocity profile (jerk-limited).

    Phases: j+ → a_const → j- → cruise → j- → -a_const → j+
    Guarantees continuous acceleration (no sudden force changes),
    which produces smoother, quieter motion with less mechanical wear.

    Parameters:
        distance: Total displacement.
        v_max: Maximum velocity.
        a_max: Maximum acceleration.
        j_max: Maximum jerk (rate of change of acceleration).
    """

    def __init__(self, distance: float, v_max: float, a_max: float,
                 j_max: Optional[float] = None):
        self._sign = 1.0 if distance >= 0 else -1.0
        d = abs(distance)
        v_max = abs(v_max)
        a_max = abs(a_max)

        if j_max is None:
            j_max = a_max * 10.0  # Default: reach max accel in 0.1s

        # Time for jerk phase to reach a_max
        tj = a_max / j_max

        # Velocity gained during jerk phases (accel + decel jerk)
        v_jerk = a_max * tj  # = j_max * tj^2 = a_max^2 / j_max

        if v_jerk >= v_max:
            # Can't even reach a_max before hitting v_max
            # Simplified: reduce to triangular acceleration
            tj = math.sqrt(v_max / j_max)
            self._segments = self._compute_no_cruise_no_const_accel(d, v_max, j_max, tj)
        else:
            # Check if we can reach v_max
            # Distance consumed by accel phase (with possible constant accel)
            v_accel_phase = v_max  # velocity gained during full acceleration
            # Time at constant acceleration
            ta = (v_max - v_jerk) / a_max if v_jerk < v_max else 0.0

            # Distance during acceleration (jerk_up + const_accel + jerk_down)
            d_accel = (0.5 * j_max * tj ** 3 +         # jerk up
                       a_max * ta * (0.5 * ta + tj) +   # const accel (trapezoid area)
                       v_max * tj - 0.5 * j_max * tj ** 3)  # jerk down (approximate)

            # Simpler approach: compute using standard S-curve formulas
            d_accel = self._accel_phase_distance(v_max, a_max, j_max, tj, ta)

            if 2 * d_accel >= d:
                # No cruise phase — scale down v_max
                self._segments = self._compute_no_cruise(d, v_max, a_max, j_max)
            else:
                # Full 7-segment
                t_cruise = (d - 2 * d_accel) / v_max
                self._segments = self._compute_full(d, v_max, a_max, j_max, tj, ta, t_cruise)

        self._duration = sum(s[0] for s in self._segments)
        self._distance = d

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> ProfilePoint:
        """Evaluate position, velocity, acceleration at time t."""
        t = max(0.0, min(t, self._duration))
        s = self._sign

        # Walk through segments
        p, v, a = 0.0, 0.0, 0.0
        t_remaining = t
        for (seg_dur, seg_jerk) in self._segments:
            if t_remaining <= seg_dur:
                dt = t_remaining
                p += v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * seg_jerk * dt ** 3
                v += a * dt + 0.5 * seg_jerk * dt ** 2
                a += seg_jerk * dt
                break
            else:
                dt = seg_dur
                p += v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * seg_jerk * dt ** 3
                v += a * dt + 0.5 * seg_jerk * dt ** 2
                a += seg_jerk * dt
                t_remaining -= seg_dur

        return ProfilePoint(
            position=p * s,
            velocity=v * s,
            acceleration=a * s,
            time=t,
        )

    def sample(self, dt: float) -> list[ProfilePoint]:
        """Sample the profile at fixed intervals."""
        n = max(1, int(math.ceil(self._duration / dt)))
        points = []
        for i in range(n + 1):
            t = min(i * dt, self._duration)
            points.append(self.evaluate(t))
        return points

    def _accel_phase_distance(self, v_max, a_max, j_max, tj, ta):
        """Distance covered during the acceleration phase (3 sub-segments)."""
        # Phase 1: jerk up (duration tj)
        d1 = (1.0 / 6.0) * j_max * tj ** 3
        v1 = 0.5 * j_max * tj ** 2  # = 0.5 * a_max * tj

        # Phase 2: constant accel (duration ta)
        d2 = v1 * ta + 0.5 * a_max * ta ** 2
        v2 = v1 + a_max * ta

        # Phase 3: jerk down (duration tj)
        d3 = v2 * tj + 0.5 * a_max * tj ** 2 - (1.0 / 6.0) * j_max * tj ** 3
        return d1 + d2 + d3

    def _compute_full(self, d, v_max, a_max, j_max, tj, ta, t_cruise):
        """Full 7-segment S-curve."""
        return [
            (tj, j_max),     # 1. Jerk up (acceleration increases)
            (ta, 0.0),       # 2. Constant acceleration
            (tj, -j_max),    # 3. Jerk down (acceleration decreases to 0)
            (t_cruise, 0.0), # 4. Cruise (constant velocity)
            (tj, -j_max),    # 5. Jerk down (deceleration increases)
            (ta, 0.0),       # 6. Constant deceleration
            (tj, j_max),     # 7. Jerk up (deceleration decreases to 0)
        ]

    def _compute_no_cruise(self, d, v_max, a_max, j_max):
        """S-curve without cruise phase — binary search for achievable peak velocity."""
        # Binary search for actual peak velocity
        v_lo, v_hi = 0.0, v_max
        for _ in range(50):
            v_mid = 0.5 * (v_lo + v_hi)
            tj = min(a_max / j_max, math.sqrt(v_mid / j_max))
            a_actual = j_max * tj
            ta = max(0.0, (v_mid - a_actual * tj) / a_actual) if a_actual > 0 else 0.0
            d_test = 2 * self._accel_phase_distance(v_mid, a_actual, j_max, tj, ta)
            if d_test < d:
                v_lo = v_mid
            else:
                v_hi = v_mid

        v_peak = 0.5 * (v_lo + v_hi)
        tj = min(a_max / j_max, math.sqrt(v_peak / j_max))
        a_actual = j_max * tj
        ta = max(0.0, (v_peak - a_actual * tj) / a_actual) if a_actual > 0 else 0.0

        return [
            (tj, j_max),
            (ta, 0.0),
            (tj, -j_max),
            (0.0, 0.0),  # no cruise
            (tj, -j_max),
            (ta, 0.0),
            (tj, j_max),
        ]

    def _compute_no_cruise_no_const_accel(self, d, v_max, j_max, tj):
        """Simplified S-curve: only jerk phases, no constant accel or cruise."""
        # 4-segment: j+, j-, j-, j+
        # Binary search for tj
        tj_lo, tj_hi = 0.0, tj * 2
        for _ in range(50):
            t_mid = 0.5 * (tj_lo + tj_hi)
            v_peak = j_max * t_mid ** 2
            d_test = 2 * (1.0 / 6.0) * j_max * t_mid ** 3 * 2
            if d_test < d:
                tj_lo = t_mid
            else:
                tj_hi = t_mid

        t_final = 0.5 * (tj_lo + tj_hi)
        return [
            (t_final, j_max),
            (0.0, 0.0),
            (t_final, -j_max),
            (0.0, 0.0),
            (t_final, -j_max),
            (0.0, 0.0),
            (t_final, j_max),
        ]


class QuinticBlend:
    """
    5th-order (quintic) polynomial for smooth transitions between segments.

    Guarantees C² continuity: position, velocity, and acceleration are
    all continuous at segment boundaries. This is the industry standard
    for multi-waypoint blending (ABB, KUKA).

    The polynomial satisfies 6 boundary conditions:
        p(0) = p0, v(0) = v0, a(0) = a0
        p(T) = p1, v(T) = v1, a(T) = a1
    """

    def __init__(self, duration: float,
                 p0: float, v0: float, a0: float,
                 p1: float, v1: float, a1: float):
        """
        Args:
            duration: Blend duration (seconds).
            p0, v0, a0: Initial position, velocity, acceleration.
            p1, v1, a1: Final position, velocity, acceleration.
        """
        self._T = duration
        self._p0 = p0
        self._v0 = v0
        self._a0 = a0

        # Solve for quintic coefficients: p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
        T = duration
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5

        self._c0 = p0
        self._c1 = v0
        self._c2 = 0.5 * a0
        # From boundary conditions at t=T:
        dp = p1 - p0 - v0 * T - 0.5 * a0 * T2
        dv = v1 - v0 - a0 * T
        da = a1 - a0
        self._c3 = (20 * dp - (8 * dv + da * T) * T - (12 * v0 + 3 * a0 * T) * T2 / T2) if T > 0 else 0.0

        if T > 0:
            self._c3 = (20 * dp / T3 - (8 * v1 + 12 * v0) / T2 -
                        (3 * a0 - a1) / T) / 2.0
            # Standard quintic solution
            self._c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T -
                        (3 * a0 - a1) * T2) / (2 * T3)
            self._c4 = (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T +
                        (3 * a0 - 2 * a1) * T2) / (2 * T4)
            self._c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * T -
                        (a0 - a1) * T2) / (2 * T5)
        else:
            self._c3 = self._c4 = self._c5 = 0.0

    @property
    def duration(self) -> float:
        return self._T

    def evaluate(self, t: float) -> ProfilePoint:
        """Evaluate position, velocity, acceleration at time t."""
        t = max(0.0, min(t, self._T))
        t2, t3, t4, t5 = t**2, t**3, t**4, t**5

        p = (self._c0 + self._c1 * t + self._c2 * t2 +
             self._c3 * t3 + self._c4 * t4 + self._c5 * t5)
        v = (self._c1 + 2 * self._c2 * t + 3 * self._c3 * t2 +
             4 * self._c4 * t3 + 5 * self._c5 * t4)
        a = (2 * self._c2 + 6 * self._c3 * t +
             12 * self._c4 * t2 + 20 * self._c5 * t3)

        return ProfilePoint(position=p, velocity=v, acceleration=a, time=t)


def compute_synchronized_duration(deltas: list[float], v_max: list[float],
                                   a_max: list[float]) -> float:
    """
    Compute the shortest duration that satisfies all joint constraints.

    For multi-DOF motion, all joints must finish at the same time.
    The slowest joint (the one requiring the most time) determines the
    overall duration. Other joints are then slowed down to match.

    Args:
        deltas: Displacement per joint (rad or m).
        v_max: Max velocity per joint.
        a_max: Max acceleration per joint.

    Returns:
        Synchronized duration (seconds).
    """
    max_duration = 0.0
    for d, v, a in zip(deltas, v_max, a_max):
        d_abs = abs(d)
        if d_abs < 1e-10:
            continue
        profile = TrapezoidalProfile(d_abs, v, a)
        max_duration = max(max_duration, profile.duration)
    return max(max_duration, 0.1)


def multi_joint_trapezoidal(start: list[float], end: list[float],
                             v_max: list[float], a_max: list[float],
                             dt: float,
                             target_duration: Optional[float] = None) -> list[list[ProfilePoint]]:
    """
    Generate synchronized trapezoidal profiles for multiple joints.

    All joints start and stop together. Faster joints are scaled down
    to match the slowest joint's duration.

    Args:
        start: Start positions per joint.
        end: End positions per joint.
        v_max: Max velocity per joint.
        a_max: Max acceleration per joint.
        dt: Sample interval (seconds).
        target_duration: If given, motion takes at least this long.
                        Velocities are scaled down to fit.

    Returns:
        List of ProfilePoint lists, one per joint.
    """
    n = len(start)
    deltas = [end[i] - start[i] for i in range(n)]
    duration = compute_synchronized_duration(deltas, v_max, a_max)
    if target_duration is not None and target_duration > duration:
        duration = target_duration

    results = []
    for i in range(n):
        d = abs(deltas[i])
        if d < 1e-10:
            # Stationary joint
            n_steps = max(1, int(math.ceil(duration / dt)))
            results.append([
                ProfilePoint(position=start[i], velocity=0.0, acceleration=0.0,
                             time=k * dt)
                for k in range(n_steps + 1)
            ])
        else:
            # Scale v/a to fit the synchronized duration
            # For trapezoidal: T = t_a + t_c + t_d, solve for v_actual given T
            v_actual = _solve_trapezoidal_velocity(d, duration, a_max[i])
            profile = TrapezoidalProfile(deltas[i], v_actual, a_max[i])
            points = profile.sample(dt)
            # Offset by start position
            results.append([
                ProfilePoint(
                    position=start[i] + p.position,
                    velocity=p.velocity,
                    acceleration=p.acceleration,
                    time=p.time,
                )
                for p in points
            ])

    return results


def _solve_trapezoidal_velocity(distance: float, duration: float,
                                 a_max: float) -> float:
    """Find the cruise velocity that makes a trapezoidal profile fit in duration."""
    # For trapezoidal: T = v/a + d/v (when triangular: T = 2*sqrt(d/a))
    # Quadratic: v^2 - T*a*v + a*d = 0 (derived from T = v/a + (d - v^2/(2a)) / v + v/a)
    # Simplified: T = 2*v/a + (d - v^2/a) / v → T*v = 2*v^2/a + d - v^2/a → T*v = v^2/a + d
    # → v^2 - a*T*v + a*d = 0
    discriminant = (a_max * duration) ** 2 - 4 * a_max * distance
    if discriminant < 0:
        # Triangular profile fits
        return math.sqrt(a_max * distance)
    v = (a_max * duration - math.sqrt(discriminant)) / 2.0
    return max(v, 1e-6)
