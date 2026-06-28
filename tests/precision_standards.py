"""
Precision standards for robot arm testing.

Tolerance levels defined from ISO 9283 and manufacturer specifications:
- UR e-Series: ±0.03 mm repeatability, ±0.1° orientation
- KUKA LBR iiwa: ±0.1 mm repeatability, ±0.01° joint repeatability
- Franka Research 3: ±0.1 mm repeatability, 1 kHz control loop
- ABB GoFa: ±0.02 mm repeatability (Ultra Accuracy mode)

We define three tiers:
- TIGHT: hardware-level precision (sensor resolution, encoder accuracy)
- NORMAL: typical cobot operational precision
- LOOSE: acceptable for simulation or high-speed motion

These are for a 7-DOF collaborative arm with ~800mm reach and ~7kg payload.
"""

import math


# ═══════════════════════════════════════════════════════════════════════════════
# Position Precision (meters)
# ═══════════════════════════════════════════════════════════════════════════════

# End-effector position repeatability
POS_REPEATABILITY_TIGHT = 0.0001    # 0.1 mm (Franka/KUKA level)
POS_REPEATABILITY_NORMAL = 0.001    # 1.0 mm (acceptable for fast motion)
POS_REPEATABILITY_LOOSE = 0.002     # 2 mm (simulation with impedance control)

# End-effector position accuracy (absolute, uncalibrated)
POS_ACCURACY_TIGHT = 0.0005         # 0.5 mm
POS_ACCURACY_NORMAL = 0.002         # 2.0 mm
POS_ACCURACY_LOOSE = 0.005          # 5.0 mm

# Path accuracy (deviation from programmed path, ISO 9283 AT)
PATH_ACCURACY_TIGHT = 0.0005        # 0.5 mm
PATH_ACCURACY_NORMAL = 0.002        # 2.0 mm
PATH_ACCURACY_LOOSE = 0.005         # 5.0 mm


# ═══════════════════════════════════════════════════════════════════════════════
# Orientation Precision (radians)
# ═══════════════════════════════════════════════════════════════════════════════

# End-effector orientation repeatability
ORI_REPEATABILITY_TIGHT = math.radians(0.05)     # 0.05° ≈ 0.87 mrad
ORI_REPEATABILITY_NORMAL = math.radians(0.1)     # 0.1° ≈ 1.7 mrad
ORI_REPEATABILITY_LOOSE = math.radians(0.5)      # 0.5° ≈ 8.7 mrad

# Joint angle precision
JOINT_REPEATABILITY_TIGHT = math.radians(0.01)   # 0.01° (KUKA iiwa level)
JOINT_REPEATABILITY_NORMAL = math.radians(0.05)  # 0.05°
JOINT_REPEATABILITY_LOOSE = math.radians(0.1)    # 0.1°

# SLERP/rotation interpolation mathematical precision
# (Pure math — should be near machine epsilon for doubles)
ROTATION_MATH_TOL = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Velocity Precision
# ═══════════════════════════════════════════════════════════════════════════════

# Velocity tracking error (fraction of commanded speed)
VEL_ACCURACY_TIGHT = 0.02    # ±2% (Franka 1kHz loop)
VEL_ACCURACY_NORMAL = 0.05   # ±5% (ISO 9283 typical)
VEL_ACCURACY_LOOSE = 0.10    # ±10% (during acceleration/deceleration)

# Maximum joint velocity for a 7-DOF cobot (rad/s)
MAX_JOINT_VELOCITY = 5.0     # ~286 deg/s
# Maximum TCP speed (m/s)
MAX_TCP_SPEED = 2.0          # typical for collaborative arms


# ═══════════════════════════════════════════════════════════════════════════════
# Force/Torque Precision
# ═══════════════════════════════════════════════════════════════════════════════

# Force control accuracy (Newtons)
FORCE_ACCURACY_TIGHT = 1.0    # ±1 N (Franka level)
FORCE_ACCURACY_NORMAL = 3.5   # ±3.5 N (UR e-Series level)
FORCE_ACCURACY_LOOSE = 5.0    # ±5 N

# Torque control resolution (Nm)
TORQUE_RESOLUTION_TIGHT = 0.05    # ±0.05 Nm (Franka joint torque sensors)
TORQUE_RESOLUTION_NORMAL = 0.2    # ±0.2 Nm
TORQUE_RESOLUTION_LOOSE = 0.5     # ±0.5 Nm

# Maximum joint torque (Nm) — varies by joint, this is a representative value
MAX_JOINT_TORQUE = 87.0      # largest joint on a 7-DOF (similar to Franka joint 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Timing Precision
# ═══════════════════════════════════════════════════════════════════════════════

# Settling time after motion
SETTLING_TIME_TIGHT = 0.050     # 50 ms
SETTLING_TIME_NORMAL = 0.100    # 100 ms
SETTLING_TIME_LOOSE = 0.200     # 200 ms

# Control loop period
CONTROL_PERIOD_200HZ = 0.005    # 5 ms (200 Hz standard)
CONTROL_PERIOD_1KHZ = 0.001     # 1 ms (high-frequency)


# ═══════════════════════════════════════════════════════════════════════════════
# Mathematical Invariant Tolerances
# ═══════════════════════════════════════════════════════════════════════════════

# SO(3) validity: R^T R = I, det(R) = 1
SO3_ORTHOGONALITY_TOL = 1e-8
SO3_DETERMINANT_TOL = 1e-8

# SLERP geodesic proportionality
SLERP_GEODESIC_TOL = 1e-5

# Trajectory velocity integral = displacement
TRAJECTORY_INTEGRAL_TOL = 0.015  # 1.5% relative error (discretization at 200Hz)

# C1 continuity: no accel spike > K × average
C1_SPIKE_FACTOR = 5.0

# Rate limiter convergence: should reach target in ceil(d/delta)+1 steps
RATE_LIMITER_OVERSHOOT_STEPS = 1
