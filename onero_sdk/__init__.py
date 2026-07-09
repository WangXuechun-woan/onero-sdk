"""
onero SDK — Python interface for onero robotic arms.

High-level API (recommended):
    from onero_sdk import Arm, DualArm, ArmState

Low-level API (advanced):
    from onero_sdk import CanBus, Motor, MotorLimits, MotorState
    from onero_sdk.protocol import encode_mit, decode_feedback
"""

# High-level API
from .arm import Arm
from .dual_arm import DualArm
from .backend import ArmBackend, ArmState
from .end_effector import EndEffector, Gripper, DexterousHand
from .trajectory import Recording, Recorder, TrajectoryPlanner, Waypoint, TrajectoryFrame
from .exceptions import (
    OneroError, OneroConnectionError, OneroHardwareError,
    OneroConfigError, OneroTimeoutError, OneroSafetyError,
)
from .safety import SafetyConfig
from .collision import CollisionConfig, CollisionEvent, CollisionDetector, CollisionResponse

# Low-level API (hardware access)
from .can_bus import CanBus, CanFrame
from .motor import Motor
from .protocol import MotorLimits, MotorState, MotorError, RegisterID

# Utilities
from .model_registry import get_model, register_model, ArmModel, JointSpec
from .config import load_arm_config, load_dual_arm_config
from . import geometry
from .clock import ControlClock
from .motion_executor import MotionExecutor, MotionFrame, VelocityFrame
from .safe_backend import SafeBackend
from .gain_policy import GainPolicy, Stiffness
from .mounting import MountingPreset, MountingConfig
from .protocols import ArmProtocol
from .realtime import apply_realtime, release_realtime, RealtimeStatus

# Optional: IK (requires pip install onero-sdk[ik])
try:
    from .ik_solver import IKSolver, PinkIKSolver
except ImportError:
    IKSolver = None
    PinkIKSolver = None

# Optional: Dynamics (requires pip install onero-sdk[dynamics])
try:
    from .dynamics import DynamicsModel
except ImportError:
    DynamicsModel = None

# Optional: Calibration (requires pip install onero-sdk[dynamics])
try:
    from .calibration import CalibrationResult, GravityCalibrator
except ImportError:
    CalibrationResult = None
    GravityCalibrator = None

# Optional: Kinematic calibration (requires scipy)
try:
    from .kinematic_calibration import KinematicCalibrator, KinematicCalibrationResult
except ImportError:
    KinematicCalibrator = None
    KinematicCalibrationResult = None

# Trajectory profiles (numpy only — always available)
from .trajectory_profiles import (
    TrapezoidalProfile, SCurveProfile, QuinticBlend,
    compute_synchronized_duration, multi_joint_trapezoidal,
)

# Optional: Ruckig (requires pip install onero-sdk[trajectory])
try:
    from .ruckig_planner import RuckigPlanner
except ImportError:
    RuckigPlanner = None

# Optional: TOPP-RA (requires pip install onero-sdk[trajectory])
try:
    from .toppra_planner import TimeOptimalPlanner
except ImportError:
    TimeOptimalPlanner = None

# Optional: MuJoCo simulation (requires pip install onero-sdk[sim])
try:
    from .mujoco_backend import MujocoBackend
except ImportError:
    MujocoBackend = None

try:
    from .dual_mujoco_backend import DualMujocoBackend
except ImportError:
    DualMujocoBackend = None

__version__ = "0.3.0"


def __getattr__(name):
    if name == "URDFChain":
        from .fk import URDFChain
        return URDFChain
    raise AttributeError(f"module 'onero_sdk' has no attribute {name!r}")


__all__ = [
    # High-level
    "Arm", "DualArm", "ArmBackend", "ArmState",
    "EndEffector", "Gripper", "DexterousHand",
    "Recording", "Recorder", "TrajectoryPlanner", "Waypoint", "TrajectoryFrame",
    "OneroError", "OneroConnectionError", "OneroHardwareError", "OneroConfigError",
    "OneroTimeoutError", "OneroSafetyError",
    "SafetyConfig",
    "CollisionConfig", "CollisionEvent", "CollisionDetector", "CollisionResponse",
    # Low-level
    "CanBus", "CanFrame", "Motor",
    "MotorLimits", "MotorState", "MotorError", "RegisterID",
    # Utilities
    "get_model", "register_model", "ArmModel", "JointSpec",
    "load_arm_config", "load_dual_arm_config",
    "geometry", "ControlClock", "MotionExecutor", "MotionFrame", "VelocityFrame",
    "SafeBackend", "GainPolicy", "Stiffness",
    "MountingPreset", "MountingConfig",
    "ArmProtocol",
    "apply_realtime", "release_realtime", "RealtimeStatus",
    "URDFChain",
    # Optional
    "IKSolver", "PinkIKSolver", "DynamicsModel",
    "CalibrationResult", "GravityCalibrator",
    "KinematicCalibrator", "KinematicCalibrationResult",
    # Trajectory profiles
    "TrapezoidalProfile", "SCurveProfile", "QuinticBlend",
    "compute_synchronized_duration", "multi_joint_trapezoidal",
    "RuckigPlanner", "TimeOptimalPlanner",
]
