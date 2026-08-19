"""
Model registry — maps arm model names to joint configurations.

Physical parameters (joint limits, max velocity, max torque) are read from
URDF so that updating the URDF is the only step needed for new hardware.
CAN bus IDs and control parameters are defined here (not in URDF).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@dataclass
class JointSpec:
    """Physical parameters for a single joint/motor."""
    p_min: float         # Physical lower joint limit (rad) — for safety clamping
    p_max: float         # Physical upper joint limit (rad) — for safety clamping
    v_max: float         # Max velocity (rad/s)
    t_max: float         # Max torque (Nm)
    can_id: int
    response_id: int
    encoding_p_max: float = 12.5  # Motor encoder range for MIT protocol (rad)


@dataclass
class ArmModel:
    """Complete specification of an arm model."""
    name: str
    num_joints: int
    joints: list[JointSpec]
    default_kp: float = 50.0
    default_kd: float = 1.0
    control_freq: float = 100.0
    gear_ratios: list[float] | None = None
    rotor_inertias: list[float] | None = None


def _parse_joint_limits_from_urdf(urdf_path: Path) -> list[dict]:
    """
    Parse joint position/velocity/effort limits from URDF.

    Returns list of dicts with keys: p_min, p_max, v_max, t_max
    for each revolute joint in chain order.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    limits = []
    for joint_elem in root.findall("joint"):
        if joint_elem.get("type") not in ("revolute", "continuous"):
            continue
        limit_elem = joint_elem.find("limit")
        if limit_elem is None:
            continue
        limits.append({
            "p_min": float(limit_elem.get("lower", 0)),
            "p_max": float(limit_elem.get("upper", 0)),
            "v_max": float(limit_elem.get("velocity", 0)),
            "t_max": float(limit_elem.get("effort", 0)),
        })
    return limits


def _parse_model_variant(name: str) -> tuple[str, str]:
    """
    Parse model name into (base_model, variant).

    Examples:
        "A1R" → ("A1", "r")
        "A1L" → ("A1", "l")
        "A1"  → ("A1", "r")  (deprecated, defaults to right)
    """
    upper = name.upper()
    if upper.endswith("R") and upper[:-1] in _CAN_CONFIG:
        return upper[:-1], "r"
    if upper.endswith("L") and upper[:-1] in _CAN_CONFIG:
        return upper[:-1], "l"
    return name, "r"


def _find_urdf(model_name: str, variant: str = "r") -> Path:
    """Locate URDF file for a given model and variant (r=right, l=left)."""
    base = Path(__file__).parent / "urdf" / model_name
    urdf_file = base / f"{model_name.lower()}_{variant}.urdf"
    if urdf_file.exists():
        return urdf_file
    # Fallback: try without variant
    for f in base.glob("*.urdf"):
        return f
    return None


# CAN bus configuration per model (hardware-specific, not in URDF)
_CAN_CONFIG = {
    "A1": {
        "can_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        "response_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17],
        "encoding_p_max": 12.5,  # Motor firmware PMAX register (encoder range)
        "default_kp": 50.0,
        "default_kd": 1.0,
        "control_freq": 200.0,
        # Gear ratios: joint 1-3 large (48.19:1), joint 4-7 small (10:1)
        "gear_ratios": [48.19, 48.19, 48.19, 10.0, 10.0, 10.0, 10.0],
        # Estimated rotor inertia (kg·m²) — typical for this motor class
        # Large joints: ~40mm BLDC, ~4.5e-6 kg·m²
        # Small joints: ~28mm BLDC, ~1.5e-6 kg·m²
        "rotor_inertias": [4.5e-6, 4.5e-6, 4.5e-6, 1.5e-6, 1.5e-6, 1.5e-6, 1.5e-6],
    },
}


def _build_model(name: str, variant: str = "r") -> ArmModel:
    """Build ArmModel by combining URDF limits with CAN config."""
    from .exceptions import OneroConfigError

    can_cfg = _CAN_CONFIG.get(name)
    if can_cfg is None:
        raise OneroConfigError(f"No CAN configuration for model '{name}'")

    urdf_path = _find_urdf(name, variant)
    if urdf_path is None:
        raise OneroConfigError(
            f"URDF not found for model '{name}'. "
            f"Expected at: onero_sdk/urdf/{name}/")

    limits = _parse_joint_limits_from_urdf(urdf_path)
    num_joints = len(can_cfg["can_ids"])

    if len(limits) != num_joints:
        raise OneroConfigError(
            f"URDF has {len(limits)} revolute joints but CAN config "
            f"expects {num_joints} for model '{name}'")

    joints = []
    encoding_p_max = can_cfg.get("encoding_p_max", 12.5)
    for i, lim in enumerate(limits):
        # SolidWorks URDF exporter leaves effort=0 / velocity=0 by default.
        # Fall back to safe simulation/hardware ceilings so encoding never
        # divides by zero.  Real limits are enforced by the safety layer.
        v_max = lim["v_max"] if lim["v_max"] > 0 else 30.0   # rad/s
        t_max = lim["t_max"] if lim["t_max"] > 0 else 15.0   # Nm
        joints.append(JointSpec(
            p_min=lim["p_min"],
            p_max=lim["p_max"],
            v_max=v_max,
            t_max=t_max,
            can_id=can_cfg["can_ids"][i],
            response_id=can_cfg["response_ids"][i],
            encoding_p_max=encoding_p_max,
        ))

    return ArmModel(
        name=name,
        num_joints=num_joints,
        joints=joints,
        default_kp=can_cfg["default_kp"],
        default_kd=can_cfg["default_kd"],
        control_freq=can_cfg["control_freq"],
        gear_ratios=can_cfg.get("gear_ratios"),
        rotor_inertias=can_cfg.get("rotor_inertias"),
    )


# Lazy-loaded registry cache
_REGISTRY_CACHE: dict[str, ArmModel] = {}


def get_model(name: str) -> ArmModel:
    """
    Look up an arm model by name. Raises OneroConfigError if not found.

    Supports explicit variant naming:
        "A1R" — right arm
        "A1L" — left arm
        "A1"  — deprecated, defaults to right arm with warning
    """
    from .exceptions import OneroConfigError

    if name in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[name]

    base_model, variant = _parse_model_variant(name)

    if base_model not in _CAN_CONFIG:
        available = []
        for k in _CAN_CONFIG:
            available.extend([f"{k}L", f"{k}R"])
        raise OneroConfigError(
            f"Unknown arm model '{name}'. Available: {', '.join(available)}")

    if name in _CAN_CONFIG and name == base_model:
        logger.warning(
            f"Arm('{name}') is deprecated — please use '{name}R' (right) "
            f"or '{name}L' (left) explicitly.")

    model = _build_model(base_model, variant)
    model = ArmModel(
        name=name,
        num_joints=model.num_joints,
        joints=model.joints,
        default_kp=model.default_kp,
        default_kd=model.default_kd,
        control_freq=model.control_freq,
        gear_ratios=model.gear_ratios,
        rotor_inertias=model.rotor_inertias,
    )
    _REGISTRY_CACHE[name] = model
    return model


def register_model(name: str, model: ArmModel) -> None:
    """Register a custom arm model (bypasses URDF parsing)."""
    _REGISTRY_CACHE[name] = model
