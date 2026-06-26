"""
Model registry — maps arm model names to joint configurations.

Users specify only the model name; SDK resolves all hardware parameters.
"""

from dataclasses import dataclass


@dataclass
class JointSpec:
    """Physical parameters for a single joint/motor."""
    p_max: float
    v_max: float
    t_max: float
    can_id: int
    response_id: int
    p_min: float = None  # If None, symmetric limits: (-p_max, p_max)


@dataclass
class ArmModel:
    """Complete specification of an arm model."""
    name: str
    num_joints: int
    joints: list[JointSpec]
    default_kp: float = 50.0
    default_kd: float = 1.0
    control_freq: float = 100.0


REGISTRY: dict[str, ArmModel] = {
    "A1": ArmModel(
        name="A1",
        num_joints=7,
        joints=[
            JointSpec(p_max=12.5, v_max=30.0, t_max=15.0, can_id=0x01, response_id=0x11),
            JointSpec(p_max=12.5, v_max=30.0, t_max=15.0, can_id=0x02, response_id=0x12),
            JointSpec(p_max=12.5, v_max=30.0, t_max=15.0, can_id=0x03, response_id=0x13),
            JointSpec(p_max=12.5, v_max=30.0, t_max=3.0, can_id=0x04, response_id=0x14),
            JointSpec(p_max=12.5, v_max=30.0, t_max=3.0, can_id=0x05, response_id=0x15),
            JointSpec(p_max=12.5, v_max=30.0, t_max=3.0, can_id=0x06, response_id=0x16),
            JointSpec(p_max=12.5, v_max=30.0, t_max=3.0, can_id=0x07, response_id=0x17),
        ],
        default_kp=50.0,
        default_kd=1.0,
        control_freq=200.0,
    ),
}


def get_model(name: str) -> ArmModel:
    """Look up an arm model by name. Raises OneroConfigError if not found."""
    from .exceptions import OneroConfigError
    model = REGISTRY.get(name)
    if model is None:
        available = ", ".join(REGISTRY.keys())
        raise OneroConfigError(f"Unknown arm model '{name}'. Available: {available}")
    return model


def register_model(name: str, model: ArmModel) -> None:
    """Register a custom arm model."""
    REGISTRY[name] = model
