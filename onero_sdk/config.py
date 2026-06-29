"""
Configuration loading — single-arm as the atomic unit.
"""

from typing import Optional

from .safety import SafetyConfig
from .exceptions import OneroConfigError


def load_arm_config(path: str) -> dict:
    """
    Load arm configuration from a YAML file.

    Minimal config:
        model: "A1R"
        channel: "can0"

    Full config:
        model: "A1R"
        channel: "can0"
        mounting: "side_mount_right"
        world_frame: true
        base_position: [0, -0.15, 0]
        kp: 50.0
        kd: 1.0
        urdf: "path/to/arm.urdf"
        safety:
          max_velocity: 5.0
          max_delta_per_step: 0.03
          timeout_frames: 10
          enabled: true

    Returns:
        Dictionary with parsed config values.
    """
    try:
        import yaml
    except ImportError:
        raise OneroConfigError("pyyaml is required: pip install pyyaml")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    return raw


def load_dual_arm_config(path: str) -> dict:
    """
    Load dual-arm configuration from a YAML file.

    Format:
        left:
          model: "A1L"
          channel: "can0"
          mounting: "side_mount_left"
          base_position: [0, 0.15, 0]
        right:
          model: "A1R"
          channel: "can1"
          mounting: "side_mount_right"
          base_position: [0, -0.15, 0]
        # Optional:
        world_frame: true
        arm_spacing: 0.3  # alternative to per-arm base_position

    Returns:
        Dictionary with 'left' and 'right' arm configs.
    """
    try:
        import yaml
    except ImportError:
        raise OneroConfigError("pyyaml is required: pip install pyyaml")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    if "left" not in raw or "right" not in raw:
        raise OneroConfigError("Dual-arm config must have 'left' and 'right' sections")

    return raw
