"""
Zero-point offset persistence.

Stores per-arm software zero offsets in ~/.onero/zero_offsets/<key>.json
where <key> is "<model>_<channel>" when the CAN channel is known and
"<model>" otherwise. Applied at runtime via ZeroOffsetBackend — no motor
firmware writes involved.

File format (v1):
    {
      "version": 1,
      "model": "A1L",
      "channel": "can0",
      "offsets_rad": [...],
      "created_at": "2026-07-15T10:23:41+00:00",
      "note": "..."
    }
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .exceptions import OneroConfigError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_ENCODING_MARGIN = 0.9  # keep 10% headroom below motor encoding p_max


def _config_dir() -> Path:
    override = os.environ.get("ONERO_CONFIG_DIR")
    if override:
        return Path(override) / "zero_offsets"
    return Path.home() / ".onero" / "zero_offsets"


@dataclass
class ZeroOffsetRecord:
    version: int
    model: str
    channel: str
    offsets_rad: list[float]
    created_at: str
    note: str = ""


def default_path(model: str, channel: Optional[str]) -> Path:
    stem = f"{model}_{channel}" if channel else model
    return _config_dir() / f"{stem}.json"


def load_zero_offset(model: str, channel: Optional[str], num_joints: int,
                     path: Optional[Path] = None) -> list[float]:
    """
    Return the stored offset array, or [0.0]*num_joints if no file exists.

    A corrupt or wrong-length file is treated as missing and logged as a
    warning — better to fall back to no calibration than to refuse to boot.
    """
    p = path or default_path(model, channel)
    if not p.exists():
        return [0.0] * num_joints
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to read zero offset file %s: %s. Using zero offsets.",
            p, e,
        )
        return [0.0] * num_joints

    version = data.get("version")
    if version != SCHEMA_VERSION:
        logger.warning(
            "Zero offset file %s has schema version %r, expected %d. "
            "Refusing to load — the semantics may differ. "
            "Delete the file or re-run arm.set_current_position_as_zero() "
            "to regenerate it. Using zero offsets.",
            p, version, SCHEMA_VERSION,
        )
        return [0.0] * num_joints

    offsets = data.get("offsets_rad", [])
    if len(offsets) != num_joints:
        logger.warning(
            "Zero offset file %s has %d entries, expected %d for model %s. "
            "Using zero offsets.",
            p, len(offsets), num_joints, model,
        )
        return [0.0] * num_joints
    return [float(o) for o in offsets]


def save_zero_offset(model: str, channel: Optional[str], offsets: list[float],
                     path: Optional[Path] = None, note: str = "") -> Path:
    """Atomically write the offset record. Returns the final file path."""
    p = path or default_path(model, channel)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = ZeroOffsetRecord(
        version=SCHEMA_VERSION,
        model=model,
        channel=channel or "",
        offsets_rad=[float(o) for o in offsets],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note=note,
    )
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2) + "\n")
    tmp.replace(p)
    return p


def remove_zero_offset(model: str, channel: Optional[str],
                       path: Optional[Path] = None) -> bool:
    """Delete the offset file. Returns True if a file was removed."""
    p = path or default_path(model, channel)
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


def validate_offsets(offsets: list[float], joint_specs) -> None:
    """
    Check that URDF joint limits + the given offsets still fit inside
    90% of the motor's encoding range (encoding_p_max).

    After applying the offset, a user-space position of user_pos becomes
    a raw position of user_pos + offset on the wire. The MIT encoder
    can only represent [-encoding_p_max, +encoding_p_max]; if either
    limit + offset overshoots, the encoded value wraps and the motor
    executes a garbage target.

    Raises OneroConfigError on the first violation.
    """
    if len(offsets) != len(joint_specs):
        raise OneroConfigError(
            f"validate_offsets: got {len(offsets)} offsets for "
            f"{len(joint_specs)} joints"
        )
    for i, (offset, spec) in enumerate(zip(offsets, joint_specs)):
        max_raw = max(abs(spec.p_min + offset), abs(spec.p_max + offset))
        limit = _ENCODING_MARGIN * spec.encoding_p_max
        if max_raw > limit:
            raise OneroConfigError(
                f"Joint {i}: offset {offset:+.3f} rad pushes raw position "
                f"beyond {int(_ENCODING_MARGIN * 100)}% of motor encoding "
                f"range (max |raw|={max_raw:.3f} > {limit:.3f} rad, "
                f"encoding_p_max={spec.encoding_p_max}). "
                f"Move the arm closer to its factory zero before "
                f"re-calibrating, or reset the offset with "
                f"arm.clear_zero_offset()."
            )
