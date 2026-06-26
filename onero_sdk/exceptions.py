"""
onero SDK exception hierarchy.

Flat design — three main exception types with attributes for detail.
"""

from typing import Optional


class OneroError(Exception):
    """Base exception for all onero SDK errors."""


class OneroConnectionError(OneroError):
    """CAN interface unavailable or connection failed."""

    def __init__(self, channel: str, detail: str = ""):
        self.channel = channel
        self.detail = detail
        msg = f"Cannot connect to CAN channel '{channel}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class OneroHardwareError(OneroError):
    """Motor hardware fault (overtemp, overload, overcurrent, etc.)."""

    def __init__(self, joint_index: int, error_code: int, description: str = ""):
        self.joint_index = joint_index
        self.error_code = error_code
        self.description = description
        msg = f"Hardware fault on joint {joint_index}: code=0x{error_code:X}"
        if description:
            msg += f" ({description})"
        super().__init__(msg)


class OneroConfigError(OneroError):
    """Configuration error (unknown model, invalid parameters)."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class OneroTimeoutError(OneroError):
    """Communication timeout (CAN frame not received within deadline)."""

    def __init__(self, joint_index: int, missed_frames: int, detail: str = ""):
        self.joint_index = joint_index
        self.missed_frames = missed_frames
        self.detail = detail
        msg = f"Timeout on joint {joint_index}: {missed_frames} consecutive missed frames"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class OneroSafetyError(OneroError):
    """Safety limit triggered (collision, joint limit, emergency stop)."""

    def __init__(self, reason: str, joint_index: Optional[int] = None):
        self.reason = reason
        self.joint_index = joint_index
        msg = f"Safety stop: {reason}"
        if joint_index is not None:
            msg += f" (joint {joint_index})"
        super().__init__(msg)
