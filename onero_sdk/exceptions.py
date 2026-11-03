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
            msg += f" (joint {joint_index + 1})"
        super().__init__(msg)


class IKConvergenceError(OneroError):
    """
    Inverse kinematics failed to reach the target within tolerance.

    Raised when the target pose cannot be reached from the given initial
    joint configuration — typically because the target is outside the
    workspace, near a singularity, or the IK got stuck in a local minimum.

    The `q` attribute holds the best-effort joint configuration returned
    by the solver (useful for diagnostics but NOT safe to command directly).
    """

    def __init__(self, pos_err: float, rot_err: float, iters: int,
                 pos_tol: float, rot_tol: float,
                 q: Optional[list] = None,
                 target_pos: Optional[list] = None):
        self.pos_err = pos_err
        self.rot_err = rot_err
        self.iters = iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.q = q
        self.target_pos = target_pos
        msg = (f"IK did not converge after {iters} iters: "
               f"pos_err={pos_err*1000:.2f}mm (tol {pos_tol*1000:.2f}mm), "
               f"rot_err={rot_err*1000:.2f}mrad (tol {rot_tol*1000:.2f}mrad).")
        if target_pos is not None:
            msg += f" Target: {target_pos}."
        super().__init__(msg)
