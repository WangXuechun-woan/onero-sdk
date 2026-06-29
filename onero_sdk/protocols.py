"""Adapter-facing protocol — what external integrations depend on from Arm."""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

from .backend import ArmState


@runtime_checkable
class ArmProtocol(Protocol):
    """
    Minimal interface that external adapters (BEAVR, LeRobot) rely on.

    This is a typing.Protocol — no runtime overhead, no ABC inheritance.
    The Arm class satisfies this implicitly via structural subtyping.
    Use isinstance(arm, ArmProtocol) in contract tests to verify conformance.
    """

    @property
    def state(self) -> ArmState: ...

    @property
    def enabled(self) -> bool: ...

    def enable(self) -> ArmState: ...

    def disable(self) -> None: ...

    def servoj(self, joints: list[float], **kwargs) -> ArmState: ...

    def servol(self, position: Union[list, np.ndarray],
               rotation: Union[list, np.ndarray, None] = None,
               *, frame: Optional[str] = None,
               **kwargs) -> ArmState: ...

    def movej(self, joints: list[float], **kwargs) -> ArmState: ...

    def movel(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray, None] = None,
              *, frame: Optional[str] = None,
              **kwargs) -> ArmState: ...

    def movep(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray, None] = None,
              *, frame: Optional[str] = None,
              **kwargs) -> ArmState: ...

    def movec(self, via_pose: Union[list, np.ndarray],
              target_pose: Union[list, np.ndarray],
              *, frame: Optional[str] = None,
              **kwargs) -> ArmState: ...

    def speedl(self, velocities: list[float], *,
               frame: Optional[str] = None, **kwargs) -> ArmState: ...

    def force_mode(self, selection: list[int], wrench: list[float], *,
                   frame: Optional[str] = None, **kwargs) -> ArmState: ...

    def get_pose(self, frame: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]: ...

    def get_external_wrench(self, frame: Optional[str] = None) -> np.ndarray: ...

    def register_frame(self, name: str, rotation, position=None) -> None: ...
