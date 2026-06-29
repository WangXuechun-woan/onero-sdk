"""Adapter-facing protocol — what external integrations depend on from Arm."""

from __future__ import annotations

from typing import Protocol, Union, runtime_checkable

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
               **kwargs) -> ArmState: ...

    def movej(self, joints: list[float], **kwargs) -> ArmState: ...
