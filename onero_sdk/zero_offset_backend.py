"""
Zero-offset backend — software zero-point calibration at the Backend seam.

Applies a per-joint software offset to translate between the SDK's
"user" coordinate frame (URDF-defined zero, exposed to callers) and
the "raw" coordinate frame (motor-firmware zero, on the wire).

    pos_raw = pos_user + offset          (on command write)
    pos_user = pos_raw - offset          (on feedback read)

The wrapper sits between SafeBackend and the transport backend so that
URDF joint limits and safety clamping keep operating in user coordinates
without any changes:

    Arm → SafeBackend → ZeroOffsetBackend → CanBackend/MujocoBackend
        (user coord)   (translate here)    (raw coord)

Velocities, gains, and torques are frame-invariant and pass through
untouched.
"""

from dataclasses import replace

from .backend import ArmBackend, ArmState


class ZeroOffsetBackend(ArmBackend):
    def __init__(self, inner: ArmBackend, offsets: list[float] | None = None):
        self._inner = inner
        n = inner.num_joints
        if offsets is None:
            self._offsets = [0.0] * n
        else:
            if len(offsets) != n:
                raise ValueError(
                    f"ZeroOffsetBackend: got {len(offsets)} offsets for "
                    f"{n}-joint backend"
                )
            self._offsets = [float(o) for o in offsets]

    @property
    def num_joints(self) -> int:
        return self._inner.num_joints

    @property
    def inner(self) -> ArmBackend:
        return self._inner

    def get_offsets(self) -> list[float]:
        return list(self._offsets)

    def set_offsets(self, offsets: list[float]) -> None:
        if len(offsets) != self.num_joints:
            raise ValueError(
                f"ZeroOffsetBackend: got {len(offsets)} offsets for "
                f"{self.num_joints}-joint backend"
            )
        self._offsets = [float(o) for o in offsets]

    def _user_from_raw(self, state: ArmState) -> ArmState:
        return replace(
            state,
            joints=[j - o for j, o in zip(state.joints, self._offsets)],
        )

    def _raw_from_user(self, positions: list[float]) -> list[float]:
        return [p + o for p, o in zip(positions, self._offsets)]

    def enable(self) -> ArmState:
        return self._user_from_raw(self._inner.enable())

    def disable(self) -> None:
        self._inner.disable()

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        raw_positions = self._raw_from_user(positions)
        return self._user_from_raw(
            self._inner.command(raw_positions, velocities, kps, kds, torques)
        )

    def command_hold(self, positions: list[float], velocities: list[float],
                     kps: list[float], kds: list[float],
                     torques: list[float]) -> ArmState:
        raw_positions = self._raw_from_user(positions)
        return self._user_from_raw(
            self._inner.command_hold(
                raw_positions, velocities, kps, kds, torques
            )
        )

    def command_velocity(self, velocities: list[float]) -> ArmState:
        return self._user_from_raw(self._inner.command_velocity(velocities))

    def command_torque(self, torques: list[float]) -> ArmState:
        return self._user_from_raw(self._inner.command_torque(torques))

    def get_state(self) -> ArmState:
        return self._user_from_raw(self._inner.get_state())

    def close(self) -> None:
        self._inner.close()
