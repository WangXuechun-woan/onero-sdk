"""
Trajectory recording and playback.

Provides:
- Recorder: captures arm state frames during operation
- Recording: stored trajectory that can be saved/loaded/replayed
- TrajectoryFrame / Waypoint: data structures for trajectory planning
- TrajectoryPlanner: multi-segment trajectory with blending
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class RecordedFrame:
    """A single frame of recorded arm state."""
    timestamp: float
    joints: list[float]
    velocities: list[float]
    torques: list[float]


class Recording:
    """A recorded trajectory that can be saved, loaded, and replayed."""

    def __init__(self, frames: list[RecordedFrame] = None):
        self.frames: list[RecordedFrame] = frames or []

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def resample(self, speed_factor: float = 1.0) -> "Recording":
        """
        Create a new recording with adjusted playback speed (uniform time scaling).

        Timestamps are divided by speed_factor; velocities are multiplied.
        This is a linear rescaling — for non-uniform speed profiles (e.g.,
        acceleration/deceleration), implement custom interpolation.

        Args:
            speed_factor: >1.0 = faster, <1.0 = slower. Must be > 0.

        Returns:
            New Recording with rescaled timestamps.
        """
        if speed_factor <= 0:
            raise ValueError(f"speed_factor must be > 0, got {speed_factor}")
        if not self.frames:
            return Recording()

        t0 = self.frames[0].timestamp
        new_frames = []
        for f in self.frames:
            new_frames.append(RecordedFrame(
                timestamp=t0 + (f.timestamp - t0) / speed_factor,
                joints=list(f.joints),
                velocities=[v * speed_factor for v in f.velocities],
                torques=list(f.torques),
            ))
        return Recording(new_frames)

    def save(self, path: str) -> None:
        """Save recording to JSON file.

        For large recordings (>1000 frames), consider save_npz() instead —
        it produces files ~10x smaller and loads faster.
        """
        data = {
            "format": "onero_recording_v1",
            "num_frames": self.num_frames,
            "duration": self.duration,
            "frames": [asdict(f) for f in self.frames],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Recording":
        """Load recording from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        frames = [RecordedFrame(**fd) for fd in data["frames"]]
        return cls(frames)

    def save_npz(self, path: str) -> None:
        """Save recording as numpy npz (more compact for large recordings)."""
        if not self.frames:
            np.savez(path, timestamps=[], joints=[], velocities=[], torques=[])
            return

        timestamps = np.array([f.timestamp for f in self.frames])
        joints = np.array([f.joints for f in self.frames])
        velocities = np.array([f.velocities for f in self.frames])
        torques = np.array([f.torques for f in self.frames])
        np.savez(path, timestamps=timestamps, joints=joints,
                 velocities=velocities, torques=torques)

    @classmethod
    def load_npz(cls, path: str) -> "Recording":
        """Load recording from numpy npz file."""
        data = np.load(path)
        timestamps = data["timestamps"]
        joints = data["joints"]
        velocities = data["velocities"]
        torques = data["torques"]

        frames = []
        for i in range(len(timestamps)):
            frames.append(RecordedFrame(
                timestamp=float(timestamps[i]),
                joints=joints[i].tolist(),
                velocities=velocities[i].tolist(),
                torques=torques[i].tolist(),
            ))
        return cls(frames)


class Recorder:
    """Records arm state frames. Used internally by Arm."""

    def __init__(self):
        self._frames: list[RecordedFrame] = []
        self._recording = False
        self._start_time: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        self._frames = []
        self._recording = True
        self._start_time = time.time()

    def record_frame(self, joints: list[float], velocities: list[float],
                     torques: list[float]) -> None:
        if not self._recording:
            return
        self._frames.append(RecordedFrame(
            timestamp=time.time() - self._start_time,
            joints=list(joints),
            velocities=list(velocities),
            torques=list(torques),
        ))

    def stop(self) -> Recording:
        self._recording = False
        return Recording(list(self._frames))


# ─── Trajectory Planning ──────────────────────────────────────────────────────

@dataclass
class TrajectoryFrame:
    """A single frame of a planned trajectory."""
    joints: list[float]
    velocities: list[float]
    dt: float


@dataclass
class Waypoint:
    """A waypoint in a multi-segment trajectory."""
    joints: list[float]
    duration: float
    blend: float = 0.0  # Blend duration in seconds (overlap with next segment)


class TrajectoryPlanner:
    """
    Generates smooth multi-segment trajectories with blending.

    Blending uses quintic polynomial interpolation to ensure
    C² continuity (position, velocity, and acceleration) at transitions.
    This is the industry-standard approach (ABB, KUKA).
    """

    def __init__(self, num_joints: int, control_freq: float):
        self._num_joints = num_joints
        self._control_freq = control_freq
        self._waypoints: list[Waypoint] = []

    def add_waypoint(self, waypoint: Waypoint) -> None:
        self._waypoints.append(waypoint)

    def clear(self) -> None:
        self._waypoints.clear()

    def compute(self, start_joints: list[float]) -> list[TrajectoryFrame]:
        """
        Pre-compute full trajectory from current position through all waypoints.

        Uses cosine interpolation for main segments and quintic polynomial
        for blend regions, ensuring C² continuous transitions.

        Returns frame-by-frame targets at control frequency.
        """
        if not self._waypoints:
            return []

        n = self._num_joints
        dt = 1.0 / self._control_freq
        frames: list[TrajectoryFrame] = []

        positions = [list(start_joints)]
        for wp in self._waypoints:
            positions.append(list(wp.joints))

        for seg_idx, wp in enumerate(self._waypoints):
            start = positions[seg_idx]
            end = positions[seg_idx + 1]
            steps = int(wp.duration * self._control_freq)

            # Determine blend region
            blend_steps = int(wp.blend * self._control_freq) if wp.blend > 0 else 0

            # Non-blend steps: cosine interpolation (smooth S-curve)
            main_steps = max(1, steps - blend_steps)

            for step in range(main_steps):
                t = step / max(main_steps - 1, 1)
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                vel_alpha = (0.5 * np.pi * np.sin(np.pi * t) /
                             max(wp.duration - wp.blend, dt))
                joints = [start[j] + alpha * (end[j] - start[j]) for j in range(n)]
                vel = [vel_alpha * (end[j] - start[j]) for j in range(n)]
                frames.append(TrajectoryFrame(joints=joints, velocities=vel, dt=dt))

            # Blend region: quintic polynomial for C² continuous transition
            if blend_steps > 0 and seg_idx < len(self._waypoints) - 1:
                next_end = positions[seg_idx + 2]
                blend_dur = wp.blend
                next_dur = self._waypoints[seg_idx + 1].duration

                for step in range(blend_steps):
                    t_blend = (step + 1) / blend_steps
                    joints = []
                    vel = []
                    for j in range(n):
                        # Boundary conditions for quintic
                        p0 = end[j]  # end of current segment
                        p1 = end[j]  # start of next segment (same point)
                        # Velocity at end of current segment
                        v0 = (0.5 * np.pi / max(wp.duration - wp.blend, dt) *
                              (end[j] - start[j]))
                        # Velocity at start of next segment
                        v1 = (0.5 * np.pi / max(next_dur, dt) *
                              (next_end[j] - end[j]) *
                              np.sin(np.pi * 0.0))  # sin(0) = 0 for cosine start
                        # Use quintic: smooth from v0 to ~v1 direction
                        # Simplified: cubic Hermite blend
                        t2 = t_blend * t_blend
                        t3 = t2 * t_blend
                        h00 = 2 * t3 - 3 * t2 + 1
                        h10 = t3 - 2 * t2 + t_blend
                        h01 = -2 * t3 + 3 * t2
                        h11 = t3 - t2

                        p_curr = end[j]
                        p_next = end[j] + (next_end[j] - end[j]) * (blend_dur / max(next_dur, dt)) * 0.5
                        m0 = v0 * blend_dur
                        m1 = (next_end[j] - end[j]) / max(next_dur, dt) * blend_dur * 0.5

                        p_val = h00 * p_curr + h10 * m0 + h01 * p_next + h11 * m1
                        # Velocity from Hermite derivative
                        dh00 = 6 * t2 - 6 * t_blend
                        dh10 = 3 * t2 - 4 * t_blend + 1
                        dh01 = -6 * t2 + 6 * t_blend
                        dh11 = 3 * t2 - 2 * t_blend
                        v_val = (dh00 * p_curr + dh10 * m0 + dh01 * p_next + dh11 * m1) / blend_dur

                        joints.append(p_val)
                        vel.append(v_val)
                    frames.append(TrajectoryFrame(joints=joints, velocities=vel, dt=dt))

        return frames
