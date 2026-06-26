# ADR-0002: Two-tier motion API — planned motion and real-time control

## Status

Accepted

## Context

Users have fundamentally different needs depending on their use case:

1. **Application developers** (pick-and-place, demos): want to say "move here" and have it happen safely.
2. **Control engineers** (teleoperation, custom planners): want to send targets every control cycle (200Hz) with minimal latency.

A single API cannot serve both well — safety/smoothing adds latency that streaming users cannot tolerate, while raw command access is dangerous for casual users.

## Decision

The SDK provides two motion modes at the high-level Arm API:

| Mode | Behavior | User responsibility |
|------|----------|-------------------|
| **Planned motion** | Blocking. SDK interpolates trajectory, enforces smooth acceleration. | Specify goal + duration. |
| **Real-time control** | Non-blocking. SDK executes each target immediately (with default safety limits). | Generate smooth trajectory externally, call at control rate. |

Both modes are available in joint space and task space (Cartesian, requires IK extras).

Safety limits (joint bounds, velocity clamping) are active by default in real-time mode but can be explicitly disabled by advanced users.

## Consequences

- Planned motion internally uses MIT impedance mode with cosine (or trapezoidal) interpolation. Users never see "MIT mode" — they see `move_to()`.
- Real-time control exposes `set_target()` / `set_target_pose()` — one call per cycle, no internal buffering.
- The control mode (MIT vs position-velocity vs velocity) is an implementation detail hidden from high-level users. Advanced users can drop to the Motor/CAN layer for direct mode control.
- Documentation must clearly guide users to the right mode for their use case.
