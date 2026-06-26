# ADR-0003: Simulation as same-repo sub-package with step-based time model

## Status

Accepted

## Context

Users need to develop and test without physical hardware. MuJoCo is the simplest physics simulator to integrate (pip-installable, fast, good robotics support). The question is how to package and how to model time.

### Packaging alternatives

1. **Independent repo/package** — separate versioning, more overhead, version drift risk.
2. **Same repo, extras_require** — `pip install onero-sdk[sim]`, version-locked, shared CI.

### Time model alternatives

1. **Wall-clock** — `movej(duration=3.0)` uses `time.sleep()`. Simulation must run 1x speed.
2. **Sim-clock via backend step** — `movej` iterates N steps (duration × frequency). Backend decides what a "step" means: real backend = sleep + CAN send; sim backend = `mj_step`. Simulation can run faster than real-time.

## Decision

- Same-repo sub-package, installed via `pip install onero-sdk[sim]`.
- Time driven by backend steps: `movej(duration=3.0)` at 200Hz = 600 backend steps. The backend interface includes a `step(dt)` concept. Real backend: send + sleep. Sim backend: `mj_step(dt)`.
- Two physics modes: **dynamics** (PD torque control, default) and **kinematic** (direct qpos assignment, for fast trajectory validation).

## Consequences

- Simulation can run 10-100x faster than real-time (no sleep in sim backend step).
- RL training and batch testing become practical without hardware.
- The Backend interface must include a step/tick method — this is a slight addition to the interface beyond pure command/query.
- MuJoCo model files (MJCF) and basic scenes ship with the sim package.
- Sim-specific operations (reset, add objects, viewer) live on the backend object, not on Arm.
