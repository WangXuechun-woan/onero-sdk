# ADR-0001: Backend abstraction at arm level, not motor level

## Status

Accepted

## Context

The SDK needs a backend abstraction to support multiple execution targets (real CAN hardware, MuJoCo simulation, potentially EtherCAT in the future). The question is where to cut the interface:

- **Motor-level**: each backend implements single-motor primitives (enable, set_mit, get_feedback). The `Motor` class holds a backend reference. Flexible, but forces simulators to model per-motor request/response which is unnatural for physics engines.
- **Arm-level**: each backend implements whole-arm operations (enable_all, set_joint_targets, get_joint_states). The `Arm` class holds a backend reference. `Motor` as an independent abstraction only exists in the CAN layer.

## Decision

Backend interface operates at the **arm level** — all joints simultaneously.

## Consequences

- MuJoCo backend is straightforward: one `mj_step` maps directly to one arm-level command cycle.
- Real CAN backend internally iterates over motors, but presents the same whole-arm interface.
- Single-motor operations (parameter tuning, error clearing) are only available via direct CAN access, not through the backend abstraction. This is intentional — these are hardware-debug operations, not user-facing workflow.
- Adding a new arm model (different DOF count) only requires a new Model Registry entry and a backend that handles that many joints.
- If we ever need per-motor backend flexibility (e.g., heterogeneous actuator buses), this decision would need revisiting.
