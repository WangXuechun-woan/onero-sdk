# ADR-0004: Tiered error handling with automatic protection

## Status

Accepted

## Context

Motors can report various faults (overvoltage, overcurrent, overtemperature, communication loss, overload). Real-time control loops (200Hz) cannot tolerate exceptions interrupting every cycle, but hardware-damaging faults must be caught immediately.

### Alternatives considered

1. **Silent** — errors stored in state, user checks. Risk: burned motors.
2. **Always throw** — immediate exception. Risk: crashes real-time loops.
3. **Callback only** — user registers handler. Risk: unregistered = silent.
4. **Tiered** — severity determines response. Automatic protection for dangerous faults.

## Decision

Tiered error handling:

| Severity | Examples | SDK response |
|----------|----------|--------------|
| Transient | Single frame timeout | Retry, use cached state, log at DEBUG |
| Warning | Intermittent timeouts (< N consecutive) | Log WARNING, increment counter |
| Critical | N=10 consecutive timeouts, overtemp, overload, overcurrent | Auto-disable all joints, log ERROR, invoke error callback |

Additional design choices:

- **Communication timeout threshold**: 10 consecutive missed frames (50ms at 200Hz), configurable.
- **Critical fault response**: disable entire arm (not just the faulting joint — partial operation is dangerous).
- **Notification**: Python `logging` always active; optional user callback (`arm.on_error = handler`).
- **`enable()` behavior**: auto-clears residual errors + retries once before failing. Failure = all-or-nothing rollback with per-joint error details.
- **Exception hierarchy**: flat — `OneroConnectionError`, `OneroHardwareError`, `OneroConfigError`, all inheriting `OneroError`. Use `.error_code` and `.joint_index` attributes for detail.

## Consequences

- Real-time loops are never interrupted by exceptions during normal operation (transient errors are absorbed).
- Hardware is protected: critical faults trigger immediate shutdown regardless of user code.
- Users who need programmatic response register a callback; others just see log output.
- Flat exception hierarchy is easy to catch (`except OneroError`) and extend later without breaking compatibility.
