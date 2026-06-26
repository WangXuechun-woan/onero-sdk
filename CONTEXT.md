# Context: onero SDK

## Glossary

### Joint
A single rotational degree of freedom in the arm. Strictly 1:1 with a physical motor — they are the same entity at different abstraction levels. Use "Joint" when discussing kinematics and user-facing APIs; use "Motor" only when referring to the physical hardware or CAN-level control.

### Arm
A 7-DOF kinematic chain. The core primitive of the SDK — all higher-level APIs build on Arm. An Arm can optionally have an End Effector attached, but the Arm itself is always exactly 7 joints.

### End Effector
An optional attachment at the end of an Arm (gripper, dexterous hand). Controlled via the same CAN bus but independent of the Arm's 7 joints. Not configured by default; SDK provides extensibility hooks for custom end effectors.

### DualArm
A convenience combination of two Arms (left + right). Not the core primitive — single-arm deployment is a first-class scenario. DualArm provides no coordination logic (no collision avoidance), just joint enable/disable/command for both arms together.

### Backend
An abstraction at the Arm level that decouples motion commands from the transport. Implementations include the real CAN backend (sending MIT commands over SocketCAN) and a simulation backend (driving MuJoCo or similar). The interface operates on all joints of an arm simultaneously, not per-motor.

### Planned Motion
A motion mode where the user specifies a goal (joint angles or Cartesian pose) and the SDK internally handles trajectory planning, interpolation, and smoothing. Blocking — returns when the motion is complete.

### Real-time Control
A motion mode where the user sends targets at each control step. The SDK executes them directly (with default safety limits). Non-blocking — user is responsible for trajectory smoothness. Used by teleoperation and other streaming applications.

### Safety Limits
Default protections active in Real-time Control mode: joint position limits, velocity clamping, acceleration bounds. Prevent hardware damage (especially on smaller wrist motors). Can be explicitly disabled by advanced users.

### Model Registry
An internal lookup table mapping arm model names (e.g. "A1") to their joint configurations (number of joints, position/velocity/torque limits per joint, CAN ID assignments). Users specify only the model name; SDK resolves all hardware parameters.

### movej / movep
Planned motion commands (blocking). `movej` moves in joint space, `movep` moves in task space (Cartesian). Named after the industrial robotics convention (UR, ABB). SDK internally handles trajectory interpolation and smoothing.

### servoj / servop
Real-time control commands (non-blocking, single-frame). `servoj` sets a joint-space target, `servop` sets a task-space target. User calls these at control frequency (e.g. 200Hz). No explicit start/stop required — arm must be enabled. Timeout protection auto-holds position if no new target arrives.

### Forward Kinematics (FK)
Built into the SDK with a lightweight URDF-based implementation. No external dependency (no pinocchio required). Computes end-effector pose from joint angles. Included in arm state.

### Inverse Kinematics (IK)
Optional capability requiring `pip install onero-sdk[ik]` (pinocchio + pink). Converts task-space targets to joint angles. Used by `movep` and `servop`.

### Simulation Backend
An optional attachment package (`pip install onero-sdk[sim]`). Implements the Backend interface using MuJoCo. Supports two physics modes: dynamics mode (PD impedance control, realistic) and kinematic mode (direct position assignment, fast). Time is driven by simulation steps, enabling accelerated execution.

## Not in this SDK

### BEAVR Integration
The OneroA1Robot, OneroA1LeftOperator, OneroA1RightOperator, and OneroA1Config classes implement the BEAVR framework's RobotWrapper/XArmOperator interfaces. They are applications built on top of the SDK, not part of it. They live in a fork of the BEAVR repository (to be upstreamed via PR), not in this repo. BEAVR is a third-party teleoperation framework.

### BeavrBridge
Historical prototype for standalone VR teleoperation. Superseded by the BEAVR integration above. To be deleted.

### CAN Interface Setup
Initializing the CAN bus (slcand, udev rules, `ip link set up`) is not a runtime concern of the SDK. The SDK assumes a CAN interface is already UP. Setup tools (scripts, udev rules, CLI) are shipped alongside the SDK as utilities but are not part of the runtime.
