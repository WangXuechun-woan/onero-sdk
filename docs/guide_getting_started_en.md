# Onero SDK Getting Started Guide

A Python SDK for controlling Onero robotic arms (A1: 7-DOF).

## Installation

```bash
pip install onero-sdk          # Core (numpy only)
pip install onero-sdk[sim]     # + MuJoCo simulation
pip install onero-sdk[dynamics] # + Pinocchio (gravity comp, freedrive)
pip install onero-sdk[full]    # Everything
```

## Quick Start (no hardware needed)

```python
from onero_sdk import Arm
from onero_sdk.mujoco_backend import MujocoBackend

# Simulation backend — identical API to real hardware
arm = Arm("A1R", backend=MujocoBackend(model_name="A1R"))
arm.enable()

# Joint-space motion: move each of 7 joints to target angles (radians)
arm.movej([0, -0.5, 0.3, -1.0, 0, 0.5, 0], duration=2.0)

# Cartesian motion: move end-effector to [x, y, z, rx, ry, rz]
arm.movel([0.3, 0.0, 0.2, 0, 0, 0], duration=2.0)

# Get current end-effector pose
pos, rot = arm.get_pose()  # pos: [x,y,z], rot: 3x3 rotation matrix
print(f"End-effector at: {pos}")

arm.disable()
```

For real hardware, replace `backend=...` with `channel="can0"`:
```python
arm = Arm("A1R", channel="can0")
```

## Core Concepts

### Motion Types

| Method | Description | Use when... |
|--------|-------------|-------------|
| `movej(joints)` | Joint-space interpolation | Moving to a known joint config |
| `movel(pos, rot)` | Straight line in Cartesian space | Precise linear path matters |
| `movep(pos, rot)` | IK → joint interpolation (faster) | Only endpoint matters, not path |
| `movec(via, target)` | Circular arc through 3 points | Curved paths (welding, polishing) |
| `servoj(joints)` | Single-frame joint command | Real-time control loop (100Hz) |
| `servop(pos, rot)` | Single-frame Cartesian command | Real-time Cartesian tracking |
| `speedl(twist)` | Velocity control (m/s, rad/s) | Continuous motion / teleoperation |

### Coordinate Frames

```
World Frame (optional, enable with world_frame=True)
  X = forward (away from operator)
  Y = left
  Z = up

Base Frame (default)
  The arm's native frame (depends on mounting orientation)

Flange Frame
  The end-effector mounting face

TCP Frame (optional, set with set_tcp())
  Tool tip = flange + offset
```

### Mounting

The current dual-arm A1 installation uses vertical shoulder mounts: URDF base +X points up on both arms and base +Z points outward. Other fixtures may still use the side-mount presets. Setting the correct mounting ensures:
- Correct gravity compensation in freedrive/force mode
- Intuitive world-frame coordinates

```python
left = Arm("A1L", channel="can0", mounting="humanoid_left")
right = Arm("A1R", channel="can1", mounting="humanoid_right")

# Available presets:
# "upright"          — base Z points up (raw default)
# "side_mount_left"  — base Z points left
# "side_mount_right" — base Z points right
# "humanoid_left"    — canonical vertical left shoulder
# "humanoid_right"   — canonical vertical right shoulder
# "ceiling"          — mounted upside-down
```

### World Frame Mode

```python
arm = Arm("A1R", channel="can0",
          mounting="humanoid_right",
          world_frame=True)

# Now all Cartesian APIs use intuitive world coordinates
pos, rot = arm.get_pose()       # world frame
arm.movel([0.3, 0, 0.2, ...])  # "forward 0.3m, up 0.2m"
arm.speedl([0.05, 0, 0, ...])  # "move forward at 0.05 m/s"
```

### TCP (Tool Center Point)

```python
arm.set_tcp([0, 0, 0.15])       # gripper tip is 15cm from flange
pos, rot = arm.get_pose()        # now returns gripper tip position
arm.movel(target)                # target refers to gripper tip
arm.clear_tcp()                  # revert to flange
```

### Freedrive (Teach/Manual Mode)

```python
arm.freedrive_enable(damping=0.5)
# User can physically move the arm — gravity is auto-compensated
# Record the motion:
arm.start_recording()
# ... move arm by hand ...
recording = arm.stop_recording()
arm.freedrive_disable()

# Replay the recorded trajectory
arm.replay(recording, speed_factor=0.8)
```

### Force Control

```python
# Hybrid force/position: force in Z, position hold in X/Y
arm.force_mode(
    selection=[0, 0, 1, 0, 0, 0],   # 1 = force-controlled
    wrench=[0, 0, -5.0, 0, 0, 0],   # 5N downward
    limits=[0, 0, 0.1, 0, 0, 0],    # max speed 0.1 m/s
)
```

## Safety

The SDK includes built-in safety:
- **Joint limits**: auto-clamped to model limits
- **Velocity limits**: commands exceeding limits are scaled down
- **Emergency stop**: `arm.emergency_stop()` immediately halts motion
- **Auto-disable on exit**: if you forget `arm.disable()`, it's called automatically at interpreter exit

Recommended pattern:
```python
with Arm("A1R", channel="can0") as arm:
    arm.movej([...])
    # arm.disable() called automatically, even on exceptions
```

## API Quick Reference

```python
# Lifecycle
arm.enable()                    # Power on motors
arm.disable()                   # Power off
arm.home(duration=2.0)          # Return to zero position

# State
arm.state                       # ArmState(joints, velocities, torques)
arm.joints                      # Current joint positions [rad]
arm.get_pose()                  # (position[3], rotation[3x3])
arm.is_moving                   # True if any joint velocity > 0.01

# Motion
arm.movej(joints, duration)     # Joint-space
arm.movel(pos, rot, duration)   # Cartesian straight-line
arm.movep(pos, rot, duration)   # Cartesian point-to-point
arm.movec(via, target, dur)     # Circular arc
arm.servoj(joints, kp, kd)     # Real-time joint (one frame)
arm.servop(pos, rot, kp, kd)   # Real-time Cartesian (one frame)
arm.speedl(twist, duration)     # Velocity control

# Teach/Record
arm.freedrive_enable(damping)
arm.freedrive_disable()
arm.start_recording()
recording = arm.stop_recording()
arm.replay(recording)

# Force control
arm.force_mode(selection, wrench, task_frame, limits)
arm.force_mode_disable()
arm.get_external_wrench()

# Configuration
arm.set_tcp(position, rotation)
arm.clear_tcp()
arm.set_world_frame(True/False)
arm.set_stiffness("soft"/"medium"/"stiff")
```

## FAQ

**Q: Freedrive doesn't hold position — arm sags?**
Set the correct `mounting` parameter. Wrong mounting = wrong gravity direction = bad compensation.

**Q: Motion direction opposite to expected?**
Enable `world_frame=True`. Without it, directions depend on the base frame orientation.

**Q: Do I need pinocchio?**
- `movej`/`movel`/`get_pose`: No (pure kinematics)
- `freedrive`/`force_mode`/`calibrate`: Yes (`pip install onero-sdk[dynamics]`)

**Q: Can I test without hardware?**
Yes. Use `MujocoBackend` — identical API, zero hardware needed.
