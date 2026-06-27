"""
ROS2 Control Hardware Interface for Onero robot arms.

Bridges ros2_control to the Onero SDK's CanBackend, providing standard
position/velocity/effort command interfaces at 200Hz.

Plugin name: onero_ros2/OneroHardwareInterface
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import State

from hardware_interface import (
    ActuatorInterface,
    HardwareInfo,
    SystemInterface,
    return_type,
)
from hardware_interface.types import HardwareInterfaceReturn

from onero_sdk.backend import ArmBackend, ArmState
from onero_sdk.can_backend import CanBackend
from onero_sdk.model_registry import get_model


class OneroHardwareInterface(SystemInterface):
    """
    ros2_control SystemInterface for Onero arms.

    Connects to real hardware via CanBackend (SocketCAN) and exposes
    position, velocity, and effort command/state interfaces.

    URDF parameters:
        can_channel: SocketCAN interface name (default: "can0")
        model: Robot model name from registry (default: "A1")
        kp: Default position gain (default: 50.0)
        kd: Default damping gain (default: 1.0)
    """

    def __init__(self):
        super().__init__()
        self._backend: CanBackend = None
        self._model_name: str = "A1"
        self._can_channel: str = "can0"
        self._kp: float = 50.0
        self._kd: float = 1.0
        self._n_joints: int = 7

        # State arrays
        self._hw_positions = np.zeros(7)
        self._hw_velocities = np.zeros(7)
        self._hw_efforts = np.zeros(7)

        # Command arrays
        self._cmd_positions = np.zeros(7)
        self._cmd_velocities = np.zeros(7)
        self._cmd_efforts = np.zeros(7)

        # Track which command interface is claimed
        self._position_interface_active = False
        self._velocity_interface_active = False
        self._effort_interface_active = False

    def on_init(self, hardware_info: HardwareInfo) -> return_type.CallbackReturn:
        """Parse URDF parameters and validate joint configuration."""
        if super().on_init(hardware_info) != return_type.CallbackReturn.SUCCESS:
            return return_type.CallbackReturn.ERROR

        # Parse hardware parameters from URDF <ros2_control> block
        params = hardware_info.hardware_parameters
        self._can_channel = params.get("can_channel", "can0")
        self._model_name = params.get("model", "A1")
        self._kp = float(params.get("kp", "50.0"))
        self._kd = float(params.get("kd", "1.0"))

        # Validate joint count
        self._n_joints = len(hardware_info.joints)
        model_info = get_model(self._model_name)
        if self._n_joints != len(model_info.joints):
            rclpy.logging.get_logger("OneroHW").error(
                f"URDF has {self._n_joints} joints but model '{self._model_name}' "
                f"has {len(model_info.joints)} joints"
            )
            return return_type.CallbackReturn.ERROR

        # Validate interfaces
        for joint in hardware_info.joints:
            state_names = {si.name for si in joint.state_interfaces}
            cmd_names = {ci.name for ci in joint.command_interfaces}

            if not {"position", "velocity", "effort"}.issubset(state_names):
                rclpy.logging.get_logger("OneroHW").error(
                    f"Joint '{joint.name}' must have position/velocity/effort state interfaces"
                )
                return return_type.CallbackReturn.ERROR

            if not cmd_names.intersection({"position", "velocity", "effort"}):
                rclpy.logging.get_logger("OneroHW").error(
                    f"Joint '{joint.name}' must have at least one command interface"
                )
                return return_type.CallbackReturn.ERROR

        # Resize arrays
        self._hw_positions = np.zeros(self._n_joints)
        self._hw_velocities = np.zeros(self._n_joints)
        self._hw_efforts = np.zeros(self._n_joints)
        self._cmd_positions = np.zeros(self._n_joints)
        self._cmd_velocities = np.zeros(self._n_joints)
        self._cmd_efforts = np.zeros(self._n_joints)

        rclpy.logging.get_logger("OneroHW").info(
            f"Initialized: model={self._model_name}, channel={self._can_channel}, "
            f"joints={self._n_joints}, kp={self._kp}, kd={self._kd}"
        )
        return return_type.CallbackReturn.SUCCESS

    def on_configure(self, previous_state: State) -> return_type.CallbackReturn:
        """Create CanBackend connection (does not enable motors yet)."""
        try:
            model_info = get_model(self._model_name)
            self._backend = CanBackend(
                channel=self._can_channel,
                joint_specs=model_info.joints,
            )
            rclpy.logging.get_logger("OneroHW").info(
                f"Connected to CAN channel '{self._can_channel}'"
            )
            return return_type.CallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger("OneroHW").error(
                f"Failed to connect: {e}"
            )
            return return_type.CallbackReturn.ERROR

    def on_activate(self, previous_state: State) -> return_type.CallbackReturn:
        """Enable motors and read initial state."""
        try:
            state = self._backend.enable()
            self._hw_positions[:] = state.joints
            self._hw_velocities[:] = state.velocities
            self._hw_efforts[:] = state.torques
            # Initialize commands to current position (hold in place)
            self._cmd_positions[:] = state.joints
            self._cmd_velocities[:] = 0.0
            self._cmd_efforts[:] = 0.0

            rclpy.logging.get_logger("OneroHW").info("Motors enabled")
            return return_type.CallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger("OneroHW").error(
                f"Failed to enable motors: {e}"
            )
            return return_type.CallbackReturn.ERROR

    def on_deactivate(self, previous_state: State) -> return_type.CallbackReturn:
        """Disable motors."""
        try:
            self._backend.disable()
            rclpy.logging.get_logger("OneroHW").info("Motors disabled")
            return return_type.CallbackReturn.SUCCESS
        except Exception as e:
            rclpy.logging.get_logger("OneroHW").error(
                f"Failed to disable motors: {e}"
            )
            return return_type.CallbackReturn.ERROR

    def on_cleanup(self, previous_state: State) -> return_type.CallbackReturn:
        """Release CAN resources."""
        if self._backend:
            self._backend.close()
            self._backend = None
        return return_type.CallbackReturn.SUCCESS

    def on_shutdown(self, previous_state: State) -> return_type.CallbackReturn:
        """Emergency shutdown."""
        if self._backend:
            try:
                self._backend.disable()
            except Exception:
                pass
            self._backend.close()
            self._backend = None
        return return_type.CallbackReturn.SUCCESS

    # ─── State Interfaces ──────────────────────────────────────────────────

    def export_state_interfaces(self):
        """Export position, velocity, effort state for each joint."""
        interfaces = []
        for i, joint in enumerate(self.info_.joints):
            interfaces.append(
                hardware_interface.StateInterface(
                    joint.name, "position", self._hw_positions[i:i+1]
                )
            )
            interfaces.append(
                hardware_interface.StateInterface(
                    joint.name, "velocity", self._hw_velocities[i:i+1]
                )
            )
            interfaces.append(
                hardware_interface.StateInterface(
                    joint.name, "effort", self._hw_efforts[i:i+1]
                )
            )
        return interfaces

    # ─── Command Interfaces ────────────────────────────────────────────────

    def export_command_interfaces(self):
        """Export position, velocity, effort commands for each joint."""
        interfaces = []
        for i, joint in enumerate(self.info_.joints):
            interfaces.append(
                hardware_interface.CommandInterface(
                    joint.name, "position", self._cmd_positions[i:i+1]
                )
            )
            interfaces.append(
                hardware_interface.CommandInterface(
                    joint.name, "velocity", self._cmd_velocities[i:i+1]
                )
            )
            interfaces.append(
                hardware_interface.CommandInterface(
                    joint.name, "effort", self._cmd_efforts[i:i+1]
                )
            )
        return interfaces

    # ─── Control Loop ──────────────────────────────────────────────────────

    def read(self, time, period) -> return_type.HardwareInterfaceReturn:
        """Read current joint state from hardware (called at 200Hz)."""
        try:
            state = self._backend.get_state()
            self._hw_positions[:] = state.joints
            self._hw_velocities[:] = state.velocities
            self._hw_efforts[:] = state.torques
            return return_type.HardwareInterfaceReturn.OK
        except Exception:
            return return_type.HardwareInterfaceReturn.ERROR

    def write(self, time, period) -> return_type.HardwareInterfaceReturn:
        """Write commands to hardware (called at 200Hz).

        Command mode is determined by which interfaces are claimed by
        the active controller:
        - position: MIT impedance (pos + kp/kd gains)
        - velocity: direct velocity command
        - effort: direct torque command
        """
        try:
            if self._position_interface_active:
                n = self._n_joints
                self._backend.command(
                    positions=self._cmd_positions.tolist(),
                    velocities=self._cmd_velocities.tolist(),
                    kps=[self._kp] * n,
                    kds=[self._kd] * n,
                    torques=self._cmd_efforts.tolist(),
                )
            elif self._velocity_interface_active:
                self._backend.command_velocity(
                    velocities=self._cmd_velocities.tolist(),
                )
            elif self._effort_interface_active:
                self._backend.command_torque(
                    torques=self._cmd_efforts.tolist(),
                )
            else:
                # No controller active — hold position with impedance
                n = self._n_joints
                self._backend.command(
                    positions=self._hw_positions.tolist(),
                    velocities=[0.0] * n,
                    kps=[self._kp] * n,
                    kds=[self._kd] * n,
                    torques=[0.0] * n,
                )
            return return_type.HardwareInterfaceReturn.OK
        except Exception:
            return return_type.HardwareInterfaceReturn.ERROR

    def prepare_command_mode_switch(self, start_interfaces, stop_interfaces):
        """Track which command interfaces are being claimed."""
        return return_type.HardwareInterfaceReturn.OK

    def perform_command_mode_switch(self, start_interfaces, stop_interfaces):
        """Activate/deactivate command interfaces."""
        for iface in stop_interfaces:
            if "position" in iface:
                self._position_interface_active = False
            elif "velocity" in iface:
                self._velocity_interface_active = False
            elif "effort" in iface:
                self._effort_interface_active = False

        for iface in start_interfaces:
            if "position" in iface:
                self._position_interface_active = True
                # Initialize position command to current to avoid jump
                self._cmd_positions[:] = self._hw_positions
            elif "velocity" in iface:
                self._velocity_interface_active = True
                self._cmd_velocities[:] = 0.0
            elif "effort" in iface:
                self._effort_interface_active = True
                self._cmd_efforts[:] = 0.0

        return return_type.HardwareInterfaceReturn.OK
