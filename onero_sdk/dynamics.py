"""
Dynamics model — gravity compensation, Jacobian, collision detection, and external force estimation.

Uses pinocchio's Recursive Newton-Euler Algorithm (RNEA) to compute
dynamics quantities from URDF. Required for freedrive, force_mode,
collision detection, and external wrench estimation.

Install: pip install onero-sdk[dynamics]
"""

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None


class DynamicsModel:
    """
    Compute dynamics quantities from URDF using pinocchio.

    Provides:
    - gravity_torques(q): gravity compensation vector
    - jacobian(q): 6×n end-effector Jacobian
    - inverse_dynamics(q, v, a): full RNEA
    - expected_torques(q, v): model-predicted torques (gravity + Coriolis)
    - estimate_external_wrench(q, v, tau_measured): external force estimation
    - set_payload(mass, cog): configure end-effector payload
    """

    def __init__(self, urdf_path: str):
        """
        Load dynamics model from URDF.

        Args:
            urdf_path: Path to URDF file.

        Raises:
            ImportError: if pinocchio is not installed.
        """
        if pin is None:
            raise ImportError(
                "DynamicsModel requires pinocchio. Install with: pip install onero-sdk[dynamics]"
            )
        self._urdf_path = urdf_path
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._nq = self._model.nq
        self._ee_frame_id = self._model.nframes - 1
        self._ee_joint_id = self._model.frames[self._ee_frame_id].parent
        # Store original last-link inertia for payload reset
        self._original_ee_inertia = self._model.inertias[self._ee_joint_id].copy()

    @property
    def num_joints(self) -> int:
        return self._nq

    def set_payload(self, mass: float, cog: list[float]) -> None:
        """
        Configure end-effector payload for accurate dynamics.

        Affects gravity compensation, collision detection, and force estimation.
        Call this whenever the tool/payload changes. Replaces any previous payload.

        Args:
            mass: Payload mass (kg).
            cog: Center of gravity relative to EE frame [x, y, z] (meters).
        """
        cog_arr = np.asarray(cog, dtype=float)
        # Reset to original inertia first (replaces, not accumulates)
        self._model.inertias[self._ee_joint_id] = self._original_ee_inertia.copy()
        # Add payload inertia
        payload_inertia = pin.Inertia(
            mass,
            cog_arr,
            np.zeros((3, 3))  # point mass approximation
        )
        self._model.inertias[self._ee_joint_id] += payload_inertia
        # Recreate data after model change
        self._data = self._model.createData()

    def clear_payload(self) -> None:
        """Remove payload configuration, restore original dynamics."""
        self._model.inertias[self._ee_joint_id] = self._original_ee_inertia.copy()
        self._data = self._model.createData()

    def gravity_torques(self, joint_positions: list[float]) -> list[float]:
        """
        Compute gravity compensation torques.

        If calibration has been applied, uses regressor-based prediction
        (τ = Y·π) for higher accuracy. Otherwise uses RNEA with URDF parameters.

        Args:
            joint_positions: Current joint angles (rad).

        Returns:
            Gravity torques per joint (Nm).
        """
        q = np.asarray(joint_positions, dtype=float)
        if hasattr(self, '_calibrated_params') and self._calibrated_params is not None:
            Y = pin.computeJointTorqueRegressor(
                self._model, self._data, q,
                np.zeros(self._nq), np.zeros(self._nq)
            )
            return (Y @ self._calibrated_params).tolist()
        v = np.zeros(self._nq)
        a = np.zeros(self._nq)
        tau = pin.rnea(self._model, self._data, q, v, a)
        return tau.tolist()

    def expected_torques(self, joint_positions: list[float],
                         joint_velocities: list[float]) -> list[float]:
        """
        Compute model-predicted torques for current state (gravity + Coriolis + friction).

        Used for collision detection: residual = measured - expected.

        Args:
            joint_positions: Current joint angles (rad).
            joint_velocities: Current joint velocities (rad/s).

        Returns:
            Expected torques per joint (Nm) assuming zero acceleration.
        """
        q = np.asarray(joint_positions, dtype=float)
        v = np.asarray(joint_velocities, dtype=float)
        a = np.zeros(self._nq)  # assume zero acceleration (steady state)
        tau = pin.rnea(self._model, self._data, q, v, a)
        return tau.tolist()

    def torque_residual(self, joint_positions: list[float],
                        joint_velocities: list[float],
                        measured_torques: list[float]) -> list[float]:
        """
        Compute torque residual (measured - expected).

        Positive residual means external force is acting on the arm.
        Used for collision detection.

        Args:
            joint_positions: Current joint angles (rad).
            joint_velocities: Current joint velocities (rad/s).
            measured_torques: Measured motor torques (Nm).

        Returns:
            Residual torque per joint (Nm). Near-zero = no external contact.
        """
        expected = self.expected_torques(joint_positions, joint_velocities)
        measured = np.asarray(measured_torques, dtype=float)
        return (measured - np.asarray(expected)).tolist()

    def estimate_external_wrench(self, joint_positions: list[float],
                                 joint_velocities: list[float],
                                 measured_torques: list[float]) -> np.ndarray:
        """
        Estimate external wrench (force/torque) at the end-effector.

        Uses: tau_ext = tau_measured - tau_model, then F_ext = (J J^T + λ²I)^{-1} J @ tau_ext
        with adaptive DLS damping to avoid blow-up near singularities.

        Args:
            joint_positions: Current joint angles (rad).
            joint_velocities: Current joint velocities (rad/s).
            measured_torques: Measured motor torques (Nm).

        Returns:
            Estimated external wrench [fx, fy, fz, tx, ty, tz] (N, Nm).
        """
        tau_residual = np.asarray(
            self.torque_residual(joint_positions, joint_velocities, measured_torques)
        )
        J = self.jacobian(joint_positions)
        sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
        lambda_max = 0.05
        sigma_threshold = 0.02
        if sigma_min < sigma_threshold:
            ratio = sigma_min / sigma_threshold
            lambda_sq = lambda_max ** 2 * (1.0 - ratio * ratio)
        else:
            lambda_sq = 0.0
        JJT = J @ J.T
        wrench = np.linalg.solve(JJT + lambda_sq * np.eye(6), J @ tau_residual)
        return wrench

    def jacobian(self, joint_positions: list[float]) -> np.ndarray:
        """
        Compute end-effector Jacobian (6×n) in world frame.

        Used by speedl (velocity mapping) and force_mode (wrench → joint torques).

        Args:
            joint_positions: Current joint angles (rad).

        Returns:
            6×n Jacobian matrix. Rows: [vx, vy, vz, wx, wy, wz].
        """
        q = np.asarray(joint_positions, dtype=float)
        pin.computeJointJacobians(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        J = pin.getFrameJacobian(
            self._model, self._data, self._ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J

    def jacobian_solve(self, joint_positions: list[float], twist: np.ndarray,
                       lambda_max: float = 0.1,
                       sigma_threshold: float = 0.02) -> np.ndarray:
        """
        DLS solve: dq = J^+ @ twist with adaptive singularity damping.

        Near singularities the minimum singular value drops, causing naive
        pseudoinverse to produce explosive joint velocities. This method
        applies Damped Least-Squares with adaptive λ that ramps smoothly
        from 0 (full precision) to lambda_max (full damping) as the arm
        approaches a singular configuration.

        Args:
            joint_positions: Current joint angles (rad).
            twist: Target EE velocity [vx, vy, vz, wx, wy, wz].
            lambda_max: Maximum damping coefficient.
            sigma_threshold: Singular values below this trigger damping.

        Returns:
            Joint velocities (n,) — always finite even at singularities.
        """
        J = self.jacobian(joint_positions)
        sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
        if sigma_min < sigma_threshold:
            ratio = sigma_min / sigma_threshold
            lambda_sq = lambda_max ** 2 * (1.0 - ratio * ratio)
        else:
            lambda_sq = 0.0
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lambda_sq * np.eye(6), twist)
        return dq

    def inverse_dynamics(self, joint_positions: list[float],
                         joint_velocities: list[float],
                         joint_accelerations: list[float]) -> list[float]:
        """
        Full inverse dynamics (RNEA): compute torques for desired motion.

        Args:
            joint_positions: Joint angles (rad).
            joint_velocities: Joint velocities (rad/s).
            joint_accelerations: Joint accelerations (rad/s^2).

        Returns:
            Required torques per joint (Nm).
        """
        q = np.asarray(joint_positions, dtype=float)
        v = np.asarray(joint_velocities, dtype=float)
        a = np.asarray(joint_accelerations, dtype=float)
        tau = pin.rnea(self._model, self._data, q, v, a)
        return tau.tolist()

    def mass_matrix(self, joint_positions: list[float]) -> np.ndarray:
        """
        Compute joint-space mass (inertia) matrix M(q).

        Args:
            joint_positions: Joint angles (rad).

        Returns:
            n×n symmetric positive-definite mass matrix.
        """
        q = np.asarray(joint_positions, dtype=float)
        M = pin.crba(self._model, self._data, q)
        return M

    def torque_regressor(self, joint_positions: list[float],
                         joint_velocities: list[float] = None,
                         joint_accelerations: list[float] = None) -> np.ndarray:
        """
        Compute the joint torque regressor matrix Y such that τ = Y @ π.

        π is the (10 * num_bodies)-dimensional dynamics parameter vector:
        [mass, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] per body.

        Args:
            joint_positions: Joint angles (rad).
            joint_velocities: Joint velocities (rad/s). Zeros if None.
            joint_accelerations: Joint accelerations (rad/s²). Zeros if None.

        Returns:
            (nq, 10*n_bodies) regressor matrix.
        """
        q = np.asarray(joint_positions, dtype=float)
        v = np.zeros(self._nq) if joint_velocities is None else np.asarray(joint_velocities, dtype=float)
        a = np.zeros(self._nq) if joint_accelerations is None else np.asarray(joint_accelerations, dtype=float)
        return pin.computeJointTorqueRegressor(self._model, self._data, q, v, a)

    def get_params(self) -> np.ndarray:
        """
        Export current model dynamics parameters as a flat vector.

        Returns:
            (10 * n_bodies,) vector: [mass, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
            per body (excluding universe).
        """
        params = []
        for i in range(1, self._model.njoints):
            inertia = self._model.inertias[i]
            m = inertia.mass
            c = inertia.lever
            I = inertia.inertia
            params.extend([
                m, m * c[0], m * c[1], m * c[2],
                I[0, 0], I[0, 1], I[1, 1], I[0, 2], I[1, 2], I[2, 2]
            ])
        return np.array(params)

    def apply_calibration(self, params: np.ndarray) -> None:
        """
        Apply identified parameters for gravity/dynamics prediction.

        Stores the parameter vector and uses the regressor (τ = Y·π) for
        subsequent gravity_torques() calls. This avoids writing non-physical
        parameters back to the model while giving exact predictions.

        Args:
            params: (10 * n_bodies,) parameter vector from calibration.
        """
        n_bodies = self._model.njoints - 1
        if len(params) != 10 * n_bodies:
            raise ValueError(
                f"Expected {10 * n_bodies} params, got {len(params)}"
            )
        self._calibrated_params = params.copy()
