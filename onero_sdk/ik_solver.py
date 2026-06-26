"""
Inverse kinematics solver using pinocchio + pink.

Supports both left (a1_l) and right (a1_r) arm URDF models.
Requires: pip install pin pink
"""

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

try:
    import pink
    from pink.tasks import FrameTask
except ImportError:
    pink = None


class IKSolver:
    def __init__(self, urdf_path: str, ee_frame: str = "Link7"):
        if pin is None:
            raise RuntimeError("pinocchio is required: pip install pin")
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._ee_frame = ee_frame
        self._ee_id = self._model.getFrameId(ee_frame)
        self._nq = self._model.nq
        self._q = pin.neutral(self._model)

        self._q_min = self._model.lowerPositionLimit.copy()
        self._q_max = self._model.upperPositionLimit.copy()

    @property
    def nq(self) -> int:
        return self._nq

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_min, self._q_max

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @q.setter
    def q(self, value: np.ndarray):
        self._q = np.array(value, dtype=float)

    def forward_kinematics(self, q: np.ndarray = None) -> np.ndarray:
        if q is None:
            q = self._q
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._ee_id]
        return oMf.homogeneous

    def get_ee_pose(self, q: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        T = self.forward_kinematics(q)
        return T[:3, 3], T[:3, :3]

    def jacobian(self, q: np.ndarray = None) -> np.ndarray:
        if q is None:
            q = self._q
        pin.computeJointJacobians(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return pin.getFrameJacobian(
            self._model, self._data, self._ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

    def solve_ik(self, target_pos: np.ndarray, target_rot: np.ndarray = None,
                 q_init: np.ndarray = None, max_iter: int = 200,
                 dt: float = 0.1, damp: float = 1e-6,
                 pos_tol: float = 1e-3, rot_tol: float = 1e-2) -> np.ndarray:
        """
        Damped least squares IK solver with joint limit clamping.
        7-DOF arm allows null-space optimization.
        """
        if q_init is not None:
            q = np.array(q_init, dtype=float)
        else:
            q = self._q.copy()

        target_se3 = pin.SE3(
            target_rot if target_rot is not None else np.eye(3),
            target_pos
        )

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.computeJointJacobians(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            oMf = self._data.oMf[self._ee_id]

            err = pin.log6(oMf.actInv(target_se3)).vector

            if target_rot is None:
                err[3:] = 0.0

            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and (target_rot is None or rot_err < rot_tol):
                break

            J = pin.getFrameJacobian(
                self._model, self._data, self._ee_id,
                pin.ReferenceFrame.LOCAL
            )

            if target_rot is None:
                J = J[:3, :]
                err = err[:3]

            JtJ = J.T @ J + damp * np.eye(self._nq)
            dq = np.linalg.solve(JtJ, J.T @ err)
            q = q + dt * dq
            q = np.clip(q, self._q_min, self._q_max)

        self._q = q
        return q

    def solve_ik_with_nullspace(self, target_pos: np.ndarray,
                                target_rot: np.ndarray = None,
                                q_init: np.ndarray = None,
                                q_ref: np.ndarray = None,
                                nullspace_gain: float = 0.5,
                                **kwargs) -> np.ndarray:
        """
        IK with null-space posture optimization (7-DOF redundancy).
        q_ref is the preferred joint configuration.
        """
        if q_init is not None:
            q = np.array(q_init, dtype=float)
        else:
            q = self._q.copy()

        if q_ref is None:
            q_ref = (self._q_min + self._q_max) / 2.0

        max_iter = kwargs.get('max_iter', 200)
        dt = kwargs.get('dt', 0.1)
        damp = kwargs.get('damp', 1e-6)
        pos_tol = kwargs.get('pos_tol', 1e-3)
        rot_tol = kwargs.get('rot_tol', 1e-2)

        target_se3 = pin.SE3(
            target_rot if target_rot is not None else np.eye(3),
            target_pos
        )

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.computeJointJacobians(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            oMf = self._data.oMf[self._ee_id]

            err = pin.log6(oMf.actInv(target_se3)).vector
            if target_rot is None:
                err[3:] = 0.0

            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and (target_rot is None or rot_err < rot_tol):
                break

            J = pin.getFrameJacobian(
                self._model, self._data, self._ee_id,
                pin.ReferenceFrame.LOCAL
            )

            if target_rot is None:
                J = J[:3, :]
                err = err[:3]

            JtJ = J.T @ J + damp * np.eye(self._nq)
            J_pinv = np.linalg.solve(JtJ, J.T)
            dq_task = J_pinv @ err

            # Null-space projection
            N = np.eye(self._nq) - J_pinv @ J
            dq_null = nullspace_gain * N @ (q_ref - q)

            dq = dq_task + dq_null
            q = q + dt * dq
            q = np.clip(q, self._q_min, self._q_max)

        self._q = q
        return q


class PinkIKSolver:
    """
    Higher-level IK using the pink library (task-based, QP solver).
    Better for real-time streaming with smooth trajectories.
    """

    def __init__(self, urdf_path: str, ee_frame: str = "Link7",
                 solver: str = "quadprog"):
        if pin is None or pink is None:
            raise RuntimeError("pinocchio and pink are required: pip install pin pin-pink")
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._ee_frame = ee_frame
        self._nq = self._model.nq
        self._solver = solver
        self._configuration = pink.Configuration(self._model, self._data,
                                                  pin.neutral(self._model))
        self._ee_task = FrameTask(ee_frame, position_cost=1.0, orientation_cost=1.0)

    @property
    def nq(self) -> int:
        return self._nq

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray = None,
              dt: float = 0.005) -> np.ndarray:
        target_se3 = pin.SE3(
            target_rot if target_rot is not None else np.eye(3),
            target_pos
        )
        self._ee_task.set_target(target_se3)
        velocity = pink.solve_ik(self._configuration, [self._ee_task], dt,
                                 solver=self._solver)
        self._configuration.integrate_inplace(velocity, dt)
        return self._configuration.q

    def reset(self, q: np.ndarray):
        self._configuration = pink.Configuration(self._model, self._data, q)
