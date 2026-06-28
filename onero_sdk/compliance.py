"""
Compliance controller — force/position hybrid control and freedrive math.

Encapsulates the Jacobian-based computations for force mode (wrench projection,
speed limiting, per-joint gain modulation) and freedrive (gravity compensation
with optional axis locking).
"""

import numpy as np


class ComplianceController:
    """
    Stateless compliance computation for force mode and freedrive.

    Requires a dynamics model providing jacobian(), gravity_torques(), and ee_pose().
    """

    def __init__(self, dynamics, num_joints: int):
        self._dynamics = dynamics
        self._n = num_joints

    def force_step(self, q, v_joints, selection, wrench, jacobian,
                   task_frame_rotation=None, limits=None,
                   kp=50.0, kd=1.0):
        """
        Compute one force-mode frame: joint kps, kds, and force torques.

        Args:
            q: Joint positions.
            v_joints: Joint velocities.
            selection: 6-element binary [fx,fy,fz,tx,ty,tz].
            wrench: Desired wrench [fx,fy,fz,tx,ty,tz] (N, Nm).
            jacobian: 6×n Jacobian matrix at q.
            task_frame_rotation: 3×3 rotation matrix for task frame, or None.
            limits: Speed limits [vx,vy,vz,wx,wy,wz] or None.
            kp: Base position gain.
            kd: Damping gain.

        Returns:
            (kps, kds, force_torques) — per-joint gains and feedforward torques.
        """
        J = jacobian
        wrench_arr = np.asarray(wrench, dtype=float)
        selection_arr = np.asarray(selection, dtype=float)
        n = self._n

        if task_frame_rotation is not None:
            R_task = task_frame_rotation
            wrench_base = selection_arr * wrench_arr
            wrench_base[:3] = R_task @ wrench_base[:3]
            wrench_base[3:] = R_task @ wrench_base[3:]
            sel_lin_task = selection_arr[:3]
            sel_ang_task = selection_arr[3:]
            R_abs = np.abs(R_task)
            sel_lin = np.max(R_abs * sel_lin_task[np.newaxis, :], axis=1)
            sel_ang = np.max(R_abs * sel_ang_task[np.newaxis, :], axis=1)
            selection_arr = np.clip(np.concatenate([sel_lin, sel_ang]), 0.0, 1.0)
            force_torques = J.T @ wrench_base
        else:
            force_torques = J.T @ (selection_arr * wrench_arr)

        if limits is not None:
            limits_arr = np.asarray(limits, dtype=float)
            v_cart = J @ np.asarray(v_joints, dtype=float)
            scale = 1.0
            for i in range(6):
                if selection_arr[i] > 0 and limits_arr[i] > 0:
                    ratio = abs(v_cart[i]) / limits_arr[i]
                    if ratio > 1.0:
                        scale = 0.0
                        break
                    elif ratio > 0.8:
                        s = 1.0 - (ratio - 0.8) / 0.2
                        scale = min(scale, s)
            force_torques *= scale

        J_norm = np.abs(J)
        col_norms = J_norm.sum(axis=0)
        col_norms[col_norms < 1e-10] = 1.0
        J_contrib = J_norm / col_norms
        force_weight = J_contrib.T @ selection_arr
        force_weight = np.clip(force_weight, 0.0, 1.0)
        kps = [(kp * (1.0 - fw)) for fw in force_weight]
        kds = [kd] * n

        return kps, kds, force_torques

    def freedrive_lock_torques(self, q, lock_mask, pos_lock, rot_lock,
                               jacobian, lock_stiffness=200.0):
        """
        Compute constraint torques for axis-locked freedrive.

        Args:
            q: Current joint positions.
            lock_mask: 6-element array, 1=locked, 0=free.
            pos_lock: 3D position to hold.
            rot_lock: 3×3 rotation to hold.
            jacobian: 6×n Jacobian at q.
            lock_stiffness: Position gain for locked axes.

        Returns:
            Lock constraint torques (n-element array).
        """
        pos_curr, rot_curr = self._dynamics.ee_pose(q)
        x_err = np.zeros(6)
        x_err[:3] = pos_curr - pos_lock

        R_err = rot_curr @ rot_lock.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        if angle > 1e-6:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]) / (2.0 * np.sin(angle))
            x_err[3:] = angle * axis

        F_lock = -lock_stiffness * lock_mask * x_err
        return jacobian.T @ F_lock
