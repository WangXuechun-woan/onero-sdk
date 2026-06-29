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
                   task_frame_rotation=None, tcp_offset_base=None,
                   limits=None, kp=50.0, kd=1.0):
        """
        Compute one force-mode frame: joint kps, kds, and force torques.

        Args:
            q: Joint positions.
            v_joints: Joint velocities.
            selection: 6-element binary [fx,fy,fz,tx,ty,tz].
            wrench: Desired wrench [fx,fy,fz,tx,ty,tz] (N, Nm).
            jacobian: 6×n Jacobian matrix at q (at flange frame origin).
            task_frame_rotation: 3×3 rotation R_base_frame (frame→base), or None.
            tcp_offset_base: Vector from flange to TCP in base frame (3,), or None.
            limits: Speed limits [vx,vy,vz,wx,wy,wz] in task frame, or None.
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
            # Rotate wrench to base frame (apply selection first in task frame)
            wrench_task = selection_arr * wrench_arr
            wrench_base = np.empty(6)
            wrench_base[:3] = R_task @ wrench_task[:3]
            wrench_base[3:] = R_task @ wrench_task[3:]
            # Proper subspace projection for selection:
            # P = R @ diag(sel) @ R^T maps task-frame selection to base-frame
            # projection matrix. Per-axis selection in base = diagonal of P.
            S_lin = np.diag(selection_arr[:3])
            S_ang = np.diag(selection_arr[3:])
            P_lin = R_task @ S_lin @ R_task.T
            P_ang = R_task @ S_ang @ R_task.T
            # Use diagonal of projection matrix as per-axis force weight
            sel_lin = np.clip(np.diag(P_lin), 0.0, 1.0)
            sel_ang = np.clip(np.diag(P_ang), 0.0, 1.0)
            selection_arr = np.concatenate([sel_lin, sel_ang])
        else:
            wrench_base = selection_arr * wrench_arr

        # TCP moment arm: wrench is at TCP, Jacobian is at flange.
        # tau_flange = tau_tcp + r_tcp × f  (both already in base frame)
        if tcp_offset_base is not None:
            wrench_base[3:] += np.cross(tcp_offset_base, wrench_base[:3])

        force_torques = J.T @ wrench_base

        if limits is not None:
            limits_arr = np.asarray(limits, dtype=float)
            v_cart = J @ np.asarray(v_joints, dtype=float)
            # Compare velocity against limits in the task frame
            if task_frame_rotation is not None:
                R_task = task_frame_rotation
                R_task_inv = R_task.T  # R_frame_base
                v_task = np.empty(6)
                v_task[:3] = R_task_inv @ v_cart[:3]
                v_task[3:] = R_task_inv @ v_cart[3:]
            else:
                v_task = v_cart
            # selection in task frame (original user selection)
            sel_task = np.asarray(selection, dtype=float)
            scale = 1.0
            for i in range(6):
                if sel_task[i] > 0 and limits_arr[i] > 0:
                    ratio = abs(v_task[i]) / limits_arr[i]
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
