"""
相机坐标系与手眼标定 — 腕部相机的坐标变换。

当机械臂腕部安装了相机时，需要知道相机坐标系和机械臂坐标系的关系，
才能把相机看到的目标位置转换到机械臂可以到达的位置。

坐标系关系:
  base (机械臂基座)
    └── flange (法兰面, FK 输出)
         ├── tcp (工具/夹爪, set_tcp 设定)
         └── camera (相机, 手眼标定求解)

变换链:
  T_base_camera = T_base_flange × T_flange_camera
                  (FK, 已知)       (手眼标定求解)

  目标在基座下的位置:
  P_base = T_base_camera × P_camera
         = T_base_flange × T_flange_camera × P_camera

手眼标定:
  通过在不同位姿下拍摄已知标定板, 同时记录机械臂位姿,
  联立方程求解 T_flange_camera (eye-in-hand) 或
  T_base_camera (eye-to-hand, 固定相机场景)。

运行:
    pip install opencv-python numpy
    python examples/14_camera_calibration.py
"""

import math
import time
import numpy as np
from onero_sdk import Arm


# ============================================================
# 坐标变换工具函数
# ============================================================

def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """[x, y, z, rx, ry, rz] → 4x4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    rvec = np.array(pose[3:6])
    angle = np.linalg.norm(rvec)
    if angle > 1e-10:
        axis = rvec / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        T[:3, :3] = R
    return T


def matrix_to_pose(T: np.ndarray) -> list[float]:
    """4x4 齐次变换矩阵 → [x, y, z, rx, ry, rz]。"""
    pos = T[:3, 3].tolist()
    R = T[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if angle < 1e-10:
        rvec = [0.0, 0.0, 0.0]
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2 * math.sin(angle))
        rvec = (axis * angle).tolist()
    return pos + rvec


def invert_transform(T: np.ndarray) -> np.ndarray:
    """求逆变换矩阵。"""
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


# ============================================================
# 手眼标定
# ============================================================

def hand_eye_calibration(T_base_flange_list: list[np.ndarray],
                         T_target_camera_list: list[np.ndarray],
                         method: str = "tsai") -> np.ndarray:
    """
    Eye-in-hand 手眼标定。

    通过多组 (机械臂位姿, 相机观测) 数据, 求解 T_flange_camera。

    方程: A × X = X × B
      A = T_flange_i⁻¹ × T_flange_j   (两次机械臂运动)
      B = T_camera_i × T_camera_j⁻¹    (两次相机观测变化)
      X = T_flange_camera               (待求)

    Args:
        T_base_flange_list: 每次采集时的机械臂法兰位姿 [4x4] × N
        T_target_camera_list: 每次采集时标定板在相机坐标系的位姿 [4x4] × N
        method: 求解方法 ("tsai", "park", "horaud")

    Returns:
        T_flange_camera: 4x4 相机相对于法兰的变换
    """
    try:
        import cv2
        # 使用 OpenCV 的手眼标定
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_bf in T_base_flange_list:
            R_gripper2base.append(T_bf[:3, :3])
            t_gripper2base.append(T_bf[:3, 3].reshape(3, 1))

        for T_tc in T_target_camera_list:
            R_target2cam.append(T_tc[:3, :3])
            t_target2cam.append(T_tc[:3, 3].reshape(3, 1))

        methods = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        }
        cv_method = methods.get(method, cv2.CALIB_HAND_EYE_TSAI)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv_method,
        )

        T_flange_camera = np.eye(4)
        T_flange_camera[:3, :3] = R_cam2gripper
        T_flange_camera[:3, 3] = t_cam2gripper.flatten()
        return T_flange_camera

    except ImportError:
        print("需要 OpenCV: pip install opencv-python")
        raise


# ============================================================
# 标定流程
# ============================================================

def detect_charuco(image, board, camera_matrix, dist_coeffs):
    """检测 ChArUco 标定板并返回位姿 (需要 OpenCV)。"""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
        gray, charucoCorners=None, charucoIds=None)

    if charuco_ids is None or len(charuco_ids) < 6:
        return None

    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None)

    if not success:
        return None

    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


# ============================================================
# 主程序
# ============================================================

arm = Arm("A1", channel="can0")
arm.enable()

try:
    print("=" * 55)
    print(" 手眼标定 (Eye-in-Hand)")
    print("=" * 55)
    print()
    print("场景: 相机安装在机械臂腕部(法兰面附近)")
    print("目的: 求解相机相对于法兰面的变换 T_flange_camera")
    print()

    # === 1. 坐标系关系说明 ===
    print("1. 坐标系关系:")
    print()
    print("   base (世界/基座)")
    print("     │")
    print("     ├── T_base_flange (FK 计算, 已知)")
    print("     │       │")
    print("     │       ├── T_flange_tcp (set_tcp 设置, 已知)")
    print("     │       │       └── [夹爪坐标系]")
    print("     │       │")
    print("     │       └── T_flange_camera (手眼标定求解)")
    print("     │               └── [相机坐标系]")
    print("     │")
    print("     └── T_base_target (标定板在世界中的位置, 固定)")
    print()

    # === 2. 相机内参 (假设已标定) ===
    print("2. 相机内参 (示例, 实际需用棋盘格标定):")
    camera_matrix = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    dist_coeffs = np.zeros(5)
    print(f"   fx=fy=600, cx=320, cy=240")
    print()

    # === 3. 采集标定数据 ===
    print("3. 采集标定数据 (示例流程):")
    print("   将标定板放在机械臂可达范围内, 保持不动。")
    print("   让机械臂运动到 10-20 个不同位姿, 每个位姿:")
    print("     (a) 记录法兰位姿 T_base_flange")
    print("     (b) 拍照并检测标定板位姿 T_target_camera")
    print()

    # 模拟采集过程
    calibration_poses = [
        [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0],
        [0.2, -0.3, 0.2, -0.8, 0.3, 0.6, 0.1],
        [-0.2, -0.4, 0.1, -1.2, -0.2, 0.7, -0.1],
        [0.1, -0.6, 0.3, -0.9, 0.1, 0.4, 0.2],
        [-0.1, -0.3, 0.4, -1.1, 0.2, 0.8, 0.0],
        [0.3, -0.5, 0.2, -0.7, -0.1, 0.5, -0.2],
        [0.0, -0.4, 0.1, -1.3, 0.3, 0.6, 0.1],
        [-0.3, -0.5, 0.3, -1.0, -0.3, 0.7, 0.0],
        [0.15, -0.6, 0.2, -0.8, 0.0, 0.4, 0.15],
        [-0.15, -0.3, 0.35, -1.1, 0.15, 0.9, -0.1],
    ]

    T_base_flange_data = []
    T_target_camera_data = []

    print(f"   模拟采集 {len(calibration_poses)} 个位姿...")
    for i, joints in enumerate(calibration_poses):
        arm.movej(joints, duration=1.5)

        # 记录法兰位姿 (从 FK 获取)
        pos, rot = arm.get_pose()  # 返回 (position[3], rotation[3x3])
        T_bf = np.eye(4)
        T_bf[:3, :3] = rot
        T_bf[:3, 3] = pos
        T_base_flange_data.append(T_bf)

        # 实际项目中: 拍照 → detect_charuco → T_target_camera
        # 这里用模拟数据
        # T_target_camera = T_camera_flange⁻¹ × T_flange_base⁻¹ × T_base_target
        # 模拟一个 T_flange_camera (这是我们要求解的真值)
        T_flange_camera_gt = pose_to_matrix([0.04, 0.0, 0.06, 0.0, -0.2, 0.0])
        T_base_target = pose_to_matrix([0.4, 0.0, 0.0, 0.0, math.pi, 0.0])
        T_camera = invert_transform(T_flange_camera_gt) @ invert_transform(T_bf) @ T_base_target
        # 加一点噪声模拟真实检测
        noise = np.random.randn(3) * 0.001
        T_camera[:3, 3] += noise
        T_target_camera_data.append(T_camera)

        if (i + 1) % 5 == 0:
            print(f"     已采集 {i+1}/{len(calibration_poses)}")

    # === 4. 求解手眼标定 ===
    print("\n4. 求解 T_flange_camera (手眼标定):")
    try:
        T_result = hand_eye_calibration(
            T_base_flange_data,
            T_target_camera_data,
            method="tsai",
        )
        result_pose = matrix_to_pose(T_result)
        print(f"   结果: {[f'{v:.4f}' for v in result_pose]}")
        print(f"   位移: x={result_pose[0]*100:.1f}cm, "
              f"y={result_pose[1]*100:.1f}cm, z={result_pose[2]*100:.1f}cm")
        print(f"   旋转: rx={math.degrees(result_pose[3]):.1f}°, "
              f"ry={math.degrees(result_pose[4]):.1f}°, "
              f"rz={math.degrees(result_pose[5]):.1f}°")
        print()
        print(f"   (真值: [0.04, 0.0, 0.06, 0.0, -0.2, 0.0])")
    except ImportError:
        print("   跳过 (需要 opencv-python)")
        T_result = pose_to_matrix([0.04, 0.0, 0.06, 0.0, -0.2, 0.0])
        result_pose = [0.04, 0.0, 0.06, 0.0, -0.2, 0.0]

    # === 5. 使用标定结果: 相机看到目标 → 机械臂去抓 ===
    print("\n5. 应用: 相机看到物体 → 机械臂去抓取")
    print()

    # 假设相机检测到物体在相机坐标系下的位置
    object_in_camera = np.array([0.05, -0.02, 0.30, 1.0])  # [x, y, z, 1]
    print(f"   物体在相机坐标系: [{object_in_camera[0]:.3f}, "
          f"{object_in_camera[1]:.3f}, {object_in_camera[2]:.3f}] m")

    # 变换到基座坐标系
    pos_now, rot_now = arm.get_pose()
    T_base_flange_now = np.eye(4)
    T_base_flange_now[:3, :3] = rot_now
    T_base_flange_now[:3, 3] = pos_now
    T_flange_camera = T_result
    T_base_camera = T_base_flange_now @ T_flange_camera

    object_in_base = T_base_camera @ object_in_camera
    print(f"   物体在基座坐标系: [{object_in_base[0]:.3f}, "
          f"{object_in_base[1]:.3f}, {object_in_base[2]:.3f}] m")
    print()
    print("   然后用 movel 运动到物体位置:")
    print(f"   arm.movel(position=[{object_in_base[0]:.3f}, "
          f"{object_in_base[1]:.3f}, {object_in_base[2]:.3f}])")

    # === 6. 坐标系之间的转换总结 ===
    print("\n" + "=" * 55)
    print(" 坐标变换速查")
    print("=" * 55)
    print()
    print("  物体在相机坐标系 → 物体在基座坐标系:")
    print("    P_base = T_base_flange × T_flange_camera × P_camera")
    print()
    print("  物体在相机坐标系 → 物体在夹爪坐标系:")
    print("    P_tcp = T_tcp_flange × T_flange_camera × P_camera")
    print("    其中 T_tcp_flange = T_flange_tcp⁻¹")
    print()
    print("  物体在基座坐标系 → 物体在相机坐标系:")
    print("    P_camera = T_camera_flange × T_flange_base × P_base")
    print("    其中 T_camera_flange = T_flange_camera⁻¹")
    print("          T_flange_base  = T_base_flange⁻¹")
    print()
    print("  保存标定结果 (重启后加载):")
    print("    np.save('T_flange_camera.npy', T_flange_camera)")
    print("    T_flange_camera = np.load('T_flange_camera.npy')")

finally:
    arm.disable()
