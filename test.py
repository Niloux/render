import numpy as np

# 直接定义为numpy数组，消除后续转换开销
CENTER = np.array([492.07811834, -147.71372052, -32.64144724])

# 相机到车辆坐标系的变换矩阵 (T_camera_to_vehicle)
ext = np.array([
    [-9.703265255827278613e-03, -1.072251344945212778e-02, 9.998954317070867237e-01, 1.538897001763444461e00],
    [-9.999406983533636328e-01, -4.840233586415512712e-03, -9.755609433373464007e-03, -2.432485553238794215e-02],
    [4.944332104809027843e-03, -9.999307975275865124e-01, -1.067491151635933944e-02, 2.115484641063037685e00],
    [0.0, 0.0, 0.0, 1.0],
])


def create_pose_matrix(heading_rad: float, world_position: np.ndarray) -> np.ndarray:
    """
    根据航向角和世界坐标创建车辆到世界坐标系的变换矩阵

    Args:
        heading_rad: 航向角，单位为弧度
        world_position: 世界坐标 [x, y, z] 的numpy数组

    Returns:
        4x4的车辆到世界坐标系变换矩阵
    """
    # 计算旋转矩阵（绕Z轴旋转）
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)

    # 减去场景center - 现在类型一致，无需转换
    position = world_position - CENTER

    # 构建4x4变换矩阵 - 直接构造，避免多次数组创建
    return np.array([
        [cos_h, -sin_h, 0.0, position[0]],
        [sin_h, cos_h, 0.0, position[1]],
        [0.0, 0.0, 1.0, position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])


if __name__ == "__main__":
    # 直接使用numpy数组，消除类型转换
    ego_position = np.array([498.28, -186.11, -31.95])
    ego_heading = 1.728

    ego_pose = create_pose_matrix(ego_heading, ego_position)
    print(f"ego pose = {ego_pose}")

    c2w = ego_pose @ ext
    w2c = np.linalg.inv(c2w)
    print(f"c2w = {c2w}")
    print(f"T_w2c = {w2c}")
    print(w2c.tolist())
