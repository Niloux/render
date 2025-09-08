import numpy as np

CENTER = [492.07811834, -147.71372052, -32.64144724]

# 相机到车辆坐标系的变换矩阵 (T_camera_to_vehicle)
ext = np.array([
    [-9.703265255827278613e-03, -1.072251344945212778e-02, 9.998954317070867237e-01, 1.538897001763444461e00],
    [-9.999406983533636328e-01, -4.840233586415512712e-03, -9.755609433373464007e-03, -2.432485553238794215e-02],
    [4.944332104809027843e-03, -9.999307975275865124e-01, -1.067491151635933944e-02, 2.115484641063037685e00],
    [0.0, 0.0, 0.0, 1.0],
])


def create_pose_matrix(heading_rad, world_x, world_y, world_z):
    """
    根据航向角和世界坐标创建车辆到世界坐标系的变换矩阵

    Args:
        heading_rad (float): 航向角，单位为弧度
        world_x (float): 世界坐标系中的X坐标
        world_y (float): 世界坐标系中的Y坐标
        world_z (float): 世界坐标系中的Z坐标

    Returns:
        np.ndarray: 4x4的车辆到世界坐标系变换矩阵
    """
    # 计算旋转矩阵（绕Z轴旋转）
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)

    # 构建4x4变换矩阵
    T_vehicle_to_world = np.array([
        [cos_h, -sin_h, 0.0, world_x - CENTER[0]],
        [sin_h, cos_h, 0.0, world_y - CENTER[1]],
        [0.0, 0.0, 1.0, world_z - CENTER[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return T_vehicle_to_world


# 测试函数
if __name__ == "__main__":
    ego_pose = create_pose_matrix(1.728, 498.28, -186.11, -31.95)
    print(f"ego pose = {ego_pose}")
    c2w = ego_pose @ ext
    w2c = np.linalg.inv(c2w)
    print(f"c2w = {c2w}")
    print(f"T_w2c = {w2c}")
    print(w2c.tolist())
