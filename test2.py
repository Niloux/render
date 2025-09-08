import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_multiply(q1, q2):
    """q1 * q2，四元数乘法 (wxyz)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# 假设数据
N = 100  # 示例高斯数量
means_object = np.random.rand(N, 3)  # [N, 3]
quats_object = np.random.rand(N, 4)  # [N, 4]，假设已归一化
position_world = np.array([1, 1, 1])  # [3,]
heading = np.pi / 4  # 示例弧度，45度

# 步骤1: heading to quat (wxyz)
rot = R.from_euler("z", heading)  # 围绕Z轴
quat_heading = rot.as_quat()  # [x,y,z,w]，SciPy是xyzw，需要转换为wxyz
quat_heading = np.array([quat_heading[3], quat_heading[0], quat_heading[1], quat_heading[2]])  # wxyz

# 步骤2: 转换means
# 先转换为旋转矩阵
rot_matrix = rot.as_matrix()  # [3,3]
means_rotated = np.dot(means_object, rot_matrix.T)  # 或 means_object @ rot_matrix.T，根据约定
means_world = means_rotated + position_world  # [N,3]

# 步骤3: 转换quats
quats_world = np.zeros((N, 4))
for i in range(N):
    quats_world[i] = quaternion_multiply(quat_heading, quats_object[i])

# 如果quats需要归一化
quats_world /= np.linalg.norm(quats_world, axis=1, keepdims=True)  # 可选
