"""从points3D_lidar.ply中读取点云，计算包围球的中心和半径

后续可以将这两个值嵌入pth权重文件中，避免额外的文件解析开销。
"""

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


def calculate_bounding_sphere(ply_path: Path, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """计算点云的包围球参数

    Args:
        ply_path: PLY文件路径
        scale: 半径缩放因子

    Returns:
        tuple: (center, radius) 包围球中心和半径的GPU张量
    """
    data = PlyData.read(ply_path)
    vertices = data["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T

    xyz_max = np.max(positions, axis=0)
    xyz_min = np.min(positions, axis=0)
    center = (xyz_max + xyz_min) / 2
    radius = (np.linalg.norm(xyz_max - xyz_min) / 2) * scale

    # 转换为GPU张量
    center_tensor = torch.from_numpy(center).float().cuda()
    radius_tensor = torch.tensor(radius).float().cuda()  # 修复：使用torch.tensor()处理标量

    return center_tensor, radius_tensor


# if __name__ == "__main__":
#     lidar_pc_path = Path("points3D_lidar.ply")
#     CENTER, RADIUS = calculate_bounding_sphere(lidar_pc_path)

lidar_pc_path = Path("points3D_lidar.ply")
CENTER, RADIUS = calculate_bounding_sphere(lidar_pc_path)
