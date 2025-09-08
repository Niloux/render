"""高斯点云组件数据类。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lidar_prepare import CENTER, RADIUS


@dataclass
class GaussianComponent:
    """表示3DGS模型中的单个高斯点云组件。

    包含位置、特征、缩放、旋转、透明度和语义信息。
    """

    name: str
    xyz: torch.Tensor  # [N, 3] 位置坐标
    feature_dc: torch.Tensor  # [N, 1, 3] 直流特征
    feature_rest: torch.Tensor  # [N, 3, 3] 其余特征
    scaling: torch.Tensor  # [N, 3] 缩放参数
    rotation: torch.Tensor  # [N, 4] 旋转四元数
    opacity: torch.Tensor  # [N, 1] 透明度
    semantic: Optional[torch.Tensor] = None  # [N, K] 语义信息

    @property
    def num_points(self) -> int:
        """获取点云数量。"""
        return self.xyz.shape[0]

    @property
    def memory_usage(self) -> float:
        """获取内存使用量（MB）。"""
        total_bytes = sum([
            self.xyz.numel() * self.xyz.element_size(),
            self.feature_dc.numel() * self.feature_dc.element_size(),
            self.feature_rest.numel() * self.feature_rest.element_size(),
            self.scaling.numel() * self.scaling.element_size(),
            self.rotation.numel() * self.rotation.element_size(),
            self.opacity.numel() * self.opacity.element_size(),
        ])

        if self.semantic is not None:
            total_bytes += self.semantic.numel() * self.semantic.element_size()

        return total_bytes / (1024 * 1024)

    def validate(self) -> bool:
        """验证数据的一致性。"""
        n = self.num_points

        # 检查所有tensor的第一维是否一致
        tensors = [self.xyz, self.feature_dc, self.feature_rest, self.scaling, self.rotation, self.opacity]

        for tensor in tensors:
            if tensor.shape[0] != n:
                return False

        # 检查语义信息
        if self.semantic is not None and self.semantic.shape[0] != n:
            return False

        # 检查tensor维度
        if (
            self.xyz.shape[1] != 3
            or self.feature_dc.shape[1:] != (1, 3)
            or self.feature_rest.shape[1:] != (3, 3)
            or self.scaling.shape[1] != 3
            or self.rotation.shape[1] != 4
            or self.opacity.shape[1] != 1
        ):
            return False

        return True

    def to_device(self, device: torch.device) -> "GaussianComponent":
        """将组件移动到指定设备。"""
        return GaussianComponent(
            name=self.name,
            xyz=self.xyz.to(device),
            feature_dc=self.feature_dc.to(device),
            feature_rest=self.feature_rest.to(device),
            scaling=self.scaling.to(device),
            rotation=self.rotation.to(device),
            opacity=self.opacity.to(device),
            semantic=self.semantic.to(device) if self.semantic is not None else None,
        )

    def __str__(self) -> str:
        """字符串表示。"""
        semantic_info = f", semantic: {self.semantic.shape}" if self.semantic is not None else ""
        return (
            f"GaussianComponent(name='{self.name}', points={self.num_points}, "
            f"memory={self.memory_usage:.2f}MB{semantic_info})"
        )

    def get_xyz(self, heading: Optional[float] = None, position: Optional[np.ndarray] = None) -> torch.Tensor:  # [N, 3]
        if self.name == "background":
            return self.xyz
        elif self.name == "sky":
            dists = torch.linalg.norm(self.xyz - CENTER, dim=1)
            ratios = dists / (2 * RADIUS)
            # 将条件张量扩展到匹配xyz的形状 (N, 3)
            condition = (ratios < 1.0).unsqueeze(1)  # (N, 1) -> 广播到 (N, 3)
            xyz = torch.where(condition, CENTER + (self.xyz - CENTER) / ratios.unsqueeze(1), self.xyz)
            return xyz
        else:
            rot = R.from_euler("z", heading)
            quat_heading = rot.as_quat()  # [x,y,z,w]，SciPy是xyzw，需要转换为wxyz
            quat_heading = np.array([quat_heading[3], quat_heading[0], quat_heading[1], quat_heading[2]])  # wxyz

            rot_matrix = rot.as_matrix()  # [3,3]
            means_objs = self.xyz.detach().cpu().numpy()
            means_rotated = np.dot(means_objs, rot_matrix.T)
            means_world = means_rotated + position  # [N,3]
            return torch.from_numpy(means_world).float().to(self.xyz.device)

    def get_quats(self, heading: Optional[float] = None) -> torch.Tensor:  # [N, 4]
        if self.name in ["background", "sky"]:
            return torch.nn.functional.normalize(self.rotation)
        else:
            rot = R.from_euler("z", heading)  # 围绕Z轴
            quat_heading = rot.as_quat()  # [x,y,z,w]，SciPy是xyzw，需要转换为wxyz
            quat_heading = np.array([quat_heading[3], quat_heading[0], quat_heading[1], quat_heading[2]])  # wxyz

            quats_object = self.rotation.detach().cpu().numpy()
            N = len(quats_object)
            quats_world = np.zeros((N, 4))
            for i in range(N):
                quats_world[i] = quaternion_multiply(quat_heading, quats_object[i])
            quats_world /= np.linalg.norm(quats_world, axis=1, keepdims=True)

            return torch.from_numpy(quats_world).float().to(self.rotation.device)

    def get_scales(self) -> torch.Tensor:  # [N, 3]
        if self.name == "sky":
            scales = torch.exp(self.scaling)
            return torch.clamp(scales, max=RADIUS)
        else:
            return torch.exp(self.scaling)

    def get_opacities(self) -> torch.Tensor:  # [N, 1]
        return torch.sigmoid(self.opacity)

    def get_colors(self) -> torch.Tensor:  # [N, 4, 3]
        return torch.cat((self.feature_dc, self.feature_rest), dim=1)


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
