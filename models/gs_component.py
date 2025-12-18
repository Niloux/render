"""高斯点云组件数据类。"""

from dataclasses import dataclass
from typing import Optional

import torch

from config import SKY_CENTER, SKY_RADIUS

if torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
else:
    # 如果不是分布式环境，则使用可用的第一个GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SKY_CENTER = torch.tensor(SKY_CENTER, device=device)
SKY_RADIUS = torch.tensor(SKY_RADIUS, device=device)


@dataclass
class GaussianComponent:
    """高斯点云组件数据类

    用于管理和操作高斯点云的各种属性，包括位置、旋转、缩放、透明度和颜色特征。
    提供了数据验证、内存使用统计和坐标变换等功能。

    """

    name: str  # 组件名称
    xyz: torch.Tensor  # 位置 [N, 3]
    feature_dc: torch.Tensor  # DC特征 [N, 1, 3]
    feature_rest: torch.Tensor  # 其余特征 [N, K, 3]
    scaling: torch.Tensor  # 对数缩放 [N, 3]
    rotation: torch.Tensor  # 四元数旋转 [N, 4] (wxyz格式)
    opacity: torch.Tensor  # logit透明度 [N, 1]
    semantic: Optional[torch.Tensor] = None  # [N, K] 语义信息

    @property
    def num_points(self) -> int:
        """获取点云数量。"""
        return self.xyz.shape[0]

    @property
    def memory_usage(self) -> float:
        """获取内存使用量（MB）。"""
        total_bytes = sum(
            [
                self.xyz.numel() * self.xyz.element_size(),
                self.feature_dc.numel() * self.feature_dc.element_size(),
                self.feature_rest.numel() * self.feature_rest.element_size(),
                self.scaling.numel() * self.scaling.element_size(),
                self.rotation.numel() * self.rotation.element_size(),
                self.opacity.numel() * self.opacity.element_size(),
            ]
        )

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
            # TODO:
            or self.feature_dc.shape[1:] != (1, 5)
            or self.feature_rest.shape[1:] != (3, 5)
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

    def get_xyz(
        self, heading: Optional[float] = None, position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [N, 3]
        """获取变换后的3D坐标

        Args:
            heading: 航向角（弧度），仅对object类型有效
            position: 世界坐标位置 [3] 的torch张量，仅对object类型有效

        Returns:
            变换后的坐标张量 [N, 3]
        """
        with torch.no_grad():
            if self.name == "background":
                return self.xyz
            elif self.name == "sky":
                dists = torch.linalg.norm(self.xyz - SKY_CENTER, dim=1)
                ratios = dists / (2 * SKY_RADIUS)
                # 将条件张量扩展到匹配xyz的形状 (N, 3)
                condition = (ratios < 1.0).unsqueeze(1)  # (N, 1) -> 广播到 (N, 3)
                xyz = torch.where(condition, SKY_CENTER + (self.xyz - SKY_CENTER) / ratios.unsqueeze(1), self.xyz)
                return xyz
            else:
                # 计算旋转矩阵，无需缓存
                device = self.xyz.device
                cos_h = torch.cos(torch.tensor(heading, device=device))
                sin_h = torch.sin(torch.tensor(heading, device=device))
                rot_matrix = torch.tensor([[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]], device=device)

                # 向量化矩阵乘法：[N, 3] @ [3, 3] -> [N, 3]
                means_rotated = torch.matmul(self.xyz, rot_matrix.T)
                means_world = means_rotated + position.to(device)
                return means_world

    def get_quats(self, heading: Optional[float] = None) -> torch.Tensor:  # [N, 4]
        """获取变换后的四元数

        Args:
            heading: 航向角（弧度），仅对object类型有效

        Returns:
            归一化的四元数张量 [N, 4] (wxyz格式)
        """
        with torch.no_grad():
            if self.name in ["background", "sky"]:
                return torch.nn.functional.normalize(self.rotation)
            else:
                # 直接计算四元数，无需缓存
                device = self.rotation.device
                half_angle = heading * 0.5
                cos_half = torch.cos(torch.tensor(half_angle, device=device))
                sin_half = torch.sin(torch.tensor(half_angle, device=device))
                quat_heading = torch.tensor([cos_half, 0.0, 0.0, sin_half], device=device)  # [w, x, y, z]

                # 向量化四元数乘法：quat_heading * self.rotation
                # q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
                #            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                #            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                #            w1*z2 + x1*y2 - y1*x2 + z1*w2]
                w1, x1, y1, z1 = quat_heading[0], quat_heading[1], quat_heading[2], quat_heading[3]
                w2, x2, y2, z2 = self.rotation[:, 0], self.rotation[:, 1], self.rotation[:, 2], self.rotation[:, 3]

                quats_world = torch.stack(
                    [
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
                    ],
                    dim=1,
                )

                # 归一化
                return torch.nn.functional.normalize(quats_world, dim=1)

    def get_scales(self) -> torch.Tensor:  # [N, 3]
        with torch.no_grad():
            if self.name == "sky":
                scales = torch.exp(self.scaling)
                return torch.clamp(scales, max=SKY_RADIUS)
            else:
                return torch.exp(self.scaling)

    def get_opacities(self) -> torch.Tensor:  # [N, 1]
        with torch.no_grad():
            return torch.sigmoid(self.opacity)

    def get_colors(self) -> torch.Tensor:  # [N, 4, 3]
        with torch.no_grad():
            return torch.cat((self.feature_dc, self.feature_rest), dim=1)
