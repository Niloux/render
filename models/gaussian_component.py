"""高斯点云组件数据类。"""

from dataclasses import dataclass
from typing import Optional

import torch


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

    def get_xyz(self) -> torch.Tensor:  # [N, 3]
        xyzs = []
        if self.name == "background":
            xyzs.append(self.xyz)
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs

    def get_quats(self) -> torch.Tensor:  # [N, 4]
        quats = []
        if self.name == "background":
            quats.append(torch.nn.functional.normalize(self.rotation))
        return torch.cat(quats, dim=0)

    def get_scales(self) -> torch.Tensor:  # [N, 3]
        scalings = []
        if self.name == "background":
            scalings.append(torch.exp(self.scaling))
        return torch.cat(scalings, dim=0)

    def get_opacities(self) -> torch.Tensor:  # [N, 1]
        opacities = []
        if self.name == "background":
            opacities.append(torch.sigmoid(self.opacity))
        return torch.cat(opacities, dim=0)

    def get_colors(self) -> torch.Tensor:  # [N, 4, 3]
        colors = []
        if self.name == "background":
            features = torch.cat((self.feature_dc, self.feature_rest), dim=1)
            colors.append(features)
        return torch.cat(colors, dim=0)
