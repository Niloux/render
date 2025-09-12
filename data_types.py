from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from models import GaussianComponent


@dataclass
class Camera:
    id: str
    extrinsics: List[List[float]]  # 4*4矩阵
    intrinsics: List[List[float]]  # 3*3矩阵
    width: int
    height: int


@dataclass
class InitParams:
    cameras: List[Camera]


@dataclass
class InitResp:
    init_status: bool
    error_msg: Optional[str] = None


@dataclass
class Vehicle:
    trajectory: List[float]  # (x, y, z)
    yaw: float
    type: Optional[str] = ""


@dataclass
class FrameParams:
    ego_trajectory: List[float]  # (x, y, z)
    ego_yaw: float
    env_vehicles: List[Vehicle]
    timestamp: int


@dataclass
class GaussianData:
    """统一管理高斯点云的5种属性数据结构

    这是Linus式"好品味"的体现：用一个统一的数据结构
    替代5个平行数组，消除重复代码和特殊情况处理。
    """

    means: torch.Tensor  # [N, 3] 位置
    quats: torch.Tensor  # [N, 4] 四元数
    scales: torch.Tensor  # [N, 3] 缩放
    opacities: torch.Tensor  # [N, 1] 透明度
    colors: torch.Tensor  # [N, 4, 3] 颜色

    def cat(self, other: Optional["GaussianData"]) -> "GaussianData":
        """连接两个高斯数据，消除重复的torch.cat调用"""
        if other is None:
            return self
        return GaussianData(
            means=torch.cat([self.means, other.means]),
            quats=torch.cat([self.quats, other.quats]),
            scales=torch.cat([self.scales, other.scales]),
            opacities=torch.cat([self.opacities, other.opacities]),
            colors=torch.cat([self.colors, other.colors]),
        )

    @classmethod
    def from_components(cls, components: List["GaussianComponent"]) -> Optional["GaussianData"]:
        """从GaussianComponent列表创建GaussianData

        这个方法消除了_static_gs和_dynamic_gs中的重复代码
        """
        if not components:
            return None

        return cls(
            means=torch.cat([c.get_xyz() for c in components]),
            quats=torch.cat([c.get_quats() for c in components]),
            scales=torch.cat([c.get_scales() for c in components]),
            opacities=torch.cat([c.get_opacities() for c in components]),
            colors=torch.cat([c.get_colors() for c in components]),
        )

    def to_render_args(self) -> tuple:
        """返回render函数需要的参数元组"""
        return (self.means, self.quats, self.scales, self.opacities, self.colors)


@dataclass
class FrameResp:
    timestamp: int
    images: Dict[str, torch.Tensor]
    error_msg: Optional[str] = None
