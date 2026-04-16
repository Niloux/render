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

    def to_json(self) -> Dict:
        """将Camera对象转换为JSON字典"""
        return {
            "id": self.id,
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Lidar:
    id: str
    extrinsics: List[List[float]]  # 4*4矩阵（传感器到车辆坐标系）
    azimuth_resolution: float
    min_azimuth: float
    max_azimuth: float
    n_elevation_channels: int
    min_elevation: float
    max_elevation: float
    near_plane: float = 0.01
    far_plane: float = 1e10
    tile_width: int = 64
    tile_height: int = 4

    def to_json(self) -> Dict:
        """将Lidar对象转换为JSON字典"""
        return {
            "id": self.id,
            "extrinsics": self.extrinsics,
            "azimuth_resolution": self.azimuth_resolution,
            "min_azimuth": self.min_azimuth,
            "max_azimuth": self.max_azimuth,
            "n_elevation_channels": self.n_elevation_channels,
            "min_elevation": self.min_elevation,
            "max_elevation": self.max_elevation,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
        }


@dataclass
class InitParams:
    cameras: List[Camera]
    lidars: Optional[List[Lidar]] = None
    render_camera: bool = True
    render_lidar: bool = False
    model_id: Optional[List[int]] = None
    model_path: Optional[str] = None

    def to_json(self) -> Dict:
        """将InitParams对象转换为JSON字典"""
        return {
            "cameras": [camera.to_json() for camera in self.cameras],
            "lidars": (
                [lidar.to_json() for lidar in self.lidars] if self.lidars else None
            ),
            "render_camera": self.render_camera,
            "render_lidar": self.render_lidar,
            "model_id": self.model_id,
            "model_path": self.model_path,
        }


@dataclass
class InitResp:
    init_status: bool
    error_msg: Optional[str] = None

    def to_json(self) -> Dict:
        """将InitResp对象转换为JSON字典

        Returns:
            包含响应信息的字典，格式如：
            {
                "init_status": true,
                "error_msg": null  # 或错误信息字符串
            }
        """
        return {"init_status": self.init_status, "error_msg": self.error_msg}


@dataclass
class Vehicle:
    trajectory: List[float]  # (x, y, z)
    yaw: float
    type: str

    def to_json(self) -> Dict:
        return {
            "trajectory": self.trajectory,
            "yaw": self.yaw,
            "type": self.type if self.type is not None else "",
        }


@dataclass
class FrameParams:
    ego_trajectory: List[float]  # (x, y, z)
    ego_yaw: float
    env_vehicles: List[Vehicle]
    timestamp: int

    def to_json(self) -> Dict:
        """将FrameParams对象转换为JSON字典"""
        return {
            "ego_trajectory": self.ego_trajectory,
            "ego_yaw": self.ego_yaw,
            "env_vehicles": [v.to_json() for v in self.env_vehicles],
            "timestamp": self.timestamp,
        }


@dataclass
class GaussianData:
    """统一管理高斯点云的5种属性数据结构"""

    means: torch.Tensor  # [N, 3] 位置
    quats: torch.Tensor  # [N, 4] 四元数
    scales: torch.Tensor  # [N, 3] 缩放
    opacities: torch.Tensor  # [N, 1] 透明度
    colors: torch.Tensor  # [N, 4, C] 颜色

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
    def from_components(
        cls, components: List["GaussianComponent"]
    ) -> Optional["GaussianData"]:
        """从GaussianComponent列表创建GaussianData"""
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
    lidars: Dict[str, torch.Tensor]
    error_msg: Optional[str] = None

    def to_json(self) -> Dict:
        """将FrameResp对象转换为字典

        Returns:
            包含响应信息的字典，格式如：
            {
                "timestamp": 1234567890,
                "images": {
                    "cam1": torch.Tensor,
                    "cam2": torch.Tensor
                },
                "error_msg": null
            }
        """
        return {
            "timestamp": self.timestamp,
            "images": self.images,
            "error_msg": self.error_msg,
            "lidars": self.lidars,
        }
