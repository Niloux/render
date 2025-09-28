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

    @classmethod
    def from_json(cls, data: Dict) -> "Camera":
        """从JSON字典创建Camera对象"""
        required_fields = ["id", "extrinsics", "intrinsics", "width", "height"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        extrinsics = data["extrinsics"]
        intrinsics = data["intrinsics"]

        if len(extrinsics) != 4 or any(len(row) != 4 for row in extrinsics):
            raise ValueError("extrinsics must be a 4x4 matrix")

        if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
            raise ValueError("intrinsics must be a 3x3 matrix")

        return cls(
            id=data["id"],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            width=int(data["width"]),
            height=int(data["height"]),
        )

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
class InitParams:
    cameras: List[Camera]

    @classmethod
    def from_json(cls, data: Dict) -> "InitParams":
        """从JSON字典创建InitParams对象"""
        if "cameras" not in data:
            raise ValueError("Missing required field: cameras")

        cameras_data = data["cameras"]
        if not isinstance(cameras_data, list):
            raise ValueError("cameras must be a list")

        cameras = [Camera.from_json(cam_data) for cam_data in cameras_data]
        return cls(cameras=cameras)

    def to_json(self) -> Dict:
        """将InitParams对象转换为JSON字典"""
        return {
            "cameras": [cam.to_json() for cam in self.cameras]
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
    type: Optional[str] = ""

    @classmethod
    def from_json(cls, data: Dict) -> "Vehicle":
        if "trajectory" not in data:
            raise ValueError("Missing required field: trajectory")
        if "yaw" not in data:
            raise ValueError("Missing required field: yaw")

        trajectory = data["trajectory"]
        if not isinstance(trajectory, list) or len(trajectory) != 3:
            raise ValueError("trajectory must be a list of 3 floats [x, y, z]")

        return cls(
            trajectory=[float(x) for x in trajectory],
            yaw=float(data["yaw"]),
            type=data.get("type", "")
        )

    def to_json(self) -> Dict:
        return {
            "trajectory": self.trajectory,
            "yaw": self.yaw,
            "type": self.type if self.type is not None else ""
        }


@dataclass
class FrameParams:
    ego_trajectory: List[float]  # (x, y, z)
    ego_yaw: float
    env_vehicles: List[Vehicle]
    timestamp: int

    @classmethod
    def from_json(cls, data: Dict) -> "FrameParams":
        # 校验必需字段
        for key in ("ego_trajectory", "ego_yaw", "env_vehicles", "timestamp"):
            if key not in data:
                raise ValueError(f"Missing required field: {key}")

        # ego_trajectory
        ego_traj = data["ego_trajectory"]
        if not isinstance(ego_traj, list) or len(ego_traj) != 3:
            raise ValueError("ego_trajectory must be a list of 3 floats [x, y, z]")
        ego_traj = [float(x) for x in ego_traj]

        # ego_yaw
        ego_yaw = float(data["ego_yaw"])

        # env_vehicles: 允许是 dict 列表或 Vehicle 实例列表
        ev_raw = data["env_vehicles"]
        if not isinstance(ev_raw, list):
            raise ValueError("env_vehicles must be a list")

        env_vehicles: List[Vehicle] = []
        for i, item in enumerate(ev_raw):
            if isinstance(item, Vehicle):
                env_vehicles.append(item)
            elif isinstance(item, dict):
                env_vehicles.append(Vehicle.from_json(item))
            else:
                raise ValueError(f"env_vehicles[{i}] must be a dict or Vehicle")

        # timestamp
        ts = data["timestamp"]
        if not isinstance(ts, int):
            raise ValueError("timestamp must be an int")

        return cls(
            ego_trajectory=ego_traj,
            ego_yaw=ego_yaw,
            env_vehicles=env_vehicles,
            timestamp=ts,
        )

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
        return {"timestamp": self.timestamp, "images": self.images, "error_msg": self.error_msg}
