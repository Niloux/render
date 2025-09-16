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
        """从JSON字典创建Camera对象

        Args:
            data: 包含camera信息的字典，格式如：
                {
                    "id": "cam1",
                    "extrinsics": [[1.0, 0.0, 0.0, 0.0], ...],  # 4x4矩阵
                    "intrinsics": [[1000.0, 0.0, 960.0], ...],   # 3x3矩阵
                    "width": 1920,
                    "height": 1080
                }

        Returns:
            Camera对象

        Raises:
            ValueError: 当数据格式不正确时
        """
        # 验证必需字段
        required_fields = ["id", "extrinsics", "intrinsics", "width", "height"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # 验证矩阵维度
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


@dataclass
class InitParams:
    cameras: List[Camera]

    @classmethod
    def from_json(cls, data: Dict) -> "InitParams":
        """从JSON字典创建InitParams对象

        Args:
            data: 包含初始化参数的字典，格式如：
                {
                    "cameras": [
                        {"id": "cam1", "extrinsics": [...], ...},
                        {"id": "cam2", "extrinsics": [...], ...}
                    ]
                }

        Returns:
            InitParams对象

        Raises:
            ValueError: 当数据格式不正确时
        """
        if "cameras" not in data:
            raise ValueError("Missing required field: cameras")

        cameras_data = data["cameras"]
        if not isinstance(cameras_data, list):
            raise ValueError("cameras must be a list")

        cameras = [Camera.from_json(cam_data) for cam_data in cameras_data]
        return cls(cameras=cameras)


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
        """从JSON字典创建Vehicle对象

        Args:
            data: 包含车辆信息的字典，格式如：
                {
                    "trajectory": [x, y, z],
                    "yaw": 1.57,
                    "type": "car"  # 可选
                }

        Returns:
            Vehicle对象

        Raises:
            ValueError: 当数据格式不正确时
        """
        # 验证必需字段
        if "trajectory" not in data:
            raise ValueError("Missing required field: trajectory")
        if "yaw" not in data:
            raise ValueError("Missing required field: yaw")

        trajectory = data["trajectory"]
        if not isinstance(trajectory, list) or len(trajectory) != 3:
            raise ValueError("trajectory must be a list of 3 floats [x, y, z]")

        return cls(trajectory=[float(x) for x in trajectory], yaw=float(data["yaw"]), type=data.get("type", ""))


@dataclass
class FrameParams:
    ego_trajectory: List[float]  # (x, y, z)
    ego_yaw: float
    env_vehicles: List[Vehicle]
    timestamp: int

    @classmethod
    def from_json(cls, data: Dict) -> "FrameParams":
        """从JSON字典创建FrameParams对象

        Args:
            data: 包含帧参数的字典，格式如：
                {
                    "ego_trajectory": [x, y, z],
                    "ego_yaw": 1.57,
                    "env_vehicles": [
                        {"trajectory": [x, y, z], "yaw": 0.0, "type": "car"},
                        ...
                    ],
                    "timestamp": 1234567890
                }

        Returns:
            FrameParams对象

        Raises:
            ValueError: 当数据格式不正确时
        """
        # 验证必需字段
        required_fields = ["ego_trajectory", "ego_yaw", "env_vehicles", "timestamp"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        ego_trajectory = data["ego_trajectory"]
        if not isinstance(ego_trajectory, list) or len(ego_trajectory) != 3:
            raise ValueError("ego_trajectory must be a list of 3 floats [x, y, z]")

        env_vehicles_data = data["env_vehicles"]
        if not isinstance(env_vehicles_data, list):
            raise ValueError("env_vehicles must be a list")

        env_vehicles = [Vehicle.from_json(vehicle_data) for vehicle_data in env_vehicles_data]

        return cls(
            ego_trajectory=[float(x) for x in ego_trajectory],
            ego_yaw=float(data["ego_yaw"]),
            env_vehicles=env_vehicles,
            timestamp=int(data["timestamp"]),
        )


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
