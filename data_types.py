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
    lidars: Optional[List["Lidar"]] = None
    render_camera: bool = True
    render_lidar: bool = False
    model_id: Optional[List[int]] = None
    model_path: Optional[str] = None

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

        lidars: Optional[List[Lidar]] = None
        if "lidars" in data and data["lidars"] is not None:
            lidars_data = data["lidars"]
            if not isinstance(lidars_data, list):
                raise ValueError("lidars must be a list")
            lidars = [Lidar.from_json(ld) for ld in lidars_data]
        else:
            # 兼容单雷达参数风格
            single_keys = [
                "lidar_extrinsics",
                "Far_lidar",
                "Near_lidar",
                "H_FOV",
                "V_FOV_up",
                "V_FOV_down",
                "H_lidar",
                "W_lidar",
            ]
            if all(k in data for k in single_keys):
                lidars = [
                    Lidar.from_json({
                        "id": data.get("lidar_id", "lidar_0"),
                        "extrinsics": data["lidar_extrinsics"],
                        "far_plane": data["Far_lidar"],
                        "near_plane": data["Near_lidar"],
                        "h_fov": data["H_FOV"],
                        "v_fov_up": data["V_FOV_up"],
                        "v_fov_down": data["V_FOV_down"],
                        "h_lidar": data["H_lidar"],
                        "w_lidar": data["W_lidar"],
                    })
                ]

        render_camera = bool(data.get("render_camera", True))
        render_lidar = bool(data.get("render_lidar", False))
        model_id = data.get("model_id")
        model_path = data.get("model_path")

        return cls(
            cameras=cameras,
            lidars=lidars,
            render_camera=render_camera,
            render_lidar=render_lidar,
            model_id=model_id,
            model_path=model_path,
        )


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
    lidar_points: Optional[Dict[str, List[List[float]]]] = None

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

        lidar_points: Optional[Dict[str, List[List[float]]]] = None
        if "lidar_points" in data and data["lidar_points"] is not None:
            raw_lp = data["lidar_points"]
            if not isinstance(raw_lp, dict):
                raise ValueError("lidar_points must be a dict of id -> points")
            # 统一转换为float列表
            lidar_points = {str(k): [[float(v) for v in pt] for pt in pts] for k, pts in raw_lp.items()}

        return cls(
            ego_trajectory=[float(x) for x in ego_trajectory],
            ego_yaw=float(data["ego_yaw"]),
            env_vehicles=env_vehicles,
            timestamp=int(data["timestamp"]),
            lidar_points=lidar_points,
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
    lidars: Dict[str, torch.Tensor]

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

    @classmethod
    def from_json(cls, data: Dict) -> "Lidar":
        """从JSON字典创建Lidar对象

        Args:
            data: 包含lidar信息的字典，示例字段：
                {
                  "id": "lidar_front",
                  "extrinsics": [[...], ...],
                  "azimuth_resolution": 0.2,
                  "min_azimuth": -180,
                  "max_azimuth": 180,
                  "n_elevation_channels": 32,
                  "min_elevation": -30,
                  "max_elevation": 30,
                  "tile_width": 64,  # 可选
                  "tile_height": 4   # 可选
                }

        Returns:
            Lidar对象

        Raises:
            ValueError: 当数据格式不正确时
        """
        # 支持两种风格：直接提供角分辨率与边界，或提供FOV与分辨率参数
        if "azimuth_resolution" in data:
            required = [
                "id",
                "extrinsics",
                "azimuth_resolution",
                "min_azimuth",
                "max_azimuth",
                "n_elevation_channels",
                "min_elevation",
                "max_elevation",
            ]
            missing = [f for f in required if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            extrinsics = data["extrinsics"]
            if len(extrinsics) != 4 or any(len(row) != 4 for row in extrinsics):
                raise ValueError("extrinsics must be a 4x4 matrix")

            return cls(
                id=str(data["id"]),
                extrinsics=extrinsics,
                azimuth_resolution=float(data["azimuth_resolution"]),
                min_azimuth=float(data["min_azimuth"]),
                max_azimuth=float(data["max_azimuth"]),
                n_elevation_channels=int(data["n_elevation_channels"]),
                min_elevation=float(data["min_elevation"]),
                max_elevation=float(data["max_elevation"]),
                near_plane=float(data.get("near_plane", 0.01)),
                far_plane=float(data.get("far_plane", 1e10)),
                tile_width=int(data.get("tile_width", 64)),
                tile_height=int(data.get("tile_height", 4)),
            )
        else:
            # 基于FOV与分辨率推导
            required = [
                "id",
                "extrinsics",
                "h_fov",
                "v_fov_up",
                "v_fov_down",
                "h_lidar",
                "w_lidar",
            ]
            missing = [f for f in required if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            extrinsics = data["extrinsics"]
            if len(extrinsics) != 4 or any(len(row) != 4 for row in extrinsics):
                raise ValueError("extrinsics must be a 4x4 matrix")

            h_fov = float(data["h_fov"])  # 度
            w_lidar = int(data["w_lidar"])  # 水平点数
            az_res = h_fov / float(w_lidar)
            min_az = -h_fov / 2.0
            max_az = h_fov / 2.0
            n_elev = int(data["h_lidar"])  # 线数
            min_el = float(data["v_fov_down"])  # 度
            max_el = float(data["v_fov_up"])  # 度

            return cls(
                id=str(data["id"]),
                extrinsics=extrinsics,
                azimuth_resolution=az_res,
                min_azimuth=min_az,
                max_azimuth=max_az,
                n_elevation_channels=n_elev,
                min_elevation=min_el,
                max_elevation=max_el,
                near_plane=float(data.get("near_plane", data.get("near_lidar", 0.01))),
                far_plane=float(data.get("far_plane", data.get("far_lidar", 1e10))),
                tile_width=int(data.get("tile_width", 64)),
                tile_height=int(data.get("tile_height", 4)),
            )
