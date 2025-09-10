from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


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
    error_msg: Optional[str]


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
class FrameResp:
    timestamp: int
    images: Dict[str, torch.Tensor]
    error_msg: Optional[str]
