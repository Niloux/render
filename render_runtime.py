"""Runtime validation and pose helpers for rendering."""

import math
from typing import List, Tuple

import torch

from data_types import Camera, FrameParams, InitParams


def validate_matrix(
    value: List[List[float]], shape: Tuple[int, int], name: str
) -> None:
    rows, cols = shape
    if len(value) != rows or any(len(row) != cols for row in value):
        raise ValueError(f"{name}期望形状为{shape}，实际为不规则矩阵")


def validate_init_params(params: InitParams) -> None:
    cameras = params.cameras or []
    if not cameras and getattr(params, "render_camera", True):
        raise ValueError("render_camera=True时必须提供至少一个camera")
    for camera in cameras:
        validate_camera(camera)

    for lidar in params.lidars or []:
        validate_matrix(lidar.extrinsics, (4, 4), f"lidar {lidar.id} extrinsics")
        if lidar.tile_width <= 0 or lidar.tile_height <= 0:
            raise ValueError(f"lidar {lidar.id} tile尺寸必须为正数")


def validate_camera(camera: Camera) -> None:
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError(
            f"camera {camera.id} 分辨率非法: {camera.width}x{camera.height}"
        )
    validate_matrix(camera.extrinsics, (4, 4), f"camera {camera.id} extrinsics")
    validate_matrix(camera.intrinsics, (3, 3), f"camera {camera.id} intrinsics")


def validate_frame_params(params: FrameParams, initialized: bool) -> None:
    if not initialized:
        raise RuntimeError("RenderManager尚未init，不能调用render_frame")
    if len(params.ego_trajectory) != 3:
        raise ValueError(f"ego_trajectory必须为长度3，实际为{params.ego_trajectory}")
    if not math.isfinite(float(params.ego_yaw)):
        raise ValueError(f"ego_yaw必须为有限数，实际为{params.ego_yaw}")
    for index, value in enumerate(params.ego_trajectory):
        if not math.isfinite(float(value)):
            raise ValueError(f"ego_trajectory[{index}]必须为有限数，实际为{value}")
    for vehicle in params.env_vehicles or []:
        if len(vehicle.trajectory) != 3:
            raise ValueError(f"车辆{vehicle.type} trajectory必须为长度3")


def calculate_viewmats(
    extrinsics_tensor: torch.Tensor,
    ego_heading: float,
    ego_position: torch.Tensor,
    ego_pitch: float = 0.0,
    ego_roll: float = 0.0,
) -> torch.Tensor:
    device = ego_position.device

    yaw = float(ego_heading)
    pitch = float(ego_pitch)
    roll = float(ego_roll)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    Rz = torch.tensor(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    Ry = torch.tensor(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        device=device,
        dtype=torch.float32,
    )
    Rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        device=device,
        dtype=torch.float32,
    )

    R_ego = Rz @ Ry @ Rx

    ego_pose = torch.eye(4, device=device, dtype=torch.float32)
    ego_pose[:3, :3] = R_ego
    ego_pose[:3, 3] = ego_position

    c2w = torch.matmul(ego_pose.unsqueeze(0), extrinsics_tensor)

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    Rt = R.transpose(-2, -1)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)

    w2c = (
        torch.eye(4, device=device, dtype=torch.float32)
        .expand(c2w.shape[0], 4, 4)
        .clone()
    )
    w2c[:, :3, :3] = Rt
    w2c[:, :3, 3] = t_inv
    return w2c
