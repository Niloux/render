import os
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


def calculate_viewmats(
    extrinsics: List[List[List[float]]], ego_heading: float, ego_position: torch.Tensor
) -> torch.Tensor:
    """
    根据相机外参、主车航向角和位置计算viewmats

    Args:
        extrinsics: 相机外参矩阵列表，每个元素为4x4的相机到车辆坐标系变换矩阵
        ego_heading: 主车航向角，单位为弧度
        ego_position: 主车在世界坐标系中的位置，已减去CENTER的torch.Tensor [x, y, z]

    Returns:
        viewmats: 世界坐标系到相机坐标系的变换矩阵，torch.Tensor类型，形状为[N, 4, 4]
    """
    device = ego_position.device

    # 直接转换外参为tensor
    ext_tensors = []
    for ext_matrix in extrinsics:
        ext = torch.tensor(ext_matrix, device=device, dtype=torch.float32)
        ext_tensors.append(ext)
    ext_tensors = torch.stack(ext_tensors)

    # 直接计算旋转矩阵
    cos_h = torch.cos(torch.tensor(ego_heading, device=device))
    sin_h = torch.sin(torch.tensor(ego_heading, device=device))
    ego_rot = torch.tensor(
        [
            [cos_h, -sin_h, 0.0, 0.0],
            [sin_h, cos_h, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    # 构建完整的ego pose
    ego_pose = ego_rot.clone()
    ego_pose[:3, 3] = ego_position

    # 向量化计算所有相机的viewmats
    c2w = torch.matmul(ego_pose.unsqueeze(0), ext_tensors)  # [N, 4, 4]

    # 批量计算逆矩阵
    w2c = torch.linalg.inv(c2w)

    return w2c


def save_colors_as_png(image: Dict[str, torch.Tensor], output_dir="output"):
    """
    将渲染的colors张量保存为PNG图像

    Args:
        image: torch.Tensor形状为 [H, W, C] 的张量,键值为camera_id
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera_id, colors_tensor in image.items():
        # 将张量移到CPU并转换为numpy
        t0 = time.time()
        pinned_cpu = torch.empty_like(colors_tensor, device="cpu", pin_memory=True)
        pinned_cpu.copy_(colors_tensor, non_blocking=True)
        rgb_colors = pinned_cpu.detach().numpy()
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"拷贝耗时: {t1 - t0:.6f}")
        # rgb_colors = colors_tensor.detach().cpu().numpy()

        # # 数值范围处理：假设输出在[0,1]范围内，转换到[0,255]
        # rgb_colors = np.clip(rgb_colors, 0, 1)
        # rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存
        img = Image.fromarray(rgb_colors)
        output_path = os.path.join(output_dir, f"{camera_id}.png")
        img.save(output_path)


def pano_to_lidar_with_intensities(
    raster_pts, out, directions=None, depth_valid_mask=None
):
    """
    将渲染输出转换为点云

    参数:
    raster_pts: [1, H, W, 5] - (azimuth, elev, depth, time, intensity)
    out: 渲染输出字典，包含:
        - "depth": [H, W] - 预测深度
        - "intensity": [H, W] - 预测强度
        - "ray_drop_prob": [H, W] - 射线丢弃概率
    directions: 可选，预计算方向向量，形状为[H, W, 3]或[H*W, 3]
    depth_valid_mask: 可选，预计算静态深度有效mask，形状为[H, W]或[H*W]

    返回:
    pred_point_cloud: [N, 4] - 预测点云 (x, y, z, intensity)
    """
    if len(raster_pts.shape) == 4 and raster_pts.shape[0] == 1:
        raster_pts = raster_pts.squeeze(0)

    H, W = raster_pts.shape[:2]

    if directions is None:
        azimuth_angles = torch.deg2rad(raster_pts[..., 0].flatten())
        elevation_angles = torch.deg2rad(raster_pts[..., 1].flatten())
        directions = torch.stack(
            [
                torch.cos(elevation_angles) * torch.cos(azimuth_angles),
                torch.cos(elevation_angles) * torch.sin(azimuth_angles),
                torch.sin(elevation_angles),
            ],
            dim=-1,
        )
    else:
        if isinstance(directions, torch.Tensor) and directions.dim() == 3:
            directions = directions.reshape(-1, 3)

    pred_depth = out["depth"]
    pred_intensity = out["intensity"]
    pred_ray_drop_logits = out["ray_drop_prob"]

    if pred_depth.dim() == 2:
        pred_depth = pred_depth.flatten()
    if pred_intensity.dim() == 2:
        pred_intensity = pred_intensity.flatten()
    if pred_ray_drop_logits.dim() == 2:
        pred_ray_drop_logits = pred_ray_drop_logits.flatten()

    ray_drop_prob = torch.sigmoid(pred_ray_drop_logits)
    pred_intensity = torch.sigmoid(pred_intensity)

    pred_points = directions * pred_depth.unsqueeze(-1)
    pred_point_cloud = torch.cat([pred_points, pred_intensity.unsqueeze(-1)], dim=-1)

    ray_drop_mask = ray_drop_prob < 0.5

    if depth_valid_mask is None:
        gt_depth = raster_pts[..., 2].flatten()
        did_return_threshold = 1000.0
        gt_did_return = gt_depth <= did_return_threshold
        depth_valid_mask = (gt_depth > 0) & gt_did_return
    else:
        if isinstance(depth_valid_mask, torch.Tensor) and depth_valid_mask.dim() == 2:
            depth_valid_mask = depth_valid_mask.flatten()

    valid_mask = depth_valid_mask & ray_drop_mask
    pred_point_cloud = pred_point_cloud[valid_mask]

    return pred_point_cloud


def affine_inverse(A: np.ndarray):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return np.concatenate([np.concatenate([R.T, -R.T @ T], axis=-1), P], axis=-2)


def get_ray_dirs_pinhole(K: torch.Tensor, width: int, height: int, c2w: torch.Tensor):
    ys = (
        torch.arange(height, device=K.device, dtype=torch.float32) + (0.5 - K[0, 1, 2])
    ) / K[0, 1, 1]
    xs = (
        torch.arange(width, device=K.device, dtype=torch.float32) + (0.5 - K[0, 0, 2])
    ) / K[0, 0, 0]
    image_coords = torch.meshgrid(ys, xs, indexing="ij")
    # flip y and z to align with nerfstudio convention
    directions = torch.stack(
        [image_coords[1], -image_coords[0], -torch.ones_like(image_coords[0])], dim=-1
    )  # (h, w, 3)
    directions = directions.view(-1, 3)
    directions = torch.matmul(directions, c2w[0, :3, :3].transpose(0, 1))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    directions = directions.view(height, width, 3)

    return directions
