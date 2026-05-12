import os
from typing import Dict

import torch
from PIL import Image


def save_colors_as_png(image: Dict[str, torch.Tensor], output_dir="output"):
    """
    将渲染的colors张量保存为PNG图像

    Args:
        image: torch.Tensor形状为 [H, W, C] 的张量,键值为camera_id
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera_id, colors_tensor in image.items():
        rgb_colors = colors_tensor.detach().cpu().numpy()
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
