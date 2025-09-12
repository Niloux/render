import os
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

# Linus式优化：全局缓存，避免重复计算
_viewmats_cache = {}


def precompute_camera_transforms(extrinsics: List[List[List[float]]], device: torch.device) -> torch.Tensor:
    """
    预计算相机外参tensor，只需要在初始化时调用一次

    Linus式优化：消除每帧的tensor转换开销
    """
    ext_tensors = []
    for ext_matrix in extrinsics:
        ext = torch.tensor(ext_matrix, device=device, dtype=torch.float32)
        ext_tensors.append(ext)
    return torch.stack(ext_tensors)


def calculate_viewmats(
    extrinsics: List[List[List[float]]], ego_heading: float, ego_position: torch.Tensor
) -> torch.Tensor:
    """
    根据相机外参、主车航向角和位置计算viewmats

    Linus式优化：缓存旋转计算，预计算外参tensor

    Args:
        extrinsics: 相机外参矩阵列表，每个元素为4x4的相机到车辆坐标系变换矩阵
        ego_heading: 主车航向角，单位为弧度
        ego_position: 主车在世界坐标系中的位置，已减去CENTER的torch.Tensor [x, y, z]

    Returns:
        viewmats: 世界坐标系到相机坐标系的变换矩阵，torch.Tensor类型，形状为[N, 4, 4]
    """
    device = ego_position.device

    # Linus式优化：缓存外参tensor转换
    ext_cache_key = f"ext_{id(extrinsics)}_{device}"
    if ext_cache_key not in _viewmats_cache:
        _viewmats_cache[ext_cache_key] = precompute_camera_transforms(extrinsics, device)
    ext_tensors = _viewmats_cache[ext_cache_key]

    # Linus式优化：缓存旋转矩阵计算
    rot_cache_key = f"ego_rot_{ego_heading}_{device}"
    if rot_cache_key not in _viewmats_cache:
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
        _viewmats_cache[rot_cache_key] = ego_rot
    else:
        ego_rot = _viewmats_cache[rot_cache_key]

    # 构建完整的ego pose（只更新位置部分）
    ego_pose = ego_rot.clone()
    ego_pose[:3, 3] = ego_position

    # 向量化计算所有相机的viewmats
    # c2w = ego_pose @ ext_tensors  # [N, 4, 4]
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
        rgb_colors = colors_tensor.detach().cpu().numpy()

        # 数值范围处理：假设输出在[0,1]范围内，转换到[0,255]
        rgb_colors = np.clip(rgb_colors, 0, 1)
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存
        img = Image.fromarray(rgb_colors)
        output_path = os.path.join(output_dir, f"{camera_id}.png")
        img.save(output_path)
