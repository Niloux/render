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