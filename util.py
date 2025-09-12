import os
from typing import List

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
    # 计算旋转矩阵（绕Z轴旋转）
    cos_h = torch.cos(torch.tensor(ego_heading, device=ego_position.device))
    sin_h = torch.sin(torch.tensor(ego_heading, device=ego_position.device))

    # 构建车辆到世界坐标系的变换矩阵 (T_vehicle_to_world)
    ego_pose = torch.tensor(
        [
            [cos_h, -sin_h, 0.0, ego_position[0]],
            [sin_h, cos_h, 0.0, ego_position[1]],
            [0.0, 0.0, 1.0, ego_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=ego_position.device,
        dtype=torch.float32,
    )

    viewmats_list = []

    for ext_matrix in extrinsics:
        # 将外参矩阵转换为torch.Tensor
        ext = torch.tensor(ext_matrix, device=ego_position.device, dtype=torch.float32)

        # 计算相机到世界坐标系的变换矩阵 c2w = T_vehicle_to_world @ T_camera_to_vehicle
        c2w = ego_pose @ ext

        # 计算世界坐标系到相机坐标系的变换矩阵 w2c = inv(c2w)
        w2c = torch.linalg.inv(c2w)

        viewmats_list.append(w2c)

    return torch.stack(viewmats_list)


def save_colors_as_png(colors_tensor, output_dir="output"):
    """
    将渲染的colors张量保存为PNG图像

    Args:
        colors_tensor: 形状为 [N_views, H, W, C] 的张量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 将张量移到CPU并转换为numpy
    colors_np = colors_tensor.detach().cpu().numpy()
    n_views, height, width, channels = colors_np.shape

    print(f"Colors张量形状: {colors_np.shape}")
    print(f"视角数量: {n_views}, 图像尺寸: {height}x{width}, 通道数: {channels}")

    for view_idx in range(n_views):
        # 获取单个视角的图像数据
        view_colors = colors_np[view_idx]  # [H, W, C]

        # 如果通道数大于3，取前3个通道作为RGB
        if channels > 3:
            rgb_colors = view_colors[:, :, :3]
            print(f"视角 {view_idx}: 使用前3个通道作为RGB")
        else:
            rgb_colors = view_colors

        # 数值范围处理：假设输出在[0,1]范围内，转换到[0,255]
        rgb_colors = np.clip(rgb_colors, 0, 1)
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存
        img = Image.fromarray(rgb_colors)
        output_path = os.path.join(output_dir, f"view_{view_idx:02d}.png")
        img.save(output_path)
        print(f"保存视角 {view_idx} 到: {output_path}")

        # 保存统计信息
        print(f"  - 像素值范围: [{rgb_colors.min()}, {rgb_colors.max()}]")
        print(f"  - 平均像素值: {rgb_colors.mean():.3f}")
