import os
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


def calculate_viewmats(
    extrinsics_inv: torch.Tensor , ego_heading: float, ego_position: torch.Tensor
) -> torch.Tensor:

    device = ego_position.device
    dtype = torch.float32


    h = torch.as_tensor(ego_heading, device=device, dtype=dtype)
    c, s = torch.cos(h), torch.sin(h)


    ego_inv = torch.eye(4, device=device, dtype=dtype)

    ego_inv[0, 0] =  c; ego_inv[0, 1] =  s
    ego_inv[1, 0] = -s; ego_inv[1, 1] =  c

    t = ego_position
    ego_inv[0, 3] = -(c * t[0] + s * t[1])
    ego_inv[1, 3] =  (s * t[0] - c * t[1])
    ego_inv[2, 3] = -t[2]


    w2c = extrinsics_inv @ ego_inv
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

        # 数值范围处理：假设输出在[0,1]范围内，转换到[0,255]
        # rgb_colors = np.clip(rgb_colors, 0, 1)
        # rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存
        img = Image.fromarray(rgb_colors)
        output_path = os.path.join(output_dir, f"{camera_id}.png")
        img.save(output_path)
