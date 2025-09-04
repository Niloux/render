import math

import torch
from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterization,
    rasterize_to_pixels,
    spherical_harmonics,
)

NATIVE = True


def extract_camera_centers(viewmats: torch.Tensor):
    """
    从view matrices提取相机中心位置

    Args:
        viewmats: 形状为 [C, 4, 4] 的view matrices, World2Camera的转换矩阵

    Returns:
        camera_centers: 形状为 [C, 3] 的相机中心位置
    """
    R = viewmats[:, :3, :3]  # 旋转矩阵 [C, 3, 3]
    t = viewmats[:, :3, 3]  # 平移向量 [C, 3]
    # camera_center = -R^T * t
    camera_centers = -torch.bmm(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)  # [C, 3]
    return camera_centers


def render_gaussian_splatting(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height):
    """
    执行高斯点云渲染的核心函数

    Args:
        means: 高斯点的3D位置 [N, 3]
        quats: 高斯点的四元数旋转 [N, 4]
        scales: 高斯点的缩放 [N, 3]
        opacities: 高斯点的不透明度 [N, 1]
        colors: 高斯点的球谐系数 [N, K, 3]
        viewmats: World2Camera转换矩阵 [C, 4, 4]
        Ks: 相机内参矩阵 [C, 3, 3]
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        render_colors: 渲染的颜色图像 [C, H, W, 4]
        render_alphas: 渲染的alpha通道 [C, H, W, 1]
    """
    # 投影
    project_results = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=img_width,
        height=img_height,
        packed=False,
        near_plane=0.001,
        far_plane=1000,
        calc_compensations=True,
    )
    radii, means2d, depths, conics, compensations = project_results

    # 处理不透明度
    batch_opacities = opacities[None, :, 0].expand(viewmats.shape[0], -1)  # [C, N]
    if compensations is not None:
        batch_opacities = batch_opacities * compensations

    # Tile处理
    tile_size = 16
    tile_width = math.ceil(img_width / float(tile_size))
    tile_height = math.ceil(img_height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False, n_images=viewmats.shape[0]
    )
    isect_offsets = isect_offset_encode(isect_ids, viewmats.shape[0], tile_width, tile_height)

    # 球谐函数处理
    camera_centers = extract_camera_centers(viewmats)  # [C, 3]
    dirs = means[None, :, :] - camera_centers[:, None, :]  # [C, N, 3]
    masks = (radii > 0).any(dim=-1)
    # masks = radii > 0  # [C, N]
    shs = colors.expand(viewmats.shape[0], -1, -1, -1)  # [C, N, K, 3]
    batch_colors = spherical_harmonics(1, dirs, shs, masks=masks)  # [C, N, 3]
    batch_colors = torch.clamp_min(batch_colors + 0.5, 0.0)
    batch_colors = torch.cat((batch_colors, depths[..., None]), dim=-1)

    # 光栅化
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        batch_colors,
        batch_opacities,
        img_width,
        img_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=None,
        packed=False,
        absgrad=True,
    )

    return render_colors, render_alphas


def render_native(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height):
    """原生的渲染方法，使用rasterization函数的内置球谐函数处理"""
    # gsplat库期望opacities形状为[N,]，而不是[N,1]
    if opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities = opacities.squeeze(1)

    # 使用rasterization函数的内置球谐函数处理
    # colors保持[N, 4, 3]形状，设置sh_degree=1让rasterization内部处理球谐函数
    # 关键修复：添加rasterize_mode="antialiased"启用抗锯齿补偿，消除渲染裂痕
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=img_width,
        height=img_height,
        near_plane=0.001,
        far_plane=1000,
        sh_degree=1,
        packed=True,
        absgrad=True,
        tile_size=16,
        radius_clip=0.0,
        distributed=False,
        rasterize_mode="antialiased",  # 启用抗锯齿模式
    )
    return render_colors, render_alphas


def render(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height):
    if NATIVE:
        return render_native(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height)
    else:
        return render_gaussian_splatting(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height)
