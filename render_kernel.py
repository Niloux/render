import math

import torch
from gsplat.cuda._wrapper import (
    map_points_to_lidar_tiles,
    points_mapping_offset_encode,
    populate_image_from_points,
)

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
    camera_centers = -torch.bmm(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(
        -1
    )  # [C, 3]
    return camera_centers


def render_gaussian_splatting(
    means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
):
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
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=viewmats.shape[0],
    )
    isect_offsets = isect_offset_encode(
        isect_ids, viewmats.shape[0], tile_width, tile_height
    )

    # 球谐函数处理
    camera_centers = extract_camera_centers(viewmats)  # [C, 3]
    dirs = means[None, :, :] - camera_centers[:, None, :]  # [C, N, 3]
    masks = (radii > 0).all(dim=-1)
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


def render_native(
    means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
):
    """原生的渲染方法，使用rasterization函数的内置球谐函数处理"""
    # gsplat库期望opacities形状为[N,]，而不是[N,1]
    if opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities = opacities.squeeze(1)

    # 使用rasterization函数的内置球谐函数处理
    # colors保持[N, 4, 3]形状，设置sh_degree=1让rasterization内部处理球谐函数
    # TODO:colors应该在前期做好camera和lidar的分离，后面再优化吧
    if colors.dim() == 3 and colors.shape[1] == 4 and colors.shape[2] == 5:
        colors = colors[..., :3]

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
        tile_size=16,
        radius_clip=3.0,
        rasterize_mode="antialiased",  # 启用抗锯齿模式
        # # 多GPU并行
        # distributed=True,
        # absgrad=False,
        # velocities=None,
    )
    return render_colors, render_alphas


def render(
    means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
):
    if NATIVE:
        return render_native(
            means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
        )
    else:
        return render_gaussian_splatting(
            means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
        )


def generate_point_cloud(
    azimuth_resolution: float,
    min_azimuth: float,
    max_azimuth: float,
    n_elevation_channels: int,
    min_elevation: float,
    max_elevation: float,
    device: torch.device,
):
    """生成用于渲染的点云（方位、俯仰、距离、时间偏移、强度）。"""
    azimuths = torch.linspace(
        min_azimuth,
        max_azimuth - azimuth_resolution,
        int((max_azimuth - azimuth_resolution - min_azimuth) / azimuth_resolution) + 1,
        device=device,
    )
    elevations = torch.linspace(
        min_elevation,
        max_elevation,
        n_elevation_channels,
        device=device,
    )
    azim_elev = torch.meshgrid(azimuths, elevations, indexing="ij")
    azim_elev = torch.stack(azim_elev, dim=-1)
    azim_elev = azim_elev + torch.randn_like(azim_elev) * 0.001
    pc_range = torch.randn_like(azim_elev[..., 0]).abs() * 50 + 2
    pc_azim_elev_range = torch.cat([azim_elev, pc_range.unsqueeze(-1)], dim=-1)

    _pc_xyz = torch.stack(
        [
            torch.cos(azim_elev[..., 1].deg2rad())
            * torch.cos(azim_elev[..., 0].deg2rad())
            * pc_range,
            torch.cos(azim_elev[..., 1].deg2rad())
            * torch.sin(azim_elev[..., 0].deg2rad())
            * pc_range,
            torch.sin(azim_elev[..., 1].deg2rad()) * pc_range,
        ],
        dim=-1,
    )

    range_filter = pc_range <= pc_range.quantile(0.7)
    pc_azim_elev_range = pc_azim_elev_range[range_filter]

    pc_timeoffset = (torch.rand_like(pc_azim_elev_range[..., 0:1]) - 0.5) * 0.1
    pc_intensity = torch.rand_like(pc_azim_elev_range[..., 0:1])
    point_cloud = torch.cat([pc_azim_elev_range, pc_timeoffset, pc_intensity], dim=-1)

    return point_cloud, azimuths, elevations


def build_raster_pts(
    point_cloud: torch.Tensor,
    azimuths: torch.Tensor,
    elevations: torch.Tensor,
    azimuth_resolution: float,
    min_azimuth: float,
    tile_width: int,
    tile_height: int,
):
    """根据点云与瓦片设置生成 raster_pts 与边界/偏移。"""
    elevation_boundaries = torch.cat([
        elevations[0:1] - 1.0,
        (
            elevations[tile_height::tile_height]
            + elevations[tile_height - 1 : -1 : tile_height]
        )
        / 2,
        elevations[-1:] + 1.0,
    ])

    points_tile_ids, flatten_ids = map_points_to_lidar_tiles(
        points2d=point_cloud[None, :, :2],
        elev_boundaries=elevation_boundaries,
        tile_azim_resolution=azimuth_resolution * tile_width,
        min_azim=min_azimuth,
    )

    tile_offsets = points_mapping_offset_encode(
        points_tile_ids,
        1,
        math.ceil((azimuths[-1] - azimuths[0]) / (azimuth_resolution * tile_width)),
        len(elevations) // tile_height,
    )

    raster_pts = populate_image_from_points(
        point_cloud[None],
        image_width=len(azimuths),
        image_height=len(elevations),
        tile_width=tile_width,
        tile_height=tile_height,
        tile_offsets=tile_offsets,
        flatten_id=flatten_ids,
    )

    return raster_pts, elevation_boundaries
