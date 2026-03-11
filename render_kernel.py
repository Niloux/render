import math
from typing import List, Optional, Sequence

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
)

NATIVE = False


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


def invert_world2camera(viewmats: torch.Tensor) -> torch.Tensor:
    """将World2Camera的4x4矩阵批量求逆，得到Camera2World矩阵。"""
    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3:4]
    Rt = R.transpose(-2, -1)
    t_inv = -Rt @ t
    c2w = (
        torch
        .eye(4, device=viewmats.device, dtype=viewmats.dtype)
        .expand(*viewmats.shape[:-2], 4, 4)
        .clone()
    )
    c2w[..., :3, :3] = Rt
    c2w[..., :3, 3:4] = t_inv
    return c2w


def get_ray_dirs_pinhole_batched(
    Ks: torch.Tensor, width: int, height: int, c2w: torch.Tensor
) -> torch.Tensor:
    """为每个相机生成归一化的世界系射线方向，输出形状为[C, H, W, 3]。"""
    device = Ks.device
    dtype = Ks.dtype
    C = Ks.shape[0]

    fx = Ks[:, 0, 0].view(C, 1, 1)
    fy = Ks[:, 1, 1].view(C, 1, 1)
    cx = Ks[:, 0, 2].view(C, 1, 1)
    cy = Ks[:, 1, 2].view(C, 1, 1)

    ys = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)

    ys = (ys + 0.5 - cy) / fy
    xs = (xs + 0.5 - cx) / fx

    xs = xs.expand(C, height, width)
    ys = ys.expand(C, height, width)

    dirs_cam = torch.stack([xs, -ys, -torch.ones_like(xs)], dim=-1)  # [C, H, W, 3]
    dirs_cam = dirs_cam.view(C, -1, 3)

    R = c2w[:, :3, :3]  # [C, 3, 3]
    dirs_world = torch.bmm(dirs_cam, R.transpose(1, 2))
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)
    return dirs_world.view(C, height, width, 3)


def render_gaussian_splatting(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    img_width,
    img_height,
    rgb_decoder=None,
    camera_ids: Optional[Sequence[str]] = None,
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
    # 保持与rasterization一致：当颜色包含额外通道时，只取相机渲染用的前三个通道
    if colors.dim() == 3 and colors.shape[1] == 4 and colors.shape[2] >= 3:
        colors = colors[..., :3]
    if colors.dim() != 3 or colors.shape[1] != 4 or colors.shape[2] != 3:
        raise ValueError(
            f"CNN渲染分支期望colors形状为[N, 4, 3] (sh_degree=1)，实际为{tuple(colors.shape)}"
        )

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
    _tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
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

    # 将每个点的SH系数展平为每个点的特征，直接光栅化到像素
    shs = colors.expand(viewmats.shape[0], -1, -1, -1)  # [C, N, K, 3]
    colors = shs.contiguous().view(viewmats.shape[0], shs.shape[1], -1)

    # 光栅化
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
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

    if rgb_decoder is None:
        raise ValueError("启用CNN渲染分支时必须传入rgb_decoder实例")
    if camera_ids is None:
        raise ValueError("启用CNN渲染分支时必须传入camera_ids")
    if len(camera_ids) != viewmats.shape[0]:
        raise ValueError(
            f"camera_ids数量与viewmats不一致: camera_ids={len(camera_ids)} viewmats={viewmats.shape[0]}"
        )

    c2w = invert_world2camera(viewmats)
    ray_dirs = get_ray_dirs_pinhole_batched(Ks, img_width, img_height, c2w)
    features = torch.cat([render_colors, ray_dirs], dim=-1)  # [C, H, W, 12+3]

    decoded: List[torch.Tensor] = []
    for cam_idx, cam_id in enumerate(camera_ids):
        decoded.append(rgb_decoder(cam_id, features[cam_idx]))
    rendered_rgb = torch.stack(decoded, dim=0)
    return rendered_rgb, render_alphas


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
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    img_width,
    img_height,
    rgb_decoder=None,
    camera_ids: Optional[Sequence[str]] = None,
):
    if NATIVE:
        return render_native(
            means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
        )
    else:
        return render_gaussian_splatting(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            img_width,
            img_height,
            rgb_decoder=rgb_decoder,
            camera_ids=camera_ids,
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
