import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from splatad.gsplat.cuda._wrapper import (
    map_points_to_lidar_tiles,
    points_mapping_offset_encode,
    populate_image_from_points,
)
from splatad.gsplat.rendering import lidar_rasterization

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


def setup_device() -> torch.device:
    """返回用于执行的设备（需 GPU）。"""
    assert torch.cuda.is_available(), "GPU 不可用"
    return torch.device("cuda")


def setup_gaussians(N: int, device: torch.device):
    """构建随机高斯场景（位置、四元数、尺度、不透明度、特征、速度）。"""
    means = (torch.rand(N, 1, device=device) * 100 + 20) * F.normalize(torch.randn(N, 3, device=device), dim=-1)
    quats = F.normalize(torch.randn(N, 4, device=device))
    scales = torch.rand(N, 3, device=device) + 0.1
    opacities = torch.rand(N, device=device)
    features = torch.randn(N, 16, device=device)
    velocities = torch.randn(N, 3, device=device) * 2
    return means, quats, scales, opacities, features, velocities


def setup_lidar_params():
    """返回激光雷达参数（方位分辨率、角度范围与通道数）。"""
    azimuth_resolution = 0.2
    min_azimuth = -180
    max_azimuth = 180
    n_elevation_channels = 32
    min_elevation = -30
    max_elevation = 30
    return (
        azimuth_resolution,
        min_azimuth,
        max_azimuth,
        n_elevation_channels,
        min_elevation,
        max_elevation,
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

    pc_xyz = torch.stack(
        [
            torch.cos(azim_elev[..., 1].deg2rad()) * torch.cos(azim_elev[..., 0].deg2rad()) * pc_range,
            torch.cos(azim_elev[..., 1].deg2rad()) * torch.sin(azim_elev[..., 0].deg2rad()) * pc_range,
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
        (elevations[tile_height::tile_height] + elevations[tile_height - 1 : -1 : tile_height]) / 2,
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


def run_rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    velocities: torch.Tensor,
    viewmats: torch.Tensor,
    raster_pts: torch.Tensor,
    elevation_boundaries: torch.Tensor,
    min_azimuth: float,
    max_azimuth: float,
    min_elevation: float,
    max_elevation: float,
    n_elevation_channels: int,
    azimuth_resolution: float,
    tile_width: int,
    tile_height: int,
):
    """执行激光雷达栅格化并返回主要结果。"""
    rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = lidar_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        lidar_features=features.unsqueeze(0),
        velocities=velocities,
        viewmats=viewmats,
        raster_pts=raster_pts[..., :4],
        tile_elevation_boundaries=elevation_boundaries.clone(),
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        n_elevation_channels=n_elevation_channels,
        azimuth_resolution=azimuth_resolution,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    return rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info


def postprocess_predictions(
    rendered_feat: torch.Tensor,
    rendered_alpha: torch.Tensor,
    raster_pts: torch.Tensor,
    meta_info: dict,
):
    """从渲染结果中提取有效像素并返回球坐标、xyz 坐标与深度图。"""
    rendered_expected_depth = rendered_feat[..., -1]
    rendered_median_depth = meta_info["median_depths"]
    rendered_feat = rendered_feat[..., :-1]

    valid_pixels = (rendered_alpha > 0).any(dim=-1)
    valid_raster_pts = raster_pts[valid_pixels]
    valid_rendered_expected_depth = rendered_expected_depth[valid_pixels]
    valid_rendered_median_depth = rendered_median_depth[valid_pixels].squeeze(-1)

    pred_points_spherical = torch.cat([valid_raster_pts[..., :2], valid_rendered_expected_depth[:, None]], dim=-1)

    pred_points_xyz_expected_depth = torch.stack(
        [
            torch.cos(pred_points_spherical[..., 1].deg2rad())
            * torch.cos(pred_points_spherical[..., 0].deg2rad())
            * pred_points_spherical[..., 2],
            torch.cos(pred_points_spherical[..., 1].deg2rad())
            * torch.sin(pred_points_spherical[..., 0].deg2rad())
            * pred_points_spherical[..., 2],
            torch.sin(pred_points_spherical[..., 1].deg2rad()) * pred_points_spherical[..., 2],
        ],
        dim=-1,
    )

    pred_points_xyz_median_depth = torch.stack(
        [
            torch.cos(pred_points_spherical[..., 1].deg2rad())
            * torch.cos(pred_points_spherical[..., 0].deg2rad())
            * valid_rendered_median_depth,
            torch.cos(pred_points_spherical[..., 1].deg2rad())
            * torch.sin(pred_points_spherical[..., 0].deg2rad())
            * valid_rendered_median_depth,
            torch.sin(pred_points_spherical[..., 1].deg2rad()) * valid_rendered_median_depth,
        ],
        dim=-1,
    )

    return (
        pred_points_spherical,
        pred_points_xyz_expected_depth,
        pred_points_xyz_median_depth,
        rendered_expected_depth,
        rendered_median_depth,
    )


def save_visualizations(
    output_dir: str,
    raster_pts: torch.Tensor,
    expected_depth_map: torch.Tensor,
    median_depth_map: torch.Tensor,
    pred_points_spherical: torch.Tensor,
    pred_points_xyz_expected_depth: torch.Tensor,
    pred_points_xyz_median_depth: torch.Tensor,
):
    """保存栅格化与预测结果的可视化图片到指定目录。"""
    os.makedirs(output_dir, exist_ok=True)

    range_img = raster_pts[0, ..., 2].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Range Image")
    im = ax.imshow(range_img, cmap="gray")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(output_dir, "raster_range.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    if raster_pts.shape[-1] >= 5:
        intensity_img = raster_pts[0, ..., 4].detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Intensity Image")
        im = ax.imshow(intensity_img, cmap="gray")
        fig.colorbar(im, ax=ax)
        fig.savefig(os.path.join(output_dir, "raster_intensity.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    exp_depth = expected_depth_map[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Expected Depth Map")
    im = ax.imshow(exp_depth, cmap="viridis")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(output_dir, "expected_depth_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    med_depth = median_depth_map[0].detach().cpu().numpy().squeeze(-1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Median Depth Map")
    im = ax.imshow(med_depth, cmap="viridis")
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(output_dir, "median_depth_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    sph = pred_points_spherical.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Spherical Scatter (Expected Depth)")
    sc = ax.scatter(sph[:, 0], sph[:, 1], s=0.2, c=sph[:, 2], cmap="viridis")
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax)
    fig.savefig(
        os.path.join(output_dir, "spherical_expected_depth_scatter.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    exp_xyz = pred_points_xyz_expected_depth.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Predicted Point Cloud (Expected Depth)")
    ax.scatter(exp_xyz[:, 0], exp_xyz[:, 1], exp_xyz[:, 2], s=0.2)
    fig.savefig(
        os.path.join(output_dir, "pred_cloud_expected_depth_3d.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    med_xyz = pred_points_xyz_median_depth.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Predicted Point Cloud (Median Depth)")
    ax.scatter(med_xyz[:, 0], med_xyz[:, 1], med_xyz[:, 2], s=0.2)
    fig.savefig(
        os.path.join(output_dir, "pred_cloud_median_depth_3d.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    """整合 notebook 逻辑，执行栅格化并打印关键维度信息。"""
    torch.manual_seed(42)
    device = setup_device()

    C, N = 1, 1_000_000
    (
        means,
        quats,
        scales,
        opacities,
        features,
        velocities,
    ) = setup_gaussians(N, device)

    (
        azimuth_resolution,
        min_azimuth,
        max_azimuth,
        n_elevation_channels,
        min_elevation,
        max_elevation,
    ) = setup_lidar_params()

    point_cloud, azimuths, elevations = generate_point_cloud(
        azimuth_resolution,
        min_azimuth,
        max_azimuth,
        n_elevation_channels,
        min_elevation,
        max_elevation,
        device,
    )

    tile_width, tile_height = 64, 4
    raster_pts, elevation_boundaries = build_raster_pts(
        point_cloud,
        azimuths,
        elevations,
        azimuth_resolution,
        min_azimuth,
        tile_width,
        tile_height,
    )

    viewmats = torch.eye(4, device=device).unsqueeze(0)

    rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = run_rasterization(
        means,
        quats,
        scales,
        opacities,
        features,
        velocities,
        viewmats,
        raster_pts,
        elevation_boundaries,
        min_azimuth,
        max_azimuth,
        min_elevation,
        max_elevation,
        n_elevation_channels,
        azimuth_resolution,
        tile_width,
        tile_height,
    )

    (
        pred_points_spherical,
        pred_points_xyz_expected_depth,
        pred_points_xyz_median_depth,
        expected_depth_map,
        median_depth_map,
    ) = postprocess_predictions(rendered_feat, rendered_alpha, raster_pts, meta_info)

    save_visualizations(
        os.path.join("output"),
        raster_pts,
        expected_depth_map,
        median_depth_map,
        pred_points_spherical,
        pred_points_xyz_expected_depth,
        pred_points_xyz_median_depth,
    )

    print({
        "rendered_feat": tuple(rendered_feat.shape),
        "rendered_alpha": tuple(rendered_alpha.shape),
        "raster_pts": tuple(raster_pts.shape),
        "pred_points_spherical": tuple(pred_points_spherical.shape),
        "pred_points_xyz_expected_depth": tuple(pred_points_xyz_expected_depth.shape),
        "pred_points_xyz_median_depth": tuple(pred_points_xyz_median_depth.shape),
    })


if __name__ == "__main__":
    main()
