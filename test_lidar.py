import math
import os

import torch
from gsplat.cuda._wrapper import (
    map_points_to_lidar_tiles,
    points_mapping_offset_encode,
    populate_image_from_points,
)
from gsplat.rendering import lidar_rasterization

import test_camera as TC
from config import MAP_CENTER
from data_types import GaussianData
from models import GSModel
from util import calculate_viewmats

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


def setup_device() -> torch.device:
    """返回用于执行的设备（需 GPU）。"""
    assert torch.cuda.is_available(), "GPU 不可用"
    return torch.device("cuda")


def setup_gaussians_from_model(device: torch.device, model_path: str):
    """从GSModel加载高斯数据并生成激光雷达特征。

    返回: means, quats, scales, opacities, lidar_features
    """
    if not (model_path and os.path.exists(model_path)):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = GSModel.load_from_pth(model_path).to_device(device)
    components = list(model.components.values())
    data = GaussianData.from_components(components)
    if data is None:
        raise RuntimeError("模型未包含任何高斯组件")
    means, quats, scales, opacities, colors = data.to_render_args()
    intensity = colors.reshape(colors.shape[0], -1).mean(dim=-1, keepdim=True)
    return means, quats, scales, opacities.squeeze(-1), intensity


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


def collect_gaussians_from_model_and_scenario(device: torch.device):
    """从模型与 test_camera 场景组装静态与动态高斯，并生成强度特征。"""
    model_path = TC.CONFIG["model_path"]
    model = GSModel.load_from_pth(model_path).to_device(device)

    background = model.get_component("background")
    sky = model.get_component("sky")
    actors = model.get_components_by_type("obj")
    actor_map = {a.name: a for a in actors}

    means_list = []
    quats_list = []
    scales_list = []
    opacities_list = []
    features_list = []

    for comp in [background, sky]:
        if comp is None:
            continue
        means_list.append(comp.get_xyz())
        quats_list.append(comp.get_quats())
        scales_list.append(comp.get_scales())
        opacities_list.append(comp.get_opacities())
        features_list.append(comp.get_colors().reshape(comp.num_points, -1).mean(dim=-1, keepdim=True))

    init_params, frame_params = TC.create_test_scenario()
    for v in frame_params.env_vehicles:
        comp = actor_map.get(v.type)
        if comp is None:
            continue
        position = torch.tensor(v.trajectory, device=device, dtype=torch.float32) - torch.tensor(
            MAP_CENTER, device=device
        )
        means_list.append(comp.get_xyz(v.yaw, position))
        quats_list.append(comp.get_quats(v.yaw))
        scales_list.append(comp.get_scales())
        opacities_list.append(comp.get_opacities())
        features_list.append(comp.get_colors().reshape(comp.num_points, -1).mean(dim=-1, keepdim=True))

    means = torch.cat(means_list)
    quats = torch.cat(quats_list)
    scales = torch.cat(scales_list)
    opacities = torch.cat(opacities_list).squeeze(-1)
    features = torch.cat(features_list)
    return means, quats, scales, opacities, features, init_params, frame_params


def build_raster_pts(
    point_cloud: torch.Tensor,
    azimuths: torch.Tensor,
    elevations: torch.Tensor,
    azimuth_resolution: float,
    min_azimuth: float,
    min_elevation: float,
    max_elevation: float,
    tile_width: int,
    tile_height: int,
):
    """根据点云与瓦片设置生成 raster_pts 与边界/偏移。"""
    n_elev_tiles = math.ceil(elevations.numel() / tile_height)
    elevation_boundaries = torch.linspace(
        min_elevation - 1.0,
        max_elevation + 1.0,
        n_elev_tiles + 1,
        device=elevations.device,
    )

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
        n_elev_tiles,
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
    velocities: torch.Tensor | None,
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
    C = viewmats.shape[0]
    features = features if features.dim() == 3 else features.unsqueeze(0).expand(C, -1, -1).contiguous()
    rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = lidar_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        lidar_features=features,
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


def save_rendered_feat_panorama(rendered_feat: torch.Tensor, out_dir: str, v_scale: int = 8):
    """保存渲染后的 LiDAR 全景图（上采样+归一化），更适合人眼观看。

    参数:
        rendered_feat: 形状 [H, W, D+1] 或 [C, H, W, D+1] 的张量。
                       - rendered_feat[..., 0]: 强度（建议范围 [0,1]）
                       - rendered_feat[..., -1]: 期望距离（需归一化）
        out_dir: 输出目录，例如 "outputs"。
        v_scale: 垂直方向上采样倍率，默认 8。32×8=256 更易读。

    生成:
        - panorama_intensity.png：上采样后的强度图，灰度。
        - panorama_expected_distance.png：上采样后的期望距离图，分位归一化灰度。
    """
    from PIL import Image
    import os
    import numpy as np
    import torch.nn.functional as F

    os.makedirs(out_dir, exist_ok=True)

    feat = rendered_feat
    if feat.dim() == 4:  # [C, H, W, *]
        feat = feat[0]

    # [H, W, *] -> [1, 1, H, W, *] 便于插值
    H, W = feat.shape[:2]
    intensity = feat[..., 0].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    expected_dist = feat[..., -1].unsqueeze(0).unsqueeze(0)

    # 仅对 H 做上采样，W 保持不变（也可按需横向缩放）
    new_H = int(H * v_scale)
    intensity_up = F.interpolate(intensity, size=(new_H, W), mode="bilinear", align_corners=False)
    expected_up = F.interpolate(expected_dist, size=(new_H, W), mode="bilinear", align_corners=False)

    # 强度裁剪到 [0,1]
    inten_img = intensity_up.squeeze().detach().cpu().numpy().astype(np.float32)
    inten_img = np.clip(inten_img, 0.0, 1.0)
    Image.fromarray((inten_img * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "panorama_intensity.png"))

    # 期望距离分位归一化到 [0,1]
    dist_img = expected_up.squeeze().detach().cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(dist_img, 1.0), np.percentile(dist_img, 99.0)
    dist_img = np.clip((dist_img - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    Image.fromarray((dist_img * 255.0).astype(np.uint8)).save(
        os.path.join(out_dir, "panorama_expected_distance.png")
    )


def save_bev_from_rendered_feat(
    rendered_feat: torch.Tensor,
    min_azimuth: float,
    max_azimuth: float,
    min_elevation: float,
    max_elevation: float,
    azimuth_resolution: float,
    out_path: str = "outputs/bev_intensity.png",
    meters_per_pixel: float = 0.2,
    bev_size: int = 512,
    use_alpha_weight: bool = True,
):
    """将渲染结果重投影为 BEV 鸟瞰图并保存，更符合人眼直觉。

    参数:
        rendered_feat: 渲染结果，形状 [H, W, D+1] 或 [C, H, W, D+1]。
                       - rendered_feat[..., 0]: 强度
                       - rendered_feat[..., -1]: 期望距离（当作 range 使用）
        min_azimuth, max_azimuth, min_elevation, max_elevation, azimuth_resolution:
            渲染所用的激光雷达参数，用于恢复每个像素的角度值。
        out_path: 输出 BEV 图片路径。
        meters_per_pixel: BEV 每像素空间分辨率（米/像素）。
        bev_size: BEV 图尺寸（方形，bev_size×bev_size）。
        use_alpha_weight: 若你也有 rendered_alpha，可用其权重提升视觉对比度（此函数假定强度已是体渲染结果，默认 True）。

    行为:
        - 用期望距离作为 range，将 [azim, elev] 转到 (x, y) 平面。
        - 在平面栅格累计强度（可做 max/mean，这里用 sum 简化）。
    """
    from PIL import Image
    import numpy as np

    feat = rendered_feat
    if feat.dim() == 4:  # [C, H, W, *]
        feat = feat[0]
    H, W = feat.shape[:2]

    # 恢复每个像素的角度网格（保证形状为 [H, W]）
    azims = torch.linspace(min_azimuth, max_azimuth - azimuth_resolution, W, device=feat.device).deg2rad()
    elevs = torch.linspace(min_elevation, max_elevation, H, device=feat.device).deg2rad()
    # 使用 indexing="ij" 并传入 (elevs, azims) 保证输出为 [H, W]
    elev_grid, azim_grid = torch.meshgrid(elevs, azims, indexing="ij")

    # 距离/强度
    ranges = feat[..., -1]  # [H, W]
    intens = feat[..., 0]   # [H, W]

    # 由球坐标到平面 (x,y)
    cos_e = torch.cos(elev_grid)
    x = cos_e * torch.cos(azim_grid) * ranges  # [H, W]
    y = cos_e * torch.sin(azim_grid) * ranges  # [H, W]

    # 将 (x,y) 映射到 BEV 栅格坐标
    u = (x / meters_per_pixel + bev_size / 2).long()
    v = (y / meters_per_pixel + bev_size / 2).long()
    mask = (u >= 0) & (u < bev_size) & (v >= 0) & (v < bev_size)

    bev = torch.zeros((bev_size, bev_size), device=feat.device, dtype=torch.float32)
    # 累计强度（sum），如需更锐利可改为最大值聚合
    bev.index_put_((u[mask], v[mask]), intens[mask], accumulate=True)
    bev = torch.clamp(bev, 0.0, 1.0)

    # 保存
    bev_img = bev.detach().cpu().numpy().astype(np.float32)
    bev_img8 = (bev_img * 255.0).astype(np.uint8)
    Image.fromarray(bev_img8).save(out_path)


def main():
    """整合 notebook 逻辑，执行栅格化并打印关键维度信息。"""
    torch.manual_seed(42)
    device = setup_device()

    (
        means,
        quats,
        scales,
        opacities,
        features,
        init_params,
        frame_params,
    ) = collect_gaussians_from_model_and_scenario(device)

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
        min_elevation,
        max_elevation,
        tile_width,
        tile_height,
    )

    cameras = init_params.cameras
    C = len(cameras)
    extrinsics_list = [cam.extrinsics for cam in cameras]
    ego_yaw = frame_params.ego_yaw
    ego_position = torch.tensor(frame_params.ego_trajectory, device=device, dtype=torch.float32) - torch.tensor(
        MAP_CENTER, device=device
    )
    viewmats = calculate_viewmats(extrinsics_list, ego_yaw, ego_position)
    if raster_pts.shape[0] != C:
        raster_pts = raster_pts.repeat(C, 1, 1, 1)

    rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = run_rasterization(
        means,
        quats,
        scales,
        opacities,
        features,
        None,
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
    print(f"{rendered_feat.shape=}")
    print(f"{rendered_alpha.shape=}")
    print(f"{alpha_sum_until_points.shape=}")

    # 更友好的可视化输出
    os.makedirs("outputs", exist_ok=True)
    save_rendered_feat_panorama(rendered_feat, out_dir="outputs", v_scale=8)
    save_bev_from_rendered_feat(
        rendered_feat,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        azimuth_resolution=azimuth_resolution,
        out_path="outputs/bev_intensity.png",
        meters_per_pixel=0.2,
        bev_size=512,
    )

if __name__ == "__main__":
    main()