"""Lidar initialization and rendering helpers."""

from typing import Dict, List, Optional, Tuple

import torch
from gsplat import spherical_harmonics
from gsplat.rendering import lidar_rasterization

from data_types import Lidar
from mlp_decoder import MLPDecoder
from render_kernel import extract_camera_centers, invert_world2camera
from render_runtime import calculate_viewmats
from util import pano_to_lidar_with_intensities


LidarData = Dict[str, Dict]


def prepare_lidar_grids_from_raster_pts(
    raster_pts: torch.Tensor, tile_height: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer lidar angular grids and tile boundaries from checkpoint raster_pts."""
    if not isinstance(raster_pts, torch.Tensor):
        raster_pts = torch.as_tensor(raster_pts)
    raster_pts = raster_pts.to(device=device, dtype=torch.float32)
    if raster_pts.dim() == 3:
        raster_pts = raster_pts.unsqueeze(0)
    if raster_pts.dim() != 4:
        raise ValueError(f"raster_pts维度必须为3或4，实际为{tuple(raster_pts.shape)}")

    az_grid = raster_pts[0, ..., 0]
    el_grid = raster_pts[0, ..., 1]

    az_axis = 0 if az_grid[:, 0].std() >= az_grid[0, :].std() else 1
    el_axis = 0 if el_grid[:, 0].std() >= el_grid[0, :].std() else 1
    if el_axis == az_axis:
        el_axis = 1 - az_axis

    azimuths = az_grid[:, 0] if az_axis == 0 else az_grid[0, :]
    elevations = el_grid[:, 0] if el_axis == 0 else el_grid[0, :]

    elevation_boundaries = torch.cat([
        elevations[0:1] - 1.0,
        (
            elevations[tile_height::tile_height]
            + elevations[tile_height - 1 : -1 : tile_height]
        )
        / 2,
        elevations[-1:] + 1.0,
    ])

    return raster_pts, azimuths, elevations, elevation_boundaries


def build_lidar_data(
    lidars: Optional[List[Lidar]],
    checkpoint_raster_pts: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, Lidar], LidarData]:
    if not lidars:
        return {}, {}
    if checkpoint_raster_pts is None:
        raise ValueError(
            "pth中缺少raster_pts，当前实现不再在运行时构建raster_pts；"
            "请在训练/导出时将raster_pts保存到checkpoint中"
        )

    lidar_by_id: Dict[str, Lidar] = {}
    lidar_data: LidarData = {}
    for lidar in lidars:
        lidar_by_id[lidar.id] = lidar
        raster_pts, azimuths, elevations, elevation_boundaries = (
            prepare_lidar_grids_from_raster_pts(
                checkpoint_raster_pts, tile_height=lidar.tile_height, device=device
            )
        )

        if azimuths.numel() > 1:
            azimuth_resolution = torch.diff(azimuths).abs().median().item()
        else:
            azimuth_resolution = float(lidar.azimuth_resolution)

        ray_dirs_lidar = build_lidar_ray_dirs(raster_pts)
        gt_depth = raster_pts[..., 2].squeeze(0)
        depth_valid_mask = (gt_depth > 0) & (gt_depth <= 1000.0)

        lidar_data[lidar.id] = {
            "extrinsics_tensor": torch.tensor(
                [lidar.extrinsics], dtype=torch.float32, device=device
            ),
            "azimuths": azimuths,
            "elevations": elevations,
            "elevation_boundaries": elevation_boundaries,
            "image_width": int(raster_pts.shape[1]),
            "image_height": int(raster_pts.shape[2]),
            "tile_width": lidar.tile_width,
            "tile_height": lidar.tile_height,
            "min_azimuth": float(lidar.min_azimuth),
            "max_azimuth": float(lidar.max_azimuth),
            "min_elevation": float(lidar.min_elevation),
            "max_elevation": float(lidar.max_elevation),
            "azimuth_resolution": azimuth_resolution,
            "raster_pts": raster_pts,
            "ray_dirs_lidar": ray_dirs_lidar,
            "pano_dirs_lidar": ray_dirs_lidar.squeeze(0),
            "depth_valid_mask": depth_valid_mask,
        }

    return lidar_by_id, lidar_data


def build_lidar_ray_dirs(raster_pts: torch.Tensor) -> torch.Tensor:
    angles = torch.deg2rad(raster_pts[..., :2])
    az = angles[..., 0:1]
    el = angles[..., 1:2]
    ray_dirs_lidar = torch.cat(
        [
            torch.cos(az) * torch.cos(el),
            torch.sin(az) * torch.cos(el),
            torch.sin(el),
        ],
        dim=-1,
    )
    return ray_dirs_lidar / (ray_dirs_lidar.norm(dim=-1, keepdim=True) + 1e-8)


def get_mlp_decoder_state_and_num_lidars(
    mlp_decoder_state: Optional[dict], lidars: Optional[List[Lidar]]
) -> Tuple[Optional[dict], int]:
    if not lidars:
        return None, 0
    if mlp_decoder_state is None:
        return None, len(lidars)

    params_state = mlp_decoder_state.get("params")
    if isinstance(params_state, dict) and "appearance_dim" in params_state:
        app = params_state["appearance_dim"]
        if isinstance(app, torch.Tensor) and app.ndim == 2:
            return mlp_decoder_state, int(app.shape[0])
    return mlp_decoder_state, len(lidars)


def init_mlp_decoder(
    render_lidar: bool,
    lidars: Optional[List[Lidar]],
    mlp_decoder_state: Optional[dict],
    device: torch.device,
) -> Tuple[Optional[MLPDecoder], Dict[str, int]]:
    if not render_lidar or not lidars:
        return None, {}

    lidar_id_to_index = {lidar.id: idx for idx, lidar in enumerate(lidars)}
    mlp_state, num_lidars_ckpt = get_mlp_decoder_state_and_num_lidars(
        mlp_decoder_state, lidars
    )
    if mlp_state is None:
        return None, lidar_id_to_index
    if len(lidars) > num_lidars_ckpt:
        raise ValueError(
            "激光雷达数量超过MLPDecoder权重支持范围: "
            f"lidars={len(lidars)} ckpt={num_lidars_ckpt}"
        )

    decoder = MLPDecoder.from_checkpoint_state(
        metadata={"num_lidars": num_lidars_ckpt},
        state_dict=mlp_state,
        device=device,
    )
    decoder.lidar_id_map = lidar_id_to_index
    decoder.eval()
    return decoder, lidar_id_to_index


def render_lidars(
    lidar_data: LidarData,
    render_enabled: bool,
    ego_heading: float,
    ego_position: torch.Tensor,
    render_params: Tuple[torch.Tensor, ...],
    mlp_decoder: Optional[MLPDecoder],
    zero_velocities: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    lidars = {}
    if not render_enabled:
        return lidars

    (
        render_means,
        render_quats,
        render_scales,
        render_opacities,
        render_colors,
    ) = render_params

    for lidar_id, lidar_cfg in lidar_data.items():
        viewmats = calculate_viewmats(
            lidar_cfg["extrinsics_tensor"], ego_heading, ego_position
        )
        if mlp_decoder is not None:
            lidar_features = compute_lidar_mlp_features_from_colors(
                render_colors,
                feature_dim=mlp_decoder.feature_dim,
                batch_size=viewmats.shape[0],
            )
        else:
            lidar_features = compute_lidar_features_from_colors(
                render_colors, render_means, viewmats
            )

        raster_pts = lidar_cfg["raster_pts"]
        rendered_feat, _, _, _ = lidar_rasterization(
            means=render_means,
            quats=render_quats,
            scales=render_scales,
            opacities=render_opacities.squeeze(-1),
            lidar_features=lidar_features,
            velocities=zero_velocities[: render_means.shape[0]],
            viewmats=viewmats,
            raster_pts=raster_pts[..., :4],
            tile_elevation_boundaries=lidar_cfg["elevation_boundaries"],
            min_azimuth=lidar_cfg["min_azimuth"],
            max_azimuth=lidar_cfg["max_azimuth"],
            min_elevation=lidar_cfg["min_elevation"],
            max_elevation=lidar_cfg["max_elevation"],
            n_elevation_channels=lidar_cfg["elevations"].shape[0],
            azimuth_resolution=lidar_cfg["azimuth_resolution"],
            tile_width=lidar_cfg["tile_width"],
            tile_height=lidar_cfg["tile_height"],
            linear_velocity=torch.zeros(
                (1, 3), device=render_means.device, dtype=render_means.dtype
            ),
            angular_velocity=torch.zeros(
                (1, 3), device=render_means.device, dtype=render_means.dtype
            ),
            rolling_shutter_time=torch.zeros(
                (1,), device=render_means.device, dtype=render_means.dtype
            ),
            near_plane=0.2,
            far_plane=300.0,
            radius_clip=0.5,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="antialiased",
            channel_chunk=128,
            eps2d=0.01718873385,
            compute_alpha_sum_until_points=False,
            compute_alpha_sum_until_points_threshold=0.8,
        )

        if mlp_decoder is not None:
            rendered_feat = decode_lidar_features(
                lidar_id, rendered_feat, raster_pts, viewmats, lidar_cfg, mlp_decoder
            )
        out = process_lidar_output(raster_pts, rendered_feat)
        lidars[lidar_id] = pano_to_lidar_with_intensities(
            raster_pts,
            out,
            directions=lidar_cfg.get("pano_dirs_lidar", None),
            depth_valid_mask=lidar_cfg.get("depth_valid_mask", None),
        )

    return lidars


def decode_lidar_features(
    lidar_id: str,
    rendered_feat: torch.Tensor,
    raster_pts: torch.Tensor,
    viewmats: torch.Tensor,
    lidar_cfg: Dict,
    mlp_decoder: MLPDecoder,
) -> torch.Tensor:
    ray_dirs_world = None
    ray_dirs_lidar = lidar_cfg.get("ray_dirs_lidar", None)
    if isinstance(ray_dirs_lidar, torch.Tensor):
        lidar_to_world = invert_world2camera(viewmats)
        R = lidar_to_world[:, :3, :3]
        B, H, W = (
            int(ray_dirs_lidar.shape[0]),
            int(ray_dirs_lidar.shape[1]),
            int(ray_dirs_lidar.shape[2]),
        )
        dirs_flat = ray_dirs_lidar.view(B, -1, 3)
        ray_world_flat = torch.bmm(dirs_flat, R.transpose(1, 2))
        ray_dirs_world = ray_world_flat.view(B, H, W, 3)

    decoded_intensity, decoded_ray_drop_logits = mlp_decoder(
        lidar_id,
        rendered_feat[..., :-1],
        raster_pts,
        viewmats,
        ray_dirs_world=ray_dirs_world,
    )
    return torch.cat(
        [decoded_intensity, decoded_ray_drop_logits, rendered_feat[..., -1:]],
        dim=-1,
    )


def process_lidar_output(
    raster_pts: torch.Tensor, rendered_feat: torch.Tensor
) -> Dict[str, torch.Tensor]:
    lidar_intensity = rendered_feat[..., 0].squeeze(0)
    lidar_ray_drop_logits = rendered_feat[..., 1].squeeze(0)
    lidar_depth_render = rendered_feat[..., -1].squeeze(0)

    valid_returns = ((raster_pts[..., 2] <= 1000) & (raster_pts[..., 2] > 0)).squeeze(0)
    gt_valid = (raster_pts[..., 2] > 0).squeeze(0)
    valid_returns = align_mask(valid_returns, lidar_depth_render)
    gt_valid = align_mask(gt_valid, lidar_depth_render)

    valid_mask = valid_returns.to(lidar_depth_render.dtype)
    lidar_intensity = lidar_intensity * valid_mask
    lidar_depth_render = lidar_depth_render * valid_mask

    gt_valid_mask = gt_valid.to(lidar_ray_drop_logits.dtype)
    lidar_ray_drop_logits = (
        lidar_ray_drop_logits * gt_valid_mask - (1.0 - gt_valid_mask) * 10000.0
    )

    lidar_depth_render, lidar_intensity = apply_depth_filter(
        lidar_depth_render, lidar_intensity
    )
    return {
        "depth": lidar_depth_render,
        "intensity": lidar_intensity,
        "ray_drop_prob": lidar_ray_drop_logits,
    }


def align_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.shape == target.shape:
        return mask
    if mask.transpose(0, 1).shape == target.shape:
        return mask.transpose(0, 1)
    if mask.numel() == target.numel():
        return mask.reshape(target.shape)
    return mask


def apply_depth_filter(
    depth: torch.Tensor, intensity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_diff = torch.abs(depth[:, 1:] - depth[:, :-1])
    full_mask = torch.ones_like(depth)
    full_mask[:, 1:] = (depth_diff <= 0.5).float()
    return depth * full_mask, intensity * full_mask


def compute_lidar_features_from_colors(
    colors: torch.Tensor,
    means: torch.Tensor,
    viewmats: torch.Tensor,
) -> torch.Tensor:
    C = viewmats.shape[0]
    N = means.shape[0]
    device = means.device

    if colors.dim() != 3:
        return torch.zeros(C, N, 2, device=device, dtype=torch.float32)

    D = colors.shape[-1]
    K = colors.shape[1]
    if D < 5 or K < 4:
        return torch.zeros(C, N, 2, device=device, dtype=colors.dtype)

    camera_centers = extract_camera_centers(viewmats)
    dirs = means[None, :, :] - camera_centers[:, None, :]
    lidar_coeffs = colors[..., 3:5]
    zeros_third = torch.zeros((N, K, 1), device=device, dtype=colors.dtype)
    lidar_coeffs_pad = torch.cat([lidar_coeffs, zeros_third], dim=-1)
    shs = lidar_coeffs_pad.unsqueeze(0).expand(C, -1, -1, -1)
    feats = spherical_harmonics(1, dirs, shs)
    return feats[..., :2]


def compute_lidar_mlp_features_from_colors(
    colors: torch.Tensor,
    feature_dim: int,
    batch_size: int = 1,
) -> torch.Tensor:
    if feature_dim <= 0:
        raise ValueError(f"feature_dim必须为正数，实际为{feature_dim}")

    device = colors.device
    dtype = colors.dtype
    if colors.dim() != 3:
        N = int(colors.shape[0]) if hasattr(colors, "shape") else 0
        return torch.zeros(batch_size, N, feature_dim, device=device, dtype=dtype)

    N = colors.shape[0]
    C = colors.shape[-1]
    if C <= 3:
        return torch.zeros(batch_size, N, feature_dim, device=device, dtype=dtype)

    flat = colors[..., 3:].reshape(N, -1)
    if flat.shape[1] < feature_dim:
        pad = torch.zeros(N, feature_dim - flat.shape[1], device=device, dtype=dtype)
        flat = torch.cat([flat, pad], dim=-1)
    feat = flat[:, :feature_dim]
    return feat.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
