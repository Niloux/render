"""Lidar initialization and rendering helpers."""

import math
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
from gsplat import (
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    SpinningDirection,
    compute_lidar_angles_to_columns_map,
    compute_lidar_tiling,
    rasterization,
    spherical_harmonics,
)

from data_types import Lidar
from mlp_decoder import MLPDecoder
from render_kernel import extract_camera_centers, invert_world2camera
from render_runtime import calculate_viewmats
from util import pano_to_lidar_with_intensities


class LidarBatchData(TypedDict):
    extrinsics_tensor: torch.Tensor
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt
    azimuths: torch.Tensor
    elevations: torch.Tensor
    elevation_boundaries: torch.Tensor
    image_width: int
    image_height: int
    tile_width: int
    tile_height: int
    min_azimuth: float
    max_azimuth: float
    min_elevation: float
    max_elevation: float
    azimuth_resolution: float
    ray_dirs_lidar: torch.Tensor
    pano_dirs_lidar: torch.Tensor
    depth_valid_mask: torch.Tensor


LidarData = Dict[str, LidarBatchData]


def build_lidar_coeffs(
    lidar: Lidar,
    device: torch.device,
    raster_pts: Optional[torch.Tensor] = None,
) -> RowOffsetStructuredSpinningLidarModelParametersExt:
    if raster_pts is not None:
        return build_lidar_coeffs_from_raster_pts(lidar, raster_pts, device)

    azimuth_span = float(lidar.max_azimuth) - float(lidar.min_azimuth)
    if azimuth_span <= 0:
        raise ValueError(
            f"lidar {lidar.id} azimuth范围非法: "
            f"min={lidar.min_azimuth} max={lidar.max_azimuth}"
        )
    if lidar.azimuth_resolution <= 0:
        raise ValueError(
            f"lidar {lidar.id} azimuth_resolution必须为正数，实际为{lidar.azimuth_resolution}"
        )

    if lidar.n_elevation_channels <= 0:
        raise ValueError(
            f"lidar {lidar.id} n_elevation_channels必须为正数，实际为{lidar.n_elevation_channels}"
        )

    image_width = lidar_image_width(
        lidar.min_azimuth,
        lidar.max_azimuth,
        lidar.azimuth_resolution,
        lidar.tile_width,
    )
    return _build_linear_lidar_coeffs(
        int(lidar.n_elevation_channels),
        image_width,
        lidar.min_azimuth,
        lidar.max_azimuth,
        lidar.min_elevation,
        lidar.max_elevation,
        lidar.azimuth_resolution,
        lidar.tile_width,
        lidar.tile_height,
        device,
    )


def lidar_image_width(
    min_azimuth: float,
    max_azimuth: float,
    azimuth_resolution: float,
    tile_width: int,
) -> int:
    azimuth_span = float(max_azimuth) - float(min_azimuth)
    image_width = max(1, int(math.ceil(azimuth_span / float(azimuth_resolution))))
    tile_width = max(1, int(tile_width))
    return tile_width * math.ceil(image_width / tile_width)


def build_lidar_coeffs_from_raster_pts(
    lidar: Lidar,
    raster_pts: torch.Tensor,
    device: torch.device,
) -> RowOffsetStructuredSpinningLidarModelParametersExt:
    grid = raster_pts
    if grid.dim() == 4:
        grid = grid[0]
    if grid.dim() != 3 or grid.shape[-1] < 2:
        raise ValueError(
            f"lidar raster_pts期望形状为[1,H,W,5]或[H,W,5]，实际为{tuple(raster_pts.shape)}"
        )

    grid = grid.to(device=device, dtype=torch.float32)
    row_elevations_rad = torch.deg2rad(grid[:, 0, 1].contiguous())
    column_azimuths_rad = torch.deg2rad(grid[0, :, 0].contiguous())
    row_azimuth_offsets_rad = torch.zeros(
        row_elevations_rad.shape[0], dtype=torch.float32, device=device
    )

    lidar_params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.CLOCKWISE,
    )
    angles_to_columns_map = compute_lidar_angles_to_columns_map(lidar_params)
    tiling = compute_lidar_tiling(
        lidar_params,
        n_bins_elevation=max(1, math.ceil(grid.shape[0] / int(lidar.tile_height))),
        max_pts_per_tile=int(lidar.tile_width) * int(lidar.tile_height),
        resolution_elevation=1600,
        densification_factor_azimuth=8,
    )
    return RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )


def build_lidar_ray_dirs_from_coeffs(
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_elevations = lidar_coeffs.row_elevations_rad
    column_azimuths = lidar_coeffs.column_azimuths_rad
    row_offsets = lidar_coeffs.row_azimuth_offsets_rad

    raw_azimuths = column_azimuths.unsqueeze(0) + row_offsets.unsqueeze(1)
    azimuths_rad = torch.remainder(raw_azimuths + math.pi, 2 * math.pi) - math.pi
    azimuths_rad = torch.where(
        torch.isclose(azimuths_rad, torch.full_like(azimuths_rad, -math.pi))
        & (raw_azimuths > 0),
        torch.full_like(azimuths_rad, math.pi),
        azimuths_rad,
    )
    elevations_rad = row_elevations[:, None].expand_as(azimuths_rad)

    azimuths = torch.rad2deg(column_azimuths)
    elevations = torch.rad2deg(row_elevations)
    bins_elevation = lidar_coeffs.tiling.n_bins_elevation
    elevation_boundaries = torch.cat([
        elevations[0:1] + 1.0,
        (
            elevations[bins_elevation::bins_elevation]
            + elevations[bins_elevation - 1 : -1 : bins_elevation]
        )
        / 2,
        elevations[-1:] - 1.0,
    ])

    ray_dirs_lidar = torch.stack(
        [
            torch.cos(azimuths_rad) * torch.cos(elevations_rad),
            torch.sin(azimuths_rad) * torch.cos(elevations_rad),
            torch.sin(elevations_rad),
        ],
        dim=-1,
    )
    ray_dirs_lidar = ray_dirs_lidar / (ray_dirs_lidar.norm(dim=-1, keepdim=True) + 1e-8)
    return ray_dirs_lidar.unsqueeze(0), azimuths, elevations, elevation_boundaries


def build_lidar_data(
    lidars: Optional[List[Lidar]],
    device: torch.device,
    raster_pts: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Lidar], LidarData]:
    if not lidars:
        return {}, {}

    lidar_by_id: Dict[str, Lidar] = {}
    lidar_data: LidarData = {}
    for lidar in lidars:
        lidar_by_id[lidar.id] = lidar
        lidar_coeffs = build_lidar_coeffs(lidar, device, raster_pts=raster_pts)
        ray_dirs_lidar, azimuths, elevations, elevation_boundaries = (
            build_lidar_ray_dirs_from_coeffs(lidar_coeffs)
        )
        image_height = int(ray_dirs_lidar.shape[1])
        image_width = int(ray_dirs_lidar.shape[2])

        if azimuths.numel() > 1:
            azimuth_resolution = torch.diff(azimuths).abs().median().item()
        else:
            azimuth_resolution = float(lidar.azimuth_resolution)

        depth_valid_mask = torch.ones(
            (image_height, image_width), device=device, dtype=torch.bool
        )

        lidar_data[lidar.id] = {
            "extrinsics_tensor": torch.tensor(
                [lidar.extrinsics], dtype=torch.float32, device=device
            ),
            "lidar_coeffs": lidar_coeffs,
            "azimuths": azimuths,
            "elevations": elevations,
            "elevation_boundaries": elevation_boundaries,
            "image_width": image_width,
            "image_height": image_height,
            "tile_width": lidar.tile_width,
            "tile_height": lidar.tile_height,
            "min_azimuth": float(lidar.min_azimuth),
            "max_azimuth": float(lidar.max_azimuth),
            "min_elevation": float(lidar.min_elevation),
            "max_elevation": float(lidar.max_elevation),
            "azimuth_resolution": azimuth_resolution,
            "ray_dirs_lidar": ray_dirs_lidar,
            "pano_dirs_lidar": ray_dirs_lidar.squeeze(0),
            "depth_valid_mask": depth_valid_mask,
        }

    return lidar_by_id, lidar_data


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
    _ = zero_velocities

    (
        render_means,
        render_quats,
        render_scales,
        render_opacities,
        render_colors,
    ) = render_params

    for lidar_id, lidar_cfg in lidar_data.items():
        ego_pitch = -0.0061
        ego_roll = 0.0443
        viewmats = calculate_viewmats(
            lidar_cfg["extrinsics_tensor"],
            ego_heading,
            ego_position,
            ego_pitch,
            ego_roll,
        )
        # print(f"viewmats wrong: {viewmats}")
        # import numpy as np

        # viewmats = np.load(
        #     "/home/saimo/work/streetcrafter_codes_for_train_0508/lidar_pose_debug_frame_0.npz"
        # )
        # viewmats = (
        #     torch.from_numpy(viewmats["train_viewmat"]).float().cuda().unsqueeze(0)
        # )
        # print(f"viewmats right: {viewmats}")

        # quit()
        has_mlp_decoder = mlp_decoder is not None
        if has_mlp_decoder:
            lidar_features = compute_lidar_mlp_features_from_colors(
                render_colors,
                feature_dim=mlp_decoder.feature_dim,
                batch_size=viewmats.shape[0],
            )
        else:
            lidar_features = compute_lidar_features_from_colors(
                render_colors, render_means, viewmats
            )

        Ks = (
            torch
            .eye(3, device=render_means.device, dtype=render_means.dtype)
            .unsqueeze(0)
            .expand(viewmats.shape[0], -1, -1)
            .contiguous()
        )
        rendered_feat, _, _ = rasterization(
            means=render_means,
            quats=render_quats,
            scales=render_scales,
            opacities=render_opacities.squeeze(-1),
            colors=lidar_features,
            viewmats=viewmats,
            Ks=Ks,
            width=lidar_cfg["image_width"],
            height=lidar_cfg["image_height"],
            near_plane=0.2,
            far_plane=300.0,
            radius_clip=0.0,
            sh_degree=None,
            packed=False,
            tile_size=16,
            render_mode="RGB-Ed",
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="antialiased",
            channel_chunk=128,
            camera_model="lidar",
            lidar_coeffs=lidar_cfg["lidar_coeffs"],
            with_ut=True,
            with_eval3d=True,
            global_z_order=False,
            eps2d=0.01718873385,
        )

        if has_mlp_decoder:
            rendered_feat = decode_lidar_features(
                lidar_id, rendered_feat, viewmats, lidar_cfg, mlp_decoder
            )
        out = process_lidar_output(
            rendered_feat, lidar_cfg["depth_valid_mask"], has_mlp_decoder
        )
        point_cloud = pano_to_lidar_with_intensities(
            out,
            directions=lidar_cfg["pano_dirs_lidar"],
            depth_valid_mask=lidar_cfg.get("depth_valid_mask", None),
        )
        lidars[lidar_id] = point_cloud

    return lidars


def decode_lidar_features(
    lidar_id: str,
    rendered_feat: torch.Tensor,
    viewmats: torch.Tensor,
    lidar_cfg: LidarBatchData,
    mlp_decoder: MLPDecoder,
) -> torch.Tensor:
    ray_dirs_lidar = lidar_cfg["ray_dirs_lidar"]
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
        ray_dirs_world=ray_dirs_world,
    )
    return torch.cat(
        [decoded_intensity, decoded_ray_drop_logits, rendered_feat[..., -1:]],
        dim=-1,
    )


def process_lidar_output(
    rendered_feat: torch.Tensor,
    depth_valid_mask: torch.Tensor,
    has_mlp_decoder: bool,
) -> Dict[str, torch.Tensor]:
    lidar_intensity = rendered_feat[..., 0].squeeze(0)
    if has_mlp_decoder:
        lidar_ray_drop_logits = rendered_feat[..., 1].squeeze(0)
    elif rendered_feat.shape[-1] >= 4:
        rayhit_logits = rendered_feat[..., 1].squeeze(0)
        raydrop_logits = rendered_feat[..., 2].squeeze(0)
        lidar_ray_drop_logits = raydrop_logits - rayhit_logits
    else:
        lidar_ray_drop_logits = rendered_feat[..., 1].squeeze(0)
    lidar_depth_render = rendered_feat[..., -1].squeeze(0)

    valid_returns = align_mask(depth_valid_mask, lidar_depth_render)
    gt_valid = valid_returns

    valid_mask = valid_returns.to(lidar_depth_render.dtype)
    lidar_intensity = lidar_intensity * valid_mask
    lidar_depth_render = lidar_depth_render * valid_mask

    gt_valid_mask = gt_valid.to(lidar_ray_drop_logits.dtype)
    lidar_ray_drop_logits = (
        lidar_ray_drop_logits * gt_valid_mask - (1.0 - gt_valid_mask) * 10000.0
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


def compute_lidar_features_from_colors(
    colors: torch.Tensor,
    means: torch.Tensor,
    viewmats: torch.Tensor,
) -> torch.Tensor:
    C = viewmats.shape[0]
    N = means.shape[0]
    device = means.device

    if colors.dim() != 3:
        return torch.zeros(C, N, 3, device=device, dtype=torch.float32)

    D = colors.shape[-1]
    K = colors.shape[1]
    if D < 3 or K < 4:
        return torch.zeros(C, N, 3, device=device, dtype=colors.dtype)

    camera_centers = extract_camera_centers(viewmats)
    dirs = means[None, :, :] - camera_centers[:, None, :]
    dirs = torch.nn.functional.normalize(dirs, dim=-1, eps=1e-8)
    shs = colors[..., :3].unsqueeze(0).expand(C, -1, -1, -1)
    feats = spherical_harmonics(1, dirs, shs)
    return feats[..., :3]


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


def _build_linear_lidar_coeffs(
    image_height: int,
    image_width: int,
    min_azimuth,
    max_azimuth,
    min_elevation,
    max_elevation,
    azimuth_resolution,
    tile_width,
    tile_height,
    device: torch.device,
) -> RowOffsetStructuredSpinningLidarModelParametersExt:
    image_height = int(image_height)
    image_width = int(image_width)

    def _scalar_float(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    min_azimuth = _scalar_float(min_azimuth)
    max_azimuth = _scalar_float(max_azimuth)
    min_elevation = _scalar_float(min_elevation)
    max_elevation = _scalar_float(max_elevation)
    azimuth_resolution = _scalar_float(azimuth_resolution)
    tile_width = int(_scalar_float(tile_width))
    tile_height = int(_scalar_float(tile_height))

    row_elevations_rad = torch.linspace(
        math.radians(max_elevation),
        math.radians(min_elevation),
        image_height,
        device=device,
        dtype=torch.float32,
    )
    azimuth_step_rad = math.radians(azimuth_resolution)
    if image_width > 1 and azimuth_step_rad * (image_width - 1) >= 2 * math.pi:
        azimuth_step_rad = (2 * math.pi - 1e-6) / (image_width - 1)
    column_azimuths_rad = (
        math.radians(max_azimuth)
        - torch.arange(image_width, device=device, dtype=torch.float32)
        * azimuth_step_rad
    )
    row_azimuth_offsets_rad = torch.zeros(
        image_height, device=device, dtype=torch.float32
    )
    lidar_params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.CLOCKWISE,
    )
    lidar_coeffs = RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params,
        compute_lidar_angles_to_columns_map(lidar_params),
        compute_lidar_tiling(
            lidar_params,
            n_bins_elevation=max(1, math.ceil(image_height / max(tile_height, 1))),
            max_pts_per_tile=max(1, tile_width * tile_height),
            resolution_elevation=1600,
            densification_factor_azimuth=8,
        ),
    )
    return lidar_coeffs
