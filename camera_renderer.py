"""Camera initialization and rendering helpers."""

from typing import Dict, List, Optional, Tuple, TypedDict

import torch

from bilateral_grids import BilGrids, get_pixel_coords
from data_types import Camera, GaussianData
from render_kernel import (
    get_ray_dirs_cam_pinhole_batched,
    invert_world2camera,
    render,
    render_sky_cubemap,
)
from render_runtime import calculate_viewmats
from rgb_decoder import RGBDecoder


class CameraBatchData(TypedDict):
    camera_ids: List[str]
    extrinsics_tensor: torch.Tensor
    intrinsics_tensor: torch.Tensor
    width: int
    height: int
    ray_dirs_cam: torch.Tensor


CameraGroups = Dict[Tuple[int, int], List[Camera]]
CameraData = Dict[Tuple[int, int], CameraBatchData]


def group_cameras_by_resolution(cameras: List[Camera]) -> CameraGroups:
    grouped: CameraGroups = {}
    for cam in cameras:
        resolution = (cam.width, cam.height)
        grouped.setdefault(resolution, []).append(cam)
    return grouped


def get_rgb_decoder_state_and_num_cams(
    rgb_decoder_state: Optional[dict], cameras: List[Camera]
) -> Tuple[Optional[dict], int]:
    if rgb_decoder_state is None or not isinstance(rgb_decoder_state, dict):
        return None, len(cameras)
    params_state = rgb_decoder_state.get("params")
    if isinstance(params_state, dict) and "appearance_embedding" in params_state:
        return rgb_decoder_state, int(params_state["appearance_embedding"].shape[0])
    return rgb_decoder_state, len(cameras)


def init_rgb_decoder(
    cameras: List[Camera],
    num_cams_ckpt: int,
    rgb_decoder_state: Optional[dict],
    device: torch.device,
) -> Tuple[Optional[RGBDecoder], Dict[str, int]]:
    camera_id_to_index = {cam.id: idx for idx, cam in enumerate(cameras)}
    if rgb_decoder_state is None:
        return None, camera_id_to_index

    decoder = RGBDecoder.from_checkpoint_state(
        metadata={"num_cams": num_cams_ckpt},
        state_dict=rgb_decoder_state,
        device=device,
        mode="image",
    )
    decoder.camera_id_map = camera_id_to_index
    decoder.eval()
    return decoder, camera_id_to_index


def init_bil_grids(
    cameras: List[Camera],
    bil_grids_state: Optional[dict],
    device: torch.device,
) -> Optional[BilGrids]:
    if bil_grids_state is None:
        return None
    return BilGrids.from_checkpoint_state(
        bil_grids_state,
        [camera.id for camera in cameras],
        device,
    )


def build_camera_data(cameras: CameraGroups, device: torch.device) -> CameraData:
    camera_data: CameraData = {}
    for resolution, camera_group in cameras.items():
        width, height = resolution
        Ks = torch.tensor(
            [cam.intrinsics for cam in camera_group],
            dtype=torch.float32,
            device=device,
        )
        extrinsics_tensor = torch.tensor(
            [cam.extrinsics for cam in camera_group],
            dtype=torch.float32,
            device=device,
        )
        camera_data[resolution] = {
            "camera_ids": [cam.id for cam in camera_group],
            "extrinsics_tensor": extrinsics_tensor,
            "intrinsics_tensor": Ks,
            "width": width,
            "height": height,
            "ray_dirs_cam": get_ray_dirs_cam_pinhole_batched(Ks, width, height),
        }
    return camera_data


def render_cameras(
    camera_data: CameraData,
    render_enabled: bool,
    ego_heading: float,
    ego_position: torch.Tensor,
    render_params: Tuple[torch.Tensor, ...],
    rgb_decoder: Optional[RGBDecoder],
    bil_grids: Optional[BilGrids],
    sky_data: Optional[GaussianData],
    sky_points: int,
    sky_cubemap: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    images = {}
    if not render_enabled:
        return images

    (
        render_means,
        render_quats,
        render_scales,
        render_opacities,
        render_colors,
    ) = render_params

    for cam_data in camera_data.values():
        camera_ids = cam_data["camera_ids"]
        extrinsics_tensor = cam_data["extrinsics_tensor"]
        Ks = cam_data["intrinsics_tensor"]
        width = cam_data["width"]
        height = cam_data["height"]

        viewmats = calculate_viewmats(extrinsics_tensor, ego_heading, ego_position)
        ray_dirs_world, ray_dirs_world_sky = build_camera_ray_dirs(
            cam_data, viewmats, rgb_decoder, sky_cubemap
        )

        batch_colors, batch_alphas = render(
            render_means,
            render_quats,
            render_scales,
            render_opacities,
            render_colors,
            viewmats,
            Ks,
            width,
            height,
            rgb_decoder=rgb_decoder,
            camera_ids=camera_ids,
            ray_dirs_world=ray_dirs_world,
        )

        if sky_data is not None and sky_points > 0:
            sky_colors, _ = render(
                sky_data.means,
                sky_data.quats,
                sky_data.scales,
                sky_data.opacities,
                sky_data.colors,
                viewmats,
                Ks,
                width,
                height,
                rgb_decoder=rgb_decoder,
                camera_ids=camera_ids,
                ray_dirs_world=ray_dirs_world,
            )
            batch_colors = batch_colors + sky_colors * (1 - batch_alphas)
        elif sky_cubemap is not None:
            if ray_dirs_world_sky is None:
                raise RuntimeError("cubemap天空渲染需要ray_dirs_world_sky，但当前未生成")
            sky_colors = render_sky_cubemap(sky_cubemap, ray_dirs_world_sky)
            batch_colors = batch_colors + sky_colors * (1 - batch_alphas)

        if bil_grids is not None:
            pixel_coords = get_pixel_coords(width, height, batch_colors.device)
            corrected = []
            for cam_id, image in zip(camera_ids, batch_colors):
                corrected_image, _ = bil_grids(cam_id, pixel_coords, image)
                corrected.append(corrected_image)
            batch_colors = torch.stack(corrected, dim=0)

        batch_colors = (batch_colors.clamp(0, 1) * 255).to(torch.uint8)
        for cam_id, image in zip(camera_ids, batch_colors):
            images[cam_id] = image

    return images


def build_camera_ray_dirs(
    cam_data: CameraBatchData,
    viewmats: torch.Tensor,
    rgb_decoder: Optional[RGBDecoder],
    sky_cubemap: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    ray_dirs_world = None
    ray_dirs_world_sky = None
    need_ray_dirs_for_sky = sky_cubemap is not None
    need_ray_dirs_for_decoder = rgb_decoder is not None and getattr(
        rgb_decoder, "use_ray_dirs", False
    )
    if not (need_ray_dirs_for_sky or need_ray_dirs_for_decoder):
        return None, None

    dirs_cam = cam_data["ray_dirs_cam"]
    c2w = invert_world2camera(viewmats)
    R = c2w[:, :3, :3]
    C = int(dirs_cam.shape[0])
    height = int(cam_data["height"])
    width = int(cam_data["width"])

    if need_ray_dirs_for_decoder:
        dirs_flat = dirs_cam.view(C, -1, 3)
        ray_world_flat = torch.bmm(dirs_flat, R.transpose(1, 2))
        ray_dirs_world = ray_world_flat.view(C, height, width, 3)

    if need_ray_dirs_for_sky:
        flip = torch.tensor(
            [1.0, -1.0, -1.0],
            device=dirs_cam.device,
            dtype=dirs_cam.dtype,
        )
        dirs_cam_sky = dirs_cam * flip.view(1, 1, 1, 3)
        dirs_flat_sky = dirs_cam_sky.view(C, -1, 3)
        ray_world_flat_sky = torch.bmm(dirs_flat_sky, R.transpose(1, 2))
        ray_dirs_world_sky = ray_world_flat_sky.view(C, height, width, 3)

    return ray_dirs_world, ray_dirs_world_sky
