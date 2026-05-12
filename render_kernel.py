import os
from typing import List, Optional, Sequence

import torch
from gsplat import (
    rasterization,
)

_RENDER_TILE_SIZE = int(os.environ.get("RENDER_TILE_SIZE", "16"))
_RENDER_RASTERIZE_MODE = os.environ.get("RENDER_RASTERIZE_MODE", "antialiased")
_RENDER_RADIUS_CLIP = float(os.environ.get("RENDER_RADIUS_CLIP", "3.0"))


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


def compute_lidar_ray_dirs_world(
    raster_pts: torch.Tensor, viewmats: torch.Tensor
) -> torch.Tensor:
    """根据raster_pts中的(azimuth,elevation)计算世界系射线方向。

    Args:
        raster_pts: [B, H, W, D] 或 [H, W, D]，其中前两维为 azimuth/elevation（单位：度）。
        viewmats: [B, 4, 4] 的 World2Lidar 变换矩阵。

    Returns:
        ray_dirs_world: [B, H, W, 3] 的单位方向向量。
    """  # noqa: E501
    if raster_pts.dim() == 3:
        raster_pts = raster_pts.unsqueeze(0)
    if viewmats.dim() == 2:
        viewmats = viewmats.unsqueeze(0)

    if raster_pts.dim() != 4 or viewmats.dim() != 3:
        raise ValueError(
            f"raster_pts/viewmats维度不匹配: raster_pts={tuple(raster_pts.shape)} viewmats={tuple(viewmats.shape)}"  # noqa: E501
        )

    angles = torch.deg2rad(raster_pts[..., :2])
    az = angles[..., 0:1]
    el = angles[..., 1:2]

    dirs_lidar = torch.cat(
        [
            torch.cos(az) * torch.cos(el),
            torch.sin(az) * torch.cos(el),
            torch.sin(el),
        ],
        dim=-1,
    )

    lidar_to_world = invert_world2camera(viewmats)
    R = lidar_to_world[:, :3, :3]  # [B, 3, 3]
    ray_dirs_world = (
        R.reshape(R.shape[0], 1, 1, 3, 3) @ dirs_lidar.unsqueeze(-1)
    ).squeeze(-1)
    ray_dirs_world = ray_dirs_world / (ray_dirs_world.norm(dim=-1, keepdim=True) + 1e-8)
    return ray_dirs_world


def get_ray_dirs_cam_pinhole_batched(
    Ks: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """为每个相机生成归一化的相机系射线方向，输出形状为[C, H, W, 3]。"""
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
    dirs_cam = dirs_cam / (dirs_cam.norm(dim=-1, keepdim=True) + 1e-8)
    return dirs_cam


def get_ray_dirs_pinhole_batched(
    Ks: torch.Tensor, width: int, height: int, c2w: torch.Tensor
) -> torch.Tensor:
    """为每个相机生成归一化的世界系射线方向，输出形状为[C, H, W, 3]。"""
    dirs_cam = get_ray_dirs_cam_pinhole_batched(Ks, width, height)
    C = dirs_cam.shape[0]
    dirs_cam = dirs_cam.view(C, -1, 3)

    R = c2w[:, :3, :3]  # [C, 3, 3]
    dirs_world = torch.bmm(dirs_cam, R.transpose(1, 2))
    return dirs_world.view(C, height, width, 3)


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
    ray_dirs_world: Optional[torch.Tensor] = None,
):
    """原生渲染接口。

    - 当 rgb_decoder 为 None：沿用 gsplat.rasterization 的 SH 渲染（sh_degree=1），直接输出 RGB。
    - 当 rgb_decoder 不为 None：使用 rasterization 的 N-D features 模式（sh_degree=None）先把每个高斯的特征
      光栅化到像素，再按 camera_ids 调用 rgb_decoder 得到最终 RGB。
    """  # noqa: E501
    if opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities = opacities.squeeze(1)

    if rgb_decoder is None:
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
            tile_size=_RENDER_TILE_SIZE,
            radius_clip=_RENDER_RADIUS_CLIP,
            rasterize_mode=_RENDER_RASTERIZE_MODE,
        )
        return render_colors, render_alphas

    if colors.dim() == 3 and colors.shape[1] == 4 and colors.shape[2] >= 3:
        colors = colors[..., :3]
    if colors.dim() != 3 or colors.shape[1] != 4 or colors.shape[2] != 3:
        raise ValueError(
            f"CNN渲染分支期望colors形状为[N, 4, 3] (sh_degree=1)，实际为{tuple(colors.shape)}"  # noqa: E501
        )

    if camera_ids is None:
        raise ValueError("启用CNN渲染分支时必须传入camera_ids")
    if len(camera_ids) != viewmats.shape[0]:
        raise ValueError(
            f"camera_ids数量与viewmats不一致: camera_ids={len(camera_ids)} viewmats={viewmats.shape[0]}"  # noqa: E501
        )

    per_gauss_feat = colors.contiguous().view(colors.shape[0], -1)

    render_feats, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=per_gauss_feat,
        viewmats=viewmats,
        Ks=Ks,
        width=img_width,
        height=img_height,
        near_plane=0.001,
        far_plane=1000,
        sh_degree=None,
        packed=True,
        tile_size=_RENDER_TILE_SIZE,
        radius_clip=_RENDER_RADIUS_CLIP,
        rasterize_mode=_RENDER_RASTERIZE_MODE,
    )

    features = render_feats
    if getattr(rgb_decoder, "use_ray_dirs", False) and ray_dirs_world is None:
        c2w = invert_world2camera(viewmats)
        ray_dirs_world = get_ray_dirs_pinhole_batched(Ks, img_width, img_height, c2w)

    if hasattr(rgb_decoder, "forward_batched"):
        rendered_rgb = rgb_decoder.forward_batched(
            camera_ids, features, ray_dirs_world=ray_dirs_world
        )
    else:
        decoded: List[torch.Tensor] = []
        for cam_idx, cam_id in enumerate(camera_ids):
            ray = ray_dirs_world[cam_idx] if (ray_dirs_world is not None) else None
            decoded.append(rgb_decoder(cam_id, features[cam_idx], ray_dirs_world=ray))
        rendered_rgb = torch.stack(decoded, dim=0)
    return rendered_rgb, render_alphas


def render_sky_cubemap(
    sky_cubemap: torch.Tensor, ray_dirs_world: torch.Tensor
) -> torch.Tensor:
    """Render sky colors from a learned cubemap texture.

    Args:
        sky_cubemap: Cubemap tensor with shape [6, R, R, 3].
        ray_dirs_world: Per-pixel world ray directions [C, H, W, 3].

    Returns:
        Sky colors [C, H, W, 3] in approximately [0, 1].
    """
    if (
        sky_cubemap.dim() != 4
        or sky_cubemap.shape[0] != 6
        or sky_cubemap.shape[-1] != 3
    ):
        raise ValueError(
            f"sky_cubemap期望形状为[6,R,R,3]，实际为{tuple(sky_cubemap.shape)}"
        )

    dirs = torch.nn.functional.normalize(ray_dirs_world, dim=-1, eps=1e-8)
    C, H, W, _ = dirs.shape
    dirs_flat = dirs.reshape(-1, 3)

    x = dirs_flat[:, 0]
    y = dirs_flat[:, 1]
    z = dirs_flat[:, 2]
    ax = x.abs()
    ay = y.abs()
    az = z.abs()

    max_axis = torch.stack((ax, ay, az), dim=-1).argmax(dim=-1)
    is_x = max_axis == 0
    is_y = max_axis == 1
    is_z = max_axis == 2

    face = torch.empty_like(max_axis, dtype=torch.long)
    gx = torch.empty_like(x)
    gy = torch.empty_like(y)

    pos_x = is_x & (x >= 0)
    neg_x = is_x & (x < 0)
    pos_y = is_y & (y >= 0)
    neg_y = is_y & (y < 0)
    pos_z = is_z & (z >= 0)
    neg_z = is_z & (z < 0)

    face[pos_x] = 0
    a = ax[pos_x].clamp_min(1e-8)
    gx[pos_x] = (-z[pos_x]) / a
    gy[pos_x] = (-y[pos_x]) / a

    face[neg_x] = 1
    a = ax[neg_x].clamp_min(1e-8)
    gx[neg_x] = (z[neg_x]) / a
    gy[neg_x] = (-y[neg_x]) / a

    face[pos_y] = 2
    a = ay[pos_y].clamp_min(1e-8)
    gx[pos_y] = (x[pos_y]) / a
    gy[pos_y] = (z[pos_y]) / a

    face[neg_y] = 3
    a = ay[neg_y].clamp_min(1e-8)
    gx[neg_y] = (x[neg_y]) / a
    gy[neg_y] = (-z[neg_y]) / a

    face[pos_z] = 4
    a = az[pos_z].clamp_min(1e-8)
    gx[pos_z] = (x[pos_z]) / a
    gy[pos_z] = (-y[pos_z]) / a

    face[neg_z] = 5
    a = az[neg_z].clamp_min(1e-8)
    gx[neg_z] = (-x[neg_z]) / a
    gy[neg_z] = (-y[neg_z]) / a

    gx = gx.clamp(-1.0, 1.0)
    gy = gy.clamp(-1.0, 1.0)

    tex = sky_cubemap.sigmoid()
    resolution = int(tex.shape[1])
    px = (gx + 1.0) * 0.5 * resolution - 0.5
    py = (gy + 1.0) * 0.5 * resolution - 0.5

    x0 = px.floor().to(torch.long).clamp(0, resolution - 1)
    y0 = py.floor().to(torch.long).clamp(0, resolution - 1)
    x1 = (x0 + 1).clamp(0, resolution - 1)
    y1 = (y0 + 1).clamp(0, resolution - 1)

    wx = (px - x0.to(px.dtype)).unsqueeze(-1)
    wy = (py - y0.to(py.dtype)).unsqueeze(-1)

    c00 = tex[face, y0, x0]
    c10 = tex[face, y0, x1]
    c01 = tex[face, y1, x0]
    c11 = tex[face, y1, x1]

    c0 = c00 * (1 - wx) + c10 * wx
    c1 = c01 * (1 - wx) + c11 * wx
    colors = c0 * (1 - wy) + c1 * wy

    return colors.view(C, H, W, 3)
