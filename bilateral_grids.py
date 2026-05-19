"""Bilateral grid color correction for camera rendering."""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pixel_coords(width: int, height: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack(((xs + 0.5) / width, (ys + 0.5) / height), dim=-1)


def color_affine_transform(affine_mats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    return (
        torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1)
        + affine_mats[..., 3]
    )


class BilateralGrid(nn.Module):
    def __init__(
        self,
        num_cams: int,
        grid_depth: int = 8,
        grid_height: int = 16,
        grid_width: int = 16,
    ) -> None:
        super().__init__()
        self.grid_depth = grid_depth
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grids = nn.Parameter(self._identity_grid().tile(num_cams, 1, 1, 1, 1))
        self.register_buffer("rgb2gray_weight", torch.tensor([[0.299, 0.587, 0.114]]))

    def _identity_grid(self) -> torch.Tensor:
        grid = torch.tensor(
            [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0],
            dtype=torch.float32,
        )
        grid = grid.repeat(self.grid_depth * self.grid_height * self.grid_width, 1)
        grid = grid.reshape(1, self.grid_depth, self.grid_height, self.grid_width, 12)
        return grid.permute(0, 4, 1, 2, 3)

    def forward(
        self, grid_xy: torch.Tensor, rgb: torch.Tensor, idx: int
    ) -> torch.Tensor:
        input_ndims = len(grid_xy.shape)
        if len(rgb.shape) != input_ndims:
            raise ValueError(
                f"bilateral grid输入维度不一致: xy={tuple(grid_xy.shape)} rgb={tuple(rgb.shape)}"
            )
        if not (1 < input_ndims < 5):
            raise ValueError(
                "bilateral grid slicing只支持2D、3D或4D输入"
            )

        for _ in range(5 - input_ndims):
            grid_xy = grid_xy.unsqueeze(1)
            rgb = rgb.unsqueeze(1)

        grids = self.grids[idx : idx + 1]
        grid_xy = (grid_xy - 0.5) * 2
        grid_z = (rgb @ self.rgb2gray_weight.T) * 2.0 - 1.0
        grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)

        affine_mats = F.grid_sample(
            grids,
            grid_xyz,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        affine_mats = affine_mats.permute(0, 2, 3, 4, 1)
        affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)

        for _ in range(5 - input_ndims):
            affine_mats = affine_mats.squeeze(1)
        return affine_mats


class BilGrids(nn.Module):
    def __init__(
        self,
        num_cams: int,
        num_levels: int = 3,
        grid_depth: int = 8,
        grid_height: int = 16,
        grid_width: int = 16,
        camera_id_map: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.bil_grids = nn.ModuleList([
            BilateralGrid(num_cams, grid_depth, grid_height, grid_width)
            for _ in range(num_levels)
        ])
        self.camera_id_map = camera_id_map or {}

    @classmethod
    def from_checkpoint_state(
        cls,
        state: dict,
        camera_ids: Sequence[str],
        device: torch.device,
    ) -> "BilGrids":
        params = state.get("params") if isinstance(state, dict) else None
        if not isinstance(params, dict):
            raise ValueError("bil_grids权重必须包含params字典")

        grid_keys = sorted(
            key for key in params.keys() if re.fullmatch(r"bil_grids\.\d+\.grids", key)
        )
        if not grid_keys:
            raise ValueError("bil_grids权重缺少bil_grids.<level>.grids")

        first_grid = params[grid_keys[0]]
        if first_grid.dim() != 5 or first_grid.shape[1] != 12:
            raise ValueError(
                f"bil_grids grid期望形状[N,12,D,H,W]，实际为{tuple(first_grid.shape)}"
            )

        num_cams = int(first_grid.shape[0])
        camera_id_map = {camera_id: idx for idx, camera_id in enumerate(camera_ids)}
        module = cls(
            num_cams=num_cams,
            num_levels=len(grid_keys),
            grid_depth=int(first_grid.shape[2]),
            grid_height=int(first_grid.shape[3]),
            grid_width=int(first_grid.shape[4]),
            camera_id_map=camera_id_map,
        ).to(device)
        module.load_state_dict(params, strict=True)
        module.eval()
        return module

    def _camera_index(self, camera_id: str) -> int:
        try:
            idx = int(camera_id)
            return idx
        except ValueError:
            pass
        if camera_id in self.camera_id_map:
            return int(self.camera_id_map[camera_id])
        raise KeyError(f"bil_grids缺少camera_id映射: {camera_id!r}")

    def forward(
        self, camera_id: str, pix_xy: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        idx = self._camera_index(camera_id)
        if idx < 0 or idx >= self.bil_grids[0].grids.shape[0]:
            raise IndexError(
                f"camera_id={camera_id!r}映射到bil_grids索引{idx}，超出范围"
            )

        rendering_rgb = [features]
        affine_mats = []
        pix_xy = pix_xy.unsqueeze(0)
        for grid in self.bil_grids:
            rgb = rendering_rgb[-1].unsqueeze(0)
            mats = grid(pix_xy, rgb, idx)
            corrected = color_affine_transform(mats, rgb)
            rendering_rgb.append(corrected.squeeze(0))
            affine_mats.append(mats.squeeze(0))
        return rendering_rgb[-1], affine_mats
