"""Preallocated Gaussian render buffers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from data_types import GaussianData, Vehicle
from models import GaussianComponent


@dataclass
class RenderBuffers:
    static_data: GaussianData
    static_points: int
    sky_data: Optional[GaussianData]
    sky_points: int
    max_dynamic_points: int
    render_buffer: Dict[str, torch.Tensor]
    zero_velocities: torch.Tensor

    def render_params(self, dynamic_points: int) -> Tuple[torch.Tensor, ...]:
        total_points = self.static_points + dynamic_points
        return (
            self.render_buffer["means"][:total_points],
            self.render_buffer["quats"][:total_points],
            self.render_buffer["scales"][:total_points],
            self.render_buffer["opacities"][:total_points],
            self.render_buffer["colors"][:total_points],
        )


def build_render_buffers(
    background: GaussianComponent,
    sky: Optional[GaussianComponent],
    actors: List[GaussianComponent],
    device: torch.device,
) -> RenderBuffers:
    static_data = GaussianData.from_components([background])
    if not static_data:
        raise ValueError("静态点云组件不能为空")

    static_points = static_data.means.shape[0]
    sh_coeffs = static_data.colors.shape[1]
    color_channels = static_data.colors.shape[2]
    sky_data = GaussianData.from_components([sky]) if sky is not None else None
    sky_points = sky_data.means.shape[0] if sky_data else 0
    max_dynamic_points = sum(component.num_points for component in actors)
    total_max_points = static_points + max_dynamic_points

    render_buffer = {
        "means": torch.empty((total_max_points, 3), device=device, dtype=torch.float32),
        "quats": torch.empty((total_max_points, 4), device=device, dtype=torch.float32),
        "scales": torch.empty((total_max_points, 3), device=device, dtype=torch.float32),
        "opacities": torch.empty(
            (total_max_points, 1), device=device, dtype=torch.float32
        ),
        "colors": torch.empty(
            (total_max_points, sh_coeffs, color_channels),
            device=device,
            dtype=torch.float32,
        ),
    }
    render_buffer["means"][:static_points] = static_data.means
    render_buffer["quats"][:static_points] = static_data.quats
    render_buffer["scales"][:static_points] = static_data.scales
    render_buffer["opacities"][:static_points] = static_data.opacities
    render_buffer["colors"][:static_points] = static_data.colors

    zero_velocities = torch.zeros(
        (total_max_points, 3), device=device, dtype=torch.float32
    )
    return RenderBuffers(
        static_data=static_data,
        static_points=static_points,
        sky_data=sky_data,
        sky_points=sky_points,
        max_dynamic_points=max_dynamic_points,
        render_buffer=render_buffer,
        zero_velocities=zero_velocities,
    )


def update_dynamic_buffer(
    buffers: RenderBuffers,
    actor_map: Dict[str, GaussianComponent],
    vehicles: List[Vehicle],
    map_center: torch.Tensor,
    device: torch.device,
) -> int:
    if not vehicles:
        return 0

    missing_types = sorted(
        {vehicle.type for vehicle in vehicles if vehicle.type not in actor_map}
    )
    if missing_types:
        raise ValueError(f"车模库中不存在车辆模型: {', '.join(missing_types)}")

    start_idx = buffers.static_points
    max_points = buffers.static_points + buffers.max_dynamic_points
    for vehicle in vehicles:
        position = (
            torch.tensor(vehicle.trajectory, device=device, dtype=torch.float32)
            - map_center
        )
        component = actor_map[vehicle.type]
        num_points = component.num_points
        end_idx = start_idx + num_points
        if end_idx > max_points:
            raise ValueError(
                "动态车辆点数超过预分配容量: "
                f"required={end_idx - buffers.static_points} "
                f"capacity={buffers.max_dynamic_points}"
            )

        buffers.render_buffer["means"][start_idx:end_idx] = component.get_xyz(
            vehicle.yaw, position
        )
        buffers.render_buffer["quats"][start_idx:end_idx] = component.get_quats(
            vehicle.yaw
        )
        buffers.render_buffer["scales"][start_idx:end_idx] = component.get_scales()
        buffers.render_buffer["opacities"][start_idx:end_idx] = (
            component.get_opacities()
        )
        buffers.render_buffer["colors"][start_idx:end_idx] = component.get_colors()
        start_idx = end_idx

    return start_idx - buffers.static_points
