from typing import Dict, List, Tuple

import torch

from config import DEVICE, MAP_CENTER
from data_types import Camera, FrameParams, FrameResp, InitParams, InitResp, Vehicle
from models import GaussianComponent, GSModel
from render_kernel import render
from util import calculate_viewmats


class RenderManager:
    def __init__(self, model: str = "model.path") -> None:
        self.device = DEVICE
        self.model = GSModel.load_from_pth(model).to_device(DEVICE)
        self.background: GaussianComponent = self.model.get_component("background")
        self.sky: GaussianComponent = self.model.get_component("sky")
        self.actors: List[GaussianComponent] = self.model.get_components_by_type("obj")
        self.map_center = torch.tensor(MAP_CENTER, device=self.device)
        # 预构建环境车name到点云的映射
        self.actor_map: Dict[str, GaussianComponent] = {actor.name: actor for actor in self.actors}
        self._static_gs()

    def _static_gs(self) -> None:
        """获取静态点云，只计算一次"""
        self.static_means = torch.cat([
            self.background.get_xyz(),
            self.sky.get_xyz(),
        ])
        self.static_quats = torch.cat([
            self.background.get_quats(),
            self.sky.get_quats(),
        ])
        self.static_scales = torch.cat([
            self.background.get_scales(),
            self.sky.get_scales(),
        ])
        self.static_opacities = torch.cat([
            self.background.get_opacities(),
            self.sky.get_opacities(),
        ])
        self.static_colors = torch.cat([
            self.background.get_colors(),
            self.sky.get_colors(),
        ])

    def _dynamic_gs(self, vehicles: List[Vehicle]) -> None:
        """获取动态点云，每帧都要计算"""
        dynamic_means = []
        dynamic_quats = []
        dynamic_scales = []
        dynamic_opacities = []
        dynamic_colors = []
        for v in vehicles:
            if v.type in self.actor_map:
                heading = v.yaw
                position = torch.tensor(v.trajectory, device=self.device) - self.map_center
                dynamic_means.append(self.actor_map[v.type].get_xyz(heading, position))
                dynamic_quats.append(self.actor_map[v.type].get_quats(heading))
                dynamic_scales.append(self.actor_map[v.type].get_scales())
                dynamic_opacities.append(self.actor_map[v.type].get_opacities())
                dynamic_colors.append(self.actor_map[v.type].get_colors())

        self.dynamic_means = torch.cat(dynamic_means)
        self.dynamic_quats = torch.cat(dynamic_quats)
        self.dynamic_scales = torch.cat(dynamic_scales)
        self.dynamic_opacities = torch.cat(dynamic_opacities)
        self.dynamic_colors = torch.cat(dynamic_colors)

    def init(self, params: InitParams) -> InitResp:
        """初始化接口，输入相机的参数"""
        # 按照分辨率对输入camera进行分类
        grouped = {}
        for i in params.cameras:
            resolution = (i.width, i.height)
            if resolution not in grouped:
                grouped[resolution] = []
            grouped[resolution].append(i)
        self.cameras: Dict[Tuple[int, int], List[Camera]] = grouped

        resp = InitResp(init_status=True)
        return resp

    def render_frame(self, params: FrameParams) -> FrameResp:
        """渲染接口，每帧调用"""
        ego_heading = params.ego_yaw
        ego_position = torch.tensor(params.ego_trajectory, device=self.device) - self.map_center

        vehicles = params.env_vehicles
        self._dynamic_gs(vehicles)

        means = torch.cat([self.static_means, self.dynamic_means])
        quats = torch.cat([self.static_quats, self.dynamic_quats])
        scales = torch.cat([self.static_scales, self.dynamic_scales])
        opacities = torch.cat([self.static_opacities, self.dynamic_opacities])
        colors = torch.cat([self.static_colors, self.dynamic_colors])

        images = {}
        for resolution, cameras in self.cameras.items():
            width, height = resolution
            camera_ids = [camera.id for camera in cameras]
            extrinsics_list = [camera.extrinsics for camera in cameras]
            intrinsics_list = [camera.intrinsics for camera in cameras]

            viewmats = calculate_viewmats(extrinsics_list, ego_heading, ego_position)
            Ks = torch.tensor(intrinsics_list, dtype=torch.float32, device=self.device)

            render_colors, render_alphas = render(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
            for id, image in zip(camera_ids, render_colors):
                images[id] = image

        resp = FrameResp(params.timestamp, images)
        return resp
