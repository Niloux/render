from typing import Dict, List, Tuple

import torch

from config import DEVICE, MAP_CENTER
from data_types import Camera, FrameParams, FrameResp, GaussianData, InitParams, InitResp, Vehicle
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
        """获取静态点云，只计算一次

        Linus式重构：消除重复代码，用统一的数据结构
        """
        static_components = [self.background, self.sky]
        self.static_data = GaussianData.from_components(static_components)

    def _dynamic_gs(self, vehicles: List[Vehicle]) -> GaussianData:
        """获取动态点云，每帧都要计算

        Linus式重构：
        1. 不要静默忽略未知车辆类型 - 这是bug！
        2. 避免每帧重复创建tensor - 性能优化
        3. 用统一的数据结构 - 消除重复代码

        修复闭包bug：直接收集数据而不是创建临时对象
        """
        if not vehicles:
            return None

        dynamic_means = []
        dynamic_quats = []
        dynamic_scales = []
        dynamic_opacities = []
        dynamic_colors = []

        for v in vehicles:
            if v.type not in self.actor_map:
                raise ValueError(f"Unknown vehicle type: {v.type}. Available types: {list(self.actor_map.keys())}")

            # 优化：预先转换为tensor，避免每次调用get_xyz时重复转换
            heading = v.yaw
            position = torch.tensor(v.trajectory, device=self.device, dtype=torch.float32) - self.map_center

            # 直接获取变换后的数据，避免闭包陷阱
            component = self.actor_map[v.type]
            dynamic_means.append(component.get_xyz(heading, position))
            dynamic_quats.append(component.get_quats(heading))
            dynamic_scales.append(component.get_scales())
            dynamic_opacities.append(component.get_opacities())
            dynamic_colors.append(component.get_colors())

        return GaussianData(
            means=torch.cat(dynamic_means),
            quats=torch.cat(dynamic_quats),
            scales=torch.cat(dynamic_scales),
            opacities=torch.cat(dynamic_opacities),
            colors=torch.cat(dynamic_colors),
        )

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
        """渲染接口，每帧调用

        Linus式重构：简化逻辑，消除重复的torch.cat调用
        """
        # 优化：预先转换ego位置，避免重复计算
        ego_heading = params.ego_yaw
        ego_position = torch.tensor(params.ego_trajectory, device=self.device, dtype=torch.float32) - self.map_center

        # 获取动态数据并与静态数据合并
        dynamic_data = self._dynamic_gs(params.env_vehicles)

        # 合并静态和动态数据 - 处理空vehicles的情况
        all_data = self.static_data.cat(dynamic_data) if dynamic_data else self.static_data

        images = {}
        for resolution, cameras in self.cameras.items():
            width, height = resolution
            camera_ids = [camera.id for camera in cameras]
            extrinsics_list = [camera.extrinsics for camera in cameras]
            intrinsics_list = [camera.intrinsics for camera in cameras]

            viewmats = calculate_viewmats(extrinsics_list, ego_heading, ego_position)
            Ks = torch.tensor(intrinsics_list, dtype=torch.float32, device=self.device)

            # 使用统一的数据结构传递参数
            render_colors, render_alphas = render(*all_data.to_render_args(), viewmats, Ks, width, height)
            for id, image in zip(camera_ids, render_colors):
                images[id] = image

        return FrameResp(params.timestamp, images)
