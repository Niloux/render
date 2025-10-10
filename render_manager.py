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
        self._setup_render_buffers()

    def _setup_render_buffers(self) -> None:
        """预分配渲染buffers，消除每帧内存分配

        预计算静态数据大小，为动态数据预留最大可能空间
        这样每帧只需要更新动态部分，零内存分配
        """
        # 计算静态点云数据
        static_components = [self.background, self.sky]
        self.static_data = GaussianData.from_components(static_components)
        self.static_points = self.static_data.means.shape[0]

        # 预计算最大可能的动态点数（假设所有车辆类型同时出现）
        max_dynamic_points = sum(component.num_points for component in self.actors)
        self.max_dynamic_points = max_dynamic_points

        # 预分配统一的渲染buffer，静态+动态
        total_max_points = self.static_points + max_dynamic_points
        device = self.device

        self.render_buffer = {
            "means": torch.empty((total_max_points, 3), device=device, dtype=torch.float32),
            "quats": torch.empty((total_max_points, 4), device=device, dtype=torch.float32),
            "scales": torch.empty((total_max_points, 3), device=device, dtype=torch.float32),
            "opacities": torch.empty((total_max_points, 1), device=device, dtype=torch.float32),
            "colors": torch.empty((total_max_points, 4, 3), device=device, dtype=torch.float32),
        }

        # 一次性拷贝静态数据到buffer前部，永不改变
        self.render_buffer["means"][: self.static_points] = self.static_data.means
        self.render_buffer["quats"][: self.static_points] = self.static_data.quats
        self.render_buffer["scales"][: self.static_points] = self.static_data.scales
        self.render_buffer["opacities"][: self.static_points] = self.static_data.opacities
        self.render_buffer["colors"][: self.static_points] = self.static_data.colors

    def _update_dynamic_buffer(self, vehicles: List[Vehicle]) -> int:
        """直接更新预分配buffer的动态部分

        返回动态点数，避免创建临时GaussianData对象
        静态数据永远不变，只更新buffer的动态部分
        """
        if not vehicles:
            return 0

        # 快速失败：批量验证所有车辆类型
        unknown_types = [v.type for v in vehicles if v.type not in self.actor_map]
        if unknown_types:
            raise ValueError(f"Unknown vehicle types: {unknown_types}. Available: {list(self.actor_map.keys())}")

        # 直接更新buffer的动态部分，从static_points开始
        start_idx = self.static_points
        device = self.device

        for v in vehicles:
            # 预先转换位置数据，避免重复计算
            heading = v.yaw
            position = torch.tensor(v.trajectory, device=device, dtype=torch.float32) - self.map_center

            # 获取对应组件并计算变换
            component = self.actor_map[v.type]
            num_points = component.num_points
            end_idx = start_idx + num_points

            # 直接写入预分配buffer的对应切片，零拷贝
            self.render_buffer["means"][start_idx:end_idx] = component.get_xyz(heading, position)
            self.render_buffer["quats"][start_idx:end_idx] = component.get_quats(heading)
            self.render_buffer["scales"][start_idx:end_idx] = component.get_scales()
            self.render_buffer["opacities"][start_idx:end_idx] = component.get_opacities()
            self.render_buffer["colors"][start_idx:end_idx] = component.get_colors()

            start_idx = end_idx

        # 返回动态点数
        return start_idx - self.static_points

    def init(self, params: InitParams) -> InitResp:
        """初始化接口，输入相机的参数

        预计算所有相机数据，消除运行时查找和转换
        """
        # 按照分辨率对输入camera进行分类
        grouped = {}
        for i in params.cameras:
            resolution = (i.width, i.height)
            if resolution not in grouped:
                grouped[resolution] = []
            grouped[resolution].append(i)
        self.cameras: Dict[Tuple[int, int], List[Camera]] = grouped

        # 计算所有相机的内参和外参tensor，消除运行时转换
        self.camera_data = {}
        for resolution, cameras in self.cameras.items():
            width, height = resolution

            # 预计算内参tensor
            intrinsics_list = [cam.intrinsics for cam in cameras]
            Ks = torch.tensor(intrinsics_list, dtype=torch.float32, device=self.device)

            # 预计算相机ID列表和外参列表
            camera_ids = [cam.id for cam in cameras]
            extrinsics_list = [cam.extrinsics for cam in cameras]

            self.camera_data[resolution] = {
                "camera_ids": camera_ids,
                "extrinsics_list": extrinsics_list,
                "intrinsics_tensor": Ks,
                "width": width,
                "height": height,
            }

        resp = InitResp(init_status=True)
        return resp

    def render_frame(self, params: FrameParams) -> FrameResp:
        """渲染接口，每帧调用"""
        # 输入验证
        if not params.ego_trajectory or len(params.ego_trajectory) != 3:
            raise ValueError(f"Invalid ego_trajectory: {params.ego_trajectory}")

        # 转换ego
        ego_heading = params.ego_yaw
        ego_position = torch.tensor(params.ego_trajectory, device=self.device, dtype=torch.float32) - self.map_center

        # 预分配buffer的动态部分
        dynamic_points = self._update_dynamic_buffer(params.env_vehicles)
        total_points = self.static_points + dynamic_points

        # 直接使用预分配buffer，避免创建临时对象和tuple解包
        render_means = self.render_buffer["means"][:total_points]
        render_quats = self.render_buffer["quats"][:total_points]
        render_scales = self.render_buffer["scales"][:total_points]
        render_opacities = self.render_buffer["opacities"][:total_points]
        render_colors = self.render_buffer["colors"][:total_points]

        images = {}
        # 使用预计算的相机数据，零查找开销
        for resolution, cam_data in self.camera_data.items():
            # 直接使用预计算的数据，无需运行时转换
            camera_ids = cam_data["camera_ids"]
            extrinsics_list = cam_data["extrinsics_list"]
            Ks = cam_data["intrinsics_tensor"]
            width = cam_data["width"]
            height = cam_data["height"]

            viewmats = calculate_viewmats(extrinsics_list, ego_heading, ego_position)

            # 渲染并收集结果 - 直接传递预分配buffer参数
            batch_colors, batch_alphas = render(
                render_means, render_quats, render_scales, render_opacities, render_colors, viewmats, Ks, width, height
            )
            batch_colors = (batch_colors.clamp(0, 1) * 255).to(torch.uint8)
            for cam_id, image in zip(camera_ids, batch_colors):
                images[cam_id] = image

        return FrameResp(params.timestamp, images)
