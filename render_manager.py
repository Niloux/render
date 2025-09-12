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
        """获取静态点云，只计算一次"""
        static_components = [self.background, self.sky]
        self.static_data = GaussianData.from_components(static_components)

    def _dynamic_gs(self, vehicles: List[Vehicle]) -> GaussianData:
        """获取动态点云，每帧都要计算"""
        if not vehicles:
            return None

        # 快速失败：批量验证所有车辆类型
        unknown_types = [v.type for v in vehicles if v.type not in self.actor_map]
        if unknown_types:
            raise ValueError(f"Unknown vehicle types: {unknown_types}. Available: {list(self.actor_map.keys())}")

        # 预分配tensor列表，避免动态增长
        num_vehicles = len(vehicles)
        dynamic_means = [None] * num_vehicles
        dynamic_quats = [None] * num_vehicles
        dynamic_scales = [None] * num_vehicles
        dynamic_opacities = [None] * num_vehicles
        dynamic_colors = [None] * num_vehicles

        # 批量处理车辆数据
        for i, v in enumerate(vehicles):
            # 预先转换位置数据，避免重复计算
            heading = v.yaw
            position = torch.tensor(v.trajectory, device=self.device, dtype=torch.float32) - self.map_center

            # 获取对应组件并计算变换
            component = self.actor_map[v.type]
            dynamic_means[i] = component.get_xyz(heading, position)
            dynamic_quats[i] = component.get_quats(heading)
            dynamic_scales[i] = component.get_scales()
            dynamic_opacities[i] = component.get_opacities()
            dynamic_colors[i] = component.get_colors()

        # 一次性拼接所有数据
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
        """渲染接口，每帧调用"""
        # 输入验证 - 快速失败原则
        if not params.ego_trajectory or len(params.ego_trajectory) != 3:
            raise ValueError(f"Invalid ego_trajectory: {params.ego_trajectory}")

        # 一次性转换ego数据，避免重复计算
        ego_heading = params.ego_yaw
        ego_position = torch.tensor(params.ego_trajectory, device=self.device, dtype=torch.float32) - self.map_center

        # 获取动态数据并与静态数据合并
        dynamic_data = self._dynamic_gs(params.env_vehicles)
        all_data = self.static_data.cat(dynamic_data) if dynamic_data else self.static_data

        # 预先解包渲染参数，避免每次调用时重复计算
        render_args = all_data.to_render_args()

        images = {}
        for resolution, cameras in self.cameras.items():
            width, height = resolution

            # 批量提取相机数据，减少循环开销
            camera_ids = [cam.id for cam in cameras]
            extrinsics_list = [cam.extrinsics for cam in cameras]

            # 使用预缓存的intrinsics tensor（如果可能）
            if not hasattr(self, "_intrinsics_cache"):
                self._intrinsics_cache = {}

            cache_key = (resolution, len(cameras))
            if cache_key not in self._intrinsics_cache:
                intrinsics_list = [cam.intrinsics for cam in cameras]
                self._intrinsics_cache[cache_key] = torch.tensor(
                    intrinsics_list, dtype=torch.float32, device=self.device
                )

            Ks = self._intrinsics_cache[cache_key]
            viewmats = calculate_viewmats(extrinsics_list, ego_heading, ego_position)

            # 渲染并收集结果
            render_colors, render_alphas = render(*render_args, viewmats, Ks, width, height)
            for cam_id, image in zip(camera_ids, render_colors):
                images[cam_id] = image

        return FrameResp(params.timestamp, images)
