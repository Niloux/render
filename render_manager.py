from typing import Dict, List, Tuple

import torch
from gsplat.rendering import lidar_rasterization

from config import MAP_CENTER
from data_types import (
    Camera,
    FrameParams,
    FrameResp,
    GaussianData,
    InitParams,
    InitResp,
    Lidar,
    Vehicle,
)
from gsplat import spherical_harmonics
from models import GaussianComponent, GSModel
from render_kernel import (
    build_raster_pts,
    extract_camera_centers,
    generate_point_cloud,
    render,
)
from rgb_decoder import RGBDecoder
from util import calculate_viewmats


class RenderManager:
    def __init__(self, model: str = "model.path") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model
        self.model = GSModel.load_from_pth(model).to_device(self.device)
        self.background: GaussianComponent = self.model.get_component("background")
        self.sky: GaussianComponent = self.model.get_component("sky")
        self.actors: List[GaussianComponent] = self.model.get_components_by_type("obj")
        # 修复 UserWarning: To copy construct from a tensor
        if isinstance(MAP_CENTER, torch.Tensor):
            self.map_center = MAP_CENTER.clone().detach().to(self.device)
        else:
            self.map_center = torch.tensor(MAP_CENTER, device=self.device)

        # 预构建环境车name到点云的映射
        self.actor_map: Dict[str, GaussianComponent] = {
            actor.name: actor for actor in self.actors
        }
        self._setup_render_buffers()
        self.lidars: Dict[str, Lidar] = {}
        self.lidar_data: Dict[str, Dict] = {}
        self.render_camera: bool = True
        self.render_lidar: bool = False
        self.rgb_decoder: RGBDecoder | None = None
        self.camera_id_to_index: Dict[str, int] = {}

    def _setup_render_buffers(self) -> None:
        """预分配渲染buffers，消除每帧内存分配

        预计算静态数据大小，为动态数据预留最大可能空间
        这样每帧只需要更新动态部分，零内存分配
        """
        # 计算静态点云数据
        static_components = [self.background, self.sky]
        self.static_data = GaussianData.from_components(static_components)
        if not self.static_data:
            raise ValueError("静态点云组件不能为空")
        self.static_points = self.static_data.means.shape[0]

        # 预计算最大可能的动态点数（假设所有车辆类型同时出现）
        max_dynamic_points = sum(component.num_points for component in self.actors)
        self.max_dynamic_points = max_dynamic_points

        # 预分配统一的渲染buffer，静态+动态
        total_max_points = self.static_points + max_dynamic_points
        device = self.device

        self.render_buffer = {
            "means": torch.empty(
                (total_max_points, 3), device=device, dtype=torch.float32
            ),
            "quats": torch.empty(
                (total_max_points, 4), device=device, dtype=torch.float32
            ),
            "scales": torch.empty(
                (total_max_points, 3), device=device, dtype=torch.float32
            ),
            "opacities": torch.empty(
                (total_max_points, 1), device=device, dtype=torch.float32
            ),
            # TODO:这里的颜色维度也需要根据模型来进行调整
            "colors": torch.empty(
                (total_max_points, 4, 3), device=device, dtype=torch.float32
            ),
        }

        # 一次性拷贝静态数据到buffer前部，永不改变
        self.render_buffer["means"][: self.static_points] = self.static_data.means
        self.render_buffer["quats"][: self.static_points] = self.static_data.quats
        self.render_buffer["scales"][: self.static_points] = self.static_data.scales
        self.render_buffer["opacities"][: self.static_points] = (
            self.static_data.opacities
        )
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
            raise ValueError(
                f"Unknown vehicle types: {unknown_types}. Available: {list(self.actor_map.keys())}"  # noqa: E501
            )

        # 直接更新buffer的动态部分，从static_points开始
        start_idx = self.static_points
        device = self.device

        for v in vehicles:
            # 预先转换位置数据，避免重复计算
            heading = v.yaw
            position = (
                torch.tensor(v.trajectory, device=device, dtype=torch.float32)
                - self.map_center
            )

            # 获取对应组件并计算变换
            component = self.actor_map[v.type]
            num_points = component.num_points
            end_idx = start_idx + num_points

            # 直接写入预分配buffer的对应切片，零拷贝
            self.render_buffer["means"][start_idx:end_idx] = component.get_xyz(
                heading, position
            )
            self.render_buffer["quats"][start_idx:end_idx] = component.get_quats(
                heading
            )
            self.render_buffer["scales"][start_idx:end_idx] = component.get_scales()
            self.render_buffer["opacities"][start_idx:end_idx] = (
                component.get_opacities()
            )
            self.render_buffer["colors"][start_idx:end_idx] = component.get_colors()

            start_idx = end_idx

        # 返回动态点数
        return start_idx - self.static_points

    def init(self, params: InitParams) -> InitResp:
        """初始化接口，输入相机的参数

        预计算所有相机数据，消除运行时查找和转换
        """
        # 渲染开关与模型路径
        self.render_camera = bool(getattr(params, "render_camera", True))
        self.render_lidar = bool(getattr(params, "render_lidar", False))
        # 按照分辨率对输入camera进行分类
        grouped = {}
        for i in params.cameras:
            resolution = (i.width, i.height)
            if resolution not in grouped:
                grouped[resolution] = []
            grouped[resolution].append(i)
        rgb_decoder_state = getattr(self.model, "rgb_decoder_state", None)
        if rgb_decoder_state is not None and isinstance(rgb_decoder_state, dict):
            params_state = rgb_decoder_state.get("params")
            if (
                isinstance(params_state, dict)
                and "appearance_embedding" in params_state
            ):
                num_cams_ckpt = int(params_state["appearance_embedding"].shape[0])
            else:
                num_cams_ckpt = len(params.cameras)
        else:
            num_cams_ckpt = len(params.cameras)

        if len(params.cameras) > num_cams_ckpt:
            raise ValueError(
                f"相机数量超过rgb_decoder权重支持范围: cameras={len(params.cameras)} ckpt={num_cams_ckpt}"
            )

        self.cameras: Dict[Tuple[int, int], List[Camera]] = grouped
        self.camera_id_to_index = {
            cam.id: idx for idx, cam in enumerate(params.cameras)
        }

        self.rgb_decoder = RGBDecoder(
            metadata={"num_cams": num_cams_ckpt},
            device=self.device,
            use_app_embed=True,
            mode="image",
        )
        self.rgb_decoder.camera_id_map = self.camera_id_to_index
        if rgb_decoder_state is not None:
            self.rgb_decoder.load_state_dict(rgb_decoder_state, strict=True)
        self.rgb_decoder.eval()

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

        # 预计算激光雷达相关数据
        if params.lidars:
            for lidar in params.lidars:
                self.lidars[lidar.id] = lidar
                point_cloud, azimuths, elevations = generate_point_cloud(
                    lidar.azimuth_resolution,
                    lidar.min_azimuth,
                    lidar.max_azimuth,
                    lidar.n_elevation_channels,
                    lidar.min_elevation,
                    lidar.max_elevation,
                    self.device,
                )

                if self.model.raster_pts is not None:
                    print(f"raster_pts shape: {self.model.raster_pts.shape}")
                    raster_pts = self.model.raster_pts
                    # 计算elevation_boundaries, [n_elevation_channels//tile_height + 1]
                    tile_height = lidar.tile_height
                    elevation_boundaries = torch.cat([
                        elevations[0:1] - 1.0,
                        (
                            elevations[tile_height::tile_height]
                            + elevations[tile_height - 1 : -1 : tile_height]
                        )
                        / 2,
                        elevations[-1:] + 1.0,
                    ])
                else:
                    raster_pts, elevation_boundaries = build_raster_pts(
                        point_cloud,
                        azimuths,
                        elevations,
                        lidar.azimuth_resolution,
                        lidar.min_azimuth,
                        lidar.tile_width,
                        lidar.tile_height,
                    )

                self.lidar_data[lidar.id] = {
                    "extrinsics": lidar.extrinsics,
                    "azimuths": azimuths,
                    "elevations": elevations,
                    "elevation_boundaries": elevation_boundaries,
                    "image_width": azimuths.shape[0],
                    "image_height": elevations.shape[0],
                    "tile_width": lidar.tile_width,
                    "tile_height": lidar.tile_height,
                    "min_azimuth": lidar.min_azimuth,
                    "max_azimuth": lidar.max_azimuth,
                    "min_elevation": lidar.min_elevation,
                    "max_elevation": lidar.max_elevation,
                    "azimuth_resolution": lidar.azimuth_resolution,
                    "raster_pts": raster_pts,
                }
        return resp

    def render_frame(self, params: FrameParams) -> FrameResp:
        """渲染接口，每帧调用"""
        # 输入验证
        if not params.ego_trajectory or len(params.ego_trajectory) != 3:
            raise ValueError(f"Invalid ego_trajectory: {params.ego_trajectory}")

        # 转换ego
        ego_heading = params.ego_yaw
        ego_position = (
            torch.tensor(params.ego_trajectory, device=self.device, dtype=torch.float32)
            - self.map_center
        )

        # 预分配buffer的动态部分
        dynamic_points = self._update_dynamic_buffer(params.env_vehicles)
        total_points = self.static_points + dynamic_points

        # 直接使用预分配buffer，避免创建临时对象和tuple解包
        # 准备渲染参数
        render_params = (
            self.render_buffer["means"][:total_points],
            self.render_buffer["quats"][:total_points],
            self.render_buffer["scales"][:total_points],
            self.render_buffer["opacities"][:total_points],
            self.render_buffer["colors"][:total_points],
        )

        # 1. 相机渲染
        images = self._render_cameras(ego_heading, ego_position, render_params)

        # 2. 激光雷达渲染
        lidars = self._render_lidars(ego_heading, ego_position, render_params)

        return FrameResp(
            timestamp=params.timestamp, images=images, error_msg=None, lidars=lidars
        )

    def _render_cameras(
        self,
        ego_heading: float,
        ego_position: torch.Tensor,
        render_params: Tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """渲染所有相机图像"""
        images = {}
        if not self.render_camera:
            return images

        (
            render_means,
            render_quats,
            render_scales,
            render_opacities,
            render_colors,
        ) = render_params

        # 使用预计算的相机数据，零查找开销
        for resolution, cam_data in self.camera_data.items():
            # 直接使用预计算的数据，无需运行时转换
            camera_ids = cam_data["camera_ids"]
            extrinsics_list = cam_data["extrinsics_list"]
            Ks = cam_data["intrinsics_tensor"]
            width = cam_data["width"]
            height = cam_data["height"]

            viewmats = calculate_viewmats(extrinsics_list, ego_heading, ego_position)

            # 渲染并收集结果
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
                rgb_decoder=self.rgb_decoder,
                camera_ids=camera_ids,
            )
            batch_colors = (batch_colors.clamp(0, 1) * 255).to(torch.uint8)
            for cam_id, image in zip(camera_ids, batch_colors):
                images[cam_id] = image
        return images

    def _render_lidars(
        self,
        ego_heading: float,
        ego_position: torch.Tensor,
        render_params: Tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """渲染所有激光雷达点云"""
        lidars = {}
        if not self.render_lidar:
            return lidars

        (
            render_means,
            render_quats,
            render_scales,
            render_opacities,
            render_colors,
        ) = render_params

        for lidar_id, lidar_cfg in self.lidar_data.items():
            viewmats = calculate_viewmats(
                [lidar_cfg["extrinsics"]], ego_heading, ego_position
            )
            lidar_features = self._compute_lidar_features_from_colors(
                render_colors, render_means, viewmats
            )
            raster_pts = lidar_cfg["raster_pts"]

            # 获取配置的远近平面
            near_plane = (
                self.lidars[lidar_id].near_plane if lidar_id in self.lidars else 0.01
            )
            far_plane = (
                self.lidars[lidar_id].far_plane if lidar_id in self.lidars else 1e10
            )

            rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = (
                lidar_rasterization(
                    means=render_means,
                    quats=render_quats,
                    scales=render_scales,
                    opacities=render_opacities.squeeze(-1),
                    lidar_features=lidar_features,
                    velocities=None,
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
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=0.0,
                    sparse_grad=False,
                    absgrad=True,
                    channel_chunk=128,
                    compute_alpha_sum_until_points=False,
                    compute_alpha_sum_until_points_threshold=0.8,
                )
            )

            # 后处理：Mask处理与深度滤波
            out = self._process_lidar_output(raster_pts, rendered_feat)

            from util import pano_to_lidar_with_intensities

            pred, gt = pano_to_lidar_with_intensities(raster_pts, out)
            lidars[lidar_id] = pred

        return lidars

    def _process_lidar_output(
        self, raster_pts: torch.Tensor, rendered_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """处理Lidar渲染的原始输出，包括Mask对齐和深度滤波"""
        lidar_intensity = rendered_feat[..., 0].squeeze(0)
        lidar_ray_drop_logits = rendered_feat[..., 1].squeeze(0)
        lidar_depth_render = rendered_feat[..., -1].squeeze(0)

        # 计算有效性Mask
        raster_pts_did_return = (raster_pts[..., 2] <= 1000).squeeze(0)
        raster_pts_valid_depth_and_did_return = raster_pts_did_return & (
            raster_pts[..., 2] > 0
        ).squeeze(0)
        gt_valid = (raster_pts[..., 2] > 0).squeeze(0)

        # 形状对齐处理 (兼容性代码)
        if raster_pts_valid_depth_and_did_return.shape != lidar_depth_render.shape:
            if (
                raster_pts_valid_depth_and_did_return.transpose(0, 1).shape
                == lidar_depth_render.shape
            ):
                raster_pts_valid_depth_and_did_return = (
                    raster_pts_valid_depth_and_did_return.transpose(0, 1)
                )
            elif (
                raster_pts_valid_depth_and_did_return.numel()
                == lidar_depth_render.numel()
            ):
                raster_pts_valid_depth_and_did_return = (
                    raster_pts_valid_depth_and_did_return.reshape(
                        lidar_depth_render.shape
                    )
                )

        if gt_valid.shape != lidar_depth_render.shape:
            if gt_valid.transpose(0, 1).shape == lidar_depth_render.shape:
                gt_valid = gt_valid.transpose(0, 1)
            elif gt_valid.numel() == lidar_depth_render.numel():
                gt_valid = gt_valid.reshape(lidar_depth_render.shape)

        # 应用Mask
        valid_mask = raster_pts_valid_depth_and_did_return.to(lidar_depth_render.dtype)
        lidar_intensity = lidar_intensity * valid_mask
        lidar_depth_render = lidar_depth_render * valid_mask

        gt_valid_mask = gt_valid.to(lidar_ray_drop_logits.dtype)
        lidar_ray_drop_logits = (
            lidar_ray_drop_logits * gt_valid_mask - (1.0 - gt_valid_mask) * 10000.0
        )

        # 深度滤波
        lidar_depth_render, lidar_intensity = self._apply_depth_filter(
            lidar_depth_render, lidar_intensity
        )

        return {
            "depth": lidar_depth_render,
            "intensity": lidar_intensity,
            "ray_drop_prob": lidar_ray_drop_logits,
        }

    def _apply_depth_filter(
        self, depth: torch.Tensor, intensity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对深度图进行滤波，去除边缘伪影"""
        # 使用相邻列深度差异进行滤波
        depth_modified = depth.clone()
        depth_sample = depth_modified  # [H, W]

        # 计算相邻列的深度差异 (列方向: width维度，对应维度1)
        depth_diff = depth_sample[:, 1:] - depth_sample[:, :-1]
        depth_diff = torch.abs(depth_diff)

        # 深度差异大于0.5的位置设为0,否则为1
        depth_diff_mask = (depth_diff <= 0.5).float()

        # 创建完整mask,第一列保持为1
        full_mask = torch.ones_like(depth_sample)
        full_mask[:, 1:] = depth_diff_mask

        # 应用mask
        return depth * full_mask, intensity * full_mask

    def _compute_lidar_features_from_colors(
        self,
        colors: torch.Tensor,
        means: torch.Tensor,
        viewmats: torch.Tensor,
    ) -> torch.Tensor:
        """从`render_colors`中提取雷达通道并转换为`[C, N, 2]`的`lidar_features`。

        - 输入`colors`形状为`[N, K, D]`，其中`K`为启用的球谐基数量（通常为4，对应degree=1），
          `D`为每个基的通道数；当`D>=5`时，假定前3个通道为相机渲染用，后2个通道为雷达特征系数。
        - 为了与球谐评估接口保持一致，将雷达的2通道系数在最后一维填充一个0，形成3通道，
          使用同样的degree=1进行方向相关的评估，然后只保留前2个通道作为雷达特征。
        - 如果`colors`不含雷达通道（例如`D==3`），则返回零特征以保持向后兼容。
        """  # noqa: E501
        C = viewmats.shape[0]
        N = means.shape[0]
        device = self.device

        if colors.dim() != 3:
            return torch.zeros(C, N, 2, device=device, dtype=torch.float32)

        D = colors.shape[-1]
        K = colors.shape[1]
        if D < 5 or K < 4:
            return torch.zeros(C, N, 2, device=device, dtype=colors.dtype)

        camera_centers = extract_camera_centers(viewmats)  # [C, 3]
        dirs = means[None, :, :] - camera_centers[:, None, :]  # [C, N, 3]

        lidar_coeffs = colors[..., 3:5]  # [N, K, 2]
        zeros_third = torch.zeros((N, K, 1), device=device, dtype=colors.dtype)
        lidar_coeffs_pad = torch.cat([lidar_coeffs, zeros_third], dim=-1)  # [N, K, 3]
        shs = lidar_coeffs_pad.unsqueeze(0).expand(C, -1, -1, -1)  # [C, N, K, 3]

        feats = spherical_harmonics(1, dirs, shs)  # [C, N, 3]
        return feats[..., :2]
