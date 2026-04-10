import math
from typing import Dict, List, Optional, Tuple

import torch
from gsplat import spherical_harmonics
from gsplat.rendering import lidar_rasterization

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
from mlp_decoder import MLPDecoder
from models import GaussianComponent, GSModel
from render_kernel import (
    extract_camera_centers,
    get_ray_dirs_cam_pinhole_batched,
    invert_world2camera,
    render,
)
from rgb_decoder import RGBDecoder


class RenderManager:
    def __init__(self, model: str = "model.path") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model
        self.model = GSModel.load_from_pth(model).to_device(self.device)
        self.background: GaussianComponent = self.model.get_component("background")
        self.sky: GaussianComponent = self.model.get_component("sky")
        self.actors: List[GaussianComponent] = self.model.get_components_by_type("obj")
        map_center = getattr(self.model, "map_center", None)
        if map_center is None:
            raise ValueError(
                "pth中缺少map_center(scene参数)，无法将ego轨迹转换到模型坐标系"
            )  # noqa: E501
        self.map_center = map_center.to(self.device)

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
        self.mlp_decoder: MLPDecoder | None = None
        self.camera_id_to_index: Dict[str, int] = {}
        self.lidar_id_to_index: Dict[str, int] = {}

    def _setup_render_buffers(self) -> None:
        """预分配渲染buffers，消除每帧内存分配

        预计算静态数据大小，为动态数据预留最大可能空间
        这样每帧只需要更新动态部分，零内存分配
        """
        # 计算静态点云数据：训练时 sky 会单独渲染并用 alpha 合成，这里保持一致
        static_components = [self.background]
        self.static_data = GaussianData.from_components(static_components)
        if not self.static_data:
            raise ValueError("静态点云组件不能为空")
        self.static_points = self.static_data.means.shape[0]
        color_channels = self.static_data.colors.shape[2]

        self.sky_data = GaussianData.from_components([self.sky])
        self.sky_points = self.sky_data.means.shape[0] if self.sky_data else 0

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
            "colors": torch.empty(
                (total_max_points, 4, color_channels),
                device=device,
                dtype=torch.float32,
            ),
        }
        self.zero_velocities = torch.zeros(
            (total_max_points, 3), device=device, dtype=torch.float32
        )

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

    def _group_cameras_by_resolution(
        self, cameras: List[Camera]
    ) -> Dict[Tuple[int, int], List[Camera]]:
        grouped: Dict[Tuple[int, int], List[Camera]] = {}
        for cam in cameras:
            resolution = (cam.width, cam.height)
            grouped.setdefault(resolution, []).append(cam)
        return grouped

    def _get_rgb_decoder_state_and_num_cams(
        self, cameras: List[Camera]
    ) -> Tuple[Optional[dict], int]:
        rgb_decoder_state = getattr(self.model, "rgb_decoder_state", None)
        if rgb_decoder_state is None or not isinstance(rgb_decoder_state, dict):
            return None, len(cameras)
        params_state = rgb_decoder_state.get("params")
        if isinstance(params_state, dict) and "appearance_embedding" in params_state:
            return rgb_decoder_state, int(params_state["appearance_embedding"].shape[0])
        return rgb_decoder_state, len(cameras)

    def _init_rgb_decoder(
        self,
        cameras: List[Camera],
        num_cams_ckpt: int,
        rgb_decoder_state: Optional[dict],
    ) -> None:
        self.camera_id_to_index = {cam.id: idx for idx, cam in enumerate(cameras)}
        if rgb_decoder_state is not None:
            self.rgb_decoder = RGBDecoder.from_checkpoint_state(
                metadata={"num_cams": num_cams_ckpt},
                state_dict=rgb_decoder_state,
                device=self.device,
                mode="image",
            )
        else:
            self.rgb_decoder = RGBDecoder(
                metadata={"num_cams": num_cams_ckpt},
                device=self.device,
                use_app_embed=True,
                mode="image",
            )
        self.rgb_decoder.camera_id_map = self.camera_id_to_index
        self.rgb_decoder.eval()

    def _build_camera_data(self) -> None:
        self.camera_data = {}
        for resolution, cameras in self.cameras.items():
            width, height = resolution
            intrinsics_list = [cam.intrinsics for cam in cameras]
            Ks = torch.tensor(intrinsics_list, dtype=torch.float32, device=self.device)
            camera_ids = [cam.id for cam in cameras]
            extrinsics_list = [cam.extrinsics for cam in cameras]
            extrinsics_tensor = torch.tensor(
                extrinsics_list, dtype=torch.float32, device=self.device
            )
            ray_dirs_cam = get_ray_dirs_cam_pinhole_batched(Ks, width, height)
            self.camera_data[resolution] = {
                "camera_ids": camera_ids,
                "extrinsics_tensor": extrinsics_tensor,
                "intrinsics_tensor": Ks,
                "width": width,
                "height": height,
                "ray_dirs_cam": ray_dirs_cam,
            }

    def _prepare_lidar_grids_from_raster_pts(
        self, raster_pts: torch.Tensor, tile_height: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 checkpoint 保存的 raster_pts 推断 azimuths/elevations 与 elevation_boundaries。

        返回:
            raster_pts_4d: [1, H, W, D] 的 raster_pts
            azimuths: [A] 方位角序列（单位：度）
            elevations: [E] 俯仰角序列（单位：度）
            elevation_boundaries: [E//tile_height + 1] 用于瓦片划分的边界
        """  # noqa: E501
        if not isinstance(raster_pts, torch.Tensor):
            raster_pts = torch.as_tensor(raster_pts)
        raster_pts = raster_pts.to(device=self.device, dtype=torch.float32)
        if raster_pts.dim() == 3:
            raster_pts = raster_pts.unsqueeze(0)
        if raster_pts.dim() != 4:
            raise ValueError(
                f"raster_pts维度必须为3或4，实际为{tuple(raster_pts.shape)}"
            )

        az_grid = raster_pts[0, ..., 0]
        el_grid = raster_pts[0, ..., 1]

        az_var0 = az_grid[:, 0].std()
        az_var1 = az_grid[0, :].std()
        az_axis = 0 if az_var0 >= az_var1 else 1

        el_var0 = el_grid[:, 0].std()
        el_var1 = el_grid[0, :].std()
        el_axis = 0 if el_var0 >= el_var1 else 1
        if el_axis == az_axis:
            el_axis = 1 - az_axis

        azimuths = az_grid[:, 0] if az_axis == 0 else az_grid[0, :]
        elevations = el_grid[:, 0] if el_axis == 0 else el_grid[0, :]

        elevation_boundaries = torch.cat([
            elevations[0:1] - 1.0,
            (
                elevations[tile_height::tile_height]
                + elevations[tile_height - 1 : -1 : tile_height]
            )
            / 2,
            elevations[-1:] + 1.0,
        ])

        return raster_pts, azimuths, elevations, elevation_boundaries

    def _load_checkpoint_dict(self) -> dict:
        """加载原始checkpoint字典，用于提取高斯组件之外的子模块权重。"""
        return torch.load(self.model_path, map_location="cpu", weights_only=False)

    def _get_mlp_decoder_state_and_num_lidars(
        self, lidars: Optional[List[Lidar]]
    ) -> Tuple[Optional[dict], int]:
        """从checkpoint解析MLPDecoder权重以及其支持的lidar数量。"""
        if not lidars:
            return None, 0

        ckpt = self._load_checkpoint_dict()
        state = None
        for key in ("MLPDecoder", "mlp_decoder", "lidar_decoder"):
            value = ckpt.get(key, None) if isinstance(ckpt, dict) else None
            if isinstance(value, dict):
                state = value
                break

        if state is None:
            return None, len(lidars)

        params_state = state.get("params") if isinstance(state, dict) else None
        if isinstance(params_state, dict) and "appearance_dim" in params_state:
            app = params_state["appearance_dim"]
            if isinstance(app, torch.Tensor) and app.ndim == 2:
                return state, int(app.shape[0])
        return state, len(lidars)

    def _init_mlp_decoder(self, lidars: Optional[List[Lidar]]) -> None:
        """初始化激光雷达的MLP解码器（可选）。"""
        if not self.render_lidar or not lidars:
            self.mlp_decoder = None
            self.lidar_id_to_index = {}
            return

        self.lidar_id_to_index = {lidar.id: idx for idx, lidar in enumerate(lidars)}
        mlp_state, num_lidars_ckpt = self._get_mlp_decoder_state_and_num_lidars(lidars)

        if mlp_state is None:
            self.mlp_decoder = None
            return

        if len(lidars) > num_lidars_ckpt:
            raise ValueError(
                "激光雷达数量超过MLPDecoder权重支持范围: "
                f"lidars={len(lidars)} ckpt={num_lidars_ckpt}"
            )

        self.mlp_decoder = MLPDecoder.from_checkpoint_state(
            metadata={"num_lidars": num_lidars_ckpt},
            state_dict=mlp_state,
            device=self.device,
        )
        self.mlp_decoder.lidar_id_map = self.lidar_id_to_index
        self.mlp_decoder.eval()

    def _build_lidar_data(self, lidars: Optional[List[Lidar]]) -> None:
        if not lidars:
            return

        if self.model.raster_pts is None:
            raise ValueError(
                "pth中缺少raster_pts，当前实现不再在运行时构建raster_pts；请在训练/导出时将raster_pts保存到checkpoint中"
            )

        for lidar in lidars:
            self.lidars[lidar.id] = lidar

            raster_pts, azimuths, elevations, elevation_boundaries = (
                self._prepare_lidar_grids_from_raster_pts(
                    self.model.raster_pts, tile_height=lidar.tile_height
                )
            )

            if azimuths.numel() > 1:
                azimuth_resolution = torch.diff(azimuths).abs().median().item()
            else:
                azimuth_resolution = float(lidar.azimuth_resolution)

            min_azimuth = float(azimuths.min().item())
            max_azimuth = float(azimuths.max().item())
            min_elevation = float(elevations.min().item())
            max_elevation = float(elevations.max().item())

            image_width = int(raster_pts.shape[1])
            image_height = int(raster_pts.shape[2])

            angles = torch.deg2rad(raster_pts[..., :2])
            az = angles[..., 0:1]
            el = angles[..., 1:2]
            ray_dirs_lidar = torch.cat(
                [
                    torch.cos(az) * torch.cos(el),
                    torch.sin(az) * torch.cos(el),
                    torch.sin(el),
                ],
                dim=-1,
            )
            ray_dirs_lidar = ray_dirs_lidar / (
                ray_dirs_lidar.norm(dim=-1, keepdim=True) + 1e-8
            )

            gt_depth = raster_pts[..., 2].squeeze(0)
            did_return_threshold = 1000.0
            gt_did_return = gt_depth <= did_return_threshold
            depth_valid_mask = (gt_depth > 0) & gt_did_return

            extrinsics_tensor = torch.tensor(
                [lidar.extrinsics], dtype=torch.float32, device=self.device
            )

            self.lidar_data[lidar.id] = {
                "extrinsics_tensor": extrinsics_tensor,
                "azimuths": azimuths,
                "elevations": elevations,
                "elevation_boundaries": elevation_boundaries,
                "image_width": image_width,
                "image_height": image_height,
                "tile_width": lidar.tile_width,
                "tile_height": lidar.tile_height,
                "min_azimuth": min_azimuth,
                "max_azimuth": max_azimuth,
                "min_elevation": min_elevation,
                "max_elevation": max_elevation,
                "azimuth_resolution": azimuth_resolution,
                "raster_pts": raster_pts,
                "ray_dirs_lidar": ray_dirs_lidar,
                "pano_dirs_lidar": ray_dirs_lidar.squeeze(0),
                "depth_valid_mask": depth_valid_mask,
            }

    def init(self, params: InitParams) -> InitResp:
        """初始化接口，输入相机的参数

        预计算所有相机数据，消除运行时查找和转换
        """
        self.render_camera = bool(getattr(params, "render_camera", True))
        self.render_lidar = bool(getattr(params, "render_lidar", False))
        self.cameras = self._group_cameras_by_resolution(params.cameras)
        rgb_decoder_state, num_cams_ckpt = self._get_rgb_decoder_state_and_num_cams(
            params.cameras
        )

        if len(params.cameras) > num_cams_ckpt:
            raise ValueError(
                "相机数量超过rgb_decoder权重支持范围: "
                f"cameras={len(params.cameras)} ckpt={num_cams_ckpt}"
            )

        self._init_rgb_decoder(params.cameras, num_cams_ckpt, rgb_decoder_state)
        self._build_camera_data()
        self._build_lidar_data(params.lidars)
        self._init_mlp_decoder(params.lidars)
        return InitResp(init_status=True)

    def _calculate_viewmats(
        self,
        extrinsics_tensor: torch.Tensor,
        ego_heading: float,
        ego_position: torch.Tensor,
    ) -> torch.Tensor:
        device = ego_position.device
        cos_h = math.cos(float(ego_heading))
        sin_h = math.sin(float(ego_heading))
        ego_pose = torch.tensor(
            [
                [cos_h, -sin_h, 0.0, 0.0],
                [sin_h, cos_h, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        ego_pose = ego_pose.clone()
        ego_pose[:3, 3] = ego_position

        c2w = torch.matmul(ego_pose.unsqueeze(0), extrinsics_tensor)

        R = c2w[:, :3, :3]
        t = c2w[:, :3, 3]
        Rt = R.transpose(-2, -1)
        t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)

        w2c = (
            torch
            .eye(4, device=device, dtype=torch.float32)
            .expand(c2w.shape[0], 4, 4)
            .clone()
        )
        w2c[:, :3, :3] = Rt
        w2c[:, :3, 3] = t_inv
        return w2c

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
            extrinsics_tensor = cam_data["extrinsics_tensor"]
            Ks = cam_data["intrinsics_tensor"]
            width = cam_data["width"]
            height = cam_data["height"]

            viewmats = self._calculate_viewmats(
                extrinsics_tensor, ego_heading, ego_position
            )

            ray_dirs_world = None
            if self.rgb_decoder is not None and getattr(
                self.rgb_decoder, "use_ray_dirs", False
            ):
                dirs_cam = cam_data["ray_dirs_cam"]
                c2w = invert_world2camera(viewmats)
                R = c2w[:, :3, :3]
                C = int(dirs_cam.shape[0])
                dirs_flat = dirs_cam.view(C, -1, 3)
                ray_world_flat = torch.bmm(dirs_flat, R.transpose(1, 2))
                ray_dirs_world = ray_world_flat.view(C, height, width, 3)

            # 先渲染前景（background + actors，不包含 sky）
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
                ray_dirs_world=ray_dirs_world,
            )

            # 再渲染 sky，并用前景 alpha 做合成：rgb = fg + sky * (1 - acc)
            if self.sky_data is not None and self.sky_points > 0:
                sky_colors, _sky_alphas = render(
                    self.sky_data.means,
                    self.sky_data.quats,
                    self.sky_data.scales,
                    self.sky_data.opacities,
                    self.sky_data.colors,
                    viewmats,
                    Ks,
                    width,
                    height,
                    rgb_decoder=self.rgb_decoder,
                    camera_ids=camera_ids,
                    ray_dirs_world=ray_dirs_world,
                )
                batch_colors = batch_colors + sky_colors * (1 - batch_alphas)

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
            extrinsics_tensor = lidar_cfg["extrinsics_tensor"]
            viewmats = self._calculate_viewmats(
                extrinsics_tensor, ego_heading, ego_position
            )
            if self.mlp_decoder is not None:
                lidar_features = self._compute_lidar_mlp_features_from_colors(
                    render_colors,
                    feature_dim=self.mlp_decoder.feature_dim,
                    batch_size=viewmats.shape[0],
                )
            else:
                lidar_features = self._compute_lidar_features_from_colors(
                    render_colors, render_means, viewmats
                )
            raster_pts = lidar_cfg["raster_pts"]

            near_plane = 0.2
            far_plane = 300.0

            lidar_linear_vel = torch.zeros(
                (1, 3), device=render_means.device, dtype=render_means.dtype
            )
            lidar_angular_vel = torch.zeros(
                (1, 3), device=render_means.device, dtype=render_means.dtype
            )
            rolling_shutter_time = torch.zeros(
                (1,), device=render_means.device, dtype=render_means.dtype
            )

            rendered_feat, rendered_alpha, alpha_sum_until_points, meta_info = (
                lidar_rasterization(
                    means=render_means,
                    quats=render_quats,
                    scales=render_scales,
                    opacities=render_opacities.squeeze(-1),
                    lidar_features=lidar_features,
                    velocities=self.zero_velocities[: render_means.shape[0]],
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
                    linear_velocity=lidar_linear_vel,
                    angular_velocity=lidar_angular_vel,
                    rolling_shutter_time=rolling_shutter_time,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=0.0,
                    sparse_grad=False,
                    absgrad=True,
                    rasterize_mode="antialiased",
                    channel_chunk=128,
                    eps2d=0.01718873385,
                    compute_alpha_sum_until_points=False,
                    compute_alpha_sum_until_points_threshold=0.8,
                )
            )

            if self.mlp_decoder is not None:
                ray_dirs_world = None
                ray_dirs_lidar = lidar_cfg.get("ray_dirs_lidar", None)
                if isinstance(ray_dirs_lidar, torch.Tensor):
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

                decoded_intensity, decoded_ray_drop_logits = self.mlp_decoder(
                    lidar_id,
                    rendered_feat[..., :-1],
                    raster_pts,
                    viewmats,
                    ray_dirs_world=ray_dirs_world,
                )
                decoded_rendered = torch.cat(
                    [
                        decoded_intensity,
                        decoded_ray_drop_logits,
                        rendered_feat[..., -1:],
                    ],
                    dim=-1,
                )
                out = self._process_lidar_output(raster_pts, decoded_rendered)
            else:
                out = self._process_lidar_output(raster_pts, rendered_feat)

            from util import pano_to_lidar_with_intensities

            pred = pano_to_lidar_with_intensities(
                raster_pts,
                out,
                directions=lidar_cfg.get("pano_dirs_lidar", None),
                depth_valid_mask=lidar_cfg.get("depth_valid_mask", None),
            )
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

    def _compute_lidar_mlp_features_from_colors(
        self,
        colors: torch.Tensor,
        feature_dim: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """从高斯的colors中构建供MLPDecoder使用的lidar_features。

        约定：
        - colors 形状为 [N, 4, C]，其中 C 的前3个通道为相机渲染(RGB)；从第3个通道开始视为额外特征。
        - 将额外特征在 [K=4] 维度上展平为每个高斯的 1D feature 向量，并裁剪/补零到 feature_dim。

        Args:
            colors: [N, 4, C] 的特征张量。
            feature_dim: 目标特征维度（与MLPDecoder期望一致）。
            batch_size: 输出批大小，通常等于 viewmats.shape[0]。

        Returns:
            lidar_features: [batch_size, N, feature_dim]
        """  # noqa: E501
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

        extra = colors[..., 3:]  # [N, 4, C-3]
        flat = extra.reshape(N, -1)  # [N, 4*(C-3)]

        if flat.shape[1] < feature_dim:
            pad = torch.zeros(
                N, feature_dim - flat.shape[1], device=device, dtype=dtype
            )
            flat = torch.cat([flat, pad], dim=-1)
        feat = flat[:, :feature_dim]

        return feat.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
