import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from camera_renderer import (
    build_camera_data,
    get_rgb_decoder_state_and_num_cams,
    group_cameras_by_resolution,
    init_rgb_decoder,
    render_cameras,
)
from data_types import (
    Camera,
    FrameParams,
    FrameResp,
    InitParams,
    InitResp,
    Lidar,
)
from gaussian_buffers import RenderBuffers, build_render_buffers, update_dynamic_buffer
from lidar_renderer import build_lidar_data, init_mlp_decoder, render_lidars
from mlp_decoder import MLPDecoder
from models import GaussianComponent, GSModel
from render_runtime import validate_frame_params, validate_init_params
from rgb_decoder import RGBDecoder


class RenderManager:
    def __init__(
        self, model: str = "model.path", enable_torch_backends: bool = True
    ) -> None:
        self._configure_torch_backends(enable_torch_backends)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model
        self.model = GSModel.load_from_pth(model).to_device(self.device)
        self.model.validate_for_render()
        self.background = self._required_component("background")
        self.sky: GaussianComponent | None = self.model.get_component("sky")
        self.sky_cubemap: torch.Tensor | None = getattr(self.model, "sky_cubemap", None)
        self.actors: List[GaussianComponent] = self.model.get_components_by_type("obj")
        self.map_center = self._required_tensor("map_center").to(self.device)

        self.actor_map: Dict[str, GaussianComponent] = {
            actor.name: actor for actor in self.actors
        }
        self.buffers: RenderBuffers = build_render_buffers(
            self.background, self.sky, self.actors, self.device
        )
        self.cameras: Dict[Tuple[int, int], List[Camera]] = {}
        self.camera_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.lidars: Dict[str, Lidar] = {}
        self.lidar_data: Dict[str, Dict[str, Any]] = {}
        self.render_camera: bool = True
        self.render_lidar: bool = False
        self.rgb_decoder: RGBDecoder | None = None
        self.mlp_decoder: MLPDecoder | None = None
        self.camera_id_to_index: Dict[str, int] = {}
        self.lidar_id_to_index: Dict[str, int] = {}
        self._initialized: bool = False
        self._frame_json_cache: Optional[dict] = None
        self._frame_json_loaded: bool = False
        self._frame_json_path: Optional[str] = None

    def _required_component(self, name: str) -> GaussianComponent:
        component = self.model.get_component(name)
        if component is None:
            raise ValueError(f"pth中缺少{name}组件，无法渲染")
        return component

    def _required_tensor(self, attr_name: str) -> torch.Tensor:
        value = getattr(self.model, attr_name, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"pth中缺少{attr_name}(scene参数)，无法渲染")
        return value

    @staticmethod
    def _configure_torch_backends(enable: bool) -> None:
        """配置PyTorch推理侧性能相关开关。

        Args:
            enable: 是否开启。
                - 默认建议开启以提升CNN推理吞吐。
                - 如需强制覆盖，可设置环境变量 RENDER_TORCH_BACKENDS=0/1。
        """
        env = os.environ.get("RENDER_TORCH_BACKENDS")
        if env is not None:
            enable = env == "1"
        if not enable:
            return

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    def init(self, params: InitParams) -> InitResp:
        """初始化接口，输入相机的参数

        预计算所有相机数据，消除运行时查找和转换
        """
        validate_init_params(params)
        self.render_camera = bool(getattr(params, "render_camera", True))
        self.render_lidar = bool(getattr(params, "render_lidar", False))
        self._reset_runtime_state()
        cameras = params.cameras or []

        if self.render_camera:
            self.cameras = group_cameras_by_resolution(cameras)
            rgb_decoder_state, num_cams_ckpt = get_rgb_decoder_state_and_num_cams(
                getattr(self.model, "rgb_decoder_state", None), cameras
            )
            if len(cameras) > num_cams_ckpt:
                raise ValueError(
                    "相机数量超过rgb_decoder权重支持范围: "
                    f"cameras={len(cameras)} ckpt={num_cams_ckpt}"
                )
            self.rgb_decoder, self.camera_id_to_index = init_rgb_decoder(
                cameras, num_cams_ckpt, rgb_decoder_state, self.device
            )
            self.camera_data = build_camera_data(self.cameras, self.device)

        self.lidars, self.lidar_data = build_lidar_data(params.lidars, self.device)
        self.mlp_decoder, self.lidar_id_to_index = init_mlp_decoder(
            self.render_lidar, params.lidars, self.model.mlp_decoder_state, self.device
        )
        self._initialized = True
        return InitResp(init_status=True)

    def _reset_runtime_state(self) -> None:
        """清理init阶段生成的运行期缓存，支持重复初始化。"""
        self.cameras = {}
        self.camera_data = {}
        self.lidars = {}
        self.lidar_data = {}
        self.rgb_decoder = None
        self.mlp_decoder = None
        self.camera_id_to_index = {}
        self.lidar_id_to_index = {}
        self._initialized = False

    def render_frame(self, params: FrameParams) -> FrameResp:
        """渲染接口，每帧调用"""
        validate_frame_params(params, self._initialized)

        ego_heading = params.ego_yaw
        ego_position = (
            torch.tensor(
                params.ego_trajectory, device=self.device, dtype=torch.float32
            )
            - self.map_center
        )

        dynamic_points = update_dynamic_buffer(
            self.buffers,
            self.actor_map,
            params.env_vehicles or [],
            self.map_center,
            self.device,
        )
        render_params = self.buffers.render_params(dynamic_points)

        images = self._render_cameras(ego_heading, ego_position, render_params)
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
        return render_cameras(
            self.camera_data,
            self.render_camera,
            ego_heading,
            ego_position,
            render_params,
            self.rgb_decoder,
            self.buffers.sky_data,
            self.buffers.sky_points,
            self.sky_cubemap,
        )

    def _render_lidars(
        self,
        ego_heading: float,
        ego_position: torch.Tensor,
        render_params: Tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        return render_lidars(
            self.lidar_data,
            self.render_lidar,
            ego_heading,
            ego_position,
            render_params,
            self.mlp_decoder,
            self.buffers.zero_velocities,
        )
