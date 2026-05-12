"""3DGS模型主管理类。"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .gs_component import GaussianComponent

SCENE_KEYS = {
    "iter",
    "raster_pts",
    "rgb_decoder",
    "MLPDecoder",
    "mlp_decoder",
    "lidar_decoder",
    "center_point",
    "sphere_center",
    "sphere_radius",
    "sky_cubemap",
}
COMPONENT_REQUIRED_KEYS = (
    "xyz",
    "feature_dc",
    "feature_rest",
    "scaling",
    "rotation",
    "opacity",
)
MLP_DECODER_KEYS = ("MLPDecoder", "mlp_decoder", "lidar_decoder")


def _as_f32_tensor(value) -> torch.Tensor:
    """将checkpoint里的场景参数统一转为float32 Tensor。"""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _optional_decoder_state(checkpoint: dict, keys: tuple[str, ...]) -> Optional[dict]:
    """按候选键查找decoder权重，返回第一个dict类型的权重。"""
    for key in keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return None


class GSModel:
    """3DGS模型管理类。

    管理模型的所有高斯点云组件，提供加载、保存和操作接口。
    """

    def __init__(self) -> None:
        """初始化空的GSModel。"""
        self.components: Dict[str, GaussianComponent] = {}
        self.iteration: int = 0
        self.raster_pts: Optional[torch.Tensor] = None
        self.rgb_decoder_state: Optional[dict] = None
        self.mlp_decoder_state: Optional[dict] = None

        self.map_center: Optional[torch.Tensor] = None
        self.sky_center: Optional[torch.Tensor] = None
        self.sky_radius: Optional[torch.Tensor] = None
        self.sky_cubemap: Optional[torch.Tensor] = None

    @classmethod
    def load_from_pth(cls, pth_path: Union[str, Path], strict: bool = True) -> "GSModel":
        """从PTH文件加载模型。

        Args:
            pth_path: PTH文件路径
            strict: 为True时组件验证失败会抛出异常；为False时跳过无效组件

        Returns:
            加载的GSModel实例
        """
        pth_path = Path(pth_path)
        if not pth_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {pth_path}")

        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"模型checkpoint必须是dict，实际为{type(checkpoint)!r}")

        model = cls()

        model._load_scene_metadata(checkpoint)
        model._load_components(checkpoint, strict=strict)
        model.validate_for_render()

        return model

    def _load_scene_metadata(self, checkpoint: dict) -> None:
        """加载非高斯组件的场景元数据和decoder权重。"""
        self.iteration = int(checkpoint.get("iter", 0))
        self.raster_pts = checkpoint.get("raster_pts")
        self.rgb_decoder_state = checkpoint.get("rgb_decoder")
        self.mlp_decoder_state = _optional_decoder_state(checkpoint, MLP_DECODER_KEYS)

        if "center_point" not in checkpoint:
            raise KeyError("pth缺少场景参数键: 需要center_point")
        self.map_center = _as_f32_tensor(checkpoint["center_point"])
        if self.map_center.shape != (3,):
            raise ValueError(
                f"center_point期望形状为(3,)，实际为{tuple(self.map_center.shape)}"
            )

        if "sphere_center" in checkpoint and "sphere_radius" in checkpoint:
            self.sky_center = _as_f32_tensor(checkpoint["sphere_center"])
            self.sky_radius = _as_f32_tensor(checkpoint["sphere_radius"])

        sky_cubemap = checkpoint.get("sky_cubemap")
        params = sky_cubemap.get("params") if isinstance(sky_cubemap, dict) else None
        cube = params.get("sky_cube_map") if hasattr(params, "get") else None
        if cube is not None:
            self.sky_cubemap = _as_f32_tensor(cube)

    def _load_components(self, checkpoint: dict, strict: bool) -> None:
        """加载checkpoint中的高斯组件。"""
        errors = []
        for name, data in checkpoint.items():
            if name in SCENE_KEYS or not isinstance(data, dict) or not data:
                continue
            if not all(key in data for key in COMPONENT_REQUIRED_KEYS):
                continue

            component = self._build_component(name, data)
            if component.validate():
                self.components[name] = component
                continue

            message = f"组件 {name} 数据验证失败"
            if strict:
                errors.append(message)

        if errors:
            raise ValueError("; ".join(errors))

    def _build_component(self, name: str, data: dict) -> GaussianComponent:
        """从checkpoint字段构建单个高斯组件。"""
        sky_args = {}
        if name == "sky":
            sky_args = {"sky_center": self.sky_center, "sky_radius": self.sky_radius}
        return GaussianComponent(
            name=name,
            xyz=data["xyz"],
            feature_dc=data["feature_dc"],
            feature_rest=data["feature_rest"],
            scaling=data["scaling"],
            rotation=data["rotation"],
            opacity=data["opacity"],
            semantic=data.get("semantic"),
            **sky_args,
        )

    def validate_for_render(self) -> None:
        """校验render_frame最小依赖的数据。"""
        if self.map_center is None:
            raise ValueError("模型缺少center_point，无法进行ego坐标转换")
        if "background" not in self.components:
            raise ValueError("模型缺少background组件，无法渲染")
        if not self.components:
            raise ValueError("模型没有可用高斯组件")

    def save_to_pth(self, pth_path: Union[str, Path]) -> None:
        """保存模型到PTH文件。

        Args:
            pth_path: 保存路径
        """
        if self.map_center is None:
            raise ValueError("保存pth前必须设置map_center")
        if "sky" in self.components and (
            self.sky_center is None or self.sky_radius is None
        ):
            raise ValueError("保存包含sky组件的pth前必须设置sky_center/sky_radius")

        checkpoint = {
            "iter": self.iteration,
            "center_point": self.map_center,
        }
        if self.sky_center is not None and self.sky_radius is not None:
            checkpoint["sphere_center"] = self.sky_center
            checkpoint["sphere_radius"] = self.sky_radius
        if self.raster_pts is not None:
            checkpoint["raster_pts"] = self.raster_pts
        if self.rgb_decoder_state is not None:
            checkpoint["rgb_decoder"] = self.rgb_decoder_state
        if self.mlp_decoder_state is not None:
            checkpoint["MLPDecoder"] = self.mlp_decoder_state

        for name, component in self.components.items():
            data = {
                "xyz": component.xyz,
                "feature_dc": component.feature_dc,
                "feature_rest": component.feature_rest,
                "scaling": component.scaling,
                "rotation": component.rotation,
                "opacity": component.opacity,
            }

            if component.semantic is not None:
                data["semantic"] = component.semantic
            checkpoint[name] = data

        torch.save(checkpoint, pth_path)

    def add_component(self, component: GaussianComponent) -> None:
        """添加组件。

        Args:
            component: 要添加的高斯组件
        """
        if not component.validate():
            raise ValueError(f"组件 {component.name} 数据验证失败")

        self.components[component.name] = component

    def remove_component(self, name: str) -> Optional[GaussianComponent]:
        """移除组件。

        Args:
            name: 组件名称

        Returns:
            被移除的组件，如果不存在则返回None
        """
        return self.components.pop(name, None)

    def get_component(self, name: str) -> Optional[GaussianComponent]:
        """获取组件。

        Args:
            name: 组件名称

        Returns:
            组件实例，如果不存在则返回None
        """
        return self.components.get(name)

    def get_components_by_type(self, component_type: str) -> List[GaussianComponent]:
        """按类型获取组件。

        Args:
            component_type: 组件类型前缀（如'obj'、'background'、'sky'）

        Returns:
            匹配的组件列表
        """
        return [
            comp
            for name, comp in self.components.items()
            if name.startswith(component_type)
        ]

    @property
    def total_points(self) -> int:
        """获取总点云数量。"""
        return sum(comp.num_points for comp in self.components.values())

    @property
    def total_memory(self) -> float:
        """获取总内存使用量（MB）。"""
        return sum(comp.memory_usage for comp in self.components.values())

    @property
    def component_names(self) -> List[str]:
        """获取所有组件名称。"""
        return list(self.components.keys())

    def to_device(self, device: torch.device) -> "GSModel":
        """将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            新的GSModel实例
        """
        new_model = GSModel()
        new_model.iteration = self.iteration
        new_model.rgb_decoder_state = self.rgb_decoder_state
        new_model.mlp_decoder_state = self.mlp_decoder_state
        if self.raster_pts is not None:
            new_model.raster_pts = self.raster_pts.to(device)

        new_model.map_center = (
            self.map_center.to(device) if self.map_center is not None else None
        )
        new_model.sky_center = (
            self.sky_center.to(device) if self.sky_center is not None else None
        )
        new_model.sky_radius = (
            self.sky_radius.to(device) if self.sky_radius is not None else None
        )
        new_model.sky_cubemap = (
            self.sky_cubemap.to(device) if self.sky_cubemap is not None else None
        )

        for name, component in self.components.items():
            new_model.components[name] = component.to_device(device)

        return new_model

    def summary(self) -> str:
        """获取模型摘要信息。"""
        lines = [
            "GSModel Summary:",
            f"  Iteration: {self.iteration}",
            f"  Components: {len(self.components)}",
            f"  Total Points: {self.total_points:,}",
            f"  Total Memory: {self.total_memory:.2f} MB",
            "  Components Detail:",
        ]

        for name, component in self.components.items():
            lines.append(f"    {component}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """字符串表示。"""
        return (
            f"GSModel(components={len(self.components)}, "
            f"points={self.total_points:,}, memory={self.total_memory:.2f}MB)"
        )

    def __len__(self) -> int:
        """返回组件数量。"""
        return len(self.components)
