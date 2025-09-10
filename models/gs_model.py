"""3DGS模型主管理类。"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .gs_component import GaussianComponent


class GSModel:
    """3DGS模型管理类。

    管理模型的所有高斯点云组件，提供加载、保存和操作接口。
    """

    def __init__(self) -> None:
        """初始化空的GSModel。"""
        self.components: Dict[str, GaussianComponent] = {}
        self.iteration: int = 0

    @classmethod
    def load_from_pth(cls, pth_path: Union[str, Path]) -> "GSModel":
        """从PTH文件加载模型。

        Args:
            pth_path: PTH文件路径

        Returns:
            加载的GSModel实例
        """
        checkpoint = torch.load(pth_path, map_location="cpu")
        model = cls()

        # 加载迭代次数
        if "iter" in checkpoint:
            model.iteration = checkpoint["iter"]

        # 加载各个组件
        for name, data in checkpoint.items():
            if name == "iter":
                continue

            if isinstance(data, dict) and len(data) > 0:
                # 检查是否包含必要的键
                required_keys = ["xyz", "feature_dc", "feature_rest", "scaling", "rotation", "opacity"]
                if all(key in data for key in required_keys):
                    component = GaussianComponent(
                        name=name,
                        xyz=data["xyz"],
                        feature_dc=data["feature_dc"],
                        feature_rest=data["feature_rest"],
                        scaling=data["scaling"],
                        rotation=data["rotation"],
                        opacity=data["opacity"],
                        semantic=data.get("semantic"),
                    )

                    if component.validate():
                        model.components[name] = component
                    else:
                        print(f"警告: 组件 {name} 数据验证失败，跳过加载")

        return model

    def save_to_pth(self, pth_path: Union[str, Path]) -> None:
        """保存模型到PTH文件。

        Args:
            pth_path: 保存路径
        """
        checkpoint = {"iter": self.iteration}

        for name, component in self.components.items():
            checkpoint[name] = {
                "xyz": component.xyz,
                "feature_dc": component.feature_dc,
                "feature_rest": component.feature_rest,
                "scaling": component.scaling,
                "rotation": component.rotation,
                "opacity": component.opacity,
            }

            if component.semantic is not None:
                checkpoint[name]["semantic"] = component.semantic

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
        return [comp for name, comp in self.components.items() if name.startswith(component_type)]

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
