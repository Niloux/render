from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from render_kernel import compute_lidar_ray_dirs_world


def _extract_params(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """从checkpoint子字典中提取真正的params权重dict。

    Args:
        state_dict: checkpoint里某个子模块对应的字典，可能形如{"params": OrderedDict(...)}，
            也可能直接就是一个state_dict。

    Returns:
        params: 可直接用于nn.Module.load_state_dict的参数字典。
    """  # noqa: E501
    params = (
        state_dict["params"]
        if isinstance(state_dict, dict)
        and "params" in state_dict
        and isinstance(state_dict["params"], dict)
        else state_dict
    )
    if not isinstance(params, dict):
        raise ValueError(
            f"MLPDecoder checkpoint格式不正确，期望dict，实际为{type(params)}"
        )
    return params


def _sorted_layer_weight_keys(
    params: Dict[str, Any], prefix: str
) -> List[Tuple[int, str]]:
    """收集并按层号排序形如 f'{prefix}.layers.{i}.weight' 的权重键。

    Args:
        params: state_dict参数字典。
        prefix: 模块前缀，例如'lidar_decoder'。

    Returns:
        layers: [(layer_idx, key), ...] 按layer_idx升序。
    """
    pat = re.compile(rf"^{re.escape(prefix)}\.layers\.(\d+)\.weight$")
    hits: List[Tuple[int, str]] = []
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            continue
        m = pat.match(k)
        if m:
            hits.append((int(m.group(1)), k))
    hits.sort(key=lambda x: x[0])
    return hits


class TorchMLP(nn.Module):
    """纯PyTorch的多层感知机（用于加载checkpoint里的MLP权重）。"""

    def __init__(self, layer_shapes: List[Tuple[int, int]]) -> None:
        """
        Args:
            layer_shapes: [(in0,out0), (in1,out1), ...]，每一层一个Linear。
        """
        super().__init__()
        if not layer_shapes:
            raise ValueError("layer_shapes不能为空")

        layers: List[nn.Module] = []
        for in_dim, out_dim in layer_shapes:
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.in_dim = int(layer_shapes[0][0])
        self.out_dim = int(layer_shapes[-1][1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            x: [..., in_dim]

        Returns:
            y: [..., out_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class MLPDecoder(nn.Module):
    """将lidar_rasterization输出的特征解码为(intensity, ray_drop_logits)。"""

    @staticmethod
    def _infer_cfg_from_state_dict(params: Dict[str, Any]) -> Dict[str, Any]:
        """从params权重推断MLPDecoder的结构超参。

        Args:
            params: 形如 nn.Module.state_dict() 的参数字典。

        Returns:
            cfg: 包含 num_lidars, appearance_dim, feature_dim, layer_shapes 等信息。
        """
        app = params.get("appearance_dim", None)
        if not isinstance(app, torch.Tensor) or app.ndim != 2:
            raise ValueError(
                "MLPDecoder权重缺少appearance_dim或其形状不正确（期望[NumLidars, AppDim]）"  # noqa: E501
            )

        num_lidars = int(app.shape[0])
        appearance_dim = int(app.shape[1])

        layer_keys = _sorted_layer_weight_keys(params, prefix="lidar_decoder")
        if not layer_keys:
            raise ValueError(
                "MLPDecoder权重中未找到lidar_decoder.layers.*.weight，无法推断MLP结构"
            )

        layer_shapes: List[Tuple[int, int]] = []
        for _, w_key in layer_keys:
            w = params[w_key]
            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                raise ValueError(
                    f"非法权重张量: {w_key} shape={getattr(w, 'shape', None)}"
                )
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            layer_shapes.append((in_dim, out_dim))

        viewdir_dim = 3
        mlp_in_dim = layer_shapes[0][0]
        feature_dim = mlp_in_dim - appearance_dim - viewdir_dim
        if feature_dim <= 0:
            raise ValueError(
                "推断feature_dim失败："
                f"mlp_in_dim={mlp_in_dim} appearance_dim={appearance_dim} viewdir_dim={viewdir_dim}"  # noqa: E501
            )

        return {
            "num_lidars": num_lidars,
            "appearance_dim": appearance_dim,
            "feature_dim": feature_dim,
            "viewdir_dim": viewdir_dim,
            "layer_shapes": layer_shapes,
        }

    @classmethod
    def from_checkpoint_state(
        cls,
        metadata: Dict[str, Any],
        state_dict: Dict[str, Any],
        device: Union[torch.device, str, None] = None,
    ) -> "MLPDecoder":
        """从checkpoint子字典构建MLPDecoder并加载权重。

        Args:
            metadata: 需要包含num_lidars（用于构建embedding），例如{"num_lidars": 1}。
            state_dict: checkpoint里的MLPDecoder子字典（可能带"params"）。
            device: 目标设备。

        Returns:
            inst: 已加载权重的MLPDecoder实例。
        """
        params = _extract_params(state_dict)
        cfg = cls._infer_cfg_from_state_dict(params)

        num_lidars_meta = int(metadata.get("num_lidars", cfg["num_lidars"]))
        inst = cls(
            metadata={"num_lidars": num_lidars_meta},
            device=device,
            feature_dim=cfg["feature_dim"],
            appearance_dim=cfg["appearance_dim"],
            layer_shapes=cfg["layer_shapes"],
            viewdir_dim=cfg["viewdir_dim"],
        )
        inst.load_state_dict(params, strict=True)
        return inst

    def __init__(
        self,
        metadata: Dict[str, Any],
        device: Union[torch.device, str, None] = None,
        feature_dim: int = 13,
        appearance_dim: int = 8,
        layer_shapes: Optional[List[Tuple[int, int]]] = None,
        viewdir_dim: int = 3,
    ) -> None:
        """初始化MLPDecoder。

        Args:
            metadata: 需要包含num_lidars。
            device: 目标设备。
            feature_dim: 光栅化特征维度（不含depth）。
            appearance_dim: 外观/校正embedding维度。
            layer_shapes: MLP层(in,out)列表；若为None则使用默认3层结构。
            viewdir_dim: 射线方向维度，固定为3。
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        num_lidars = int(metadata["num_lidars"])
        self.feature_dim = int(feature_dim)
        self.viewdir_dim = int(viewdir_dim)
        self.appearance_dim_dim = int(appearance_dim)

        self.lidar_id_map: Optional[Dict[str, int]] = None

        self.appearance_dim = nn.Parameter(
            torch.zeros(
                num_lidars,
                appearance_dim,
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        if layer_shapes is None:
            mlp_in = self.feature_dim + self.appearance_dim_dim + self.viewdir_dim
            layer_shapes = [(mlp_in, 32), (32, 32), (32, 2)]
        self.lidar_decoder = TorchMLP(layer_shapes).to(self.device)

    def _get_id(self, lidar_id: Union[str, int, None]) -> int:
        """将lidar_id映射到embedding索引。

        Args:
            lidar_id: 雷达ID（字符串/整数/None）。

        Returns:
            idx: embedding索引。
        """
        if lidar_id is None:
            return 0
        if isinstance(lidar_id, int):
            return int(lidar_id)
        if self.lidar_id_map is None:
            return 0
        return int(self.lidar_id_map.get(lidar_id, 0))

    def forward(
        self,
        lidar_id: Union[str, int, None],
        features: torch.Tensor,
        raster_pts: torch.Tensor,
        viewmats: torch.Tensor,
        ray_dirs_world: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向：把光栅化特征解码为 intensity 与 ray_drop_logits。

        Args:
            lidar_id: 当前lidar的ID（用于选择embedding）。
            features: [B, H, W, F] 或 [H, W, F]，不包含depth通道。
            raster_pts: [B, H, W, D] 或 [H, W, D]，前两维为azimuth/elevation（度）。
            viewmats: [B, 4, 4] 或 [4, 4]，World2Lidar矩阵。

        Returns:
            intensity: [B, H, W, 1]
            ray_drop_logits: [B, H, W, 1]
        """
        if features.dim() == 3:
            features = features.unsqueeze(0)
        if raster_pts.dim() == 3:
            raster_pts = raster_pts.unsqueeze(0)
        if viewmats.dim() == 2:
            viewmats = viewmats.unsqueeze(0)
        if ray_dirs_world is not None and ray_dirs_world.dim() == 3:
            ray_dirs_world = ray_dirs_world.unsqueeze(0)

        if features.dim() != 4:
            raise ValueError(f"features维度必须为3或4，实际为{tuple(features.shape)}")
        if raster_pts.dim() != 4:
            raise ValueError(
                f"raster_pts维度必须为3或4，实际为{tuple(raster_pts.shape)}"
            )
        if viewmats.dim() != 3:
            raise ValueError(f"viewmats维度必须为2或3，实际为{tuple(viewmats.shape)}")
        if ray_dirs_world is not None and ray_dirs_world.dim() != 4:
            raise ValueError(
                f"ray_dirs_world维度必须为3或4，实际为{tuple(ray_dirs_world.shape)}"
            )

        features = features.to(device=self.device)
        raster_pts = raster_pts.to(device=self.device)
        viewmats = viewmats.to(device=self.device)
        if ray_dirs_world is not None:
            ray_dirs_world = ray_dirs_world.to(device=self.device)

        B, H, W, F = features.shape
        if F != self.feature_dim:
            raise ValueError(
                f"features最后一维应为feature_dim={self.feature_dim}，实际为{F}"
            )

        if viewmats.shape[0] == 1 and B > 1:
            viewmats = viewmats.expand(B, -1, -1)
        if raster_pts.shape[0] == 1 and B > 1:
            raster_pts = raster_pts.expand(B, -1, -1, -1)

        if ray_dirs_world is None:
            ray_dirs_world = compute_lidar_ray_dirs_world(raster_pts, viewmats)
        else:
            if ray_dirs_world.shape[-1] != 3:
                raise ValueError(
                    f"ray_dirs_world最后一维应为3，实际为{ray_dirs_world.shape[-1]}"
                )
            if ray_dirs_world.shape[0] == 1 and B > 1:
                ray_dirs_world = ray_dirs_world.expand(B, -1, -1, -1)
            if ray_dirs_world.shape[:3] != (B, H, W):
                raise ValueError(
                    "ray_dirs_world形状与features不一致: "
                    f"ray_dirs_world={tuple(ray_dirs_world.shape)} features={(B, H, W, F)}"
                )
        idx = self._get_id(lidar_id)
        embedding = self.appearance_dim[idx]
        embedding_expanded = embedding.expand(B, H, W, -1)

        x = torch.cat([features, embedding_expanded, ray_dirs_world], dim=-1)
        x = x.reshape(-1, x.shape[-1])
        y = self.lidar_decoder(x).reshape(B, H, W, -1)

        intensity, ray_drop_logits = y.split([1, 1], dim=-1)
        return intensity, ray_drop_logits
