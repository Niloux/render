from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn


def get_expon_lr_func(
    lr_init,
    lr_final,
    lr_delay_steps=0,
    lr_delay_mult=1.0,
    max_steps=1000000,
    warmup_steps=0,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or (step < warmup_steps):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def assert_not_none(value):
    assert value is not None
    return value


class ResidualBlock(nn.Module):
    """Abstract Residual Block class."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        if in_dim != dim:
            self.res_branch = nn.Conv2d(in_dim, dim, kernel_size=1)
        else:
            self.res_branch = nn.Identity()
        self.main_branch = nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.res_branch(x) + self.main_branch(x))


class BasicBlock(ResidualBlock):
    """Basic residual block."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        kernel_size: int,
        padding: int,
        use_bn: bool = False,
    ):
        super().__init__(in_dim, dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
        )


class RGBDecoderCNN(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
    ):
        super().__init__()
        last_layer = torch.nn.Conv2d(hidden_dim, out_dim, 1)
        last_layer.weight.data *= weight_init_scale
        layers = [
            BasicBlock(
                in_dim - skip_dim,
                hidden_dim,
                kernel_size,
                padding=kernel_size // 2,
                use_bn=False,
            )
        ]
        for _ in range(num_hidden_blocks):
            layers.append(
                BasicBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    padding=kernel_size // 2,
                    use_bn=False,
                )
            )
        layers.append(last_layer)
        self.net = torch.nn.Sequential(*layers)
        self.skip_dim = skip_dim
        self.out_dim = out_dim

    def forward(self, features):
        """将每像素特征解码为RGB，输入形状为[H, W, C]或[1, H, W, C]。"""
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split(
            [self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1
        )

        spec = spec.permute(0, 3, 1, 2)
        spec = self.net(spec)

        spec = spec.permute(0, 2, 3, 1)

        return (albedo * (1 + spec[..., :3]) + spec[..., 3:]).squeeze(0)


class RGBDecoder(nn.Module):
    @staticmethod
    def _infer_cfg_from_state_dict(params: Dict[str, Any]) -> Dict[str, Any]:
        appearance = params.get("appearance_embedding", None)
        appearance_dim = (
            int(appearance.shape[1]) if isinstance(appearance, torch.Tensor) else 0
        )

        conv_w = params.get("rgb_decoder.net.0.main_branch.0.weight", None)
        if not isinstance(conv_w, torch.Tensor) or conv_w.ndim != 4:
            return {
                "appearance_dim": appearance_dim,
                "use_app_embed": bool(appearance_dim),
                "use_ray_dirs": True,
                "sh_degree": 1,
            }

        in_channels = int(conv_w.shape[1])
        input_dim = in_channels + 3
        sh_degree = 1
        sh_dim = 3 * (sh_degree + 1) ** 2

        candidates = [
            (sh_dim, False, False),
            (sh_dim + 3, False, True),
            (sh_dim + appearance_dim, True, False),
            (sh_dim + 3 + appearance_dim, True, True),
        ]

        use_app_embed = False
        use_ray_dirs = False
        for cand_dim, cand_app, cand_ray in candidates:
            if cand_dim == input_dim:
                use_app_embed = cand_app
                use_ray_dirs = cand_ray
                break

        return {
            "appearance_dim": appearance_dim,
            "use_app_embed": use_app_embed,
            "use_ray_dirs": use_ray_dirs,
            "sh_degree": sh_degree,
        }

    @classmethod
    def from_checkpoint_state(
        cls,
        metadata: Dict[str, Any],
        state_dict: Dict[str, Any],
        device: Union[torch.device, str, None] = None,
        mode: str = "sensor",
    ) -> "RGBDecoder":
        params = (
            state_dict["params"]
            if isinstance(state_dict, dict)
            and "params" in state_dict
            and isinstance(state_dict["params"], dict)
            else state_dict
        )
        cfg = cls._infer_cfg_from_state_dict(params)
        inst = cls(
            metadata,
            device=device,
            use_app_embed=cfg["use_app_embed"],
            mode=mode,
            appearance_dim=cfg["appearance_dim"],
            use_ray_dirs=cfg["use_ray_dirs"],
            sh_degree=cfg["sh_degree"],
        )
        inst.load_state_dict(state_dict, strict=True)
        return inst

    def __init__(
        self,
        metadata,
        # device: torch.device | str | None = None,
        device: Union[torch.device, str, None] = None,
        use_app_embed: bool = True,
        mode: str = "sensor",
        appearance_dim: int = 8,
        use_ray_dirs: bool = True,
        sh_degree: int = 1,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.mode = mode
        self.use_app_embed = use_app_embed
        self.use_ray_dirs = use_ray_dirs
        self.sh_degree = sh_degree
        self.camera_id_map: dict | None = None

        num_corrections = metadata["num_cams"]

        self.appearance_embedding = nn.Parameter(
            torch.zeros(
                num_corrections,
                appearance_dim,
                device=self.device,
                dtype=torch.float,
            ),
            requires_grad=True,
        )
        sh_dim = 3 * (sh_degree + 1) ** 2
        feature_dim = sh_dim + (3 if use_ray_dirs else 0)
        if self.use_app_embed:
            input_dim = feature_dim + appearance_dim
        else:
            input_dim = feature_dim
        self.rgb_decoder = torch.compile(
            RGBDecoderCNN(
                input_dim,
                hidden_dim=32,
                kernel_size=3,
                num_hidden_blocks=1,
            ),
            disable=True,  # TODO: enable automatically if we don't use the viewer
        )
        self.rgb_decoder.to(self.device)

        # 1223, yyf, for arbitrary cams
        self.id_trans = {0: 0, 1: 1, 2: 2, 3: 3}

    def save_state_dict(self, is_final):
        state_dict = {}
        state_dict["params"] = self.state_dict()
        if not is_final:
            state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        params = (
            state_dict["params"]
            if isinstance(state_dict, dict)
            and "params" in state_dict
            and isinstance(state_dict["params"], dict)
            else state_dict
        )
        return super().load_state_dict(params, strict=strict)

    def training_setup(self):
        color_correction_lr_init = 5e-4
        color_correction_lr_final = 5e-5
        color_correction_max_steps = 100000

        params = [
            {
                "params": self.appearance_embedding,
                "lr": color_correction_lr_init,
                "name": "appearance_embedding",
            },
            {
                "params": list(self.rgb_decoder.parameters()),
                "lr": color_correction_lr_init,
                "name": "rgb_decoder",
            },
        ]

        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)

        self.color_correction_scheduler_args = get_expon_lr_func(
            lr_init=color_correction_lr_init,
            lr_final=color_correction_lr_final,
            max_steps=color_correction_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.color_correction_scheduler_args(iteration)
            param_group["lr"] = lr

    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def get_id(self, camera):
        if self.mode == "image":
            cam_id = getattr(camera, "id", camera)
            if isinstance(cam_id, int):
                return cam_id
            if isinstance(cam_id, str) and isinstance(self.camera_id_map, dict):
                return self.camera_id_map[cam_id]
            raise TypeError(f"invalid camera id type for mode=image: {type(cam_id)}")
        elif self.mode == "sensor":
            meta = getattr(camera, "meta", None)
            if isinstance(meta, dict) and "cam" in meta:
                return self.id_trans[meta["cam"]]
            raise AttributeError("camera.meta['cam'] is required for mode=sensor")
        else:
            raise ValueError(f"invalid mode: {self.mode}")

    def forward(self, camera, features):
        if self.use_app_embed:
            id = self.get_id(camera)
            embedding = self.appearance_embedding[id]
            embedding_expanded = embedding.expand(*features.shape[:-1], -1)
            rendered_features = torch.cat([features, embedding_expanded], dim=-1)
            return self.rgb_decoder(rendered_features)
        else:
            return self.rgb_decoder(features)
