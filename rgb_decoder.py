import os
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _forward_plain(
        self, albedo: torch.Tensor, spec_in: torch.Tensor, orig_dim: int
    ) -> torch.Tensor:
        spec = spec_in.permute(0, 3, 1, 2)
        if spec.is_cuda:
            spec = spec.contiguous(memory_format=torch.channels_last)
        spec = self.net(spec)
        spec = spec.permute(0, 2, 3, 1)

        out = albedo * (1 + spec[..., :3]) + spec[..., 3:]
        if orig_dim == 3:
            return out.squeeze(0)
        return out

    def _prepare_embedding(
        self, embedding: torch.Tensor | None, ref: torch.Tensor
    ) -> tuple[torch.Tensor | None, int]:
        if embedding is None:
            return None, 0

        B = int(ref.shape[0])
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.dim() != 2 or int(embedding.shape[0]) != B:
            raise ValueError(
                "embedding形状必须为[Emb]或[B,Emb]且与features的batch一致，"
                f"实际为{tuple(embedding.shape)} features_batch={B}"
            )
        emb = embedding.to(device=ref.device, dtype=ref.dtype)
        return emb, int(emb.shape[1])

    def _prepare_ray_dirs(
        self, ray_dirs: torch.Tensor | None, ref: torch.Tensor
    ) -> tuple[torch.Tensor | None, int]:
        if ray_dirs is None:
            return None, 0

        B = int(ref.shape[0])
        if ray_dirs.dim() == 3:
            ray_dirs = ray_dirs.unsqueeze(0)
        if (
            ray_dirs.dim() != 4
            or int(ray_dirs.shape[0]) != B
            or int(ray_dirs.shape[-1]) != 3
        ):
            raise ValueError(
                "ray_dirs形状必须为[H,W,3]或[B,H,W,3]且与features的batch一致，"
                f"实际为{tuple(ray_dirs.shape)} features_batch={B}"
            )

        ray = ray_dirs.to(device=ref.device, dtype=ref.dtype).permute(0, 3, 1, 2)
        if ray.is_cuda:
            ray = ray.contiguous(memory_format=torch.channels_last)
        return ray, int(ray.shape[1])

    def _conv0_with_extras(
        self,
        conv0: nn.Conv2d,
        spec_base: torch.Tensor,
        ray: torch.Tensor | None,
        ray_dim: int,
        emb: torch.Tensor | None,
        emb_dim: int,
    ) -> torch.Tensor:
        in_base = int(spec_base.shape[1])
        w0 = conv0.weight
        y0 = F.conv2d(
            spec_base,
            w0[:, :in_base],
            bias=conv0.bias,
            stride=conv0.stride,
            padding=conv0.padding,
            dilation=conv0.dilation,
            groups=conv0.groups,
        )
        if ray is not None:
            y0 = y0 + F.conv2d(
                ray,
                w0[:, in_base : in_base + ray_dim],
                bias=None,
                stride=conv0.stride,
                padding=conv0.padding,
                dilation=conv0.dilation,
                groups=conv0.groups,
            )
        if emb is not None and emb_dim > 0:
            w0_emb = w0[:, in_base + ray_dim :]
            emb_bias0 = emb @ w0_emb.sum(dim=(2, 3)).transpose(0, 1)
            y0 = y0 + emb_bias0.view(int(spec_base.shape[0]), -1, 1, 1)
        return y0

    def _res_with_extras(
        self,
        res: nn.Module,
        spec_base: torch.Tensor,
        ray: torch.Tensor | None,
        ray_dim: int,
        emb: torch.Tensor | None,
        emb_dim: int,
    ) -> torch.Tensor:
        in_base = int(spec_base.shape[1])
        B = int(spec_base.shape[0])

        if not isinstance(res, nn.Conv2d):
            return res(spec_base)

        wr = res.weight
        yr = F.conv2d(
            spec_base,
            wr[:, :in_base],
            bias=res.bias,
            stride=res.stride,
            padding=res.padding,
            dilation=res.dilation,
            groups=res.groups,
        )
        if ray is not None:
            yr = yr + F.conv2d(
                ray,
                wr[:, in_base : in_base + ray_dim],
                bias=None,
                stride=res.stride,
                padding=res.padding,
                dilation=res.dilation,
                groups=res.groups,
            )
        if emb is not None and emb_dim > 0:
            wr_emb = wr[:, in_base + ray_dim :].squeeze(-1).squeeze(-1)
            emb_biasr = emb @ wr_emb.transpose(0, 1)
            yr = yr + emb_biasr.view(B, -1, 1, 1)
        return yr

    def _forward_fold(
        self,
        albedo: torch.Tensor,
        spec_in: torch.Tensor,
        embedding: torch.Tensor | None,
        ray_dirs: torch.Tensor | None,
        orig_dim: int,
    ) -> torch.Tensor:
        emb, emb_dim = self._prepare_embedding(embedding, spec_in)
        ray, ray_dim = self._prepare_ray_dirs(ray_dirs, spec_in)

        spec_base = spec_in.permute(0, 3, 1, 2)
        if spec_base.is_cuda:
            spec_base = spec_base.contiguous(memory_format=torch.channels_last)

        first = self.net[0]
        conv0 = first.main_branch[0]

        in_full = int(conv0.in_channels)
        in_base = int(spec_base.shape[1])
        if in_base + ray_dim + emb_dim != in_full:
            raise ValueError(
                "features/spec通道与ray/embedding维度不匹配: "
                f"spec_base={in_base} ray={ray_dim} emb={emb_dim} expected_in={in_full}"
            )

        y0 = self._conv0_with_extras(conv0, spec_base, ray, ray_dim, emb, emb_dim)
        y0 = first.main_branch[1](y0)
        y0 = first.main_branch[2](y0)

        conv1 = first.main_branch[3]
        y1 = conv1(y0)
        y1 = first.main_branch[4](y1)

        yr = self._res_with_extras(
            first.res_branch, spec_base, ray, ray_dim, emb, emb_dim
        )

        x = first.final_activation(yr + y1)
        for i in range(1, len(self.net)):
            x = self.net[i](x)

        spec = x.permute(0, 2, 3, 1)
        out = albedo * (1 + spec[..., :3]) + spec[..., 3:]
        if orig_dim == 3:
            return out.squeeze(0)
        return out

    def forward(
        self,
        features: torch.Tensor,
        embedding: torch.Tensor | None = None,
        ray_dirs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将每像素特征解码为RGB。"""
        orig_dim = features.dim()
        if orig_dim == 3:
            features = features.unsqueeze(0)
        elif orig_dim != 4:
            raise ValueError(f"features维度必须为3或4，实际为{tuple(features.shape)}")

        albedo, spec_in = features.split(
            [self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1
        )

        if embedding is None and ray_dirs is None:
            return self._forward_plain(albedo, spec_in, orig_dim)

        return self._forward_fold(albedo, spec_in, embedding, ray_dirs, orig_dim)


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
        decoder = RGBDecoderCNN(
            input_dim,
            hidden_dim=32,
            kernel_size=3,
            num_hidden_blocks=1,
        )
        decoder.to(self.device)
        if self.device.type == "cuda":
            decoder = decoder.to(memory_format=torch.channels_last)

        compile_enabled = os.environ.get("RENDER_TORCH_COMPILE", "1") == "1"
        self.rgb_decoder = torch.compile(decoder, disable=not compile_enabled)

        self._amp_enabled = (
            self.device.type == "cuda"
            and os.environ.get("RENDER_DECODER_AMP", "1") == "1"
        )
        amp_dtype = os.environ.get("RENDER_DECODER_AMP_DTYPE", "fp16").lower()
        self._amp_dtype = (
            torch.bfloat16 if amp_dtype in ("bf16", "bfloat16") else torch.float16
        )

        # 1223, yyf, for arbitrary cams
        self.id_trans = {0: 0, 1: 1, 2: 2, 3: 3}

    def save_state_dict(self, is_final):
        state_dict = {}
        state_dict["params"] = self.state_dict()
        if not is_final:
            state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """加载权重，兼容 torch.compile 前后 state_dict 的 key 命名差异。"""
        params = (
            state_dict["params"]
            if isinstance(state_dict, dict)
            and "params" in state_dict
            and isinstance(state_dict["params"], dict)
            else state_dict
        )

        if isinstance(params, dict):
            has_plain = any(k.startswith("rgb_decoder.net.") for k in params.keys())
            has_compiled = any(
                k.startswith("rgb_decoder._orig_mod.net.") for k in params.keys()
            )
            is_compiled = hasattr(getattr(self, "rgb_decoder", None), "_orig_mod")

            if is_compiled and has_plain and (not has_compiled):
                remapped = {}
                for k, v in params.items():
                    if k.startswith("rgb_decoder.") and (
                        not k.startswith("rgb_decoder._orig_mod.")
                    ):
                        remapped[
                            "rgb_decoder._orig_mod." + k[len("rgb_decoder.") :]
                        ] = v
                    else:
                        remapped[k] = v
                params = remapped
            elif (not is_compiled) and has_compiled and (not has_plain):
                remapped = {}
                for k, v in params.items():
                    if k.startswith("rgb_decoder._orig_mod."):
                        remapped[
                            "rgb_decoder." + k[len("rgb_decoder._orig_mod.") :]
                        ] = v
                    else:
                        remapped[k] = v
                params = remapped

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

    def forward(self, camera, features, ray_dirs_world: torch.Tensor | None = None):
        """解码单个相机的像素特征为RGB。"""
        ray = (
            ray_dirs_world
            if (
                self.use_ray_dirs
                and ray_dirs_world is not None
                and int(features.shape[-1]) == 12
            )
            else None
        )
        if self.use_app_embed:
            id = self.get_id(camera)
            embedding = self.appearance_embedding[id]
            if (
                self._amp_enabled
                and isinstance(features, torch.Tensor)
                and features.is_cuda
            ):
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    return self.rgb_decoder(features, embedding, ray_dirs=ray)
            return self.rgb_decoder(features, embedding, ray_dirs=ray)

        if (
            self._amp_enabled
            and isinstance(features, torch.Tensor)
            and features.is_cuda
        ):
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                return self.rgb_decoder(features, ray_dirs=ray)
        return self.rgb_decoder(features, ray_dirs=ray)

    def forward_batched(
        self,
        cameras: Sequence[object],
        features: torch.Tensor,
        ray_dirs_world: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """批量解码多相机的像素特征为RGB。

        Args:
            cameras: 相机ID/相机对象序列，长度为B。
            features: [B, H, W, C] 或 [H, W, C]。
            ray_dirs_world: 可选，[B, H, W, 3]。
                仅当模型启用use_ray_dirs且features通道不含ray时使用。

        Returns:
            rgb: [B, H, W, 3] 或 [H, W, 3]。
        """
        orig_dim = features.dim()
        if orig_dim == 3:
            batch = features.unsqueeze(0)
        elif orig_dim == 4:
            batch = features
        else:
            raise ValueError(f"features维度必须为3或4，实际为{tuple(features.shape)}")

        B = int(batch.shape[0])
        if len(cameras) != B:
            raise ValueError(
                f"cameras数量与batch不一致: cameras={len(cameras)} batch={B}"
            )

        embedding = None
        if self.use_app_embed:
            ids: List[int] = [self.get_id(cam) for cam in cameras]
            ids_t = torch.tensor(ids, device=batch.device, dtype=torch.long)
            embedding = self.appearance_embedding[ids_t]

        ray = (
            ray_dirs_world
            if (
                self.use_ray_dirs
                and ray_dirs_world is not None
                and int(batch.shape[-1]) == 12
            )
            else None
        )

        if self._amp_enabled and batch.is_cuda:
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                out = (
                    self.rgb_decoder(batch, embedding, ray_dirs=ray)
                    if embedding is not None
                    else self.rgb_decoder(batch, ray_dirs=ray)
                )
        else:
            out = (
                self.rgb_decoder(batch, embedding, ray_dirs=ray)
                if embedding is not None
                else self.rgb_decoder(batch, ray_dirs=ray)
            )

        if orig_dim == 3:
            return out.squeeze(0)
        return out
