import numpy as np
import torch
import torch.nn as nn
from easyvolcap.utils.console_utils import *
from street_gaussian.config import cfg
from street_gaussian.utils.camera_utils import Camera


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
        hidden_dim=16,
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
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split(
            [self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1
        )

        spec = spec.permute(0, 3, 1, 2)
        spec = self.net(spec)

        spec = spec.permute(0, 2, 3, 1)

        return albedo * (1 + spec[..., :3]) + spec[..., 3:]


class RGBDecoder(nn.Module):
    def __init__(self, metadata):

        super().__init__()
        self.config = cfg.model.color_correction
        self.mode = self.config.mode
        self.use_app_embed = True

        num_corrections = metadata["num_cams"]

        self.appearance_embedding = nn.Parameter(
            torch.zeros(
                num_corrections,
                self.config.appearance_embedding_dim,
                device="cuda",
                dtype=torch.float,
            ),
            requires_grad=self.use_app_embed,
        )
        if self.use_app_embed:
            input_dim = (
                3 * (cfg.model.gaussian.sh_degree + 1) ** 2
                + self.config.appearance_embedding_dim
                + 3
            )
        else:
            input_dim = 3 * (cfg.model.gaussian.sh_degree + 1) ** 2
        hidden_dim = int(
            getattr(
                self.config,
                "rgb_decoder_hidden_dim",
                os.environ.get("RENDER_RGB_DECODER_HIDDEN_DIM", "16"),
            )
        )
        num_hidden_blocks = int(
            getattr(
                self.config,
                "rgb_decoder_num_hidden_blocks",
                os.environ.get("RENDER_RGB_DECODER_NUM_HIDDEN_BLOCKS", "1"),
            )
        )

        self.rgb_decoder = torch.compile(
            RGBDecoderCNN(
                input_dim,
                hidden_dim=hidden_dim,
                kernel_size=3,
                num_hidden_blocks=num_hidden_blocks,
            ),
            disable=True,  # TODO: enable automatically if we don't use the viewer
        )
        self.rgb_decoder.cuda()

        # 1223, yyf, for arbitrary cams
        self.id_trans = {cfg.data.cameras[i]: i for i in range(len(cfg.data.cameras))}

    def save_state_dict(self, is_final):
        state_dict = dict()
        state_dict["params"] = self.state_dict()
        if not is_final:
            state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["params"])
        if cfg.mode == "train" and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def training_setup(self):
        args = cfg.optim
        fields_lr_init = args.get("fields_lr_init", 1e-3)
        fields_lr_final = args.get("fields_lr_final", fields_lr_init)
        fields_max_steps = args.get("fields_max_steps", 20001)
        fields_warmup_steps = args.get("fields_warmup_steps", 500)
        fields_lr_pre_warmup = args.get("fields_lr_pre_warmup", 1e-8)

        params = [
            {
                "params": list(self.rgb_decoder.parameters()),
                "lr": fields_lr_init,
                "name": "rgb_decoder",
            },
        ]

        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)

        def fields_scheduler(step):
            if step < 0 or (fields_lr_init == 0.0 and fields_lr_final == 0.0):
                return 0.0
            if step < fields_warmup_steps:
                if fields_warmup_steps <= 0:
                    return fields_lr_init
                return float(
                    fields_lr_pre_warmup
                    + (fields_lr_init - fields_lr_pre_warmup)
                    * np.sin(0.5 * np.pi * np.clip(step / fields_warmup_steps, 0, 1))
                )

            if fields_max_steps <= fields_warmup_steps:
                return float(fields_lr_final)

            t = np.clip(
                (step - fields_warmup_steps) / (fields_max_steps - fields_warmup_steps),
                0,
                1,
            )
            return float(
                np.exp(np.log(fields_lr_init) * (1 - t) + np.log(fields_lr_final) * t)
            )

        self.fields_scheduler_args = fields_scheduler

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.fields_scheduler_args(iteration)
            param_group["lr"] = lr

    def update_optimizer_step(self):
        self.optimizer.step()

    def update_optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def update_optimizer(self):
        self.update_optimizer_step()
        self.update_optimizer_zero_grad()

    def get_id(self, camera: Camera):
        if self.mode == "image":
            return camera.id
        elif self.mode == "sensor":
            return self.id_trans[camera.meta["cam"]]
        else:
            raise ValueError(f"invalid mode: {self.mode}")

    def forward(self, camera, features):
        if self.use_app_embed:
            id = self.get_id(camera)
            embedding = self.appearance_embedding[id]
            # torch.Size([1, 1066, 1600, 16])
            embedding_expanded = embedding.expand(*features.shape[:-1], -1)
            rendered_features = torch.cat([features, embedding_expanded], dim=-1)
            return self.rgb_decoder(rendered_features)
        else:
            return self.rgb_decoder(features)
