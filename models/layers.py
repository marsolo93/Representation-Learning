import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Literal

import numpy as np

from utils import drop_path


class FullyConnectedLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bn: bool = False,
        act_fn: nn.Module | None = nn.GELU(),
    ) -> None:
        super().__init__()
        self.bn_norm: nn.Module | None = None
        self.act_fn: nn.Module | None = None
        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features
        )
        if use_bn:
            self.bn_norm = nn.BatchNorm1d(num_features=out_features)
        if act_fn:
            self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn_norm:
            x = self.bn_norm(x)
        if self.act_fn:
            x = self.act_fn(x)
        return x


class DropPath(nn.Module):
    """
    Reference:
    https://github.com/huggingface/pytorch-image-models/blob/
    a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RopePositionEmbedding(nn.Module):
    """
    Copied from: https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py
    and a bit modified.
    """

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % (4 * config.num_heads) == 0
        both_periods = (
            config.min_period is not None and config.max_period is not None
        )
        if (config.base is None and not both_periods) or (
            config.base is not None and both_periods
        ):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        D_head = config.embed_dim // config.num_heads
        self.base = config.base
        self.min_period = config.min_period
        self.max_period = config.max_period
        self.D_head = D_head
        self.normalize_coords = config.normalize_coords
        self.shift_coords = config.shift_coords
        self.jitter_coords = config.jitter_coords
        self.rescale_coords = config.rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = config.dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=config.device, dtype=config.dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(
                f"Unknown normalize_coords: {self.normalize_coords}"
            )
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = (
                torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            )
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = (
                torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            )
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = (
                periods * self.max_period
            )  # range [min_period, max_period]
        self.periods.data = periods
