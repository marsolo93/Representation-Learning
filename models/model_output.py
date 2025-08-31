from dataclasses import dataclass

import torch


@dataclass
class ViTOutput:
    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    register_tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None
