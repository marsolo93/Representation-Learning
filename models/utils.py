import torch
from typing import Tuple


def drop_path(
        x: torch.Tensor,
        drop_prob: float = 0.,
        training: bool = False
) -> torch.Tensor:
    """
    Reference:
    https://github.com/huggingface/pytorch-image-models/blob/
    a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# RoPE-related functions:
def rope_rotate_half(
        x: torch.Tensor
) -> torch.Tensor:
    """
    Copied from: https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/attention.py#L66
    """
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor
) -> torch.Tensor:
    """
    Copied from: https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/attention.py#L66
    """
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        rope: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from: https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/attention.py#L66
    """
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    N = q.shape[-2]
    prefix = N - sin.shape[-2]
    assert prefix >= 0
    q_prefix = q[:, :, :prefix, :]
    q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
    k_prefix = k[:, :, :prefix, :]
    k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)
    return q, k
