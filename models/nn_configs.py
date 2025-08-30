from typing import Optional, List, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GatedMLPConfig:
    hidden_dim: int
    hidden_dim_factor: int = 2
    activation_fn: nn.Module = nn.GELU()
    dropout: float = 0.2


@dataclass
class MultiHeadAttentionConfig:
    hidden_dim: int
    num_heads: int
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_flash_attn: bool = False


@dataclass
class TransformerBlockConfig:
    multi_head_attention_config: MultiHeadAttentionConfig
    mlp_config: GatedMLPConfig
    drop_path_prob: float = 0.2

    def __post_init__(self):
        assert self.mlp_config.hidden_dim != self.multi_head_attention_config, (
            f"Please ensure, that the hidden dimension of the "
            f"MLP (size: {self.mlp_config.hidden_dim}) "
            f"corresponds to the hidden dimension of the multi "
            f"head attention layer (size: "
            f"{self.multi_head_attention_config.hidden_dim})"
        )


@dataclass
class PatchingEncodingConfig:
    hidden_dim: int
    img_size: int = 224
    num_patches_in_dimension: int = 16
    in_channels: int = 3


@dataclass
class RoPEConfig:
    embed_dim: int
    num_heads: int
    base: float | None = 100.0
    min_period: float | None = None
    max_period: float | None = None
    normalize_coords: Literal["min", "max", "separate"] = "separate"
    shift_coords: float | None = None
    jitter_coords: float | None = None
    rescale_coords: float | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None


@dataclass
class ViTConfig:
    # TODO: add absolute and relative positional embedding
    positional_encoding: list[Literal["absolute_trainable", "rotary_meta"]]
    num_layers: int
    patch_encode_config: PatchingEncodingConfig
    transformer_block_config: TransformerBlockConfig
    rope_config: RoPEConfig | None = None
    registers: Optional[int] = None
    positional_drop: float = 0.2
    rope_theta: Optional[float] = None

    def __post_init__(self):
        assert (
            self.patch_encode_config.hidden_dim
            == self.transformer_block_config.multi_head_attention_config.hidden_dim
        ), (
            f"Please ensure, that the hidden dimension of the "
            f"MLP (size: {self.patch_encode_config.hidden_dim})"
            f"corresponds to the hidden dimension of the multi "
            f"head attention layer (size: "
            f"{self.transformer_block_config.multi_head_attention_config.hidden_dim})"
        )

        if "rotary_meta" in self.positional_encoding:
            assert (
                self.rope_config
            ), "If you want to select the rotary_meta encoding, you need to specify also the config."
