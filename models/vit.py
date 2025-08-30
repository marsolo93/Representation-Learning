"""
Inspired by: https://github.com/huggingface/pytorch-image-models/
tree/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models

"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_

import math

from loguru import logger

from typing import Optional, Tuple
from layers import DropPath

from nn_configs import (
    GatedMLPConfig,
    MultiHeadAttentionConfig,
    TransformerBlockConfig,
    PatchingEncodingConfig,
    ViTConfig,
    RoPEConfig,
)
from utils import apply_rope
from layers import RopePositionEmbedding
from model_output import ViTOutput


class GatedMLP(nn.Module):
    """
    A gated mlp for the processing of inputs. Normally used in post-processing
    steps after a multi-head attention layer.
    """

    def __init__(self, config: GatedMLPConfig) -> None:
        """
        Constructs the linear-layers, the activation function, and the drop
        out layer.

        Parameters
        ----------
        config: GatedMLPConfig
            Contains all parameters for constructing the layers.
        """
        super().__init__()
        self.linear_gated = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim * config.hidden_dim_factor,
        )
        self.linear_source = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim * config.hidden_dim_factor,
        )
        self.linear_post_gating = nn.Linear(
            in_features=config.hidden_dim * config.hidden_dim_factor,
            out_features=config.hidden_dim,
        )
        self.activation_fn = config.activation_fn
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        It has two branches. The data is at first processed in the gating
        branch, consisting of a sequence of a linear layer and an activation
        function. The other branch executes the linear transformation by a
        linear layer. Then, an elementwise-multiplication is performed,
        followed by a last linear layer and drop out.

        Parameters
        ----------
        x: torch.Tensor
            Data for processing.

        Returns
        -------
        x: torch.Tensor
            The processed data.
        """
        gating = self.activation_fn(self.linear_gated(x))
        x = self.linear_source(x)
        x = gating * x
        x = self.drop(self.linear_post_gating(x))
        return x


class MultiHeadAttention(nn.Module):
    """
    Implementation of the multi-head attention layer in a transformer.
    """

    def __init__(
        self,
        config: MultiHeadAttentionConfig,
        is_last: bool = False,
        num_registers: Optional[int] = None,
    ) -> None:
        """
        It prepares the MultiHeadAttention layer. It initializes a linear
        layer for upsampling of input data to generate queries, keys, and
        values. One can decide, if the flash attention module (provided by
        PyTorch) is used or a self-written version. However, one cannot return
        the attention map, when the flash attention version is used. Thus, the
        last layer of a transformer model should have the self-written
        self-attention mechanism.

        Parameters
        ----------
        config: MultiHeadAttentionConfig
            Contains all relevant parameters for building the
            MultiHeadAttention layer.

        is_last:
            Determines, if this is the last layer of a transformer for
            returning also the attention maps.

        num_registers:
            Determines the number of registers. For more information see:
                https://arxiv.org/html/2309.16588v2
        """
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0, (
            "Please ensure that the hidden dimension (hidden_dim) "
            "is a multiplier of the number of heads (num_heads)."
        )
        self.linear = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim * 3,
            bias=config.qkv_bias,
        )
        self.num_heads = config.num_heads
        self.use_flash_attn = config.use_flash_attn
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim
        self.qk_scale = config.qk_scale or self.head_dim**-0.5
        self.is_last = is_last
        self.num_registers = num_registers
        if not self.use_flash_attn or self.is_last:
            self.attn_dropout = nn.Dropout(p=config.attn_drop)
        else:
            self.attn_drop = config.attn_drop
        self.proj_dropout = nn.Dropout(p=config.proj_drop)
        self.proj_linear = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_sincos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        It performs the multi-head attention with the option to perform
        rotary positional embedding (RoPE) for meta. For more information see:
            https://github.com/facebookresearch/dinov3/blob/main/dinov3/
            layers/rope_position_encoding.py

        Metalvl
        -------
        1. It performs the linear transformation and the information is
            upsampled to generate the attached queries, keys, and values.
        2. The resulting tensor is split into the queries, keys, and values.
        3. Optional: RoPE
        4. Self-attention. Either in the self-written way, or using the flash
            attention implementation of PyTorch.
            For more information see:
                FlashAttentionV2: https://arxiv.org/abs/2307.08691
                Scaled Dot-product: https://arxiv.org/abs/1706.03762
        5. The post projection is performed.

        Parameters
        ----------
        x: torch.Tensor
            The input data.

        rope_sincos: Optional[torch.Tensor]
            The sine cosine encoding for the rotary positional embedding.

        Returns
        -------
        x: torch.Tensor
            The output data.
        attn:
            The attention maps.
        """
        B, T, C = x.shape
        assert C == self.hidden_dim, (
            "Ooops, something went wrong. The channel size of this tensor "
            "does not correspond the registered hidden dimension (hidden_dim)!"
        )
        logger.trace(f"Queries, keys and values generation...")
        #  [B, T, C] -> [B, T, 3*C]
        qkv = self.linear(x)
        #  [B, T, 3*C] -> [B, T, C], [B, T, C], [B, T, C]
        q, k, v = torch.split(qkv, split_size_or_sections=C, dim=-1)
        #  [B, T, C] -> [B, T, NUM_HEADS, C] -> [B, NUM_HEADS, T, C]
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        logger.trace(f"Queries, keys and values were generated.")
        if rope_sincos is not None:
            logger.trace(f"Rotary positional embedding...")
            q, k = apply_rope(q=q, k=k, rope=rope_sincos)
            logger.trace(
                f"Rotary positional embedding was added to the "
                f"queries and keys."
            )
        if not self.use_flash_attn or self.is_last:
            logger.trace(f"Attention score computation...")
            #  q:[B, NUM_HEADS, T, C] * k:[B, NUM_HEADS, C, T] ->
            #  attn:[B, NUM_HEADS, T, T]
            qk = q @ k.transpose(-1, -2) * self.qk_scale
            attn = self.attn_dropout(torch.softmax(qk, dim=-1))
            logger.trace(f"Attention score were computed.")
            #  attn:[B, NUM_HEADS, T, T] * v:[B, NUM_HEADS, T, C] ->
            #  x:[B, NUM_HEADS, T, C]
            x = attn @ v
            logger.trace(f"Attention and values were multiplied.")
        else:
            logger.trace(f"Scaled dot product...")
            x = nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
            logger.trace(f"Scaled dot product was computed.")
        x = x.transpose(1, 2).flatten(2)
        x = self.proj_linear(x)
        x = self.proj_dropout(x)
        logger.trace(f"Projection was performed.")
        if not self.use_flash_attn or self.is_last:
            return x, attn
        else:
            return x, None


class TransformerBlock(nn.Module):
    """
    The transformer block, encapsulating the self-attention and the
    post-processing and considering the residual connections.
    """

    def __init__(
        self,
        config: TransformerBlockConfig,
        is_last: bool = False,
        num_registers: Optional[int] = None,
    ) -> None:
        """
        Builds the TransformerBlock.

        Parameters
        ----------
        config: TransformerBlockConfig
            Contains the required parameters for building the TransformerBlock.

        is_last: bool
            Determines, if a TransformerBlock is the last TransformerBlock
            in the Transformer.

        num_registers: int
            Determines the number of registers, which are used in the
            TransformerBlock.
        """
        super().__init__()
        self.config = config
        if is_last:
            self.multi_head_attention = MultiHeadAttention(
                config=config.multi_head_attention_config,
                is_last=True,
                num_registers=num_registers,
            )
        else:
            self.multi_head_attention = MultiHeadAttention(
                config=config.multi_head_attention_config,
                num_registers=num_registers,
            )
        self.mlp = GatedMLP(config=config.mlp_config)
        self.norm_1 = nn.LayerNorm(config.mlp_config.hidden_dim)
        self.norm_2 = nn.LayerNorm(config.mlp_config.hidden_dim)
        self.drop_path = DropPath(drop_prob=config.drop_path_prob)

    def forward(
        self,
        x: torch.Tensor,
        rope_sincos: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Performs the computation of the TransformerBlock.

        Parameters
        ----------
        x: torch.Tensor
            Input data.

        rope_sincos: torch.Tensor | None
            Contains the sine and cosine encoding.

        return_attn: bool
            Determines, if the attention maps should be returned by this
            TransformerBlock.

        Returns
        -------
        x: torch.Tensor
            Output data.
        """
        res = x
        logger.trace(f"Residual data was stored...")
        x = self.norm_1(x)
        logger.trace(f"Pre layer norm was performed.")
        if return_attn:
            assert (
                not self.config.multi_head_attention_config.use_flash_attn
            ), (
                "Sorry, there is a mismatch. You cannot return attention "
                "scores, while using flash attention. Please, ensure you are "
                "using flash attention only in layers, where you do not need "
                "the attention scores."
            )
            logger.trace(f"Multi head attention...")
            _, attn = self.multi_head_attention(
                x,
                rope_sincos=(rope_sincos if rope_sincos is not None else None),
            )
            return attn

        logger.trace(f"Multi head attention...")
        x, _ = self.multi_head_attention(
            x,
            rope_sincos=(rope_sincos if rope_sincos is not None else None),
        )
        x = res + self.drop_path(x)
        logger.trace(f"Addition with residual was performed")
        logger.trace(f"Multi-layer perceptron computation...")
        x = res + self.drop_path(self.mlp(self.norm_2(x)))
        logger.trace(
            f"Multi-layer perceptron computation and addition "
            f"to the residual was done."
        )
        return x


class PatchingEncoding(nn.Module):
    """
    Executes the patch encoding of an image for vision transformers.
    """

    def __init__(self, config: PatchingEncodingConfig) -> None:
        """
        Constructs the PatchingEncoding layer.

        Parameters
        ----------
        config: PatchingEncodingConfig
            Contains all variables for building the PatchingEncoding layer.
        """
        super().__init__()
        assert config.img_size % config.num_patches_in_dimension == 0, (
            "Please ensure, that the img size divided by the "
            "number of patches in both dimensions is an integer!"
        )
        kernel_size = config.img_size // config.num_patches_in_dimension
        self.num_patches_in_dimension = config.num_patches_in_dimension
        self.conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x: torch.Tensor):
        """
        Performs the PatchingEncoding step. It applies the 2-D convolution
        with a stride of the kernel size. This means only non-overlapping data
        is processed (separate patches of the input image). Then, those patches
        are flattened, yielding a sequence of flat image patches.

        Parameters
        ----------
        x: torch.Tensor
            Input image [B, 3, H, W]. 3 is the number of channels for an RGB
            image.

        Returns
        -------
        x: torch.Tensor
            Output sequence [B, H*W=T, C].
        """
        x = self.conv(x)
        logger.trace(f"Patching performed! Data shape: {x.shape}")
        return x.flatten(2).transpose(-1, -2)


class ViT(nn.Module):

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.patch_encode_config.hidden_dim
        self.patch_encode = PatchingEncoding(config.patch_encode_config)
        self.cls_token = nn.Parameter(data=torch.empty([1, 1, hidden_dim]))
        if config.registers:
            self.register_token = nn.Parameter(
                data=torch.empty([1, config.registers, hidden_dim])
            )
        if "absolute_trainable" in config.positional_encoding:
            if config.registers:
                length = (
                    config.patch_encode_config.num_patches_in_dimension**2
                    + config.registers
                    + 1
                )
            else:
                length = (
                    config.patch_encode_config.num_patches_in_dimension**2 + 1
                )
            self.positional_encoding = nn.Parameter(
                data=torch.empty(
                    [1, length, config.patch_encode_config.hidden_dim]
                )
            )

        if "rotary_meta" in config.positional_encoding:
            self.rope_embed = RopePositionEmbedding(config=config.rope_config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=config.transformer_block_config,
                    num_registers=config.registers,
                )
                for _ in range(config.num_layers - 1)
            ]
        )
        self.blocks.append(
            TransformerBlock(
                config=config.transformer_block_config,
                num_registers=config.registers,
                is_last=True,
            )
        )
        self.pos_drop = nn.Dropout(p=config.positional_drop)
        self.norm = nn.LayerNorm(hidden_dim)

        if "absolute_trainable" in config.positional_encoding:
            trunc_normal_(self.positional_encoding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if config.registers:
            trunc_normal_(self.register_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if "rotary_meta" in self.config.positional_encoding:
            self.rope_embed._init_weights()
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> ViTOutput:
        logger.trace(f"Image was parsed to the model. DATA SHAPE: {x.shape}")
        x, num_patch = self.__preprocess_patch(x)
        logger.trace(f"Image is preprocessed. DATA SHAPE: {x.shape}")
        if "rotary_meta" in self.config.positional_encoding:
            rope_sincos = self.rope_embed(H=num_patch, W=num_patch)
            for i, blk in enumerate(self.blocks):
                logger.trace(
                    f"TRANSFORMER BLOCK {i} OUT OF {len(self.blocks) - 1}"
                )
                x = blk(x, rope_sincos=rope_sincos)
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)
        logger.trace(f"Normalization and CLS token extration...")
        x = self.norm(x)

        vit_output = ViTOutput(
            cls_token=x[:, 0:1, :],
            patch_tokens=x[:, 1 : int(num_patch**2), :],
            register_tokens=(
                x[:, int(num_patch**2) :, :] if self.config.registers else None
            ),
        )
        logger.trace(f"Output generated with id: {id(vit_output)}")
        return vit_output

    def __preprocess_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, _, _, _ = x.shape
        logger.trace(f"Image patching and encoding...")
        x = self.patch_encode(x)
        num_patches_single_dim = int(math.sqrt(x.shape[1]))
        logger.trace(f"Image patching and encoding...")
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.config.registers:
            register_tokens = self.register_token.expand(B, -1, -1)
            x = torch.cat((x, register_tokens), dim=1)
        if "absolute_trainable" in self.config.positional_encoding:
            x = x + self.positional_encoding
        return self.pos_drop(x), num_patches_single_dim

    def get_final_attention_maps(self, x):
        x, num_patch = self.__preprocess_patch(x)
        if "rotary_meta" in self.config.positional_encoding:
            rope_sincos = self.rope_embed(H=num_patch, W=num_patch)
            for i, blk in enumerate(self.blocks[:-1]):
                logger.trace(
                    f"TRANSFORMER BLOCK {i} OUT OF {len(self.blocks) - 1}"
                )
                x = blk(x, rope_sincos=rope_sincos)
            attn = self.blocks[-1](
                x, rope_sincos=rope_sincos, return_attn=True
            )
        else:
            for i, blk in enumerate(self.blocks[:-1]):
                x = blk(x)
            attn = self.blocks[-1](x, return_attn=True)
        return attn


if __name__ == "__main__":
    import sys

    logger.add(sys.stderr, level="TRACE")
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    NUM_HEADS = 16

    IMG_SIZE = 224
    NUM_PATCH_PER_DIM = 16
    IN_CHANNELS = 3

    POSITIONAL_EMBEDDING = ["rotary_meta"]

    self_attention_config = MultiHeadAttentionConfig(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        attn_drop=0.2,
        proj_drop=0.2,
        use_flash_attn=True,
    )

    gated_mlp = GatedMLPConfig(
        hidden_dim=HIDDEN_DIM,
    )

    transformer_block_config = TransformerBlockConfig(
        multi_head_attention_config=self_attention_config,
        mlp_config=gated_mlp,
    )

    patch_encoding_config = PatchingEncodingConfig(
        hidden_dim=HIDDEN_DIM,
        img_size=IMG_SIZE,
        num_patches_in_dimension=NUM_PATCH_PER_DIM,
        in_channels=IN_CHANNELS,
    )

    rope_config = RoPEConfig(
        embed_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    )

    vit_config = ViTConfig(
        positional_encoding=POSITIONAL_EMBEDDING,
        registers=16,
        num_layers=NUM_LAYERS,
        patch_encode_config=patch_encoding_config,
        rope_config=rope_config,
        transformer_block_config=transformer_block_config,
        rope_theta=100.0,
    )

    model = ViT(config=vit_config)

    BATCH_SIZE = 4

    img = torch.rand(size=[BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE])

    out = model(img)
    print(out)
