"""
Inspired by:
https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/dino_head.py
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from layers import FullyConnectedLayer
from nn_configs import DINOHeadConfig


class DINOHead(nn.Module):
    """
    Implementation of the DINOHead module.
    """

    def __init__(self, config: DINOHeadConfig) -> None:
        """
        Constructs the DINOHead as a multi-layer perceptor.

        Parameters
        ----------
        config: DINOHeadConfig
            Contains all parameters for building the DINOHead
        """
        super().__init__()
        layers = []
        if config.num_layers == 1:
            layers.append(
                FullyConnectedLayer(
                    in_features=config.representation_dim,
                    out_features=config.bottleneck_dim,
                    use_bn=False,
                    act_fn=None,
                )
            )
        else:
            for i in range(config.num_layers):
                use_bn = config.use_batchnorm
                act_fn = True
                if i == 0:
                    in_features = config.representation_dim
                    out_features = config.hidden_dim
                elif i == config.num_layers - 1:
                    in_features = config.hidden_dim
                    out_features = config.bottleneck_dim
                    use_bn = False
                    act_fn = False
                else:
                    in_features = out_features = config.hidden_dim

                layers.append(
                    FullyConnectedLayer(
                        in_features=in_features,
                        out_features=out_features,
                        use_bn=True if use_bn else False,
                        act_fn=nn.GELU() if act_fn else None,
                    )
                )
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(
            in_features=config.bottleneck_dim,
            out_features=config.num_classes,
            bias=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Weight initialization.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, do_last_layer: bool = True, do_mlp: bool = True
    ) -> torch.Tensor:
        """
        Runs the DINOHead with different options. Either you can apply the MLP
        and the last layer for generating the classes, or you can omit a both
        parts individually.

        Parameters
        ----------
        x: torch.Tensor
            Input data.

        do_last_layer: bool
            Determines, if the classes should be generated.

        do_mlp: bool
            Determines, if the mlp should be applied.

        Returns
        -------
        x: torch.Tensor
            Output data.
        """
        if do_mlp:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if do_last_layer:
            x = self.last_layer(x)
        return x
