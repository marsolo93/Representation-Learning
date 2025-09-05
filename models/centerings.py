import torch
import torch.nn as nn
# TODO: maybe it is better to implement a abstract class and to inherit from it


class EMAPatchCentering(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.center_momentum = center_momentum
        self.center = nn.Parameter(
            torch.zeros([1, 1, hidden_dim])
        )

    @torch.no_grad()
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == self.center_momentum.ndim, (
            "Sorry, please ensure that the number of dimensions between the "
            "input tensor and the output centering tensor is the same."
        )
        return x - self.center

    @torch.no_grad()
    def update_center(
            self,
            x: torch.Tensor
    ) -> None:
        self.center = self.center * self.center_momentum + x * (
            1 - self.center_momentum
        )


class EMAClassCentering(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.center_momentum = center_momentum
        self.center = nn.Parameter(
            torch.zeros([1, hidden_dim])
        )

    @torch.no_grad()
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == self.center_momentum.ndim, (
            "Sorry, please ensure that the number of dimensions between the "
            "input tensor and the output centering tensor is the same."
        )
        return x - self.center

    @torch.no_grad()
    def update_center(
            self,
            x: torch.Tensor
    ) -> None:
        self.center = self.center * self.center_momentum + x * (
            1 - self.center_momentum
        )
