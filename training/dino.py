import torch
import torch.nn as nn

from training_config import DINOV1Config


class DINOV1(nn.Module):

    def __init__(self, config: DINOV1Config) -> None:
        super().__init__()
        pass

    def forward(self, data):
        pass
