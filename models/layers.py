import torch
import torch.nn as nn

from utils import drop_path


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
