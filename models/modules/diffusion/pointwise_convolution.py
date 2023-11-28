import torch
import torch.nn as nn

from torch import Tensor

from .embeddings import Modulator

class PointwiseConvolution_Modulated(nn.Module):
    def __init__(self, in_channels, out_channels, channel_last=True):
        super().__init__()
        
        self.mod = Modulator(in_channels)
        if channel_last:
            self.pointwise = nn.Linear(in_channels, out_channels)
        else:
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, c: Tensor):
        return self.pointwise(self.mod(x, c))
