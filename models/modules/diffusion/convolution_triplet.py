import torch.nn as nn

from .embeddings import Modulator

class ConvolutionTripletDiffusion(nn.Module):
    def __init__(self, in_channels, layer_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            Modulator(in_channels),
            nn.Conv2d(in_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            Modulator(layer_channels),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            Modulator(layer_channels),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layers(x)
