from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._network import _Network
from .xswin import XNetSwinTransformer

patch_size = [4, 4]
embed_dim = 64
depths = [1, 1, 1]
num_heads = [4, 8, 16]
window_size = [4, 4]
num_classes = 1

IMG_H, IMG_W = 151, 309

global_stages = 1
input_size = [IMG_H, IMG_W]
final_downsample = True
residual_cross_attention = True

class DenseFusion(_Network):
    def __init__(self, feature_size=64, **kwargs):
        super().__init__(**kwargs)

        self.segment = XNetSwinTransformer(patch_size, embed_dim, depths, 
                           num_heads, window_size, num_classes=num_classes,
                           global_stages=global_stages, input_size=input_size,
                           final_downsample=final_downsample, residual_cross_attention=residual_cross_attention,
                           )
