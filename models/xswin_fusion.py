from typing import *
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._network import _Network
from .xswin import XNetSwinTransformer
from .pointnet import PointNet

patch_size = [16, 16] # [4, 4]
embed_dim = 16 # 64
depths = [1, 1] # [3, 3, 3]
num_heads = [1, 2, 4] # [4, 8, 16]
window_size = [4, 4]
global_stages = 0 # 3
final_downsample = True
residual_cross_attention = True
smooth_conv = True

class XSwinFusion(_Network):
    def __init__(self, feature_dims=64, resize=None, xswin_weights=None, **kwargs):
        super().__init__(**kwargs)

        self.segment_net = XNetSwinTransformer(patch_size, embed_dim, depths, 
                           num_heads, window_size, num_classes=feature_dims,
                           global_stages=global_stages, input_size=resize,
                           final_downsample=final_downsample, residual_cross_attention=residual_cross_attention,
                           smooth_conv=smooth_conv, weights=xswin_weights,
                           )
        
        self.point_net = PointNet(out_channels=feature_dims)

    def forward(self, pcd, rgb, mask_indices):
        
        rgb = self.segment_net(rgb)

        pcd, transforms = self.point_net(pcd)

        rgb = torch.permute(rgb, (0, 2, 3, 1)) # B C H W -> B H W C (channel last for masking)

        batch_indices = np.arange(pcd.size(0))[:, None]
        row_indices = mask_indices[:, 0, :]
        col_indices = mask_indices[:, 1, :]

        rgb = rgb[batch_indices, row_indices, col_indices] # RGB POINT CLOUD
        rgb = torch.permute(rgb, (0, 2, 1)) # B, L, C -> B, C, L

        print(rgb.shape, pcd.shape)

        x = torch.cat((rgb, pcd), dim=-2) # concat along channel dim (B, C, L)

        print(x.shape)

        return x
