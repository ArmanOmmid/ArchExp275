from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def _unfold_padding_prep(x, window_height, window_width):

    *B, H, W, C = x.shape

    pad_height = (H % window_height) // 2
    pad_width = (W % window_width) // 2

    if pad_height > 0 or pad_width > 0:
        x = x.permute(0, -1, -3, -2)
        x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), 'constant', 0)
        x = x.permute(0, -2, -1, -3)

    padding_info = (pad_height, pad_width)
    return x, padding_info
    
def _fold_unpadding_prep(x, padding_info):

    pad_height, pad_width = padding_info

    if pad_height > 0:
        x = x[:, pad_height:-pad_height, :, :]
    if pad_width > 0:
        x = x[:, :, pad_width:-pad_width, :]

    return x

def _extract_windows(feature_map: torch.Tensor, window_height, window_width, stride_height=None, stride_width=None):
    """
    Extract local windows from a feature map for non-square windows and flatten them.

    Parameters:
    - feature_map: the input feature map, shape [B*, H, W, C]
    - window_height: the height of the window
    - window_width: the width of the window
    - stride_height: the stride of the windows across the height of the feature map
    - stride_width: the stride of the windows across the width of the feature map

    Returns:
    - windows: the local windows ready for attention, shape [B*, Window, S, C]
    """

    if stride_height is None:
        stride_height = window_height
    if stride_width is None:
        stride_width = window_width
    
    *B, H, W, C = feature_map.shape

    feature_map, padding_info = _unfold_padding_prep(feature_map, window_height, window_width)

    # Unfold the feature map into windows for both dimensions
    unfolded_height = feature_map.unfold(-3, window_height, stride_height) # New dim (from H) created at the end

    unfolded_both = unfolded_height.unfold(-3, window_width, stride_width) # So we do -3 again for W
    # Send C to the back
    windows = unfolded_both.permute(0, -5, -4, -2, -1, -3)

    # The shape after unfold will be [B*, H_unfolded, W_unfolded, window_height, window_width, C]
    *B, H_unfolded, W_unfolded, H_window, W_window, C = windows.shape
    # We need [*B, H_unfolded, W_unfolded, <WindowContext>, C]
    windows = windows.reshape(*B, H_unfolded, W_unfolded, window_height * window_width, C)

    return windows, padding_info

class SwinResidualCrossAttention(nn.Module):
    def __init__(self, 
                 window_size, 
                 embed_dim, 
                 num_heads,
                 stride = [None, None],
                 attention_dropout = 0.0, 
                 norm_layer = partial(nn.LayerNorm, eps=1e-5),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert len(window_size) == 2
        if not stride[0] == stride[1] == None: 
            raise NotImplementedError("Striding is not implemented (or wanted)") # -Arman

        self.window_height, self.window_width = window_size
        self.stride_height = stride[0] if stride is not None else window_size[0]
        self.stride_width = stride[1] if stride is not None else window_size[1]

        self.norm_x = norm_layer(embed_dim)
        self.norm_residual = norm_layer(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)

    def forward(self, x, residual):

        x, residual = self.norm_x(x), self.norm_residual(residual)

        # Unfolding
        x, _ = _extract_windows(x, self.window_height, self.window_width)
        residual, padding_info = _extract_windows(residual, self.window_height, self.window_width,
                                                                   self.stride_height, self.stride_width)

        assert x.shape == residual.shape, f"{x.shape} != {residual.shape}"

        *B, H, W, WINDOW, C = residual.shape

        Q = x.reshape(-1, WINDOW, C)
        K = V = residual.reshape(-1, WINDOW, C)

        # output = attended residuals
        output, attention_weights = self.cross_attention(Q, K, V)

        # Folding
        output = output.reshape(*B, H, W, self.window_height, self.window_width, C)
        output = output.permute(0, -5, -3, -4, -2, -1)
        output = output.reshape(*B, H * self.window_height, W * self.window_width, C)

        # Unpadding
        output = _fold_unpadding_prep(output, padding_info)

        return output
