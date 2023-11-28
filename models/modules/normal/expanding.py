from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def _pad_expansion(x: torch.Tensor, distributed: bool = False) -> torch.Tensor:
    *B, H, W, C = x.shape
    pad_c = (4 - C % 4) % 4  # Number of channels to add

    # No padding needed
    if pad_c == 0:
        return x 
    
    if not distributed:
        x = F.pad(x, (0, pad_c), "constant", 0)  # Pad with zeros to the end of channels
        return x
    else:
        raise NotImplementedError()

def _patch_expanding_pad(x: torch.Tensor) -> torch.Tensor:
    *B, H_HALF, W_HALF, C_QUAD = x.shape

    C = C_QUAD // 4

    x = _pad_expansion(x)

    x = x.view(*B, H_HALF, W_HALF, 2, 2, C)

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    x = x.view(*B, H_HALF * 2, W_HALF * 2, C)

    return x

class PatchExpandingV2(nn.Module):
    """Patch Expanding Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim # C
        self.expansion = nn.Linear(dim, 2 * dim, bias=False) # Linear expansion first to share more information
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        # Linear expansion first to share more information
        x = self.expansion(x)
        x = self.norm(x)
        x = _patch_expanding_pad(x)
        return x
    
    @staticmethod
    def _post_expand_trim(x, trimmed_shape):
        # Because of patch expanding, there can be an unexpected additional dimension (just 1)
        H_trim = x.size(-3) - trimmed_shape[-3] > 0
        W_trim = x.size(-2) - trimmed_shape[-2] > 0

        if H_trim and W_trim:
            x = x[:, :-1, :-1, :]
        elif H_trim:
            x = x[:, :-1, :, :]
        elif W_trim:
            x = x[:, :, :-1, :]

        return x