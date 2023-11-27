
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once
from torchvision.ops.misc import MLP, Permute
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
from torchvision.models.vision_transformer import EncoderBlock as ViTEncoderBlock

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
        print(x.shape)
        x = self.expansion(x)
        print(x.shape)
        x = self.norm(x)
        x = _patch_expanding_pad(x)
        return x

class PointwiseConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        final_downsample (bool): Do a final downsampling for the encoder towards the mid-network
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-5),
        # swin_block: Optional[Callable[..., nn.Module]] = SwinTransformerBlockV2,
        # downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
        middle_stages: int = 1,
        final_downsample: bool = False,
        cross_attention_skip: bool = False,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        # split image into non-overlapping patches
        self.patching = nn.Sequential(
            nn.Conv2d(
                3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]), # B C H W -> B H W C
            norm_layer(embed_dim),
        )

        self.cross_attention_skip = cross_attention_skip
        self.final_downsample = final_downsample
        total_stage_blocks = sum(depths)

        ################################################
        # ENCODER
        ################################################
        
        self.encoder : List[nn.Module] = []
        stage_block_id = 0
        # Encoder Swin Blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double
                stage.append(
                    SwinTransformerBlockV2(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.encoder.append(nn.Sequential(*stage))
            # Patch Merging Layer
            if i_stage < (len(depths) - 1) or self.final_downsample:
                self.encoder.append(PatchMergingV2(dim, norm_layer))

        self.encoder = nn.ModuleList(self.encoder)

        current_features = embed_dim * 2 ** (len(depths) - int(not self.final_downsample))

        ################################################
        # MIDDLE
        ################################################

        self.middle : List[nn.Module] = []
        for _ in range(min(1, middle_stages)):
            self.middle.append(
                ViTEncoderBlock(
                    num_heads = num_heads[-1], # Use number of heads as final Swin Encoder Block
                    hidden_dim = current_features,
                    mlp_dim = int(current_features * mlp_ratio),
                    dropout = dropout,
                    attention_dropout = attention_dropout,
                    norm_layer = norm_layer,
                )
            )
        self.middle = nn.Sequential(*self.middle)

        ################################################
        # DECODER
        ################################################

        self.decoder : List[nn.Module] = []

        # stage_block_id = 0 # NOTE : Not reseting dropout scheduler
        # Decoder Swin Blocks
        for i_stage in range(len(depths)-1, 0, -1): # NOTE : Count Backwards!
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage

            # if self.cross_attention_skip:
            #   stage.append() X-Attn Skip Connection

            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double

                # add patch merging layer
                if i_stage < (len(depths) - 1) or self.final_downsample:
                    self.decoder.append(PatchExpandingV2(2*dim, norm_layer)) # NOTE : Double input dim

                stage.append(
                    SwinTransformerBlockV2(
                        dim * (1 + (i_layer == 0)), # First Swin Block in Decoder Stage gets Stacked
                        num_heads[i_stage] * (1 + (i_layer == 0)), # Double heads in this case too
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage.append(
                    nn.Identity()
                )
                stage_block_id += 1
            self.decoder.append(nn.Sequential(*stage))


        # self.decoder : List[nn.Module] = []
        # stage_block_id = 0

        # num_features = embed_dim * 2 ** (len(depths) - int(not self.final_downsample))
        # self.norm = norm_layer(num_features)
        # self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.flatten = nn.Flatten(1)
        # self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.patching(x)
        
        residuals = []
        for i in range(0, len(self.encoder), 2):

            print(x.shape)

            x = self.encoder[i](x) # Encoder Stage
            residuals.append(x)
            if i+1 < (len(self.encoder) - 1) or self.final_downsample:
                x = self.encoder[i+1](x) # Downsample (PatchMerge)

        print(x.shape)

        *B, H, W, C = x.shape
        x = x.contiguous().view(*B, H*W, C)
        x = self.middle(x)
        x = x.contiguous().view(*B, H, W, C)

        print(x.shape)

        print("DECODER")

        for i_residual, i in zip(
            range(len(residuals)-1, 0, -1), # Count backwards for residual indices 
            range(0 - int(not self.final_downsample), len(self.decoder), 2 + int(self.cross_attention_skip))
        ):

            print(x.shape)

            if i > 0 or self.final_downsample:
                x = self.decoder[i](x) # Upsample (PatchExpand)

            residual = residuals[i_residual]
            if self.cross_attention_skip:
                residual = self.decoder[i+1](residual) # Cross Attention Skip Connection

            x = torch.cat((x, residual), dim=-1) # Dumb Skip Connection

            print(x.shape)

            x = self.decoder[i+(1 + int(self.cross_attention_skip))](x)

            print(x.shape)

        # x = self.norm(x)
        # x = self.permute(x)
        # x = self.avgpool(x)
        # x = self.flatten(x)
        # x = self.head(x)
        return x
