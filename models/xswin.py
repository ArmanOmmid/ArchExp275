
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import MLP, Permute
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
from torchvision.models.vision_transformer import EncoderBlock as ViTEncoderBlock

from ._network import _Network
from .modules import PatchExpandingV2, SwinResidualCrossAttention, ConvolutionTripletLayer, PointwiseConvolution 


class XNetSwinTransformer(_Network):
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
        final_downsample (bool): Do a final downsampling for the encoder towards the mid-network.
        cross_attention_residual: Use cross attention for Swin residual connections
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
        num_classes: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-5),
        middle_stages: int = 1,
        final_downsample: bool = False,
        residual_cross_attention: bool = True,
        weights=None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Smooth Patch Partitioning

        self.smooth_conv_in = ConvolutionTripletLayer(3, embed_dim, kernel_size=3)

        self.patching = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]), # B C H W -> B H W C
            norm_layer(embed_dim),
        )

        self.residual_cross_attention = residual_cross_attention
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
        for i_stage in range(len(depths)-1, -1, -1): # NOTE : Count Backwards!
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage

            # add patch merging layer
            if i_stage < (len(depths) - 1) or self.final_downsample:
                self.decoder.append(PatchExpandingV2(2*dim, norm_layer)) # NOTE : Double input dim

            if self.residual_cross_attention:
              self.decoder.append(
                SwinResidualCrossAttention(window_size=window_size, embed_dim=dim, 
                                           num_heads=num_heads[i_stage], attention_dropout=attention_dropout,
                                           norm_layer=norm_layer))

            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double

                stage.append(
                    SwinTransformerBlockV2(
                        dim * (1 + int(i_layer == 0)), # First Swin Block in Decoder Stage gets Stacked
                        num_heads[i_stage] * (1 + int(i_layer == 0)), # Double heads in this case too
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                if i_layer == 0:
                    stage.append(
                        PointwiseConvolution(2*dim, dim) # Reduce the dimensionality after first Swin in Stage
                    )
                stage_block_id += 1
            self.decoder.append(nn.Sequential(*stage))

        self.decoder = nn.ModuleList(self.decoder)

        self.unpatching = nn.Sequential(
            Permute([0, 3, 1, 2]), # B H W C -> B C H W
            nn.ConvTranspose2d(
                embed_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            nn.BatchNorm2d(embed_dim, eps=1e-5), # NOTE : Swapped out from LayerNorm because we are Conv-ing
        )

        # This will concatonate
        self.smooth_conv_out = ConvolutionTripletLayer(2*embed_dim, embed_dim, kernel_size=3)

        self.head = PointwiseConvolution(embed_dim, num_classes, channel_last=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.load(weights)

    def forward(self, x):

        conv_residual = self.smooth_conv_in(x)
    
        x = self.patching(conv_residual)
        
        residuals = []
        for i in range(0, len(self.encoder), 2):

            x = self.encoder[i](x) # Encoder Stage
            residuals.append(x)
            if i+1 < (len(self.encoder) - 1) or self.final_downsample:
                x = self.encoder[i+1](x) # Downsample (PatchMerge)

        *B, H, W, C = x.shape
        x = x.contiguous().view(*B, H*W, C)
        x = self.middle(x)
        x = x.contiguous().view(*B, H, W, C)

        for i_residual, i in zip(
            range(len(residuals)-1, -1, -1), # Count backwards for residual indices 
            range(0 - int(not self.final_downsample), len(self.decoder), 2 + int(self.residual_cross_attention))
        ):
            if i > 0 or self.final_downsample:
                x = self.decoder[i](x) # Upsample (PatchExpand)

            residual = residuals[i_residual]

            if self.residual_cross_attention:
                residual = self.decoder[i+1](x, residual) # Cross Attention Skip Connection

            x = torch.cat((x, residual), dim=-1) # Dumb Skip Connection

            x = self.decoder[i+(1 + int(self.residual_cross_attention))](x)

        x = self.unpatching(x)

        x = torch.cat((x, conv_residual), dim=-3) # ..., C, H, W

        x = self.smooth_conv_out(x)

        x = self.head(x)

        if self.num_classes == 1:
            x = x.squeeze(1)

        return x


# import torch
# import torch.nn.functional as F

# # Assume feature_map is the input tensor of shape [batch_size, channels, height, width]
# # Assume window_size is the size of the square window (e.g., 7 for a 7x7 window)

# batch_size, channels, height, width = feature_map.shape
# window_area = window_size * window_size

# # Step 1: Partition the feature map into windows
# windows = feature_map.unfold(2, window_size, window_size).unfold(3, window_size, window_size)
# windows = windows.contiguous().view(batch_size, channels, -1, window_area)  # [B, C, num_windows, window_area]

# # Step 2: Flatten and concatenate
# windows = windows.permute(0, 2, 1, 3).contiguous().view(-1, channels, window_area)  # [B*num_windows, C, window_area]

# # Step 3: Compute MSA in parallel for all windows
# # self_attn is a layer/module that computes multihead self-attention
# attn_windows = self_attn(windows)  # [B*num_windows, C, window_area]

# # Step 4: Reshape and merge back to the original feature map shape
# attn_windows = attn_windows.view(batch_size, -1, channels, window_area).permute(0, 2, 1, 3)
# attn_feature_map = F.fold(attn_windows, output_size=(height, width), kernel_size=(window_size, window_size))
