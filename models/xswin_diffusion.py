
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._network import _Network
from .modules import SwinTransformerBlockV2_Modulated, ViTEncoderBlock_Modulated, \
                        PatchMergingV2_Modulated, PatchExpandingV2_Modulated, Patching_Modulated, UnPatching_Modulated, \
                        SwinResidualCrossAttention_Modulated, ConvolutionTriplet_Modulated, PointwiseConvolution_Modulated, \
                        create_positional_embedding, initialize_weights, TimestepEmbedder, LabelEmbedder

class XNetSwinTransformer_Diffusion(_Network):
    """
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
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: False.
        final_downsample (bool): Do a final downsampling for the encoder towards the mid-network.
        cross_attention_residual (bool): Use cross attention for Swin residual connections
        weights (str): Path to load weights
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
        final_downsample: bool = True,
        residual_cross_attention: bool = True,
        input_size: List[int] = None, # Needed to deduce the positional encodings in the global ViT layers
        class_dropout_prob=0.1,
        weights=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        if input_size is None:
            print("WARNING: Not Providing The Argument 'input_size' Means Latent ViT Blocks will NOT Have Positional Embedings")


        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)

        # Smooth Patch Partitioning

        self.smooth_conv_in = ConvolutionTriplet_Modulated(3, embed_dim, kernel_size=3)

        self.patching = Patching_Modulated(embed_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)

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
                    SwinTransformerBlockV2_Modulated(
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
                self.encoder.append(PatchMergingV2_Modulated(dim, norm_layer))

        self.encoder = nn.ModuleList(self.encoder)

        current_features = embed_dim * 2 ** (len(depths) - int(not self.final_downsample))

        ################################################
        # MIDDLE
        ################################################

        if input_size is None:
            self.pos_embed = 0
        else:
            downsample_count = len(depths) - int(not final_downsample) # downsample is one less in this case
            latent_H = input_size[0] // window_size[0]
            latent_W = input_size[1] // window_size[1]
            for i in range(downsample_count):
                latent_H = latent_H // 2
                latent_W = latent_W // 2
            self.pos_embed = create_positional_embedding(current_features, latent_H*latent_H)

        self.middle : List[nn.Module] = []
        for _ in range(min(1, middle_stages)):
            self.middle.append(
                ViTEncoderBlock_Modulated(
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
                self.decoder.append(PatchExpandingV2_Modulated(2*dim, norm_layer)) # NOTE : Double input dim

            if self.residual_cross_attention:
              self.decoder.append(
                SwinResidualCrossAttention_Modulated(window_size=window_size, embed_dim=dim, 
                                           num_heads=num_heads[i_stage], attention_dropout=attention_dropout,
                                           norm_layer=norm_layer))

            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double

                stage.append(
                    SwinTransformerBlockV2_Modulated(
                        dim * (1 + int(i_layer == 0)), # First Swin Block in Decoder Stage gets Stacked from Residuals
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
                        PointwiseConvolution_Modulated(2*dim, dim) # Reduce the dimensionality after first Swin in Stage
                    )
                stage_block_id += 1
            self.decoder.append(nn.Sequential(*stage))

        self.decoder = nn.ModuleList(self.decoder)

        self.unpatching = UnPatching_Modulated(embed_dim=embed_dim, patch_size=patch_size) # Norm Layer uses default BatchNorm

        # This will concatonate as a residual connection
        self.smooth_conv_out = ConvolutionTriplet_Modulated(2*embed_dim, embed_dim, kernel_size=3)

        self.head = PointwiseConvolution_Modulated(embed_dim, num_classes, channel_last=False)

        initialize_weights(self)

        self.load(weights)

    def forward(self, x, t, y):

        spatial_shape = (x.size(-2), x.size(-1))

        conv_residual = self.smooth_conv_in(x)
    
        x = self.patching(conv_residual) # B C H W -> B H W C
        
        residuals = []
        for i in range(0, len(self.encoder), 2):

            x = self.encoder[i](x) # Encoder Stage

            residuals.append(x)
            if i+1 < (len(self.encoder) - 1) or self.final_downsample:
                x = self.encoder[i+1](x) # Downsample (PatchMerge)

        *B, H, W, C = x.shape
        x = x.contiguous().view(*B, H*W, C)
        
        x = self.middle(x) + self.pos_embed

        x = x.contiguous().view(*B, H, W, C)

        for i_residual, i in zip(
            range(len(residuals)-1, -1, -1), # Count backwards for residual indices 
            range(0 - int(not self.final_downsample), len(self.decoder), 2 + int(self.residual_cross_attention))
        ):
            
            if i > 0 or self.final_downsample:
                x = self.decoder[i](x) # Upsample (PatchExpand)
                x = self.decoder[i]._post_expand_trim(x, residuals[i_residual].shape)

            if self.residual_cross_attention:
                residual = self.decoder[i+1](x, residuals[i_residual]) # Cross Attention Skip Connection
            else:
                residual = residuals[i_residual]

            x = torch.cat((x, residual), dim=-1) # Dumb Skip Connection

            x = self.decoder[i+(1 + int(self.residual_cross_attention))](x)

        # Does equally spaced padding to recover the original shape to concat with
        x = self.unpatching(x, target_spatial_shape=spatial_shape) # B H W C -> B C H W

        x = torch.cat((x, conv_residual), dim=-3) # ..., C, H, W

        x = self.smooth_conv_out(x)

        x = self.head(x)

        if self.num_classes == 1:
            x = x.squeeze(1)

        return x
