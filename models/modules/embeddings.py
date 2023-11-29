import torch
import torch.nn as nn
import numpy as np
import math

from torch import Tensor
from torchvision.ops.misc import MLP, Permute

class Modulator(nn.Module):
    def __init__(self, hidden_size, gate=False, channel_last=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.gate = gate
        self.chunks = 3 if gate else 2

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # For Conv Layers, channels are in the 3rd to last position!
        self.channel_last = channel_last
        if not channel_last:
            self.permute_in = Permute([0, 2, 3, 1])
            self.permute_out = Permute([0, 3, 1, 2])

    def _get_modulation(self, c):
        return self.adaLN_modulation(c).chunk(self.chunks, dim=1)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, x: Tensor, c: Tensor, apply_gate=True):

        if not self.channel_last:
            x = self.permute_in(x)

        if not self.gate:
            shift, scale = self._get_modulation(c)
            x = self._modulate(x, shift, scale)
            if not self.channel_last:
                x = self.permute_out(x)
            return x
        else:
            shift, scale, gate = self._get_modulation(c)
            gate = gate.unsqueeze(1)
            x = self._modulate(x, shift, scale)
            if apply_gate:
                x = x * gate
                if not self.channel_last:
                    x = self.permute_out(x)
                return x
            else:
                x = x, gate
                if not self.channel_last:
                    x = self.permute_out(x)
                    gate = self.permute_out(gate)
                    return x, gate
                return x, gate

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
