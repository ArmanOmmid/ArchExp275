
import torch
import torch.nn as nn

def initialize_weights(model):
    # Initialize transformer layers:
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    model.apply(_basic_init)
