
from abc import ABC

import os
import torch
import torch.nn as nn

WEIGHTS_EXTENSION = ".pth"

class _Network(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, load_path: str=None, map_location: str="cpu"):
        """
        Load weights if weights were specified
        """
        if not load_path: return
        load_path = load_path.split(".")[0] + WEIGHTS_EXTENSION
        self.load_state_dict(torch.load(load_path, map_location=torch.device(map_location)))

    def save(self, save_path: str):
        """
        All saves should be under the same path folder, under different tag folders, with the same filename
        """
        save_path = save_path.split(".")[0] + WEIGHTS_EXTENSION
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(self.state_dict(), save_path)
