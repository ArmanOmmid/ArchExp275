from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._network import _Network
from .xswin import XNetSwinTransformer

class DenseFusion(_Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs):

        seg_net = 
