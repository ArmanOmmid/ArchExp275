
import torch
import torch.nn as nn
from torchvision.ops.misc import Permute

from ._network import _Network

class TNet(nn.Module):
    def __init__(self, k=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self. k = k
        self.permute = Permute([0, 2, 1])
        self.conv = nn.Sequential(
            nn.Conv1d(k ,64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128 ,1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(100)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, k*k)

    def forward(self, x: torch.Tensor):

        x = self.permute(x)
        # B C L
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        x = self.head(x)
        x += torch.eye(self.k).flatten().to(x.device)
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(_Network):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_tnet = TNet(3)

        self.feature_tnet = TNet(64)

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        
        num_points = x.size(2)

        T1 = self.input_tnet(x)

        x = x.permute(2, 1)
        x = torch.bmm(x, T1)
        x = x.permute(2, 1)

        x = self.conv1(x)

        return x
