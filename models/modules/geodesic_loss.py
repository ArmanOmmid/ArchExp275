
import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss

class SpecialEuclideanGeodesicLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.translation_loss = nn.MSELoss()

    def normalize(self, rot_matrix):
        u, s, v = torch.svd(rot_matrix)
        return torch.bmm(u, v.transpose(-2, -1))

    def forward(self, predicted_transform, target_transform):
        # Transforms are 3x4 with a 3x3 in SO(3) and a 3x1 in R(3)

        p_T = predicted_transform[:, :3, 3]
        t_T = target_transform[:, :3, 3]

        translation_loss = self.translation_loss(p_T, t_T)

        p_R = predicted_transform[:, :3, :3]
        t_R = target_transform[:, :3, :3]

        # p_R = self.normalize(p_R)
        I = torch.eye(3, device=p_R.device).expand_as(p_R)
        ortho_loss = torch.norm(torch.matmul(p_R, p_R.transpose(-2, -1)) - I, dim=(-2, -1)).mean()

        relative_rotation = torch.bmm(p_R, t_R.transpose(-2, -1))

        batch_trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=-1)

        cos_theta = (batch_trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -0.99, 0.99)  # Numerical stability
        theta = torch.acos(cos_theta)

        rotation_loss = torch.mean(theta)

        return rotation_loss + translation_loss + ortho_loss








    
    
