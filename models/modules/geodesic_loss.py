
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class SpecialEuclideanGeodesicLoss(_Loss):
    def __init__(self, SO_weight=0.1) -> None:
        super().__init__()
        
        self.SO_weight = SO_weight
        if type(SO_weight) == int or type(SO_weight) == float:
            self.SO_Loss = True
        else:
            self.SO_Loss = False

    def normalize(self, rot_matrix):
        u, s, v = torch.svd(rot_matrix)
        return torch.bmm(u, v.transpose(-2, -1))

    def forward(self, predicted_transform, target_transform, components=False):
        # Transforms are 3x4 with a 3x3 in SO(3) and a 3x1 in R(3)

        losses = []

        p_T = predicted_transform[:, :3, 3]
        t_T = target_transform[:, :3, 3]

        translation_loss = torch.norm(p_T - t_T).mean() # Over Batch

        p_R = predicted_transform[:, :3, :3]
        t_R = target_transform[:, :3, :3]

        relative_rotation = torch.bmm(p_R, t_R.transpose(-2, -1))

        batch_trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=-1)

        cos_theta = (batch_trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)  # Numerical stability
        theta = torch.acos(cos_theta)

        rotation_loss = torch.mean(theta) # Over Batch

        losses.append(rotation_loss)
        losses.append(translation_loss)

        if self.SO_Loss:
            I = torch.eye(3, device=p_R.device).expand_as(p_R)
            ortho_loss = torch.norm(torch.bmm(p_R, p_R.transpose(-2, -1)) - I, dim=(-2, -1)).mean() # Over Batch
            losses.append(ortho_loss * self.SO_weight)

        losses = torch.stack(losses)

        if components:
            return losses
        
        return losses.mean()

class SpecialOrthogonalLoss(_Loss):
    def __init__(self, weight=0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, rotations):

        if type(rotations) not in [list, tuple]:
            rotations = [rotations]
        
        losses = []
        for R in rotations:
            row, col = R.shape[-2:]
            assert row == col
            I = torch.eye(row, device=R.device).expand_as(R)
            ortho_loss = torch.norm(torch.bmm(R, R.transpose(-2, -1)) - I, dim=(-2, -1)).mean() # Over Batch
            losses.append(ortho_loss)
        loss = torch.stack(losses).mean()

        return loss * self.weight
    
class PointCloudMSELoss(_Loss):
    def __init__(self, target_type="pose", weight=1.0) -> None:
        super().__init__()
        self.weight = weight
        assert target_type in ["pose", "point"]
        self.target_type = target_type

    def forward(self, source, pose, target):

        R = pose[:, :, :3]
        T = pose[:, :, 3]

        source_transformed = torch.bmm(source, R.transpose(-2, -1)) + T.unsqueeze(-2)

        if target == "pose":
            # In this case, target is the ground truth pose
            target = torch.bmm(source, target.transpose(-2, -1)) + T.unsqueeze(-2)
        # Otherwise, target is a point cloud (WHICH REQUIRES CORRESPONDENCES)

        loss = F.mse_loss(source_transformed, target)

        return loss * self.weight
