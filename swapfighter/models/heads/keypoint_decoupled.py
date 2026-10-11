import torch
import torch.nn as nn

from .decoupled import DecoupledHead, DecoupledBranch

class KeypointDecoupledHead(DecoupledHead):
    def __init__(self, in_channels, num_classes,
                 num_keypoints=17,
                 with_angle=False, angle_format='sin_cos',
                 num_convs=2, reg_max=16,
                 norm_layer='group', act_layer='silu',
                 cls_num_convs=None, reg_num_convs=None,
                 angle_num_convs=None,
                 kpt_num_convs=None):
        super().__init__(
            in_channels=in_channels, num_classes=num_classes,
            with_angle=with_angle, angle_format=angle_format,
            num_convs=num_convs, reg_max=reg_max,
            norm_layer=norm_layer, act_layer=act_layer,
            cls_num_convs=cls_num_convs,
            reg_num_convs=reg_num_convs,
            angle_num_convs=angle_num_convs,
        )

        kpt_n = (kpt_num_convs if kpt_num_convs is not None
                 else num_convs)
        self.kpt_branch = DecoupledBranch(
            in_channels, num_keypoints * 2,
            num_convs=kpt_n,
            norm_layer=norm_layer, act_layer=act_layer,
        )
        self.num_keypoints = num_keypoints

        with torch.no_grad():
            nn.init.normal_(self.kpt_branch.pred.weight, std=0.001)
            if self.kpt_branch.pred.bias is not None:
                nn.init.constant_(self.kpt_branch.pred.bias, 0.5)

    def forward(self, x):
        out = super().forward(x)
        out['kpt'] = self.kpt_branch(x)
        return out

__all__ = ['KeypointDecoupledHead']