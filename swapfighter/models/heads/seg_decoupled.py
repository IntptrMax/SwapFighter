import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoupled import DecoupledHead, DecoupledBranch

def _make_norm(num_channels, norm_layer):
    if norm_layer in ('group', None):
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        norm = nn.GroupNorm(num_groups, num_channels)
        nn.init.constant_(norm.weight, 1)
        nn.init.constant_(norm.bias, 0)
        return norm
    elif norm_layer == 'batch':
        norm = nn.BatchNorm2d(num_channels)
        nn.init.constant_(norm.weight, 1)
        nn.init.constant_(norm.bias, 0)
        return norm
    elif norm_layer == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown norm_layer: {norm_layer}")

def _make_act(act_layer):
    if act_layer == 'relu':
        return nn.ReLU(inplace=True)
    elif act_layer == 'silu':
        return nn.SiLU(inplace=True)
    elif act_layer == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unknown act_layer: {act_layer}")

class SegDecoupledHead(DecoupledHead):

    def __init__(self, in_channels, num_classes,
                 num_mask_coef=32,
                 with_angle=False, angle_format='sin_cos',
                 num_convs=2, reg_max=16,
                 norm_layer='group', act_layer='silu',
                 cls_num_convs=None, reg_num_convs=None,
                 angle_num_convs=None):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            with_angle=with_angle,
            angle_format=angle_format,
            num_convs=num_convs,
            reg_max=reg_max,
            norm_layer=norm_layer,
            act_layer=act_layer,
            cls_num_convs=cls_num_convs,
            reg_num_convs=reg_num_convs,
            angle_num_convs=angle_num_convs,
        )

        self.coef_branch = DecoupledBranch(
            in_channels, num_mask_coef,
            num_convs=num_convs,
            norm_layer=norm_layer, act_layer=act_layer,
        )
        self.num_mask_coef = num_mask_coef

        with torch.no_grad():
            nn.init.normal_(self.coef_branch.pred.weight, std=0.001)
            if self.coef_branch.pred.bias is not None:
                nn.init.constant_(self.coef_branch.pred.bias, 0)

    def forward(self, x):
        out = super().forward(x)
        out['coef'] = self.coef_branch(x)
        return out

class SegPrototypeBranch(nn.Module):

    def __init__(self, in_channels_list, proto_dim=32,
                 fuse_channels=64,
                 proto_stride=4,
                 num_convs=2, norm_layer='group', act_layer='silu'):
        super().__init__()
        assert len(in_channels_list) == 3, \
            f"SegPrototypeBranch 需要 3 个 level，实际 {len(in_channels_list)}"
        assert proto_stride in (2, 4, 8), \
            f"proto_stride 必须是 2/4/8，实际 {proto_stride}"

        c3, c4, c5 = in_channels_list
        self.proj3 = nn.Conv2d(c3, fuse_channels, 1)
        self.proj4 = nn.Conv2d(c4, fuse_channels, 1)
        self.proj5 = nn.Conv2d(c5, fuse_channels, 1)

        layers = []
        layers.append(nn.Conv2d(fuse_channels * 3, fuse_channels,
                                kernel_size=3, padding=1, bias=False))
        layers.append(_make_norm(fuse_channels, norm_layer))
        layers.append(_make_act(act_layer))
        for _ in range(max(1, num_convs) - 1):
            layers.append(nn.Conv2d(fuse_channels, fuse_channels,
                                    kernel_size=3, padding=1, bias=False))
            layers.append(_make_norm(fuse_channels, norm_layer))
            layers.append(_make_act(act_layer))
        layers.append(nn.Conv2d(fuse_channels, proto_dim, 1))
        self.fuse = nn.Sequential(*layers)

        self.proto_dim = proto_dim
        self.fuse_channels = fuse_channels
        self.proto_stride = proto_stride

    def forward(self, neck_out):
        p3, p4, p5 = neck_out[0], neck_out[1], neck_out[2]
        target_size = p3.shape[2:]

        p3_proj = self.proj3(p3)
        p4_proj = F.interpolate(self.proj4(p4), size=target_size,
                                mode='bilinear', align_corners=False)
        p5_proj = F.interpolate(self.proj5(p5), size=target_size,
                                mode='bilinear', align_corners=False)

        fused = torch.cat([p3_proj, p4_proj, p5_proj], dim=1)
        proto = self.fuse(fused)

        upsample_factor = max(1, 8 // self.proto_stride)
        if upsample_factor > 1:
            proto = F.interpolate(
                proto, scale_factor=upsample_factor,
                mode='bilinear', align_corners=False)

        return proto

__all__ = ['SegDecoupledHead', 'SegPrototypeBranch']