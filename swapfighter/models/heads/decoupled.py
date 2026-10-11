import numpy as np
import torch
import torch.nn as nn

class DecoupledBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2,
                 norm_layer='group', act_layer='silu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        for _ in range(num_convs):
            conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            layers.append(conv)
            layers.append(self._make_norm(in_channels, norm_layer))
            layers.append(self._make_act(act_layer))

        self.tower = nn.Sequential(*layers)
        self.pred = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.pred.is_pred_layer = True
        nn.init.kaiming_normal_(self.pred.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.pred.bias, 0)

    @staticmethod
    def _make_norm(num_channels, norm_layer):
        if norm_layer in ('group', None):
            num_groups = min(32, num_channels)
            while num_channels % num_groups != 0:
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

    @staticmethod
    def _make_act(act_layer):
        if act_layer == 'relu':
            return nn.ReLU(inplace=True)
        elif act_layer == 'silu':
            return nn.SiLU(inplace=True)
        elif act_layer == 'gelu':
            return nn.GELU()
        raise ValueError(f"Unknown act_layer: {act_layer}")

    def forward(self, x):
        return self.pred(self.tower(x))

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes,
                 with_angle=False, angle_format='sin_cos',
                 num_convs=2, reg_max=16,
                 norm_layer='group', act_layer='silu',
                 cls_num_convs=None, reg_num_convs=None, angle_num_convs=None):
        super().__init__()
        self.with_angle = with_angle
        self.reg_max = reg_max
        self.angle_format = angle_format
        self.in_channels = in_channels
        self.num_classes = num_classes

        cls_n = cls_num_convs if cls_num_convs is not None else num_convs
        reg_n = reg_num_convs if reg_num_convs is not None else num_convs
        ang_n = angle_num_convs if angle_num_convs is not None else num_convs

        self.cls_branch = DecoupledBranch(
            in_channels, num_classes, num_convs=cls_n,
            norm_layer=norm_layer, act_layer=act_layer)

        reg_out = 4 * (reg_max + 1) if reg_max > 0 else 4
        self.reg_branch = DecoupledBranch(
            in_channels, reg_out, num_convs=reg_n,
            norm_layer=norm_layer, act_layer=act_layer)

        if with_angle:
            angle_out = 2 if angle_format == 'sin_cos' else 1
            self.angle_branch = DecoupledBranch(
                in_channels, angle_out, num_convs=ang_n,
                norm_layer=norm_layer, act_layer=act_layer)

        self._init_weights()

    def _init_weights(self):

        cls_pred = self.cls_branch.pred
        nn.init.normal_(cls_pred.weight, std=0.01)
        if cls_pred.bias is not None:
            with torch.no_grad():
                prior_prob = 0.01
                cls_pred.bias.fill_(-np.log((1 - prior_prob) / prior_prob))

        reg_pred = self.reg_branch.pred
        nn.init.normal_(reg_pred.weight, std=0.01)
        if reg_pred.bias is not None:
            with torch.no_grad():
                reg_pred.bias.zero_()
                if self.reg_max > 0:
                    bias = reg_pred.bias.view(4, self.reg_max + 1)
                    idx = torch.arange(self.reg_max + 1, dtype=bias.dtype, device=bias.device)
                    target_init = min(2.0, self.reg_max * 0.25)
                    prior = -0.5 * (idx - target_init) ** 2
                    bias.add_(prior.unsqueeze(0))
                else:
                    if reg_pred.bias.numel() >= 4:
                        reg_pred.bias[2:4].fill_(1.386)

        if self.with_angle:
            angle_pred = self.angle_branch.pred
            nn.init.normal_(angle_pred.weight, std=0.005)
            if angle_pred.bias is not None:
                with torch.no_grad():
                    angle_pred.bias.zero_()
                    if self.angle_format == 'sin_cos' and angle_pred.bias.numel() >= 2:
                        angle_pred.bias[0] = 0.0
                        angle_pred.bias[1] = 1.0

    def forward(self, x):
        out = {
            'cls': self.cls_branch(x),
            'reg': self.reg_branch(x),
        }
        if self.with_angle:
            out['angle'] = self.angle_branch(x)
        return out