import torch.nn as nn
import torch.nn.functional as F


class LightFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128,
                 use_extra_conv=True, use_pan=True):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.use_extra_conv = use_extra_conv
        self.use_pan = use_pan

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list])

        if use_extra_conv:
            self.td_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ) for _ in range(self.num_levels - 1)
            ])

        if use_pan:
            self.downsample_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ) for _ in range(self.num_levels - 1)
            ])
            if use_extra_conv:
                self.bu_convs = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ) for _ in range(self.num_levels - 1)
                ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        td_outputs = [None] * self.num_levels
        td_outputs[-1] = laterals[-1]
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(td_outputs[i + 1], size=laterals[i].shape[2:], mode='nearest')
            fused = laterals[i] + up
            if self.use_extra_conv:
                fused = self.td_convs[i](fused)
            td_outputs[i] = fused

        if not self.use_pan:
            return td_outputs

        bu_outputs = [None] * self.num_levels
        bu_outputs[0] = td_outputs[0]
        for i in range(1, self.num_levels):
            down = self.downsample_convs[i - 1](bu_outputs[i - 1])
            if down.shape[2:] != td_outputs[i].shape[2:]:
                down = F.interpolate(down, size=td_outputs[i].shape[2:], mode='nearest')
            fused = td_outputs[i] + down
            if self.use_extra_conv:
                fused = self.bu_convs[i - 1](fused)
            bu_outputs[i] = fused

        return bu_outputs