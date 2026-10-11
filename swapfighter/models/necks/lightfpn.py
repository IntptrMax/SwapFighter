import torch.nn as nn
import torch.nn.functional as F

class LightFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128,
                 use_extra_conv=True, use_pan=True):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.use_extra_conv = use_extra_conv
        self.use_pan = use_pan

        if isinstance(out_channels, int):
            out_channels_list = [out_channels] * self.num_levels
        else:
            out_channels_list = list(out_channels)
            if len(out_channels_list) != self.num_levels:
                raise ValueError(
                    f"out_channels 长度 ({len(out_channels_list)}) "
                    f"必须等于 level 数 ({self.num_levels})")
        self.out_channels = out_channels_list
        self.out_channels_unified = isinstance(out_channels, int)

        print(f"[LightFPN] out_channels={out_channels_list}, "
              f"use_extra_conv={use_extra_conv}, use_pan={use_pan}")

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1)
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])

        self.td_proj = nn.ModuleList()
        self.td_convs = nn.ModuleList()

        for i in range(self.num_levels - 1):

            self.td_proj.append(nn.Conv2d(out_channels_list[i + 1],
                                          out_channels_list[i], 1))
            if use_extra_conv:
                self.td_convs.append(nn.Sequential(
                    nn.Conv2d(out_channels_list[i], out_channels_list[i],
                              3, padding=1),
                    self._make_norm(out_channels_list[i]),
                    nn.ReLU(inplace=True),
                ))

        if use_pan:
            self.bu_proj = nn.ModuleList()
            self.bu_down = nn.ModuleList()
            self.bu_convs = nn.ModuleList()

            for i in range(1, self.num_levels):

                self.bu_down.append(nn.Sequential(
                    nn.Conv2d(out_channels_list[i - 1],
                              out_channels_list[i], 3, stride=2, padding=1),
                    self._make_norm(out_channels_list[i]),
                    nn.ReLU(inplace=True),
                ))
                if use_extra_conv:
                    self.bu_convs.append(nn.Sequential(
                        nn.Conv2d(out_channels_list[i], out_channels_list[i],
                                  3, padding=1),
                        self._make_norm(out_channels_list[i]),
                        nn.ReLU(inplace=True),
                    ))

    @staticmethod
    def _make_norm(num_channels):

        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0:
            num_groups //= 2
        return nn.BatchNorm2d(num_channels)

    def forward(self, features):

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        td_outputs = [None] * self.num_levels
        td_outputs[-1] = laterals[-1]
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(td_outputs[i + 1],
                               size=laterals[i].shape[2:],
                               mode='nearest')
            up = self.td_proj[i](up)
            fused = laterals[i] + up
            if self.use_extra_conv:
                fused = self.td_convs[i](fused)
            td_outputs[i] = fused

        if not self.use_pan:
            return td_outputs

        bu_outputs = [None] * self.num_levels
        bu_outputs[0] = td_outputs[0]
        for i in range(1, self.num_levels):
            down = self.bu_down[i - 1](bu_outputs[i - 1])
            if down.shape[2:] != td_outputs[i].shape[2:]:
                down = F.interpolate(down, size=td_outputs[i].shape[2:],
                                     mode='nearest')
            fused = td_outputs[i] + down
            if self.use_extra_conv:
                fused = self.bu_convs[i - 1](fused)
            bu_outputs[i] = fused

        return bu_outputs