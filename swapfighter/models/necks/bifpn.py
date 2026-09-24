import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNLayer(nn.Module):
    def __init__(self, channels, num_levels=3, epsilon=1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon

        self.td_weights = nn.ParameterList()
        for _ in range(num_levels - 1):
            self.td_weights.append(nn.Parameter(torch.ones(2)))

        self.bu_weights = nn.ParameterList()
        for _ in range(1, num_levels):
            self.bu_weights.append(nn.Parameter(torch.ones(2)))

        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ) for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, inputs):
        td_outputs = [None] * self.num_levels
        td_outputs[-1] = inputs[-1]
        for i in range(self.num_levels - 2, -1, -1):
            cur = inputs[i]
            up = F.interpolate(td_outputs[i + 1], size=cur.shape[2:], mode='nearest')
            w = torch.softmax(self.td_weights[i], dim=0)
            fused = (w[0] * cur + w[1] * up) / (w.sum() + self.epsilon)
            td_outputs[i] = self.td_convs[i](fused)

        bu_outputs = [None] * self.num_levels
        bu_outputs[0] = td_outputs[0]
        for i in range(1, self.num_levels):
            cur = td_outputs[i]
            down = F.interpolate(bu_outputs[i - 1], size=cur.shape[2:], mode='nearest')
            w = torch.softmax(self.bu_weights[i - 1], dim=0)
            fused = (w[0] * cur + w[1] * down) / (w.sum() + self.epsilon)
            bu_outputs[i] = self.bu_convs[i - 1](fused)

        return bu_outputs


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_layers=2):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list])
        self.bifpn_layers = nn.ModuleList(
            [BiFPNLayer(out_channels, self.num_levels) for _ in range(num_layers)])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(self.num_levels)
        ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        outputs = laterals
        for layer in self.bifpn_layers:
            outputs = layer(outputs)
        outputs = [conv(out) for conv, out in zip(self.output_convs, outputs)]
        return outputs