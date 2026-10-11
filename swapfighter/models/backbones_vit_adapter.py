import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_norm(num_channels, norm_layer='group'):
    if norm_layer == 'group':
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
    else:
        raise ValueError(f"未知 norm_layer: {norm_layer!r}")

def _make_act(act_layer='gelu'):
    if act_layer == 'gelu':
        return nn.GELU()
    elif act_layer == 'silu':
        return nn.SiLU(inplace=True)
    elif act_layer == 'relu':
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"未知 act_layer: {act_layer!r}")

class ViTAdapter(nn.Module):

    def __init__(self, in_channels,
                 out_channels_list=None,
                 norm_layer='group',
                 act_layer='gelu',
                 fuse=True):
        super().__init__()

        if out_channels_list is None:
            C3 = in_channels // 4
            C4 = in_channels // 2
            C5 = in_channels
            out_channels_list = (C3, C4, C5)

        C3, C4, C5 = out_channels_list
        self.in_channels = in_channels
        self.out_channels = [C3, C4, C5]
        self.fuse = fuse

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, C3, kernel_size=2, stride=2),
            _make_norm(C3, norm_layer),
            _make_act(act_layer),
        )
        self.same = nn.Sequential(
            nn.Conv2d(in_channels, C4, kernel_size=1, bias=False),
            _make_norm(C4, norm_layer),
            _make_act(act_layer),
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, C5, kernel_size=3,
                      stride=2, padding=1, bias=False),
            _make_norm(C5, norm_layer),
            _make_act(act_layer),
        )

        if fuse:

            self.proj_4to3 = nn.Conv2d(C4, C3, kernel_size=1, bias=False)

            self.proj_4to5 = nn.Conv2d(C4, C5, kernel_size=1, bias=False)

        self.out3 = nn.Sequential(
            nn.Conv2d(C3, C3, kernel_size=3, padding=1, bias=False),
            _make_norm(C3, norm_layer),
            _make_act(act_layer),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(C4, C4, kernel_size=3, padding=1, bias=False),
            _make_norm(C4, norm_layer),
            _make_act(act_layer),
        )
        self.out5 = nn.Sequential(
            nn.Conv2d(C5, C5, kernel_size=3, padding=1, bias=False),
            _make_norm(C5, norm_layer),
            _make_act(act_layer),
        )

        print(f"[ViTAdapter] in={in_channels}, "
              f"out={self.out_channels}, fuse={fuse}")

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"ViTAdapter 需要 4D 特征图 (B, C, H, W)，"
                f"实际得到 shape={tuple(x.shape)}")

        up = self.up(x)
        same = self.same(x)
        down = self.down(x)

        if self.fuse:

            same_up = F.interpolate(same, size=up.shape[2:],
                                    mode='nearest')

            same_up = self.proj_4to3(same_up)
            up = up + same_up

            same_down = F.avg_pool2d(same, kernel_size=2)

            if same_down.shape[2:] != down.shape[2:]:
                same_down = F.interpolate(same_down, size=down.shape[2:],
                                          mode='nearest')

            same_down = self.proj_4to5(same_down)
            down = down + same_down

        f3 = self.out3(up)
        f4 = self.out4(same)
        f5 = self.out5(down)

        return [f3, f4, f5]

def to_spatial_4d(x, img_size=None, patch_size=None):
    if isinstance(x, (list, tuple)):
        x = x[-1]

    if x.dim() == 4:
        return x

    if x.dim() != 3:
        raise ValueError(f"无法处理的 tensor 维度: {x.dim()}")

    if img_size is None or patch_size is None:
        raise ValueError(
            "3D token 输入需要提供 img_size 和 patch_size 以恢复空间维度")

    B, N, C = x.shape
    H = W = img_size // patch_size

    if N == H * W + 1:
        x = x[:, 1:, :]
    elif N != H * W:
        raise ValueError(
            f"token 数 N={N} 与 img_size/patch_size 推算的 "
            f"{H}x{W}={H*W} 不匹配")

    return x.transpose(1, 2).reshape(B, C, H, W)

__all__ = ['ViTAdapter', 'to_spatial_4d']