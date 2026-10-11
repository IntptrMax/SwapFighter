import os
import re
import torch
import torch.nn as nn
import timm
import torch.utils.checkpoint as cp

from .backbones_vit_adapter import ViTAdapter, to_spatial_4d

class Backbone(nn.Module):
    def __init__(self, name='convnextv2_tiny', pretrained=True,
                 pretrained_path=None, pretrained_source='auto',
                 img_size=640, use_grad_checkpoint=False):
        super().__init__()
        self.name = name
        self.pretrained_source = pretrained_source
        self.pretrained_path = pretrained_path
        self.img_size = img_size
        self.use_grad_checkpoint = use_grad_checkpoint

        if pretrained_source == 'auto':
            if pretrained_path is not None and os.path.exists(pretrained_path):
                resolved = 'simmim'
            elif pretrained:
                resolved = 'timm'
            else:
                resolved = 'none'
        elif pretrained_source == 'simmim':
            if pretrained_path is None or not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"pretrained_source='simmim' 但权重路径无效: {pretrained_path}")
            resolved = 'simmim'
        elif pretrained_source in ('timm', 'none'):
            resolved = pretrained_source
        else:
            raise ValueError(f"未知 pretrained_source: {pretrained_source}")

        self.resolved_source = resolved

        name_lower = name.lower()

        is_vit_like = any(x in name_lower for x in
                          ['vit_', 'vit-', 'dinov2', 'dinov3', 'eva',
                           'beit', 'deit'])

        is_swin = 'swin' in name_lower
        is_hiera = 'hiera' in name_lower

        use_timm_pretrained = (resolved == 'timm')

        create_kwargs = dict(
            pretrained=use_timm_pretrained,
            features_only=True,
            num_classes=0,
        )
        if is_vit_like:

            create_kwargs['dynamic_img_size'] = True
        elif is_swin:
            create_kwargs['img_size'] = img_size

        try:
            self.backbone = timm.create_model(name, **create_kwargs)
        except TypeError as e:
            print(f"[Backbone] create_model 参数回退: {e}")
            create_kwargs.pop('img_size', None)
            create_kwargs.pop('dynamic_img_size', None)
            self.backbone = timm.create_model(name, **create_kwargs)

        if resolved == 'simmim':
            print(f"[Backbone] 加载 SimMIM 权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('encoder.'):
                    k = k[8:]
                cleaned[k] = v

            target_keys = set(self.backbone.state_dict().keys())
            remapped = self._normalize_state_dict_keys(cleaned, target_keys)
            missing, unexpected = self.backbone.load_state_dict(
                remapped, strict=False)

            ignored = ('relative_position_index', 'attn_mask',
                       'num_batches_tracked')
            real_missing = [k for k in missing if not k.endswith(ignored)]
            real_unexpected = [k for k in unexpected if not k.endswith(ignored)]
            total = len(remapped)
            loaded = total - len(unexpected)
            print(f"  权重: {loaded}/{total} | "
                  f"缺失 {len(real_missing)} | 意外 {len(real_unexpected)}")

        self.feature_info = self.backbone.feature_info
        all_channels = self.feature_info.channels()
        all_strides = [info['reduction'] for info in self.feature_info]

        print(f"Backbone: {name}")
        print(f"所有通道: {all_channels}")
        print(f"所有步长: {all_strides}")

        self.adapter = None
        self.is_vit = False
        self.need_permute = False
        self.feature_indices = []
        self.out_channels = []
        self.strides = []

        if len(set(all_strides)) == 1:

            self.is_vit = True
            C = all_channels[0]
            S = all_strides[0]

            self.strides_raw = all_strides

            adapter_out_channels = [C // 4, C // 2, C]

            adapter_out_channels = [
                max(32, (c // 32) * 32) for c in adapter_out_channels
            ]

            self.adapter = ViTAdapter(
                in_channels=C,
                out_channels_list=adapter_out_channels,
                norm_layer='group',
                act_layer='gelu',
                fuse=True,
            )

            self.out_channels = adapter_out_channels
            self.strides = [max(1, S // 2), S, S * 2]

            print(f"检测到 ViT 风格模型 | "
                  f"ViTAdapter: in={C}, out={self.out_channels}, "
                  f"strides={self.strides}")

        elif is_swin or is_hiera:

            self.need_permute = True
            self.feature_indices = []
            self.out_channels = []
            self.strides = []
            for idx, stride in enumerate(all_strides):
                if stride in (8, 16, 32):
                    self.feature_indices.append(idx)
                    self.out_channels.append(all_channels[idx])
                    self.strides.append(stride)

            if len(self.feature_indices) < 3:
                self.feature_indices = list(range(len(all_channels)))[-3:]
                self.out_channels = all_channels[-3:]
                self.strides = all_strides[-3:]
            print(f"Swin/Hiera 风格 | 需 NHWC→NCHW")

        else:

            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            print(f"CNN 风格 | 天然多尺度")

        print(f"选择索引: {self.feature_indices}")
        print(f"输出通道: {self.out_channels}")
        print(f"输出步长: {self.strides}")
        print(f"需 NHWC→NCHW: {self.need_permute}")

    @staticmethod
    def _normalize_state_dict_keys(state_dict, target_keys):
        def normalize(k):
            return re.sub(r'\.(\d+)', r'_\1', k)

        target_map = {normalize(tk): tk for tk in target_keys}
        new_sd = {}
        hit = 0
        for k, v in state_dict.items():
            nk = normalize(k)
            if nk in target_map:
                new_sd[target_map[nk]] = v
                hit += 1
            else:
                new_sd[k] = v
        print(f"  键名归一化: {hit}/{len(state_dict)}")
        return new_sd

    def forward(self, x):

        if self.training and getattr(self, 'use_grad_checkpoint', False):
            features = cp.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            features = self.backbone(x)

        if self.is_vit and self.adapter is not None:

            if isinstance(features, (list, tuple)):
                feat = features[-1]
            else:
                feat = features

            if feat.dim() == 3:
                patch_size = self.strides_raw[0]
                feat = to_spatial_4d(feat, self.img_size, patch_size)

            return self.adapter(feat)

        selected = [features[i] for i in self.feature_indices]

        if self.need_permute:
            out = []
            for f in selected:
                if f.dim() == 4 and f.shape[-1] != self.out_channels[0]:

                    f = f.permute(0, 3, 1, 2).contiguous()
                out.append(f)
            selected = out

        return selected

    def set_freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad = not freeze

        n_train = sum(p.numel() for p in self.parameters()
                      if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        status = 'frozen' if freeze else 'unfrozen'
        print(f"[Backbone] {status} | "
              f"可训练={n_train:,} / 总={n_total:,}")

    def get_freeze_status(self):
        return not next(self.backbone.parameters()).requires_grad

def list_available_backbones(filter_str=None, pretrained_only=False):
    names = timm.list_models(filter_str or '*', pretrained=pretrained_only)
    return names

__all__ = ['Backbone', 'list_available_backbones']