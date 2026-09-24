import os
import re
import torch
import torch.nn as nn
import timm


class Backbone(nn.Module):
    def __init__(self, name='convnextv2_tiny', pretrained=True,
                 pretrained_path=None, pretrained_source='auto',
                 img_size=640):
        super().__init__()
        self.name = name
        self.pretrained_source = pretrained_source
        self.pretrained_path = pretrained_path
        self.img_size = img_size

        # ---------- 解析来源 ----------
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
        print(f"[Backbone] name={name}, source={resolved}, img_size={img_size}")

        use_timm_pretrained = (resolved == 'timm')
        name_lower = name.lower()
        is_vit_like = any(x in name_lower for x in ['vit', 'dinov2'])
        is_swin = 'swin' in name_lower

        create_kwargs = dict(pretrained=use_timm_pretrained,
                             features_only=True, num_classes=0)
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

        # ---------- SimMIM ----------
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
            missing, unexpected = self.backbone.load_state_dict(remapped, strict=False)

            ignored = ('relative_position_index', 'attn_mask', 'num_batches_tracked')
            real_missing = [k for k in missing if not k.endswith(ignored)]
            real_unexpected = [k for k in unexpected if not k.endswith(ignored)]
            total = len(remapped)
            loaded = total - len(unexpected)
            print(f"  权重: {loaded}/{total} | 缺失 {len(real_missing)} | 意外 {len(real_unexpected)}")

        # ---------- 特征图信息 ----------
        self.feature_info = self.backbone.feature_info
        all_channels = self.feature_info.channels()
        all_strides = [info['reduction'] for info in self.feature_info]

        print(f"Backbone: {name}")
        print(f"所有通道: {all_channels}")
        print(f"所有步长: {all_strides}")

        if len(set(all_strides)) == 1:
            # ViT 风格（所有 stride 相同）
            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            self.is_vit = True
            print("检测到 ViT 风格模型")
        elif 'swin' in name.lower():
            self.feature_indices, self.out_channels, self.strides = [], [], []
            for idx, stride in enumerate(all_strides):
                if stride in [8, 16, 32]:
                    self.feature_indices.append(idx)
                    self.out_channels.append(all_channels[idx])
                    self.strides.append(stride)
            if len(self.feature_indices) < 3:
                self.feature_indices = [1, 2, 3]
                self.out_channels = all_channels[1:4]
                self.strides = all_strides[1:4]
            self.is_vit = False
        else:
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.is_vit = False

        print(f"选择索引: {self.feature_indices}")
        print(f"通道: {self.out_channels}")
        print(f"步长: {self.strides}")

        self.need_permute = 'swin' in name.lower()
        print(f"需 NHWC→NCHW: {self.need_permute}")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(self, x):
        features = self.backbone(x)
        selected = [features[i] for i in self.feature_indices]
        if self.need_permute:
            out = []
            for f in selected:
                if len(f.shape) == 4:
                    f = f.permute(0, 3, 1, 2)
                out.append(f)
            selected = out
        return selected