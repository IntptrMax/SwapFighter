import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F


# ============================================================
# Backbone
# ============================================================
class Backbone(nn.Module):
    def __init__(self, name='convnextv2_tiny', pretrained=True,
                 pretrained_path=None, pretrained_source='auto',
                 img_size=640):
        super().__init__()
        self.name = name
        self.pretrained_source = pretrained_source
        self.pretrained_path = pretrained_path
        self.img_size = img_size

        if pretrained_source == 'auto':
            if pretrained_path is not None and os.path.exists(pretrained_path):
                resolved_source = 'simmim'
            elif pretrained:
                resolved_source = 'timm'
            else:
                resolved_source = 'none'
        elif pretrained_source == 'simmim':
            if pretrained_path is None or not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"pretrained_source='simmim' 但权重路径无效: {pretrained_path}"
                )
            resolved_source = 'simmim'
        elif pretrained_source in ('timm', 'none'):
            resolved_source = pretrained_source
        else:
            raise ValueError(f"pretrained_source 必须是 'auto'/'timm'/'simmim'/'none'")

        self.resolved_source = resolved_source
        print(f"[Backbone] 初始化: name={name}, source={resolved_source}, img_size={img_size}")

        use_timm_pretrained = (resolved_source == 'timm')

        name_lower = name.lower()
        is_vit_like = any(x in name_lower for x in ['vit', 'dinov2'])
        is_swin = 'swin' in name_lower

        create_kwargs = dict(pretrained=use_timm_pretrained, features_only=True, num_classes=0)
        if is_vit_like:
            create_kwargs['dynamic_img_size'] = True
        elif is_swin:
            create_kwargs['img_size'] = img_size

        try:
            self.backbone = timm.create_model(name, **create_kwargs)
        except TypeError as e:
            print(f"[Backbone] create_model 参数不被接受，回退: {e}")
            create_kwargs.pop('img_size', None)
            create_kwargs.pop('dynamic_img_size', None)
            self.backbone = timm.create_model(name, **create_kwargs)

        if resolved_source == 'simmim':
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

            ignored_suffixes = ('relative_position_index', 'attn_mask', 'num_batches_tracked')
            real_missing = [k for k in missing if not k.endswith(ignored_suffixes)]
            real_unexpected = [k for k in unexpected if not k.endswith(ignored_suffixes)]
            total = len(remapped)
            loaded = total - len(unexpected)
            print(f"  权重加载: {loaded}/{total} 个键")
            print(f"  缺失键: {len(real_missing)}, 意外键: {len(real_unexpected)}")

        self.feature_info = self.backbone.feature_info
        all_channels = self.feature_info.channels()
        all_strides = [info['reduction'] for info in self.feature_info]

        print(f"Backbone: {name}")
        print(f"所有特征图通道数: {all_channels}")
        print(f"所有特征图步长: {all_strides}")

        if len(set(all_strides)) == 1:
            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            self.is_vit = True
            print("检测到 ViT 风格模型")
        elif 'swin' in name.lower():
            self.feature_indices = []
            self.out_channels = []
            self.strides = []
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

        print(f"选择的特征图索引: {self.feature_indices}")
        print(f"选择的通道数: {self.out_channels}")
        print(f"选择的步长: {self.strides}")

        self.need_permute = 'swin' in name.lower()
        print(f"需要格式转换 (NHWC->NCHW): {self.need_permute}")

    @staticmethod
    def _normalize_state_dict_keys(state_dict, target_keys):
        import re
        def normalize(k):
            return re.sub(r'\.(\d+)', r'_\1', k)
        target_map = {}
        for tk in target_keys:
            target_map[normalize(tk)] = tk
        new_sd = {}
        hit = 0
        for k, v in state_dict.items():
            nk = normalize(k)
            if nk in target_map:
                new_sd[target_map[nk]] = v
                hit += 1
            else:
                new_sd[k] = v
        print(f"  键名归一化: {hit}/{len(state_dict)} 个键完成映射")
        return new_sd

    def forward(self, x):
        features = self.backbone(x)
        selected_features = [features[i] for i in self.feature_indices]
        if self.need_permute:
            converted_features = []
            for feat in selected_features:
                if len(feat.shape) == 4:
                    feat = feat.permute(0, 3, 1, 2)
                converted_features.append(feat)
            selected_features = converted_features
        return selected_features


# ============================================================
# BiFPN
# ============================================================
class BiFPNLayer(nn.Module):
    def __init__(self, channels, num_levels=3, epsilon=1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon

        self.td_weights = nn.ParameterList()
        for _ in range(num_levels - 1):
            self.td_weights.append(nn.Parameter(torch.ones(2, dtype=torch.float32)))

        self.bu_weights = nn.ParameterList()
        for _ in range(1, num_levels):
            self.bu_weights.append(nn.Parameter(torch.ones(2, dtype=torch.float32)))

        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, inputs):
        td_outputs = [None] * self.num_levels
        td_outputs[-1] = inputs[-1]

        for i in range(self.num_levels - 2, -1, -1):
            cur = inputs[i]
            up = nn.functional.interpolate(td_outputs[i + 1], size=cur.shape[2:], mode='nearest')
            weights = self.td_weights[i]
            norm_weights = torch.softmax(weights, dim=0)
            fused = (norm_weights[0] * cur + norm_weights[1] * up) / (norm_weights.sum() + self.epsilon)
            td_outputs[i] = self.td_convs[i](fused)

        bu_outputs = [None] * self.num_levels
        bu_outputs[0] = td_outputs[0]
        for i in range(1, self.num_levels):
            cur = td_outputs[i]
            down = nn.functional.interpolate(bu_outputs[i - 1], size=cur.shape[2:], mode='nearest')
            weights = self.bu_weights[i - 1]
            norm_weights = torch.softmax(weights, dim=0)
            fused = (norm_weights[0] * cur + norm_weights[1] * down) / (norm_weights.sum() + self.epsilon)
            bu_outputs[i] = self.bu_convs[i - 1](fused)

        return bu_outputs


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_layers=2):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list])
        self.bifpn_layers = nn.ModuleList([BiFPNLayer(out_channels, self.num_levels) for _ in range(num_layers)])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        outputs = laterals
        for layer in self.bifpn_layers:
            outputs = layer(outputs)
        outputs = [conv(out) for conv, out in zip(self.output_convs, outputs)]
        return outputs


# ============================================================
# LightFPN
# ============================================================
class LightFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128, use_extra_conv=True, use_pan=True):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.use_extra_conv = use_extra_conv
        self.use_pan = use_pan

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list])

        if use_extra_conv:
            self.td_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(self.num_levels - 1)
            ])

        if use_pan:
            self.downsample_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(self.num_levels - 1)
            ])
            if use_extra_conv:
                self.bu_convs = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
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


# ============================================================
# ★ 解耦头：单任务分支
# ============================================================
class DecoupledBranch(nn.Module):
    """
    解耦分支：单个任务（cls / reg / angle）的独立卷积塔。

    YOLOv8-OBB 风格：
        Conv3x3 → Norm → Act  ×  num_convs
        → Conv1x1  →  任务输出

    每个任务拥有独立的卷积参数，从根本上消除任务间梯度冲突；
    预测头使用 1×1 卷积，相比 3×3 参数量降低 9 倍。
    """
    def __init__(self, in_channels, out_channels, num_convs=2,
                 norm_layer='group', act_layer='silu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ---------- 卷积塔 ----------
        layers = []
        for _ in range(num_convs):
            conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            layers.append(conv)

            layers.append(self._make_norm(in_channels, norm_layer))
            layers.append(self._make_act(act_layer))

        self.tower = nn.Sequential(*layers)

        # ---------- 1×1 预测头 ----------
        self.pred = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.pred.is_pred_layer = True

        # 默认初始化（会被 DecoupledHead._init_weights 覆盖）
        nn.init.kaiming_normal_(self.pred.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.pred.bias, 0)

    @staticmethod
    def _make_norm(num_channels, norm_layer):
        if norm_layer == 'group' or norm_layer is None:
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
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

    @staticmethod
    def _make_act(act_layer):
        if act_layer == 'relu':
            return nn.ReLU(inplace=True)
        elif act_layer == 'silu':
            return nn.SiLU(inplace=True)
        elif act_layer == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown act_layer: {act_layer}")

    def forward(self, x):
        return self.pred(self.tower(x))


# ============================================================
# ★ 解耦头：多任务容器
# ============================================================
class DecoupledHead(nn.Module):
    """
    YOLOv8-OBB 风格的解耦检测头。

    结构：
        neck_out (B, C, H, W)
            ├── cls_branch   → (B, num_classes,   H, W)
            ├── reg_branch   → (B, 4*(reg_max+1), H, W)  # DFL
            └── angle_branch → (B, 2 or 1,        H, W)  # 可选

    每个分支是完全独立的卷积塔，参数互不共享，
    彻底消除分类 / 回归 / 角度之间的梯度冲突。
    """
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

        # 允许每个分支使用不同的卷积层数
        cls_num_convs = cls_num_convs if cls_num_convs is not None else num_convs
        reg_num_convs = reg_num_convs if reg_num_convs is not None else num_convs
        angle_num_convs = angle_num_convs if angle_num_convs is not None else num_convs

        # ---------- 分类分支 ----------
        self.cls_branch = DecoupledBranch(
            in_channels, num_classes,
            num_convs=cls_num_convs,
            norm_layer=norm_layer, act_layer=act_layer,
        )

        # ---------- 回归分支（DFL 通道） ----------
        reg_out = 4 * (reg_max + 1) if reg_max > 0 else 4
        self.reg_branch = DecoupledBranch(
            in_channels, reg_out,
            num_convs=reg_num_convs,
            norm_layer=norm_layer, act_layer=act_layer,
        )

        # ---------- 角度分支 ----------
        if with_angle:
            angle_out = 2 if angle_format == 'sin_cos' else 1
            self.angle_branch = DecoupledBranch(
                in_channels, angle_out,
                num_convs=angle_num_convs,
                norm_layer=norm_layer, act_layer=act_layer,
            )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """任务特定的预测头初始化（与 YOLOv8 / DFL 一致）"""
        # ---------- 分类：focal loss 先验 ----------
        cls_pred = self.cls_branch.pred
        nn.init.normal_(cls_pred.weight, std=0.01)
        if cls_pred.bias is not None:
            with torch.no_grad():
                prior_prob = 0.01
                cls_pred.bias.fill_(-np.log((1 - prior_prob) / prior_prob))

        # ---------- 回归：DFL 先验 ----------
        reg_pred = self.reg_branch.pred
        nn.init.normal_(reg_pred.weight, std=0.01)
        if reg_pred.bias is not None:
            with torch.no_grad():
                reg_pred.bias.zero_()
                if self.reg_max > 0:
                    bias = reg_pred.bias.view(4, self.reg_max + 1)
                    idx = torch.arange(
                        self.reg_max + 1, dtype=bias.dtype, device=bias.device
                    )
                    target_init = min(2.0, self.reg_max * 0.25)
                    prior = -0.5 * (idx - target_init) ** 2
                    bias.add_(prior.unsqueeze(0))
                else:
                    if reg_pred.bias.numel() >= 4:
                        # reg_pred.bias[2:4].fill_(0.0)
                        reg_pred.bias[2:4].fill_(1.386)

        # ---------- 角度：sin=0, cos=1 → 初始角度 0° ----------
        if self.with_angle:
            angle_pred = self.angle_branch.pred
            nn.init.normal_(angle_pred.weight, std=0.005)
            if angle_pred.bias is not None:
                with torch.no_grad():
                    angle_pred.bias.zero_()
                    if self.angle_format == 'sin_cos' and angle_pred.bias.numel() >= 2:
                        angle_pred.bias[0] = 0.0   # sin
                        angle_pred.bias[1] = 1.0   # cos

    # ------------------------------------------------------------------
    def forward(self, x):
        out = {
            'cls': self.cls_branch(x),
            'reg': self.reg_branch(x),
        }
        if self.with_angle:
            out['angle'] = self.angle_branch(x)
        return out


# ============================================================
# 主检测网络
# ============================================================
class DetectFCOS(nn.Module):
    def __init__(self, num_classes, backbone_name='convnextv2_tiny', pretrained=True,
                 pretrained_path=None, pretrained_source='auto',
                 with_angle=False, angle_format='sin_cos', neck_out_channels=128,
                 head_num_convs=2, use_extra_conv=True,
                 img_size=640, neck_type='lightfpn', neck_num_layers=2,
                 reg_max=16,
                 # ★ 解耦头可选配置
                 head_type='decoupled',          # 'decoupled' 或 'multi'
                 head_norm_layer='group',
                 head_act_layer='silu',
                 head_cls_num_convs=None,
                 head_reg_num_convs=None,
                 head_angle_num_convs=None):
        super().__init__()
        self.with_angle = with_angle
        self.angle_format = angle_format
        self.img_size = img_size
        self.neck_type = neck_type
        self.neck_num_layers = neck_num_layers
        self.head_type = head_type

        # ★ OBB 和 detect 都启用 DFL
        self.reg_max = reg_max
        reg_channels = 4 * (reg_max + 1)

        self.backbone = Backbone(
            backbone_name, pretrained,
            pretrained_path=pretrained_path,
            pretrained_source=pretrained_source,
            img_size=img_size,
        )
        in_channels_list = self.backbone.out_channels

        if neck_type == 'bifpn':
            print(f"[DetectFCOS] 使用 BiFPN, num_layers={neck_num_layers}, out_channels={neck_out_channels}")
            self.neck = BiFPN(in_channels_list, out_channels=neck_out_channels, num_layers=neck_num_layers)
        elif neck_type == 'lightfpn':
            print(f"[DetectFCOS] 使用 LightFPN, out_channels={neck_out_channels}, use_extra_conv={use_extra_conv}")
            self.neck = LightFPN(in_channels_list, out_channels=neck_out_channels,
                                 use_extra_conv=use_extra_conv, use_pan=True)
        else:
            raise ValueError(f"不支持的 neck_type: {neck_type}")

        # ---------- 检测头 ----------
        if head_type == 'decoupled':
            print(f"[DetectFCOS] 使用解耦头 (DecoupledHead): "
                  f"num_classes={num_classes}, with_angle={with_angle}, "
                  f"angle_format={angle_format}, reg_max={self.reg_max}, "
                  f"use_dfl={self.reg_max > 0}, num_convs={head_num_convs}, "
                  f"norm={head_norm_layer}, act={head_act_layer}")
            self.head = DecoupledHead(
                in_channels=neck_out_channels,
                num_classes=num_classes,
                with_angle=with_angle,
                angle_format=angle_format,
                num_convs=head_num_convs,
                reg_max=self.reg_max,
                norm_layer=head_norm_layer,
                act_layer=head_act_layer,
                cls_num_convs=head_cls_num_convs,
                reg_num_convs=head_reg_num_convs,
                angle_num_convs=head_angle_num_convs,
            )
        else:
            # 保留旧路径（需要 MultiTaskHead，此处未定义，可自行添加）
            raise ValueError(
                "head_type='multi' 需要 MultiTaskHead 定义。"
                "当前 model.py 仅提供 decoupled 头。"
            )

        # 打印任务头信息
        tasks = {'cls': num_classes, 'reg': reg_channels}
        if with_angle:
            tasks['angle'] = 2 if angle_format == 'sin_cos' else 1
        print(f"[DetectFCOS] 任务头: {tasks}, with_angle={with_angle}, "
              f"angle_format={angle_format}, reg_max={self.reg_max}, "
              f"use_dfl={self.reg_max > 0}")

        self.strides = self.backbone.strides
        self.is_backbone_frozen = False
        self._init_neck_weights()

    def _init_neck_weights(self):
        """只初始化 Neck；检测头的初始化已在 DecoupledHead 内部完成"""
        def init_neck(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.neck.apply(init_neck)

    def set_freeze_backbone(self, freeze=True):
        self.is_backbone_frozen = freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        if freeze:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Backbone 已冻结")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  总参数: {total_params:,}")
            print(f"  冻结比例: {(1 - trainable_params / total_params) * 100:.1f}%")
        else:
            print("Backbone 已解冻")

    def get_freeze_status(self):
        return self.is_backbone_frozen

    def forward(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)

        cls_logits_list = []
        reg_preds_list = []
        angle_preds_list = []
        for feat in neck_out:
            preds = self.head(feat)
            cls_logits_list.append(preds['cls'])
            reg_preds_list.append(preds['reg'])
            if self.with_angle:
                angle_preds_list.append(preds['angle'])

        if self.with_angle:
            return cls_logits_list, reg_preds_list, angle_preds_list
        else:
            return cls_logits_list, reg_preds_list