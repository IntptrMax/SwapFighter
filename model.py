import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F


# ---------- Backbone ----------
class Backbone(nn.Module):
    def __init__(self, name='convnextv2_tiny', pretrained=True):
        super().__init__()
        self.name = name
        is_vit = any(x in name.lower() for x in ['vit', 'dinov2'])
        if is_vit:
            self.backbone = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                num_classes=0,
                dynamic_img_size=True
            )
        else:
            self.backbone = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                num_classes=0
            )
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


# ---------- BiFPN 核心模块（模型太重，暂时抛弃） ----------
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
            up = nn.functional.interpolate(
                td_outputs[i + 1], size=cur.shape[2:], mode='nearest'
            )
            weights = self.td_weights[i]
            norm_weights = torch.softmax(weights, dim=0)
            fused = (norm_weights[0] * cur + norm_weights[1] * up) / (norm_weights.sum() + self.epsilon)
            td_outputs[i] = self.td_convs[i](fused)

        bu_outputs = [None] * self.num_levels
        bu_outputs[0] = td_outputs[0]

        for i in range(1, self.num_levels):
            cur = td_outputs[i]
            down = nn.functional.interpolate(
                bu_outputs[i - 1], size=cur.shape[2:], mode='nearest'
            )
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

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list
        ])

        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels) for _ in range(num_layers)
        ])

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


# ---------- 轻量级 FPN ----------
class LightFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128, use_extra_conv=True):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list
        ])
        self.use_extra_conv = use_extra_conv
        if use_extra_conv:
            self.extra_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(len(in_channels_list))
            ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        outputs = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(outputs[0], size=laterals[i].shape[2:], mode='nearest')
            fused = laterals[i] + up
            if self.use_extra_conv:
                fused = self.extra_convs[i](fused)
            outputs.insert(0, fused)
        return outputs


# ---------- 解耦检测头 ----------
class TaskHead(nn.Module):
    """
    单个任务的检测头，包含一个卷积塔和输出层。
    支持任务类型标记，便于特殊初始化。
    """
    def __init__(self, in_channels, out_channels, task_type, num_convs=3, norm_layer=nn.GroupNorm):
        super().__init__()
        self.task_type = task_type          # 'cls', 'reg', 'angle', etc.
        self.out_channels = out_channels

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
            # 使用 kaiming_normal 初始化卷积
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            layers.append(conv)

            norm = norm_layer(32, in_channels) if norm_layer == nn.GroupNorm else norm_layer(in_channels)
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
            layers.append(norm)
            layers.append(nn.ReLU(inplace=True))

        self.tower = nn.Sequential(*layers)

        # 预测层，先做默认初始化，稍后在 MultiTaskHead 中覆盖特殊初始化
        self.pred = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.pred.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.pred.bias, 0)
        self.pred.is_pred_layer = True   # 标记，用于特殊初始化

    def forward(self, x):
        feat = self.tower(x)
        return self.pred(feat)


class MultiTaskHead(nn.Module):
    """
    多任务检测头，支持分类、回归、角度等任务。
    提供 init_weights() 方法对每个任务的预测层进行特殊初始化。
    """
    def __init__(self, in_channels, num_classes, tasks, angle_format='sin_cos', num_convs=3):
        super().__init__()
        self.tasks = tasks
        self.angle_format = angle_format
        self.heads = nn.ModuleDict()
        for task_name, out_channels in tasks.items():
            self.heads[task_name] = TaskHead(
                in_channels, out_channels, task_name, num_convs=num_convs
            )

    def init_weights(self, is_obb=False):
        """
        对每个任务头的预测层进行特殊初始化。

        修复说明（detect 训练异常问题）：
            reg 头的 bias 初始化必须区分模式：
              - OBB 模式：输出 [dx, dy, dw, dh]，dw/dh 初始化为 -1，
                对应初始框 w=h=exp(-1)*stride≈0.37*stride，利于小目标收敛。
              - detect / segment 模式：输出 [l, t, r, b]。
                * 全部为 0 会导致初始框宽高为 0，generalized_box_iou_loss
                  无法提供有效梯度，训练一开始就崩。
                * 因此这里填 16.0，使初始框具有约 32 像素的宽高，保证
                  回归损失从第一步开始就有有意义的梯度。

        角度头初始化：
            - sin_cos 模式：bias 初始化为 [0, 1]，对应角度 0，单位向量稳定。
            - direct 模式：bias 保持 0。

        Args:
            is_obb: 是否为 OBB 模式（由 DetectFCOS._init_weights 传入 self.with_angle）
        """
        for task_name, head in self.heads.items():
            if hasattr(head.pred, 'is_pred_layer'):
                m = head.pred
                if task_name == 'cls':
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        prior_prob = 0.01
                        nn.init.constant_(m.bias, -np.log((1 - prior_prob) / prior_prob))
                elif task_name == 'reg':
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        with torch.no_grad():
                            m.bias.zero_()
                            if is_obb and m.bias.numel() >= 4:
                                # OBB: [dx, dy, dw, dh] -> dw/dh = -1
                                m.bias[2:4].fill_(-1.0)
                            elif (not is_obb) and m.bias.numel() >= 4:
                                # detect: [l, t, r, b] -> 初始距离设为 16 像素
                                m.bias.fill_(16.0)
                elif task_name == 'angle':
                    nn.init.normal_(m.weight, std=0.005)
                    if m.bias is not None:
                        with torch.no_grad():
                            m.bias.zero_()
                            if self.angle_format == 'sin_cos' and m.bias.numel() >= 2:
                                m.bias[0] = 0.0   # sin
                                m.bias[1] = 1.0   # cos
                            # direct 模式保持 0
                else:
                    # 其他任务采用默认初始化（已由 TaskHead 完成）
                    pass

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}


# ---------- 主检测网络 ----------
class DetectFCOS(nn.Module):
    def __init__(self, num_classes, backbone_name='convnextv2_tiny', pretrained=True,
                 with_angle=False, angle_format='sin_cos', neck_out_channels=128,
                 head_num_convs=2, use_extra_conv=True):
        super().__init__()
        self.with_angle = with_angle
        self.angle_format = angle_format
        self.backbone = Backbone(backbone_name, pretrained)
        in_channels_list = self.backbone.out_channels

        # 使用 LightFPN
        self.neck = LightFPN(in_channels_list, out_channels=neck_out_channels,
                             use_extra_conv=use_extra_conv)

        tasks = {'cls': num_classes, 'reg': 4}
        if with_angle:
            tasks['angle'] = 2 if angle_format == 'sin_cos' else 1

        self.head = MultiTaskHead(neck_out_channels, num_classes, tasks,
                                  angle_format, num_convs=head_num_convs)
        self.strides = self.backbone.strides
        self.is_backbone_frozen = False
        self._init_weights()

    def _init_weights(self):
        """初始化 Neck 和 Head（Head 内部会特殊初始化预测层）"""
        # 1. 初始化 Neck 中的所有卷积和 BN
        def init_neck(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.neck.apply(init_neck)

        # 2. 初始化 Head
        #    关键：把 with_angle 作为 is_obb 传入，
        #    确保只有 OBB 模式才用 dw/dh = -1 的回归头初始化；
        #    detect 模式 reg 头 bias 用 16.0，避免初始框面积为零。
        self.head.init_weights(is_obb=self.with_angle)

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