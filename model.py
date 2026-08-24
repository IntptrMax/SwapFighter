import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm
import numpy as np

class Backbone(nn.Module):
    def __init__(self, name='convnextv2_tiny', pretrained=True):
        super().__init__()
        self.name = name
        if 'vit' in name.lower():
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
        
        # 获取所有特征图的通道数和步长
        all_channels = self.feature_info.channels()
        all_strides = [info['reduction'] for info in self.feature_info]
        
        print(f"Backbone: {name}")
        print(f"所有特征图通道数: {all_channels}")
        print(f"所有特征图步长: {all_strides}")
        
        # 判断是否为 ViT（步长都相同）
        if len(set(all_strides)) == 1:
            # ViT 等模型：所有特征图尺寸相同
            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            self.is_vit = True
            print("检测到 ViT 风格模型，使用简化 FPN")
        # 对于 Swin Transformer，需要特殊处理
        elif 'swin' in name.lower():
            # Swin Transformer 返回 4 个特征图，选择步长为 8, 16, 32 的
            self.feature_indices = []
            self.out_channels = []
            self.strides = []
            for idx, stride in enumerate(all_strides):
                if stride in [8, 16, 32]:
                    self.feature_indices.append(idx)
                    self.out_channels.append(all_channels[idx])
                    self.strides.append(stride)
            
            # 如果没找到合适的步长，使用索引 1, 2, 3
            if len(self.feature_indices) < 3:
                self.feature_indices = [1, 2, 3]
                self.out_channels = all_channels[1:4]
                self.strides = all_strides[1:4]
            self.is_vit = False
        else:
            # 其他模型：取最后3个特征图
            self.out_channels = all_channels[-3:]
            self.strides = all_strides[-3:]
            self.feature_indices = list(range(len(all_channels)))[-3:]
            self.is_vit = False
        
        print(f"选择的特征图索引: {self.feature_indices}")
        print(f"选择的通道数: {self.out_channels}")
        print(f"选择的步长: {self.strides}")
        
        # 标记是否为 Swin（需要格式转换）
        self.need_permute = 'swin' in name.lower()
        print(f"需要格式转换 (NHWC->NCHW): {self.need_permute}")

    def forward(self, x):
        features = self.backbone(x)
        
        # 根据索引取特征图
        selected_features = [features[i] for i in self.feature_indices]
        
        # 只有 Swin Transformer 需要转换格式（NHWC -> NCHW）
        if self.need_permute:
            converted_features = []
            for feat in selected_features:
                # Swin 输出格式为 [B, H, W, C]，转换为 [B, C, H, W]
                if len(feat.shape) == 4:
                    feat = feat.permute(0, 3, 1, 2)
                converted_features.append(feat)
            selected_features = converted_features
        
        # 打印调试信息（仅在第一次迭代）
        if not hasattr(self, '_debug_printed'):
            print("\n特征图形状（调试信息）:")
            for idx, feat in enumerate(selected_features):
                print(f"  特征图 {idx}: 形状 {feat.shape}")
            self._debug_printed = True
        
        return selected_features


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, is_vit=False):
        super().__init__()
        self.is_vit = is_vit
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        if is_vit:
            # ViT 模式：所有特征图尺寸相同
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features):
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        
        if self.is_vit:
            # ViT 模式：所有特征图尺寸相同，不需要金字塔融合
            fpn_features = laterals
        else:
            # 标准 FPN 融合
            fpn_features = []
            for i in range(len(laterals) - 1, -1, -1):
                if i == len(laterals) - 1:
                    fpn_features.append(laterals[i])
                else:
                    up = self.upsample(fpn_features[-1])
                    # 尺寸匹配检查
                    if up.shape[2:] != laterals[i].shape[2:]:
                        up = nn.functional.interpolate(
                            up, size=laterals[i].shape[2:], mode='nearest'
                        )
                    fpn_features.append(laterals[i] + up)
            fpn_features = fpn_features[::-1]
        
        # 应用 FPN 卷积
        out = [self.fpn_convs[i](fpn_features[i]) for i in range(len(fpn_features))]
        return out


class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

    def forward(self, x):
        cls_feat = self.cls_tower(x)
        cls_logits = self.cls_logits(cls_feat)

        reg_feat = self.reg_tower(x)
        reg_pred = self.reg_pred(reg_feat)

        return cls_logits, reg_pred


class DetectFCOS(nn.Module):
    def __init__(self, num_classes, backbone_name='convnextv2_tiny', pretrained=True):
        super().__init__()
        self.backbone = Backbone(backbone_name, pretrained)
        in_channels_list = self.backbone.out_channels
        self.is_vit = self.backbone.is_vit
        self.fpn = FPN(in_channels_list, out_channels=256, is_vit=self.is_vit)
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)
        self.strides = self.backbone.strides
        
        # 是否冻结 Backbone（由外部控制）
        self.is_backbone_frozen = False

        self._init_weights()

    def _init_weights(self):
        import numpy as np
        for module in [self.fpn, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    # 判断是否为输出层
                    if hasattr(module, 'cls_logits') and m is module.cls_logits:
                        nn.init.normal_(m.weight, std=0.01)
                        if m.bias is not None:
                            prior_prob = 0.01
                            nn.init.constant_(m.bias, -np.log((1 - prior_prob) / prior_prob))
                    elif hasattr(module, 'reg_pred') and m is module.reg_pred:
                        nn.init.normal_(m.weight, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def set_freeze_backbone(self, freeze=True):
        """冻结或解冻 Backbone 参数"""
        self.is_backbone_frozen = freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # 统计参数数量
        if freeze:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Backbone 已冻结")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  总参数: {total_params:,}")
            print(f"  冻结比例: {(1 - trainable_params/total_params) * 100:.1f}%")
        else:
            print("Backbone 已解冻")

    def get_freeze_status(self):
        """获取当前冻结状态"""
        return self.is_backbone_frozen

    def forward(self, x):
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        cls_logits_list, reg_preds_list = [], []
        for feat in fpn_feats:
            cls_logits, reg_pred = self.head(feat)
            cls_logits_list.append(cls_logits)
            reg_preds_list.append(reg_pred)
        return cls_logits_list, reg_preds_list