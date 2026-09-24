import torch
import torch.nn as nn

from ..backbones import Backbone
from ..necks import LightFPN, BiFPN
from ..heads import DecoupledHead


class DetectFCOS(nn.Module):
    def __init__(self, num_classes,
                 backbone_name='convnextv2_tiny',
                 pretrained=True,
                 pretrained_path=None,
                 pretrained_source='auto',
                 with_angle=False,
                 angle_format='sin_cos',
                 neck_out_channels=128,
                 head_num_convs=2,
                 use_extra_conv=True,
                 img_size=640,
                 neck_type='lightfpn',
                 neck_num_layers=2,
                 reg_max=16,
                 head_type='decoupled',
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

        self.reg_max = reg_max
        reg_channels = 4 * (reg_max + 1) if reg_max > 0 else 4

        self.backbone = Backbone(
            backbone_name, pretrained,
            pretrained_path=pretrained_path,
            pretrained_source=pretrained_source,
            img_size=img_size,
        )
        in_channels_list = self.backbone.out_channels

        if neck_type == 'bifpn':
            print(f"[DetectFCOS] BiFPN out={neck_out_channels}, layers={neck_num_layers}")
            self.neck = BiFPN(in_channels_list, out_channels=neck_out_channels,
                              num_layers=neck_num_layers)
        elif neck_type == 'lightfpn':
            print(f"[DetectFCOS] LightFPN out={neck_out_channels}, extra_conv={use_extra_conv}")
            self.neck = LightFPN(in_channels_list, out_channels=neck_out_channels,
                                 use_extra_conv=use_extra_conv, use_pan=True)
        else:
            raise ValueError(f"不支持的 neck_type: {neck_type}")

        if head_type == 'decoupled':
            print(f"[DetectFCOS] DecoupledHead: cls={num_classes}, "
                  f"angle={with_angle}, reg_max={reg_max}")
            self.head = DecoupledHead(
                in_channels=neck_out_channels,
                num_classes=num_classes,
                with_angle=with_angle,
                angle_format=angle_format,
                num_convs=head_num_convs,
                reg_max=reg_max,
                norm_layer=head_norm_layer,
                act_layer=head_act_layer,
                cls_num_convs=head_cls_num_convs,
                reg_num_convs=head_reg_num_convs,
                angle_num_convs=head_angle_num_convs,
            )
        else:
            raise ValueError(f"当前仅支持 head_type='decoupled'")

        self.strides = self.backbone.strides
        self.is_backbone_frozen = False
        self._init_neck_weights()

    def _init_neck_weights(self):
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
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            print(f"Backbone 已冻结 | 可训练 {n_train:,} / 总 {n_total:,}")

    def get_freeze_status(self):
        return self.is_backbone_frozen

    def forward(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)

        cls_list, reg_list, angle_list = [], [], []
        for feat in neck_out:
            preds = self.head(feat)
            cls_list.append(preds['cls'])
            reg_list.append(preds['reg'])
            if self.with_angle:
                angle_list.append(preds['angle'])

        if self.with_angle:
            return cls_list, reg_list, angle_list
        return cls_list, reg_list


# ============================================================
# 工厂函数
# ============================================================
def build_detect_model(cfg) -> DetectFCOS:
    """从 Config 构建 DetectFCOS。"""
    return DetectFCOS(
        num_classes=cfg.num_classes,
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        pretrained_path=cfg.backbone_pretrained_path,
        pretrained_source=cfg.backbone_pretrained_source,
        with_angle=cfg.with_angle,
        angle_format=cfg.angle_format,
        neck_out_channels=cfg.neck_out_channels,
        head_num_convs=cfg.head_num_convs,
        use_extra_conv=cfg.use_extra_conv,
        img_size=cfg.img_size,
        neck_type=cfg.neck_type,
        neck_num_layers=cfg.neck_num_layers,
        reg_max=cfg.reg_max,
        head_type=cfg.head_type,
        head_norm_layer=cfg.head_norm_layer,
        head_act_layer=cfg.head_act_layer,
        head_cls_num_convs=cfg.head_cls_num_convs,
        head_reg_num_convs=cfg.head_reg_num_convs,
        head_angle_num_convs=cfg.head_angle_num_convs,
    )