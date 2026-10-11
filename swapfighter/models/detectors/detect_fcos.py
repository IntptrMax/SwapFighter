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
                 head_channels=128,
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
                 head_angle_num_convs=None,
                 frozen_backbone=False,
                 freeze_stage='none',
                 freeze_last_n=2):
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

        if isinstance(neck_out_channels, int):
            neck_out_list = [neck_out_channels] * len(in_channels_list)
        else:
            neck_out_list = list(neck_out_channels)
            if len(neck_out_list) != len(in_channels_list):
                raise ValueError(
                    f"neck_out_channels 长度 ({len(neck_out_list)}) "
                    f"必须等于 backbone 输出层数 ({len(in_channels_list)})")

        self.neck_out_channels_list = neck_out_list

        if neck_type == 'bifpn':
            if len(set(neck_out_list)) > 1:
                raise NotImplementedError(
                    "BiFPN 暂不支持分层通道，请使用单值或改用 lightfpn")
            print(f"[DetectFCOS] BiFPN out={neck_out_list[0]}, "
                  f"layers={neck_num_layers}")
            self.neck = BiFPN(in_channels_list,
                              out_channels=neck_out_list[0],
                              num_layers=neck_num_layers)
        elif neck_type == 'lightfpn':
            print(f"[DetectFCOS] LightFPN out={neck_out_list}, "
                  f"extra_conv={use_extra_conv}")
            self.neck = LightFPN(in_channels_list,
                                 out_channels=neck_out_list,
                                 use_extra_conv=use_extra_conv,
                                 use_pan=True)
        else:
            raise ValueError(f"不支持的 neck_type: {neck_type}")

        if head_channels is None:
            head_channels = min(neck_out_list)
        self.head_channels = head_channels

        self.head_proj = nn.ModuleList()
        for c in neck_out_list:
            if c == head_channels:
                self.head_proj.append(nn.Identity())
            else:
                self.head_proj.append(nn.Conv2d(c, head_channels,
                                                kernel_size=1, bias=True))

        print(f"[DetectFCOS] Head 投影: "
              f"{neck_out_list} → {head_channels}")

        if head_type == 'decoupled':
            print(f"[DetectFCOS] DecoupledHead (★ 共享): "
                  f"cls={num_classes}, with_angle={with_angle}, "
                  f"reg_max={reg_max}, "
                  f"in_channels={head_channels} (所有 level 共享)")

            self.head_shared = DecoupledHead(
                in_channels=head_channels,
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
            raise ValueError(
                f"当前仅支持 head_type='decoupled'，"
                f"当前为 {head_type!r}")

        self.heads = nn.ModuleList([self.head_shared])

        self.strides = self.backbone.strides
        self._init_neck_weights()
        self._init_head_proj_weights()

        self._frozen_backbone = False
        self._freeze_stage = freeze_stage
        self._freeze_last_n = freeze_last_n

        if frozen_backbone and freeze_stage == 'none':
            freeze_stage = 'neck_head'
            self._freeze_stage = freeze_stage

        if freeze_stage in ('head_only', 'neck_head'):
            self.set_freeze_backbone(True)
        elif freeze_stage == 'last_n':
            self._freeze_backbone_except_last(n=freeze_last_n)
        elif freeze_stage != 'none':
            raise ValueError(
                f"未知 freeze_stage: {freeze_stage!r}，"
                f"必须为 none / head_only / neck_head / last_n")

    def _init_neck_weights(self):
        def init_neck(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.neck.apply(init_neck)

    def _init_head_proj_weights(self):
        for m in self.head_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_freeze_backbone(self, freeze=True):
        self._frozen_backbone = freeze
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        status = 'frozen' if freeze else 'unfrozen'
        print(f"[DetectFCOS] backbone {status} | "
              f"可训练={n_train:,} / 总={n_total:,} "
              f"({n_train / max(n_total, 1) * 100:.1f}%)")

    def _freeze_backbone_except_last(self, n=2):
        self._frozen_backbone = True

        for p in self.backbone.parameters():
            p.requires_grad = False

        inner = getattr(self.backbone, 'backbone', None) or self.backbone
        blocks = None
        for attr in ('stages', 'layers', 'blocks', 'features'):
            if hasattr(inner, attr):
                blocks = getattr(inner, attr)
                break

        if blocks is None:
            print(f"[DetectFCOS] ⚠️ 无法定位 backbone 的 block 列表，"
                  f"跳过部分解冻")
            return

        blk_list = None
        if hasattr(blocks, '__len__'):
            blk_list = list(blocks)
        else:
            blk_list = list(blocks.children())

        if len(blk_list) == 0:
            print(f"[DetectFCOS] ⚠️ backbone block 列表为空，跳过部分解冻")
            return

        for blk in blk_list[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[DetectFCOS] backbone 部分解冻（最后 {n} 个 block） | "
              f"可训练={n_train:,} / 总={n_total:,} "
              f"({n_train / max(n_total, 1) * 100:.1f}%)")

    def get_freeze_status(self):
        return self._frozen_backbone

    def _forward_neck_head(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)
        return self._forward_heads(neck_out)

    def _forward_heads(self, neck_out):
        cls_list, reg_list, angle_list = [], [], []
        for lvl, feat in enumerate(neck_out):
            feat_proj = self.head_proj[lvl](feat)
            preds = self.head_shared(feat_proj)
            cls_list.append(preds['cls'])
            reg_list.append(preds['reg'])
            if self.with_angle:
                angle_list.append(preds['angle'])

        if self.with_angle:
            return cls_list, reg_list, angle_list
        return cls_list, reg_list

    def forward(self, x):
        return self._forward_neck_head(x)

    @property
    def neck_out_channels(self):
        if len(set(self.neck_out_channels_list)) == 1:
            return self.neck_out_channels_list[0]
        return self.neck_out_channels_list

    @property
    def head(self):
        return self.head_shared

def build_detect_model(cfg) -> DetectFCOS:
    return DetectFCOS(
        num_classes=cfg.num_classes,
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        pretrained_path=cfg.backbone_pretrained_path,
        pretrained_source=cfg.backbone_pretrained_source,
        with_angle=cfg.with_angle,
        angle_format=cfg.angle_format,
        neck_out_channels=cfg.neck_out_channels,
        head_channels=getattr(cfg, 'head_channels', 128),
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
        frozen_backbone=getattr(cfg, 'frozen_backbone', False),
        freeze_stage=getattr(cfg, 'freeze_stage', 'none'),
        freeze_last_n=getattr(cfg, 'freeze_last_n', 2),
    )

__all__ = ['DetectFCOS', 'build_detect_model']