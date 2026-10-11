import torch
import torch.nn as nn

from .detect_fcos import DetectFCOS
from ..heads.seg_decoupled import SegDecoupledHead, SegPrototypeBranch

class DetectFCOSSeg(DetectFCOS):

    def __init__(self, num_classes,
                 proto_dim=32,
                 proto_stride=4,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        num_mask_coef = proto_dim

        self.num_mask_coef = num_mask_coef
        self.proto_dim = proto_dim
        self.proto_stride = proto_stride

        head_channels = self.head_channels
        head_num_convs = kwargs.get('head_num_convs', 2)
        self.head_shared = SegDecoupledHead(
            in_channels=head_channels,
            num_classes=num_classes,
            num_mask_coef=num_mask_coef,
            with_angle=self.with_angle,
            angle_format=self.angle_format,
            num_convs=head_num_convs,
            reg_max=self.reg_max,
            norm_layer=kwargs.get('head_norm_layer', 'group'),
            act_layer=kwargs.get('head_act_layer', 'silu'),
            cls_num_convs=kwargs.get('head_cls_num_convs', None),
            reg_num_convs=kwargs.get('head_reg_num_convs', None),
            angle_num_convs=kwargs.get('head_angle_num_convs', None),
        )
        self.heads = nn.ModuleList([self.head_shared])

        self.proto_branch = SegPrototypeBranch(
            in_channels_list=self.neck_out_channels_list,
            proto_dim=proto_dim,
            fuse_channels=64,
            proto_stride=proto_stride,
            num_convs=head_num_convs,
            norm_layer=kwargs.get('head_norm_layer', 'group'),
            act_layer=kwargs.get('head_act_layer', 'silu'),
        )

        print(f'[DetectFCOSSeg] num_mask_coef={num_mask_coef}, '
              f'proto_dim={proto_dim}, proto_stride={proto_stride}')

    def _forward_neck_head(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)

        proto = self.proto_branch(neck_out)

        cls_list, reg_list, coef_list, angle_list = [], [], [], []
        for lvl, feat in enumerate(neck_out):
            feat_proj = self.head_proj[lvl](feat)
            preds = self.head_shared(feat_proj)
            cls_list.append(preds['cls'])
            reg_list.append(preds['reg'])
            coef_list.append(preds['coef'])
            if self.with_angle:
                angle_list.append(preds['angle'])

        if self.with_angle:
            return cls_list, reg_list, coef_list, angle_list, proto
        return cls_list, reg_list, coef_list, proto

    def forward(self, x):
        return self._forward_neck_head(x)

def build_detect_seg_model(cfg) -> DetectFCOSSeg:
    return DetectFCOSSeg(
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
        proto_dim=getattr(cfg, 'proto_dim', 32),
        proto_stride=getattr(cfg, 'proto_stride', 4),
        frozen_backbone=getattr(cfg, 'frozen_backbone', False),
        freeze_stage=getattr(cfg, 'freeze_stage', 'none'),
        freeze_last_n=getattr(cfg, 'freeze_last_n', 2),
    )

__all__ = ['DetectFCOSSeg', 'build_detect_seg_model']