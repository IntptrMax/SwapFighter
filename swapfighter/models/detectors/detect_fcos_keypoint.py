import torch
import torch.nn as nn

from .detect_fcos import DetectFCOS
from ..heads.keypoint_decoupled import KeypointDecoupledHead

class DetectFCOSKeypoint(DetectFCOS):

    def __init__(self, num_classes, num_keypoints=17, **kwargs):
        super().__init__(num_classes, **kwargs)

        self.num_keypoints = num_keypoints

        head_channels = self.head_channels
        head_num_convs = kwargs.get('head_num_convs', 2)
        self.head_shared = KeypointDecoupledHead(
            in_channels=head_channels,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            with_angle=self.with_angle,
            angle_format=self.angle_format,
            num_convs=head_num_convs,
            reg_max=self.reg_max,
            norm_layer=kwargs.get('head_norm_layer', 'group'),
            act_layer=kwargs.get('head_act_layer', 'silu'),
            cls_num_convs=kwargs.get('head_cls_num_convs', None),
            reg_num_convs=kwargs.get('head_reg_num_convs', None),
            angle_num_convs=kwargs.get('head_angle_num_convs', None),
            kpt_num_convs=kwargs.get('head_kpt_num_convs', None),
        )
        self.heads = nn.ModuleList([self.head_shared])

        print(f'[DetectFCOSKeypoint] num_keypoints={num_keypoints}')

    def _forward_neck_head(self, x):
        features = self.backbone(x)
        neck_out = self.neck(features)

        cls_list, reg_list, kpt_list = [], [], []
        for lvl, feat in enumerate(neck_out):
            feat_proj = self.head_proj[lvl](feat)
            preds = self.head_shared(feat_proj)
            cls_list.append(preds['cls'])
            reg_list.append(preds['reg'])
            kpt_list.append(preds['kpt'])

        return cls_list, reg_list, kpt_list

    def forward(self, x):
        return self._forward_neck_head(x)

def build_detect_keypoint_model(cfg) -> DetectFCOSKeypoint:
    return DetectFCOSKeypoint(
        num_classes=cfg.num_classes,
        num_keypoints=getattr(cfg, 'num_keypoints', 17),
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        pretrained_path=cfg.backbone_pretrained_path,
        pretrained_source=cfg.backbone_pretrained_source,
        with_angle=False,
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

__all__ = ['DetectFCOSKeypoint', 'build_detect_keypoint_model']