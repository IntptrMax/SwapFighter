import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..heads.ml_decoder import MLDecoderHead

def _detect_backbone_kind(name):
    n = name.lower()
    if any(x in n for x in
           ['vit', 'dinov2', 'dinov3', 'eva', 'beit', 'deit', 'clip']):
        return 'vit'
    if 'swin' in n or 'hiera' in n:
        return 'swin'
    if 'convnext' in n:
        return 'convnext'
    if 'mobilenet' in n or 'efficientnet' in n or 'ghostnet' in n:
        return 'mobilenet'
    return 'cnn'

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):

        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x.flatten(1)

class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=8, num_queries=1, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, seq, seq, need_weights=False)
        out = self.norm(out)
        if self.num_queries == 1:
            return out.mean(dim=1)
        return out.reshape(B, -1)

class MultiLabelClassifier(nn.Module):
    def __init__(self,
                 backbone_name='convnextv2_atto.fcmae_ft_in1k',
                 num_classes=80,
                 pretrained=True,
                 pretrained_source='auto',
                 pretrained_path=None,
                 img_size=224,
                 dropout=0.0,
                 hidden_dim=0,
                 drop_path_rate=0.0,

                 pool_type='gem',
                 head_type='mlp',
                 head_norm=True,
                 multi_scale=False,
                 ms_pool='gem',
                 grad_checkpointing=False,

                 ml_decoder_layers=3,
                 ml_decoder_heads=8,
                 ml_decoder_group_size=1,

                 pool_from='auto',
                 frozen_backbone=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.pool_type = pool_type
        self.head_type = head_type
        self.multi_scale = multi_scale
        self.head_norm = head_norm

        self.ml_decoder_layers = ml_decoder_layers
        self.ml_decoder_heads = ml_decoder_heads
        self.ml_decoder_group_size = ml_decoder_group_size

        backbone_kind = _detect_backbone_kind(backbone_name)
        if pool_from == 'auto':
            if backbone_kind == 'vit':
                pool_from = 'cls_token'
            else:

                pool_from = 'spatial'

        if head_type == 'ml_decoder':
            pool_from = 'spatial'

        self.pool_from = pool_from
        self.backbone_kind = backbone_kind

        if pool_from not in ('cls_token', 'avg_pool', 'spatial'):
            raise ValueError(f"未知 pool_from: {pool_from!r}")

        if pool_from == 'cls_token':
            if backbone_kind != 'vit':
                print(f"[MultiLabelClassifier] ⚠️ pool_from=cls_token "
                      f"但 backbone 是 {backbone_kind}，自动回退到 avg_pool")
                pool_from = 'avg_pool'
                self.pool_from = pool_from
                global_pool = 'avg'
            else:
                global_pool = 'token'
        elif pool_from == 'avg_pool':
            global_pool = 'avg'
        else:
            global_pool = ''

        if pretrained_source == 'auto':
            use_timm_pretrained = bool(pretrained)
        elif pretrained_source == 'timm':
            use_timm_pretrained = True
        elif pretrained_source in ('none', 'simmim'):
            use_timm_pretrained = False
        else:
            raise ValueError(f"未知 pretrained_source: {pretrained_source}")

        create_kwargs = dict(pretrained=use_timm_pretrained)
        if drop_path_rate > 0:
            create_kwargs['drop_path_rate'] = drop_path_rate
        if grad_checkpointing:
            create_kwargs['grad_checkpointing'] = True

        self.feature_info = None
        if multi_scale and pool_from == 'spatial':

            try:
                self.backbone = timm.create_model(
                    backbone_name, features_only=True,
                    out_indices=(-3, -2, -1),
                    **create_kwargs)
            except TypeError:
                create_kwargs.pop('grad_checkpointing', None)
                self.backbone = timm.create_model(
                    backbone_name, features_only=True,
                    out_indices=(-3, -2, -1),
                    **create_kwargs)
            self.feature_info = self.backbone.feature_info.channels()
            self.num_features = sum(self.feature_info)
        else:

            self.backbone, actual_pool = self._create_backbone(
                backbone_name, global_pool, create_kwargs)
            if actual_pool != global_pool:
                print(f"[MultiLabelClassifier] global_pool "
                      f"从 {global_pool!r} 回退到 {actual_pool!r}")
                global_pool = actual_pool

                if global_pool == '':
                    self.pool_from = 'spatial'
                elif global_pool in ('avg', 'avgpool'):
                    self.pool_from = 'avg_pool'

        if pretrained_source == 'simmim':
            if pretrained_path is None or not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f'SimMIM 权重不存在: {pretrained_path}')
            sd = torch.load(pretrained_path, map_location='cpu')
            cleaned = {}
            for k, v in sd.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('encoder.'):
                    k = k[8:]
                cleaned[k] = v
            missing, unexpected = self.backbone.load_state_dict(
                cleaned, strict=False)
            print(f'  SimMIM: missing={len(missing)}, '
                  f'unexpected={len(unexpected)}')

        self.feature_dim_per_level = None
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.backbone(dummy)

            if multi_scale and isinstance(out, (list, tuple)):
                self.feature_dim_per_level = [t.shape[1] for t in out]
                self.num_features = sum(self.feature_dim_per_level)
            else:
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                if out.dim() == 4:
                    self.num_features = out.shape[1]
                    self.feature_dim_per_level = [self.num_features]
                elif out.dim() == 2:
                    self.num_features = out.shape[-1]
                    self.feature_dim_per_level = [self.num_features]
                else:
                    raise ValueError(
                        f"backbone 输出维度异常: {tuple(out.shape)}")

            if was_training:
                self.backbone.train()

        print(f"[MultiLabelClassifier] backbone_kind={backbone_kind}, "
              f"pool_from={self.pool_from}, "
              f"global_pool={global_pool!r}, "
              f"num_features={self.num_features}")

        if head_type == 'ml_decoder':

            if multi_scale:
                raise NotImplementedError(
                    "ML-Decoder 暂不支持 multi_scale，请用 multi_scale=False")

            self.pool = nn.Identity()
            self.head = MLDecoderHead(
                in_channels=self.num_features,
                num_classes=num_classes,
                num_heads=ml_decoder_heads,
                num_layers=ml_decoder_layers,
                group_size=ml_decoder_group_size,
                dropout=dropout,
                norm_first=True,
            )
            self._init_head()

        else:

            if multi_scale and pool_from == 'spatial':
                self.level_pools = nn.ModuleList([
                    self._make_pool(pool_type, c)
                    for c in self.feature_dim_per_level
                ])
                pooled_dim = self.num_features
            elif pool_from in ('cls_token', 'avg_pool'):

                self.pool = nn.Identity()
                pooled_dim = self.num_features
            else:

                self.pool = self._make_pool(pool_type, self.num_features)
                pooled_dim = self.num_features

            head_layers = []
            if head_norm:
                head_layers.append(nn.LayerNorm(pooled_dim))
            head_layers.append(nn.Dropout(dropout))

            if head_type == 'mlp':
                mid_dim = hidden_dim if hidden_dim > 0 else pooled_dim
                head_layers.append(nn.Linear(pooled_dim, mid_dim))
                head_layers.append(nn.GELU())
                head_layers.append(nn.Dropout(dropout))
                head_layers.append(nn.Linear(mid_dim, num_classes))
            else:
                head_layers.append(nn.Linear(pooled_dim, num_classes))

            self.head = nn.Sequential(*head_layers)
            self._init_head()

        self._frozen_backbone = False
        if frozen_backbone:
            self.set_freeze_backbone(True)

        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_head = sum(p.numel() for p in self.head.parameters())
        n_pool = sum(
            p.numel() for m in self.modules()
            if isinstance(m, (GeM, AttentionPooling))
            for p in m.parameters()
        )
        print(f'[MultiLabelClassifier] {backbone_name}')
        print(f'  pool={pool_type} | head={head_type} | '
              f'multi_scale={multi_scale} | head_norm={head_norm}')
        print(f'  num_features={self.num_features} | '
              f'backbone={n_backbone:,} | head={n_head:,} | '
              f'pool={n_pool:,} | classes={num_classes}')

    def _create_backbone(self, backbone_name, target_pool, create_kwargs):

        candidates = [target_pool]
        if target_pool not in ('avg', 'avgpool'):
            candidates.append('avg')
        if target_pool != '':
            candidates.append('')

        last_err = None
        for gp in candidates:
            try:
                kwargs = dict(create_kwargs)
                model = timm.create_model(
                    backbone_name, num_classes=0, global_pool=gp,
                    **kwargs)
                return model, gp
            except TypeError as e:

                create_kwargs.pop('grad_checkpointing', None)
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"无法创建 backbone {backbone_name}：{last_err}")

    def _make_pool(self, pool_type, dim):
        if pool_type == 'gap':
            return nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'gem':
            return GeM(p=3.0, learnable=True)
        elif pool_type == 'attention':
            return AttentionPooling(dim, num_heads=8, num_queries=1)
        raise ValueError(f"未知 pool_type: {pool_type}")

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _head_forward(self, out):

        if self.head_type == 'ml_decoder':
            if out.dim() != 4:
                raise ValueError(
                    f"ML-Decoder 需要 4D 特征图 (B, C, H, W)，"
                    f"实际得到 {tuple(out.shape)}")
            return self.head(out)

        if self.multi_scale and self.pool_from == 'spatial':
            feats = []
            for lvl, pool in enumerate(self.level_pools):
                t = out[lvl]
                if isinstance(pool, nn.AdaptiveAvgPool2d):
                    f = pool(t).flatten(1)
                else:
                    f = pool(t)
                feats.append(f)
            feats = torch.cat(feats, dim=1)
        elif self.pool_from in ('cls_token', 'avg_pool'):

            if out.dim() == 2:
                feats = out
            elif out.dim() == 4:

                feats = out.mean(dim=(2, 3))
            else:
                feats = out.flatten(1)
        else:

            if out.dim() == 4:
                if isinstance(self.pool, nn.AdaptiveAvgPool2d):
                    feats = self.pool(out).flatten(1)
                else:
                    feats = self.pool(out)
            elif out.dim() == 2:
                feats = out
            else:
                raise ValueError(
                    f"pool_from=spatial 但 backbone 输出维度异常: "
                    f"{tuple(out.shape)}")

        return self.head(feats)

    def forward(self, x):

        if self._frozen_backbone:
            with torch.no_grad():
                out = self.backbone(x)
            return self._head_forward(out)

        out = self.backbone(x)
        return self._head_forward(out)

    def set_freeze_backbone(self, freeze=True):
        self._frozen_backbone = freeze
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        n_train = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        status = 'frozen' if freeze else 'unfrozen'
        print(f'[MultiLabelClassifier] backbone {status} | '
              f'可训练={n_train:,} / 总={n_total:,}')

    def get_freeze_status(self):
        return self._frozen_backbone

def build_classifier(cfg) -> MultiLabelClassifier:
    return MultiLabelClassifier(
        backbone_name=cfg.backbone_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        pretrained_source=cfg.backbone_pretrained_source,
        pretrained_path=cfg.backbone_pretrained_path,
        img_size=cfg.img_size,
        dropout=cfg.dropout,
        hidden_dim=cfg.hidden_dim,
        drop_path_rate=cfg.drop_path_rate,
        pool_type=cfg.pool_type,
        head_type=cfg.head_type,
        head_norm=True,
        multi_scale=cfg.multi_scale,
        grad_checkpointing=cfg.grad_checkpointing,

        ml_decoder_layers=getattr(cfg, 'ml_decoder_layers', 3),
        ml_decoder_heads=getattr(cfg, 'ml_decoder_heads', 8),
        ml_decoder_group_size=getattr(cfg, 'ml_decoder_group_size', 1),

        pool_from=getattr(cfg, 'pool_from', 'auto'),
        frozen_backbone=getattr(cfg, 'frozen_backbone', False),
    )

build_multilabel_classifier = build_classifier

__all__ = [
    'MultiLabelClassifier',
    'build_classifier',
    'build_multilabel_classifier',
    'GeM',
    'AttentionPooling',
]