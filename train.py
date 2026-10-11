import os

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'OMP_DYNAMIC', 'OMP_SCHEDULE', 'GOMP_CPU_AFFINITY'):
    os.environ.pop(_key, None)

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import platform

if platform.system() == 'Linux':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                          'expandable_segments:True')

import platform
import argparse

from swapfighter import SF
from swapfighter.tasks import list_tasks

def build_argparser():
    p = argparse.ArgumentParser('SWAP-Fighter Training')
    p.add_argument('--task', type=str, default='detect',
                   choices=list_tasks(),
                   help='detect / obb / multilabel / linear_probe')

    p.add_argument('--data-root', type=str, default=None)
    p.add_argument('--train-split', type=str, default=None)
    p.add_argument('--val-split', type=str, default=None)
    p.add_argument('--num-classes', type=int, default=None)

    p.add_argument('--train-labels', type=str, default=None)
    p.add_argument('--val-labels', type=str, default=None)
    p.add_argument('--train-img-subdir', type=str, default=None)
    p.add_argument('--val-img-subdir', type=str, default=None)

    p.add_argument('--hf-dataset', type=str, default=None,
                   help='HuggingFace 数据集 ID，例如 user/dataset')
    p.add_argument('--hf-config', type=str, default=None)
    p.add_argument('--hf-train-split', type=str, default=None)
    p.add_argument('--hf-val-split', type=str, default=None)
    p.add_argument('--hf-cache-dir', type=str, default=None)

    p.add_argument('--class-names-file', type=str, default=None,
                   help='类名文件路径（每行一个类名）')
    p.add_argument('--hf-class-names-file', type=str, default=None,
                   help='[兼容旧参数] 同 --class-names-file')
    p.add_argument('--class-name', '--class-names',
                   dest='class_names', nargs='+', default=None,
                   help='类别名称列表（空格分隔），会写入 meta 供推理时自动读取')

    p.add_argument('--backbone', type=str, default=None,
                   help='timm 模型名，如 convnextv2_tiny / '
                        'vit_small_patch16_dinov3.lvd1689m')
    p.add_argument('--img-size', type=int, default=None)

    p.add_argument('--pretrained', dest='pretrained',
                   action='store_true', default=None,
                   help='加载 timm 预训练权重（默认开启）')
    p.add_argument('--no-pretrained', dest='pretrained',
                   action='store_false',
                   help='不加载预训练权重')

    p.add_argument('--pretrained-source',
                   dest='backbone_pretrained_source',
                   type=str, default=None,
                   choices=['auto', 'timm', 'simmim', 'none'],
                   help='预训练来源：auto / timm / simmim / none')

    p.add_argument('--pretrained-path',
                   dest='backbone_pretrained_path',
                   type=str, default=None,
                   help='本地预训练权重路径（SimMIM 场景使用）')

    p.add_argument('--neck-type', type=str, default=None,
                   choices=['lightfpn', 'bifpn'])
    p.add_argument('--neck-out-channels', type=int, nargs='+', default=None,
                   help='Neck 输出通道：单值（统一）或 3 值（分层，如 64 128 256）')
    p.add_argument('--neck-num-layers', type=int, default=None)

    p.add_argument('--use-extra-conv', dest='use_extra_conv',
                   action='store_true', default=None,
                   help='Neck 是否用额外 3×3 conv（默认开启）')
    p.add_argument('--no-extra-conv', dest='use_extra_conv',
                   action='store_false',
                   help='Neck 关闭额外 3×3 conv')

    p.add_argument('--head-channels', type=int, default=None,
                   help='Head 共享模式的通道数（默认 128）。'
                        '所有 level 的 neck 输出会通过 1×1 conv 投影到该通道，'
                        '然后进入共享 head。'
                        '配合 --neck-out-channels 64 128 256 使用，'
                        '既保留分层 neck，又让 head 参数量减少 ~3 倍')

    p.add_argument('--head-num-convs', type=int, default=None)
    p.add_argument('--head-norm-layer', type=str, default=None,
                   choices=['group', 'batch', 'none'])
    p.add_argument('--head-act-layer', type=str, default=None,
                   choices=['silu', 'relu', 'gelu'])
    p.add_argument('--head-cls-num-convs', type=int, default=None)
    p.add_argument('--head-reg-num-convs', type=int, default=None)
    p.add_argument('--head-angle-num-convs', type=int, default=None)
    p.add_argument('--head-kpt-num-convs', type=int, default=None,
                   help='关键点分支卷积层数（仅 keypoint 任务）')

    p.add_argument('--reg-max', type=int, default=None,
                   help='DFL 分布通道数（0=关闭，16=YOLOv8 默认）')
    p.add_argument('--dfl-weight', type=float, default=None,
                   help='DFL loss 权重（需 reg_max>0）')

    p.add_argument('--angle-format', type=str, default=None,
                   choices=['sin_cos', 'direct'])
    p.add_argument('--angle-range', type=str, default=None,
                   choices=['pi', 'pi2'])
    p.add_argument('--obb-reg-loss', type=str, default=None,
                   choices=['probiou', 'kld'])
    p.add_argument('--kld-weight', type=float, default=None)
    p.add_argument('--kld-tau', type=float, default=None)
    p.add_argument('--kld-fun', type=str, default=None,
                   choices=['log1p', 'sqrt', 'none'])
    

    p.add_argument('--proto-dim', type=int, default=None,
                   help='prototype 通道数（默认 32）')
    p.add_argument('--proto-stride', type=int, default=None,
                   choices=[2, 4, 8],
                   help='prototype 下采样倍数（默认 4）')
    p.add_argument('--mask-weight', type=float, default=None,
                   help='mask loss 权重（默认 1.0）')
    

    p.add_argument('--num-keypoints', type=int, default=None,
                   help='关键点数量（必须与标注一致）')
    p.add_argument('--keypoint-weight', type=float, default=None,
                   help='关键点 loss 权重（默认 1.0）')
    p.add_argument('--keypoint-visibility', type=str, default=None,
                   choices=['auto', 'yes', 'no'],
                   help='标签是否带 visibility（默认 auto 自动判断）')
    p.add_argument('--flip-idx', type=int, nargs='+', default=None,
                   help='水平翻转时的关键点索引映射，如 0 2 1 4 3 ...')

    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--accum-steps', type=int, default=None,
                   help='梯度累积步数（有效 batch = batch_size × accum_steps）')
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--backbone-lr-scale', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None,
                   choices=['sgd', 'adamw'])
    p.add_argument('--scheduler', type=str, default=None,
                   choices=['lambda', 'step', 'multistep',
                            'cosine', 'onecycle'])
    p.add_argument('--scheduler-gamma', type=float, default=None)
    p.add_argument('--scheduler-step-size', type=int, default=None)
    p.add_argument('--scheduler-milestones', type=str, default=None,
                   help='逗号分隔，如 30,60')
    p.add_argument('--warmup-epochs', type=int, default=None)

    p.add_argument('--grad-clip', dest='grad_clip_norm',
                   type=float, default=None,
                   help='梯度裁剪范数上限（默认 3.0）')

    p.add_argument('--patience', type=int, default=None,
                   help='连续 N 个 epoch 未刷新 best fitness 则早停，0=禁用')
    p.add_argument('--precision', type=str, default=None,
                   choices=['fp32', 'fp16', 'bf16'])
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--resume', type=str, default=None,
                   help='从 checkpoint 恢复训练')
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--freeze-epochs', type=int, default=None,
                   help='前 N 个 epoch 冻结 backbone')

    p.add_argument('--grad-checkpointing', dest='grad_checkpointing',
                   action='store_true', default=None,
                   help='启用梯度检查点（省显存，慢 20%）')

    p.add_argument('--cls-weight', type=float, default=None)
    p.add_argument('--reg-weight', type=float, default=None)
    p.add_argument('--angle-weight', type=float, default=None)

    p.add_argument('--tal-topk', type=int, default=None)
    p.add_argument('--tal-alpha', type=float, default=None)
    p.add_argument('--tal-beta', type=float, default=None)

    p.add_argument('--mosaic-prob', type=float, default=None)
    p.add_argument('--mosaic-disable-epoch', type=int, default=None)
    p.add_argument('--degrees', type=float, default=None,
                   help='OBB 旋转增强角度范围（±degrees）')
    p.add_argument('--translate', type=float, default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--shear', type=float, default=None)
    p.add_argument('--perspective', type=float, default=None)

    p.add_argument('--hsv-h', type=float, default=None)
    p.add_argument('--hsv-s', type=float, default=None)
    p.add_argument('--hsv-v', type=float, default=None)
    p.add_argument('--color-blur-prob', type=float, default=None)
    p.add_argument('--color-median-blur-prob', type=float, default=None)
    p.add_argument('--color-to-gray-prob', type=float, default=None)
    p.add_argument('--color-clahe-prob', type=float, default=None)

    p.add_argument('--use-color-aug', dest='use_color_aug',
                   action='store_true', default=None,
                   help='启用颜色增强（默认按 task 决定）')
    p.add_argument('--no-color-aug', dest='use_color_aug',
                   action='store_false',
                   help='关闭颜色增强')

    p.add_argument('--fliplr', type=float, default=None)
    p.add_argument('--flipud', type=float, default=None)

    p.add_argument('--use-flip', dest='use_flip',
                   action='store_true', default=None,
                   help='启用翻转增强（默认开启）')
    p.add_argument('--no-flip', dest='use_flip',
                   action='store_false',
                   help='关闭翻转增强')

    p.add_argument('--mixup-prob', type=float, default=None)
    p.add_argument('--mixup-alpha', type=float, default=None)
    p.add_argument('--copy-paste-prob', type=float, default=None)
    p.add_argument('--copy-paste-max-paste', type=int, default=None)

    p.add_argument('--use-ema', dest='use_ema',
                   action='store_true', default=None,
                   help='启用 EMA（默认开启）')
    p.add_argument('--no-ema', dest='use_ema',
                   action='store_false',
                   help='关闭 EMA')
    p.add_argument('--ema-decay', type=float, default=None,
                   help='None=自动按训练步数计算；指定值则使用固定 decay')

    p.add_argument('--eval-interval', type=int, default=None)
    p.add_argument('--eval-map-start-epoch', type=int, default=None)
    p.add_argument('--eval-conf-threshold', type=float, default=None)
    p.add_argument('--eval-diag', action='store_true', default=None)
    p.add_argument('--eval-start-epoch', type=int, default=None)
    p.add_argument('--eval-tta', action='store_true', default=None)
    p.add_argument('--eval-threshold', type=float, default=None,
                   help='多标签分类判定阈值（默认 0.5）')

    p.add_argument('--pool-type', type=str, default=None,
                   choices=['gap', 'gem', 'attention'])
    p.add_argument('--head-type', type=str, default=None,
                   choices=['linear', 'mlp', 'ml_decoder', 'decoupled'])
    p.add_argument('--loss', type=str, default=None,
                   choices=['bce', 'asl', 'ce'])
    p.add_argument('--multi-scale', action='store_true', default=None)
    p.add_argument('--hidden-dim', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--drop-path-rate', type=float, default=None)

    p.add_argument('--use-rrc', dest='use_rrc',
                   action='store_true', default=None,
                   help='启用 RandomResizedCrop（默认开启）')
    p.add_argument('--no-rrc', dest='use_rrc',
                   action='store_false',
                   help='关闭 RandomResizedCrop')
    p.add_argument('--rrc-min-scale', type=float, default=None)

    p.add_argument('--asl-gamma-neg', type=float, default=None)
    p.add_argument('--asl-gamma-pos', type=float, default=None)
    p.add_argument('--asl-clip', type=float, default=None)

    p.add_argument('--ml-decoder-layers', type=int, default=None,
                   help='ML-Decoder 层数（1=纯 cross-attn，3=多层融合）')
    p.add_argument('--ml-decoder-heads', type=int, default=None,
                   help='ML-Decoder attention 头数')
    p.add_argument('--ml-decoder-group-size', type=int, default=None,
                   help='ML-Decoder 每组共享 query 的类数（1=每类独立）')

    p.add_argument('--pool-from', type=str, default=None,
                   choices=['auto', 'cls_token', 'avg_pool', 'spatial'],
                   help='backbone 池化来源：'
                        'auto=自动判断（ViT→cls_token，其他→spatial）')
    p.add_argument('--frozen-backbone', action='store_true', default=None,
                   help='永久冻结 backbone（Linear Probe）')
    p.add_argument('--linear-probe', action='store_true', default=None,
                   help='等价于 --frozen-backbone，语义化标志')

    p.add_argument('--freeze-stage', type=str, default=None,
                   choices=['none', 'head_only', 'neck_head', 'last_n'],
                   help='冻结策略：'
                        'none=不冻结，'
                        'head_only=只训head，'
                        'neck_head=训neck+head（检测推荐），'
                        'last_n=解冻backbone最后N个block')
    p.add_argument('--freeze-last-n', type=int, default=None,
                   help='freeze_stage=last_n 时解冻的 block 数（默认 2）')

    p.add_argument('--conf-threshold', type=float, default=None,
                   help='推理置信度阈值（默认 0.7）')
    p.add_argument('--nms-threshold', type=float, default=None,
                   help='NMS IoU 阈值（默认 0.5）')
    p.add_argument('--obb-nms-threshold', type=float, default=None,
                   help='OBB 旋转 NMS 阈值（默认 0.5）')

    p.add_argument('--mean', type=float, nargs=3, default=None,
                   help='3 个值，如 0.485 0.456 0.406（默认 ImageNet）')
    p.add_argument('--std', type=float, nargs=3, default=None,
                   help='3 个值，如 0.229 0.224 0.225（默认 ImageNet）')

    p.add_argument('--save-dir', type=str, default=None,
                   help='输出目录。不填则自动使用 ./outputs/{task}')
    p.add_argument('--save-interval', type=int, default=None)
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--no-plot', action='store_true', default=None)
    p.add_argument('--scan-threshold', action='store_true', default=None,
                   help='训练后扫描最优阈值')
    p.add_argument('--verbose-metrics', dest='verbose_metrics',
                   action='store_true', default=None,
                   help='验证时逐类打印 mAP（默认只打印 all 行）')

    p.add_argument('overrides', nargs='*', default=[],
                   help='额外覆盖 key=value，例如 cls_weight=1.0（可选）')
    return p

def parse_extra_overrides(overrides):
    extra = {}
    for kv in overrides:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        try:
            if '.' in v or 'e' in v.lower():
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            if v.lower() in ('true', 'yes'):
                v = True
            elif v.lower() in ('false', 'no'):
                v = False
        extra[k.replace('-', '_')] = v
    return extra

def _normalize_neck_out_channels(cfg):
    if 'neck_out_channels' not in cfg:
        return
    v = cfg['neck_out_channels']
    if v is None:
        return
    if isinstance(v, list):
        if len(v) == 1:
            cfg['neck_out_channels'] = v[0]
        elif len(v) == 3:
            cfg['neck_out_channels'] = v
        else:
            raise ValueError(
                f"--neck-out-channels 只支持 1 或 3 个值，"
                f"收到 {len(v)} 个：{v}")

def _normalize_linear_probe(cfg):
    if cfg.get('linear_probe'):
        cfg['frozen_backbone'] = True
    cfg.pop('linear_probe', None)

def _normalize_scheduler_milestones(cfg):
    if 'scheduler_milestones' in cfg and \
       isinstance(cfg['scheduler_milestones'], str):
        s = cfg['scheduler_milestones']
        cfg['scheduler_milestones'] = [
            int(x.strip()) for x in s.split(',') if x.strip().isdigit()
        ]

def _normalize_legacy_keys(cfg):

    if 'grad_clip' in cfg and 'grad_clip_norm' not in cfg:
        cfg['grad_clip_norm'] = cfg.pop('grad_clip')

    if 'pretrained_source' in cfg and \
       'backbone_pretrained_source' not in cfg:
        cfg['backbone_pretrained_source'] = cfg.pop('pretrained_source')

    if 'pretrained_path' in cfg and \
       'backbone_pretrained_path' not in cfg:
        cfg['backbone_pretrained_path'] = cfg.pop('pretrained_path')

def main():
    args = build_argparser().parse_args()
    extra = parse_extra_overrides(args.overrides)

    cfg = {k: v for k, v in vars(args).items()
           if k not in ('task', 'overrides', 'resume') and v is not None}
    cfg.update(extra)

    if 'backbone' in cfg:
        cfg['backbone_name'] = cfg.pop('backbone')

    _normalize_legacy_keys(cfg)

    _normalize_neck_out_channels(cfg)
    _normalize_linear_probe(cfg)
    _normalize_scheduler_milestones(cfg)

    model = SF(weights=args.resume, task=args.task, cfg=cfg)
    model.train(**cfg)

if __name__ == '__main__':
    main()