import argparse

from swapfighter import SF
from swapfighter.tasks import list_tasks


def build_argparser():
    p = argparse.ArgumentParser('SWAP-Fighter Training')
    p.add_argument('--task', type=str, default='detect', choices=list_tasks())

    # 数据
    p.add_argument('--data-root', type=str, default=None)
    p.add_argument('--train-split', type=str, default=None)
    p.add_argument('--val-split', type=str, default=None)
    p.add_argument('--num-classes', type=int, default=None)
    p.add_argument('--class-name', '--class-names',
                   dest='class_names', nargs='+', default=None,
                   help='类别名称列表（空格分隔），会写入 meta 供推理时使用')

    # 分类数据集专用
    p.add_argument('--train-labels', type=str, default=None)
    p.add_argument('--val-labels', type=str, default=None)
    p.add_argument('--train-img-subdir', type=str, default=None)
    p.add_argument('--val-img-subdir', type=str, default=None)
    p.add_argument('--hf-dataset', type=str, default=None)
    p.add_argument('--hf-config', type=str, default=None)
    p.add_argument('--hf-train-split', type=str, default=None)
    p.add_argument('--hf-val-split', type=str, default=None)
    p.add_argument('--class-names-file', type=str, default=None)

    # 模型
    p.add_argument('--backbone', type=str, default=None)
    p.add_argument('--img-size', type=int, default=None)
    p.add_argument('--pretrained-source', type=str, default=None)
    p.add_argument('--pretrained-path', type=str, default=None)
    p.add_argument('--neck-type', type=str, default=None)
    p.add_argument('--neck-out-channels', type=int, default=None)
    p.add_argument('--head-num-convs', type=int, default=None)
    p.add_argument('--reg-max', type=int, default=None)
    p.add_argument('--dfl-weight', type=float, default=None)     # ★ 新增
    p.add_argument('--head-cls-num-convs', type=int, default=None)
    p.add_argument('--head-reg-num-convs', type=int, default=None)
    p.add_argument('--head-angle-num-convs', type=int, default=None)

    # OBB
    p.add_argument('--angle-format', type=str, default=None)
    p.add_argument('--angle-range', type=str, default=None)
    p.add_argument('--obb-reg-loss', type=str, default=None)

    # 训练
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None)
    p.add_argument('--scheduler', type=str, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--precision', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--num-workers', type=int, default=None)

    # 输出
    p.add_argument('--save-dir', type=str, default=None)
    p.add_argument('--no-ema', action='store_true')

    # 分类专用
    p.add_argument('--pool-type', type=str, default=None)
    p.add_argument('--head-type', type=str, default=None)
    p.add_argument('--loss', type=str, default=None)
    p.add_argument('--multi-scale', action='store_true', default=None)
    p.add_argument('--eval-threshold', type=float, default=None)

    # 参数覆盖
    p.add_argument('overrides', nargs='*', default=[],
                   help='额外覆盖 key=value')
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


def main():
    args = build_argparser().parse_args()
    extra = parse_extra_overrides(args.overrides)

    # 收集所有非 None 的 kwargs
    cfg = {k: v for k, v in vars(args).items()
           if k not in ('task', 'overrides', 'resume') and v is not None}
    if args.no_ema:
        cfg['use_ema'] = False
    cfg.update(extra)

    # 参数名映射（backbone -> backbone_name）
    if 'backbone' in cfg:
        cfg['backbone_name'] = cfg.pop('backbone')

    model = SF(weights=args.resume, task=args.task, cfg=cfg)
    model.train(**cfg)


if __name__ == '__main__':
    main()