import os
import sys
import io
import time
import json
import csv
import math
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from contextlib import contextmanager

from config import Config
from dataset import YOLODataset
from model import DetectFCOS

from utils import (
    compute_fcos_loss, evaluate_mAP, evaluate_map_obb_rotated,
    post_process, post_process_obb
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将无法生成训练曲线图。请安装: pip install matplotlib")


# ============================================================
# 日志打印辅助
# ============================================================
@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


LINE = "─" * 64
LINE_EQ = "═" * 64


def _title(text):
    print(f"\n  [{text}]")


def _kv(key, value, indent=4, total_width=40):
    prefix = " " * indent
    label = f"{key} "
    pad_len = max(2, total_width - len(label))
    dots = " " + "." * pad_len + " "
    print(f"{prefix}{label}{dots} {value}")


def _bool_en(b):
    return "yes" if b else "no"


def _get_gpu_mem_str():
    if torch.cuda.is_available():
        try:
            mem = torch.cuda.memory_reserved() / (1024 ** 3)
            return f"{mem:.2f}G"
        except Exception:
            return "0.00G"
    return "cpu"


def _count_instances(targets):
    return int(sum(t['labels'].shape[0] for t in targets))


# ============================================================
# YOLOv8 风格表头
# ============================================================
TABLE_TRAIN_HEADER = (
    f"{'Epoch':>10}  {'GPU_mem':>7}  {'loss':>8}  "
    f"{'lr_bb':>9}  {'lr_nh':>9}  {'Instances':>9}  {'Size':>4}"
)

TABLE_VAL_HEADER = (
    f"{'Class':>17}  {'Images':>6}  {'Instances':>9}  "
    f"{'Box(P':>9}  {'R':>9}  {'mAP50':>9}  {'mAP50-95)':>11}"
)


# ============================================================
# YOLOv8 fitness 判定
# ============================================================
def compute_fitness(ap50, ap95):
    return 0.1 * ap50 + 0.9 * ap95


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# model_meta
# ============================================================
def build_model_meta(cfg):
    return {
        'mode': cfg.mode,
        'backbone_name': cfg.backbone_name,
        'num_classes': cfg.num_classes,
        'img_size': cfg.img_size,
        'reg_max': getattr(cfg, 'reg_max', 0),
        'neck_out_channels': cfg.neck_out_channels,
        'head_num_convs': cfg.head_num_convs,
        'use_extra_conv': cfg.use_extra_conv,
        'neck_type': getattr(cfg, 'neck_type', 'lightfpn'),
        'neck_num_layers': getattr(cfg, 'neck_num_layers', 2),
        'head_type': getattr(cfg, 'head_type', 'decoupled'),
        'head_norm_layer': getattr(cfg, 'head_norm_layer', 'group'),
        'head_act_layer': getattr(cfg, 'head_act_layer', 'silu'),
        'head_cls_num_convs': getattr(cfg, 'head_cls_num_convs', None),
        'head_reg_num_convs': getattr(cfg, 'head_reg_num_convs', None),
        'head_angle_num_convs': getattr(cfg, 'head_angle_num_convs', None),
        'obb_reg_loss': getattr(cfg, 'obb_reg_loss', 'probiou'),
        'kld_weight': getattr(cfg, 'kld_weight', 1.0),
        'kld_tau': getattr(cfg, 'kld_tau', 1.0),
        'kld_fun': getattr(cfg, 'kld_fun', 'log1p'),
        'with_angle': cfg.with_angle,
        'angle_range': cfg.angle_range,
        'angle_format': cfg.angle_format,
        'angle_loss_type': getattr(cfg, 'angle_loss_type', 'l1'),
        'mean': list(cfg.mean),
        'std': list(cfg.std),
        'strides': list(cfg.strides) if cfg.strides else None,
        'conf_threshold': cfg.conf_threshold,
        'nms_threshold': cfg.nms_threshold,
        'obb_nms_threshold': getattr(cfg, 'obb_nms_threshold', None),
    }


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k].detach(), alpha=1.0 - decay)
            else:
                v.copy_(msd[k])


def get_ema_decay(base_decay, global_step, warmup_steps=200):
    if global_step >= warmup_steps:
        return base_decay
    return base_decay * (global_step + 1) / (warmup_steps + 1)


# ============================================================
# 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg,
                    scaler=None, grad_clip_norm=5.0, lr_scheduler=None,
                    amp_dtype=torch.float16, use_amp=True,
                    ema_model=None, ema_decay=0.996, global_step=0):
    model.train()
    total_loss = 0

    print()
    print(TABLE_TRAIN_HEADER)

    pbar = tqdm(
        dataloader,
        bar_format='{desc}  {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
        leave=True,
        ncols=150,
        dynamic_ncols=False,
    )

    total_steps = len(dataloader) * cfg.epochs
    ema_warmup_steps = max(1, int(total_steps * 0.15))

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        for t in targets:
            if 'boxes' in t:
                t['boxes'] = t['boxes'].to(device, non_blocking=True)
            if 'obboxes' in t:
                t['obboxes'] = t['obboxes'].to(device, non_blocking=True)
                t['cxcywh'] = t['cxcywh'].to(device, non_blocking=True)
                t['angles'] = t['angles'].to(device, non_blocking=True)
            t['labels'] = t['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            if cfg.mode == 'detect':
                cls_logits_list, reg_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list,
                    targets, strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    dfl_weight=getattr(cfg, 'dfl_weight', 0.0),
                    mode='detect',
                    reg_max=getattr(cfg, 'reg_max', 0),
                )
            else:
                cls_logits_list, reg_preds_list, angle_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list, targets,
                    strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    angle_weight=cfg.angle_weight,
                    dfl_weight=getattr(cfg, 'dfl_weight', 0.0),
                    mode='obb', angle_format=cfg.angle_format,
                    angle_preds_list=angle_preds_list,
                    tal_topk=cfg.tal_topk,
                    tal_alpha=cfg.tal_alpha,
                    tal_beta=cfg.tal_beta,
                    reg_max=getattr(cfg, 'reg_max', 0),
                    kld_weight=getattr(cfg, 'kld_weight', 1.0),
                    kld_tau=getattr(cfg, 'kld_tau', 1.0),
                    kld_fun=getattr(cfg, 'kld_fun', 'log1p'),
                    obb_reg_loss=getattr(cfg, 'obb_reg_loss', 'probiou'),
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if lr_scheduler is not None and cfg.scheduler == 'onecycle':
            lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            cur_decay = get_ema_decay(ema_decay, global_step, ema_warmup_steps)
            update_ema(ema_model, model, cur_decay)
            global_step += 1

        total_loss += loss.item()

        n_inst = _count_instances(targets)
        gpu_mem = _get_gpu_mem_str()
        lr_bb = optimizer.param_groups[0]['lr']
        lr_nh = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else lr_bb

        desc = (
            f"{epoch+1:>5d}/{cfg.epochs:<5d}  "
            f"{gpu_mem:>7}  "
            f"{loss.item():>8.3f}  "
            f"{lr_bb:>9.2e}  "
            f"{lr_nh:>9.2e}  "
            f"{n_inst:>9d}  "
            f"{cfg.img_size:>4d}"
        )
        pbar.set_description_str(desc, refresh=True)

    pbar.close()
    return total_loss / max(len(dataloader), 1), global_step


# ============================================================
# 验证一个 epoch
# ============================================================
@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch, cfg, compute_map=True,
                       amp_dtype=torch.float16, use_amp=True,
                       eval_use_fp32=False):
    model.eval()
    total_loss = 0

    val_use_amp = use_amp and not eval_use_fp32
    val_amp_dtype = amp_dtype if not eval_use_fp32 else torch.float32

    n_images_total = len(dataloader.dataset)

    pbar = tqdm(
        dataloader,
        desc=TABLE_VAL_HEADER,
        bar_format='{desc}: {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
        leave=True,
        ncols=150,
        dynamic_ncols=False,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        for t in targets:
            if 'boxes' in t:
                t['boxes'] = t['boxes'].to(device, non_blocking=True)
            if 'obboxes' in t:
                t['obboxes'] = t['obboxes'].to(device, non_blocking=True)
                t['cxcywh'] = t['cxcywh'].to(device, non_blocking=True)
                t['angles'] = t['angles'].to(device, non_blocking=True)
            t['labels'] = t['labels'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=val_amp_dtype, enabled=val_use_amp):
            if cfg.mode == 'detect':
                cls_logits_list, reg_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list,
                    targets, strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    dfl_weight=getattr(cfg, 'dfl_weight', 0.0),
                    mode='detect',
                    reg_max=getattr(cfg, 'reg_max', 0),
                )
            else:
                cls_logits_list, reg_preds_list, angle_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list, targets,
                    strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    angle_weight=cfg.angle_weight,
                    dfl_weight=getattr(cfg, 'dfl_weight', 0.0),
                    mode='obb', angle_format=cfg.angle_format,
                    angle_preds_list=angle_preds_list,
                    tal_topk=cfg.tal_topk,
                    tal_alpha=cfg.tal_alpha,
                    tal_beta=cfg.tal_beta,
                    reg_max=getattr(cfg, 'reg_max', 0),
                    kld_weight=getattr(cfg, 'kld_weight', 1.0),
                    kld_tau=getattr(cfg, 'kld_tau', 1.0),
                    kld_fun=getattr(cfg, 'kld_fun', 'log1p'),
                    obb_reg_loss=getattr(cfg, 'obb_reg_loss', 'probiou'),
                )

        total_loss += loss.item()

    pbar.close()
    avg_loss = total_loss / max(len(dataloader), 1)

    # ---------- mAP 计算 ----------
    precision_val = recall_val = 0.0
    mAP_50 = mAP_50_95 = mar_100 = 0.0

    if compute_map:
        eval_conf = getattr(cfg, 'eval_conf_threshold', 0.05)
        if cfg.mode == 'obb':
            if getattr(cfg, 'eval_diag', False):
                for ct in [0.001, 0.01, 0.05, 0.1, 0.2]:
                    try:
                        diag_res = evaluate_map_obb_rotated(
                            model, dataloader, device, cfg,
                            conf_threshold=ct,
                            nms_threshold=0.5,
                            max_det=1000,
                            pre_nms_max_det=1000,
                        )
                        diag_map50 = diag_res[2] if len(diag_res) == 5 else diag_res[0]
                        if isinstance(diag_map50, torch.Tensor):
                            diag_map50 = diag_map50.item()
                        print(f"       [diag] conf={ct:.3f} → mAP@0.5={diag_map50:.4f}")
                    except Exception as e:
                        print(f"       [diag] conf={ct:.3f} 失败: {e}")

            result = evaluate_map_obb_rotated(
                model, dataloader, device, cfg,
                conf_threshold=eval_conf,
                nms_threshold=0.5,
                max_det=1000,
                pre_nms_max_det=1000,
            )
        else:
            result = evaluate_mAP(
                model, dataloader, device, cfg,
                conf_threshold=eval_conf,
            )

        # ★ 兼容 3 值 / 5 值返回
        if len(result) == 5:
            precision_val, recall_val, mAP_50, mAP_50_95, mar_100 = result
        else:
            mAP_50, mAP_50_95, mar_100 = result
            precision_val = mAP_50
            recall_val = mar_100

        if isinstance(mAP_50, torch.Tensor): mAP_50 = mAP_50.item()
        if isinstance(mAP_50_95, torch.Tensor): mAP_50_95 = mAP_50_95.item()
        if isinstance(mar_100, torch.Tensor): mar_100 = mar_100.item()
        if isinstance(precision_val, torch.Tensor): precision_val = precision_val.item()
        if isinstance(recall_val, torch.Tensor): recall_val = recall_val.item()

    n_inst_total = 0
    try:
        for _, targets in dataloader:
            n_inst_total += _count_instances(targets)
    except Exception:
        n_inst_total = -1

    if compute_map:
        print(f"{'all':>17}  {n_images_total:>6d}  {n_inst_total:>9d}  "
              f"{precision_val:>9.4f}  {recall_val:>9.4f}  "
              f"{mAP_50:>9.4f}  {mAP_50_95:>11.4f}")
    else:
        print(f"{'all':>17}  {n_images_total:>6d}  {n_inst_total:>9d}  "
              f"{'--':>9}  {'--':>9}  {'--':>9}  {'--':>11}")

    return avg_loss, mAP_50, mAP_50_95, mar_100


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter 训练脚本 (支持 Detect 和 OBB)')

    parser.add_argument('--mode', type=str, default=None, choices=['detect', 'obb'])
    parser.add_argument('--angle-format', type=str, default=None, choices=['sin_cos', 'direct'])
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--train-split', type=str, default=None)
    parser.add_argument('--val-split', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--num-class', type=int, default=None)

    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true', default=None)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--pretrained-source', type=str, default=None,
                        choices=['auto', 'timm', 'simmim', 'none'])
    parser.add_argument('--backbone-pretrained-path', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--neck-out-channels', type=int, default=None)
    parser.add_argument('--head-num-convs', type=int, default=None)
    parser.add_argument('--use-extra-conv', action='store_true', default=None)
    parser.add_argument('--no-extra-conv', dest='use_extra_conv', action='store_false')

    parser.add_argument('--neck-type', type=str, default=None,
                        choices=['lightfpn', 'bifpn'])
    parser.add_argument('--neck-num-layers', type=int, default=None)

    parser.add_argument('--head-type', type=str, default=None, choices=['decoupled'])
    parser.add_argument('--head-norm-layer', type=str, default=None,
                        choices=['group', 'batch', 'none'])
    parser.add_argument('--head-act-layer', type=str, default=None,
                        choices=['silu', 'relu', 'gelu'])
    parser.add_argument('--head-cls-num-convs', type=int, default=None)
    parser.add_argument('--head-reg-num-convs', type=int, default=None)
    parser.add_argument('--head-angle-num-convs', type=int, default=None)

    parser.add_argument('--reg-max', type=int, default=None)
    parser.add_argument('--dfl-weight', type=float, default=None)

    parser.add_argument('--obb-reg-loss', type=str, default=None,
                        choices=['kld', 'probiou'])
    parser.add_argument('--kld-weight', type=float, default=None)
    parser.add_argument('--kld-tau', type=float, default=None)
    parser.add_argument('--kld-fun', type=str, default=None,
                        choices=['log1p', 'sqrt', 'none'])

    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--backbone-lr-scale', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--freeze-epochs', type=int, default=None)

    parser.add_argument('--cls-weight', type=float, default=None)
    parser.add_argument('--reg-weight', type=float, default=None)
    parser.add_argument('--angle-weight', type=float, default=None)

    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--eval-interval', type=int, default=None)
    parser.add_argument('--eval-map-start-epoch', type=int, default=None)
    parser.add_argument('--eval-conf-threshold', type=float, default=None)
    parser.add_argument('--eval-diag', action='store_true', default=None)
    parser.add_argument('--eval-fp32', action='store_true')
    parser.add_argument('--no-plot', action='store_true')

    parser.add_argument('--patience', type=int, default=None,
                        help='连续 N 个 epoch 未刷新 best fitness 则早停，0 = 禁用')

    parser.add_argument('--precision', type=str, default=None,
                        choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--scaler-init-scale', type=float, default=2. ** 12)

    parser.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'adamw'])
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['lambda', 'step', 'multistep', 'cosine', 'onecycle'])
    parser.add_argument('--scheduler-gamma', type=float, default=None)
    parser.add_argument('--scheduler-step-size', type=int, default=None)
    parser.add_argument('--scheduler-milestones', type=str, default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)

    parser.add_argument('--mosaic-prob', type=float, default=None)
    parser.add_argument('--mosaic-disable-epoch', type=int, default=None)

    parser.add_argument('--degrees', type=float, default=None)
    parser.add_argument('--degree', type=float, default=None)
    parser.add_argument('--translate', type=float, default=None)
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--shear', type=float, default=None)
    parser.add_argument('--perspective', type=float, default=None)

    parser.add_argument('--hsv-h', type=float, default=None)
    parser.add_argument('--hsv-s', type=float, default=None)
    parser.add_argument('--hsv-v', type=float, default=None)
    parser.add_argument('--color-blur-prob', type=float, default=None)
    parser.add_argument('--color-median-blur-prob', type=float, default=None)
    parser.add_argument('--color-to-gray-prob', type=float, default=None)
    parser.add_argument('--color-clahe-prob', type=float, default=None)

    parser.add_argument('--fliplr', type=float, default=None)
    parser.add_argument('--flipud', type=float, default=None)
    parser.add_argument('--no-flip', dest='use_flip', action='store_false', default=None)

    parser.add_argument('--mixup-prob', type=float, default=None)
    parser.add_argument('--mixup-alpha', type=float, default=None)
    parser.add_argument('--copy-paste-prob', type=float, default=None)
    parser.add_argument('--copy-paste-max-paste', type=int, default=None)

    parser.add_argument('--use-ema', action='store_true', default=None)
    parser.add_argument('--no-ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema-decay', type=float, default=None)

    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--tal-topk', type=int, default=None)
    parser.add_argument('--tal-alpha', type=float, default=None)
    parser.add_argument('--tal-beta', type=float, default=None)

    return parser.parse_args()


# ============================================================
# 参数 → Config
# ============================================================
def update_config_from_args(cfg, args):
    if args.mode is not None: cfg.mode = args.mode
    if args.angle_format is not None: cfg.angle_format = args.angle_format
    if args.data_root is not None: cfg.data_root = args.data_root
    if args.train_split is not None: cfg.train_split = args.train_split
    if args.val_split is not None: cfg.val_split = args.val_split

    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    elif args.num_class is not None:
        cfg.num_classes = args.num_class

    if args.backbone is not None: cfg.backbone_name = args.backbone
    if args.pretrained is not None: cfg.pretrained = args.pretrained
    if args.pretrained_source is not None: cfg.backbone_pretrained_source = args.pretrained_source
    if args.backbone_pretrained_path is not None: cfg.backbone_pretrained_path = args.backbone_pretrained_path
    if args.img_size is not None: cfg.img_size = args.img_size
    if args.neck_out_channels is not None: cfg.neck_out_channels = args.neck_out_channels
    if args.head_num_convs is not None: cfg.head_num_convs = args.head_num_convs
    if args.use_extra_conv is not None: cfg.use_extra_conv = args.use_extra_conv
    if args.neck_type is not None: cfg.neck_type = args.neck_type
    if args.neck_num_layers is not None: cfg.neck_num_layers = args.neck_num_layers

    if args.head_type is not None: cfg.head_type = args.head_type
    if args.head_norm_layer is not None: cfg.head_norm_layer = args.head_norm_layer
    if args.head_act_layer is not None: cfg.head_act_layer = args.head_act_layer
    if args.head_cls_num_convs is not None: cfg.head_cls_num_convs = args.head_cls_num_convs
    if args.head_reg_num_convs is not None: cfg.head_reg_num_convs = args.head_reg_num_convs
    if args.head_angle_num_convs is not None: cfg.head_angle_num_convs = args.head_angle_num_convs

    if args.reg_max is not None: cfg.reg_max = args.reg_max
    if args.dfl_weight is not None: cfg.dfl_weight = args.dfl_weight

    if args.obb_reg_loss is not None: cfg.obb_reg_loss = args.obb_reg_loss
    if args.kld_weight is not None: cfg.kld_weight = args.kld_weight
    if args.kld_tau is not None: cfg.kld_tau = args.kld_tau
    if args.kld_fun is not None: cfg.kld_fun = None if args.kld_fun == 'none' else args.kld_fun

    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None: cfg.lr = args.lr
    if args.backbone_lr_scale is not None: cfg.backbone_lr_scale = args.backbone_lr_scale
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.grad_clip is not None: cfg.grad_clip_norm = args.grad_clip
    if args.freeze_epochs is not None: cfg.freeze_epochs = args.freeze_epochs

    if args.cls_weight is not None: cfg.cls_weight = args.cls_weight
    if args.reg_weight is not None: cfg.reg_weight = args.reg_weight
    if args.angle_weight is not None: cfg.angle_weight = args.angle_weight

    if args.save_dir is not None: cfg.save_dir = args.save_dir
    if args.save_interval is not None: cfg.save_interval = args.save_interval
    if args.log_interval is not None: cfg.log_interval = args.log_interval
    if args.eval_interval is not None: cfg.eval_interval = args.eval_interval
    if args.eval_map_start_epoch is not None: cfg.eval_map_start_epoch = args.eval_map_start_epoch
    if args.eval_conf_threshold is not None: cfg.eval_conf_threshold = args.eval_conf_threshold
    if args.eval_diag is not None: cfg.eval_diag = args.eval_diag
    if args.patience is not None: cfg.patience = args.patience

    if args.precision is not None: cfg.precision = args.precision

    if args.optimizer is not None: cfg.optimizer = args.optimizer
    if args.scheduler is not None: cfg.scheduler = args.scheduler
    if args.scheduler_gamma is not None: cfg.scheduler_gamma = args.scheduler_gamma
    if args.scheduler_step_size is not None: cfg.scheduler_step_size = args.scheduler_step_size
    if args.scheduler_milestones is not None:
        cfg.scheduler_milestones = [int(x.strip()) for x in args.scheduler_milestones.split(',') if x.strip().isdigit()]
    if args.warmup_epochs is not None: cfg.warmup_epochs = args.warmup_epochs

    if args.mosaic_prob is not None: cfg.mosaic_prob = args.mosaic_prob
    if args.mosaic_disable_epoch is not None: cfg.mosaic_disable_epoch = args.mosaic_disable_epoch

    if args.degrees is not None:
        cfg.degrees = args.degrees
    elif args.degree is not None:
        cfg.degrees = args.degree

    if args.translate is not None: cfg.translate = args.translate
    if args.scale is not None: cfg.scale = args.scale
    if args.shear is not None: cfg.shear = args.shear
    if args.perspective is not None: cfg.perspective = args.perspective

    if args.hsv_h is not None: cfg.hsv_h = args.hsv_h
    if args.hsv_s is not None: cfg.hsv_s = args.hsv_s
    if args.hsv_v is not None: cfg.hsv_v = args.hsv_v
    if args.color_blur_prob is not None: cfg.color_blur_prob = args.color_blur_prob
    if args.color_median_blur_prob is not None: cfg.color_median_blur_prob = args.color_median_blur_prob
    if args.color_to_gray_prob is not None: cfg.color_to_gray_prob = args.color_to_gray_prob
    if args.color_clahe_prob is not None: cfg.color_clahe_prob = args.color_clahe_prob

    if args.fliplr is not None: cfg.fliplr = args.fliplr
    if args.flipud is not None: cfg.flipud = args.flipud
    if args.use_flip is not None: cfg.use_flip = args.use_flip

    if args.mixup_prob is not None: cfg.mixup_prob = args.mixup_prob
    if args.mixup_alpha is not None: cfg.mixup_alpha = args.mixup_alpha
    if args.copy_paste_prob is not None: cfg.copy_paste_prob = args.copy_paste_prob
    if args.copy_paste_max_paste is not None: cfg.copy_paste_max_paste = args.copy_paste_max_paste

    if args.use_ema is not None: cfg.use_ema = args.use_ema
    if args.ema_decay is not None: cfg.ema_decay = args.ema_decay

    if args.device is not None: cfg.device = args.device
    if args.seed is not None: cfg.seed = args.seed

    if args.tal_topk is not None: cfg.tal_topk = args.tal_topk
    if args.tal_alpha is not None: cfg.tal_alpha = args.tal_alpha
    if args.tal_beta is not None: cfg.tal_beta = args.tal_beta

    cfg.validate_and_setup()
    return cfg


def resolve_precision(cfg, device, scaler_init_scale=2. ** 12):
    precision = getattr(cfg, 'precision', None)
    if precision is None:
        precision = 'fp16' if getattr(cfg, 'use_amp', True) else 'fp32'

    if device.type != 'cuda':
        if precision != 'fp32':
            print(f"警告: 设备 {device.type} 不支持混合精度，自动回退到 fp32")
        return torch.float32, False, None, 'fp32'

    if precision == 'fp32':
        return torch.float32, False, None, 'fp32'

    if precision == 'fp16':
        if not torch.cuda.is_available():
            return torch.float32, False, None, 'fp32'
        if scaler_init_scale and scaler_init_scale > 0:
            scaler = torch.amp.GradScaler(
                'cuda', init_scale=float(scaler_init_scale),
                growth_factor=2.0, backoff_factor=0.5, growth_interval=1000)
        else:
            scaler = torch.amp.GradScaler('cuda')
        return torch.float16, True, scaler, 'fp16'

    if precision == 'bf16':
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except AttributeError:
            try:
                bf16_ok = torch.cuda.get_device_properties(0).major >= 8
            except Exception:
                bf16_ok = False
        if not bf16_ok:
            print("警告: 当前 GPU 不支持 BF16，自动回退到 FP16")
            if scaler_init_scale and scaler_init_scale > 0:
                scaler = torch.amp.GradScaler(
                    'cuda', init_scale=float(scaler_init_scale),
                    growth_factor=2.0, backoff_factor=0.5, growth_interval=1000)
            else:
                scaler = torch.amp.GradScaler('cuda')
            return torch.float16, True, scaler, 'fp16'
        return torch.bfloat16, True, None, 'bf16'

    raise ValueError(f"未知精度: {precision}")


def build_optimizer(model, cfg):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            other_params.append(param)

    backbone_lr = cfg.lr * cfg.backbone_lr_scale
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': other_params,    'lr': cfg.lr,        'name': 'neck_head'},
    ]

    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, lr=cfg.lr, momentum=0.937,
                              weight_decay=cfg.weight_decay, nesterov=False)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer}")

    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_epochs = getattr(cfg, 'warmup_epochs', 0) or 0
    total_epochs = cfg.epochs

    if cfg.scheduler == 'onecycle':
        total_steps = max(steps_per_epoch * cfg.epochs, 1)
        max_lr = [pg['lr'] for pg in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=max(0.05, warmup_epochs / max(total_epochs, 1)) if warmup_epochs > 0 else 0.1,
            anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
        return scheduler, 'step'

    def make_lambda(base_lambda):
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            adjusted_epoch = epoch - warmup_epochs
            adjusted_total = max(total_epochs - warmup_epochs, 1)
            return base_lambda(adjusted_epoch, adjusted_total)
        return lr_lambda

    if cfg.scheduler == 'cosine':
        def base_lambda(epoch, total):
            return 0.5 * (1.0 + math.cos(math.pi * epoch / total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lambda(base_lambda))
        return scheduler, 'epoch'

    if cfg.scheduler == 'lambda':
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            return gamma ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lambda(base_lambda))
        return scheduler, 'epoch'

    if cfg.scheduler == 'step':
        step_size = max(cfg.scheduler_step_size, 1)
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            return gamma ** (epoch // step_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lambda(base_lambda))
        return scheduler, 'epoch'

    if cfg.scheduler == 'multistep':
        milestones = sorted(cfg.scheduler_milestones or [])
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            count = sum(1 for m in milestones if epoch >= m)
            return gamma ** count
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lambda(base_lambda))
        return scheduler, 'epoch'

    raise ValueError(f"不支持的调度器: {cfg.scheduler}")


def check_and_update_freeze(model, epoch, freeze_epochs):
    if freeze_epochs > 0 and epoch >= freeze_epochs and model.get_freeze_status():
        print()
        print(LINE)
        print(f"  达到第 {freeze_epochs} 轮，解冻 Backbone")
        print(LINE)
        model.set_freeze_backbone(False)


def save_config_json(cfg, save_path):
    config_dict = cfg.to_dict()
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def init_csv_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'AP50', 'AP95', 'AR100',
                             'time_sec', 'lr_bb', 'lr_nh', 'fitness'])


def append_csv_log(log_path, epoch, train_loss, val_loss, ap50, ap95, ar100,
                   time_sec, lr_bb=None, lr_nh=None, fitness=None):
    def to_float(val):
        if val is None:
            return ''
        if isinstance(val, torch.Tensor):
            return val.item()
        try:
            return float(val)
        except (TypeError, ValueError):
            return val

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            int(epoch), to_float(train_loss), to_float(val_loss),
            to_float(ap50), to_float(ap95), to_float(ar100), to_float(time_sec),
            to_float(lr_bb) if lr_bb is not None else '',
            to_float(lr_nh) if lr_nh is not None else '',
            to_float(fitness) if fitness is not None else '',
        ])


# ============================================================
# 平滑函数（保证输出长度与输入相同）
# ============================================================
def smooth(y, f=0.05):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        return y.copy()
    nf = round(n * f * 2) // 2 + 1
    if nf % 2 == 0:
        nf += 1
    if nf < 3 or nf > n:
        return y.copy()
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    smoothed = np.convolve(yp, np.ones(nf) / nf, mode='valid')
    if len(smoothed) > n:
        smoothed = smoothed[:n]
    elif len(smoothed) < n:
        smoothed = np.concatenate([smoothed, [smoothed[-1]] * (n - len(smoothed))])
    return smoothed


# ============================================================
# 训练曲线
# ============================================================
def plot_training_curves(log_path, save_dir):
    if not HAS_MATPLOTLIB:
        print("跳过绘图：matplotlib 未安装")
        return
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        return

    epochs, train_losses, val_losses = [], [], []
    ap50s, ap95s, ar100s = [], [], []
    lr_bbs, lr_nhs = [], []

    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        for row in reader:
            if len(row) < 8:
                continue
            try:
                epochs.append(int(row[0]))
                train_losses.append(float(row[1]) if row[1] else np.nan)
                val_losses.append(float(row[2]) if row[2] else np.nan)
                ap50s.append(float(row[3]) if row[3] else np.nan)
                ap95s.append(float(row[4]) if row[4] else np.nan)
                ar100s.append(float(row[5]) if row[5] else np.nan)
                lr_bbs.append(float(row[7]) if row[7] else np.nan)
                lr_nhs.append(float(row[8]) if len(row) > 8 and row[8] else np.nan)
            except (ValueError, IndexError):
                continue

    if not epochs:
        print("日志无有效数据，无法绘图")
        return

    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    ap50s = np.array(ap50s)
    ap95s = np.array(ap95s)
    ar100s = np.array(ar100s)
    lr_bbs = np.array(lr_bbs)
    lr_nhs = np.array(lr_nhs)

    panels = [
        ('train/loss',       'train loss',   train_losses, '#1f77b4', '#ff7f0e'),
        ('val/loss',         'val loss',     val_losses,   '#1f77b4', '#ff7f0e'),
        ('metrics/lr_bb',    'backbone lr',  lr_bbs,       '#1f77b4', '#ff7f0e'),
        ('metrics/mAP50',    'mAP@0.5',      ap50s,        '#1f77b4', '#ff7f0e'),
        ('metrics/mAP50-95', 'mAP@0.5:0.95', ap95s,        '#1f77b4', '#ff7f0e'),
        ('metrics/AR100',    'AR@100',       ar100s,       '#1f77b4', '#ff7f0e'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (title, ylabel, y, color_raw, color_smooth) in enumerate(panels):
        ax = axes[i]
        mask = ~np.isnan(y)
        x_valid = epochs[mask]
        y_valid = y[mask]

        if len(y_valid) > 0:
            ax.plot(x_valid, y_valid, 'o-', color=color_raw,
                    markersize=5, linewidth=1.2, label='results')
            if len(y_valid) > 1:
                y_smooth = smooth(y_valid, f=0.05)
                x_smooth = x_valid[:len(y_smooth)]
                ax.plot(x_smooth, y_smooth, '--', color=color_smooth,
                        linewidth=2.2, label='smooth')

        ax.set_title(title, fontsize=13, fontweight='normal')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        if i == 0:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  训练曲线已保存至: {plot_path}")


# ============================================================
# 主训练入口
# ============================================================
def main():
    args = parse_args()

    cfg = Config()
    cfg = update_config_from_args(cfg, args)

    # ---------- 默认值补齐 ----------
    defaults = {
        'mosaic_prob': 0.0, 'mosaic_disable_epoch': None,
        'grad_clip_norm': 5.0, 'cls_weight': 1.0, 'reg_weight': 5.0,
        'angle_weight': 1.5, 'freeze_epochs': 0,
        'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
        'optimizer': 'adamw', 'scheduler': 'cosine',
        'scheduler_gamma': 0.95, 'scheduler_step_size': 1,
        'scheduler_milestones': [30, 60], 'warmup_epochs': 5,
        'neck_out_channels': 128, 'head_num_convs': 2, 'use_extra_conv': True,
        'eval_map_start_epoch': 0,
        'backbone_pretrained_source': 'auto', 'backbone_pretrained_path': None,
        'angle_loss_type': 'l1', 'neck_type': 'lightfpn', 'neck_num_layers': 2,
        'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0,
        'backbone_lr_scale': 0.1, 'eval_conf_threshold': 0.05, 'eval_diag': False,
        'use_ema': True, 'ema_decay': 0.99,
        'tal_topk': 13, 'tal_alpha': 1.0, 'tal_beta': 6.0,
        'reg_max': 0, 'dfl_weight': 0.0,
        'obb_reg_loss': 'probiou', 'kld_weight': 1.0, 'kld_tau': 1.0, 'kld_fun': 'log1p',
        'head_type': 'decoupled', 'head_norm_layer': 'group', 'head_act_layer': 'silu',
        'head_cls_num_convs': None, 'head_reg_num_convs': None, 'head_angle_num_convs': None,
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'color_blur_prob': 0.01, 'color_median_blur_prob': 0.01,
        'color_to_gray_prob': 0.01, 'color_clahe_prob': 0.01,
        'use_color_aug': True,
        'fliplr': 0.5, 'flipud': 0.0, 'use_flip': True,
        'mixup_prob': 0.0, 'mixup_alpha': 32.0,
        'copy_paste_prob': 0.0, 'copy_paste_max_paste': 3,
        'log_interval': 10, 'save_interval': 10,
        'patience': 20,
    }
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    if not hasattr(cfg, 'precision') or cfg.precision is None:
        cfg.precision = 'fp16' if getattr(cfg, 'use_amp', True) else 'fp32'

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    amp_dtype, use_amp, scaler, actual_precision = resolve_precision(
        cfg, device, scaler_init_scale=args.scaler_init_scale
    )

    # ============================================================
    # 折叠所有子模块打印
    # ============================================================
    with suppress_stdout():
        train_dataset = YOLODataset(
            root=cfg.data_root, img_size=cfg.img_size, split=cfg.train_split,
            use_flip=cfg.use_flip, use_color_aug=getattr(cfg, 'use_color_aug', True),
            mean=cfg.mean, std=cfg.std, mode=cfg.mode,
            angle_range=cfg.angle_range if cfg.mode == 'obb' else None,
            mosaic_prob=cfg.mosaic_prob, mosaic_disable_epoch=cfg.mosaic_disable_epoch,
            degrees=getattr(cfg, 'degrees', 0.0),
            translate=getattr(cfg, 'translate', 0.1),
            scale=getattr(cfg, 'scale', 0.5),
            shear=getattr(cfg, 'shear', 0.0),
            perspective=getattr(cfg, 'perspective', 0.0),
            hsv_h=getattr(cfg, 'hsv_h', 0.015),
            hsv_s=getattr(cfg, 'hsv_s', 0.7),
            hsv_v=getattr(cfg, 'hsv_v', 0.4),
            color_blur_prob=getattr(cfg, 'color_blur_prob', 0.01),
            color_median_blur_prob=getattr(cfg, 'color_median_blur_prob', 0.01),
            color_to_gray_prob=getattr(cfg, 'color_to_gray_prob', 0.01),
            color_clahe_prob=getattr(cfg, 'color_clahe_prob', 0.01),
            fliplr=getattr(cfg, 'fliplr', 0.5),
            flipud=getattr(cfg, 'flipud', 0.0),
            mixup_prob=getattr(cfg, 'mixup_prob', 0.0),
            mixup_alpha=getattr(cfg, 'mixup_alpha', 32.0),
            copy_paste_prob=getattr(cfg, 'copy_paste_prob', 0.0),
            copy_paste_max_paste=getattr(cfg, 'copy_paste_max_paste', 3),
            debug_vis=False
        )
        val_dataset = YOLODataset(
            root=cfg.data_root, img_size=cfg.img_size, split=cfg.val_split,
            use_flip=False, use_color_aug=False,
            mean=cfg.mean, std=cfg.std, mode=cfg.mode,
            angle_range=cfg.angle_range if cfg.mode == 'obb' else None,
            debug_vis=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None
        )

        model = DetectFCOS(
            num_classes=cfg.num_classes,
            backbone_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            pretrained_path=getattr(cfg, 'backbone_pretrained_path', None),
            pretrained_source=getattr(cfg, 'backbone_pretrained_source', 'auto'),
            with_angle=cfg.with_angle,
            angle_format=cfg.angle_format,
            neck_out_channels=cfg.neck_out_channels,
            head_num_convs=cfg.head_num_convs,
            use_extra_conv=cfg.use_extra_conv,
            img_size=cfg.img_size,
            neck_type=cfg.neck_type,
            neck_num_layers=cfg.neck_num_layers,
            reg_max=getattr(cfg, 'reg_max', 0),
            head_type=getattr(cfg, 'head_type', 'decoupled'),
            head_norm_layer=getattr(cfg, 'head_norm_layer', 'group'),
            head_act_layer=getattr(cfg, 'head_act_layer', 'silu'),
            head_cls_num_convs=getattr(cfg, 'head_cls_num_convs', None),
            head_reg_num_convs=getattr(cfg, 'head_reg_num_convs', None),
            head_angle_num_convs=getattr(cfg, 'head_angle_num_convs', None),
        ).to(device)

        if cfg.freeze_epochs > 0:
            model.set_freeze_backbone(True)
        else:
            model.set_freeze_backbone(False)

        cfg.strides = model.strides
        model_meta = build_model_meta(cfg)

        optimizer = build_optimizer(model, cfg)

        steps_per_epoch = max(len(train_loader), 1)
        scheduler, scheduler_type = build_scheduler(optimizer, cfg, steps_per_epoch)

    # ============================================================
    # 打印训练配置
    # ============================================================
    print()
    print(LINE_EQ)
    print("  SWAP-Fighter Training Configuration")
    print(LINE_EQ)

    _title("Dataset")
    _kv("Root", cfg.data_root)
    _kv("Classes", cfg.num_classes)
    _kv("Image Size", f"{cfg.img_size} x {cfg.img_size}")
    _kv("Train / Val", f"{len(train_dataset)} / {len(val_dataset)}")

    _title("Model")
    mode_str = ("OBB (rotated bbox, angle=sin_cos)" if cfg.mode == 'obb'
                else "Detect (axis-aligned bbox)")
    _kv("Mode", mode_str)
    _kv("Backbone", f"{cfg.backbone_name} ({cfg.backbone_pretrained_source})")
    _kv("Strides", str(cfg.strides))
    neck_extra = _bool_en(cfg.use_extra_conv)
    neck_line = f"{cfg.neck_type} (out={cfg.neck_out_channels}, extra_conv={neck_extra})"
    if cfg.neck_type == 'bifpn':
        neck_line += f", layers={cfg.neck_num_layers}"
    _kv("Neck", neck_line)
    _kv("Head", f"{cfg.head_type} (convs={cfg.head_num_convs}, "
                f"norm={cfg.head_norm_layer}, act={cfg.head_act_layer})")

    if cfg.mode == 'obb':
        _kv("Task", f"cls={cfg.num_classes}, reg=4, angle=2")
    else:
        _kv("Task", f"cls={cfg.num_classes}, reg=4")

    _kv("DFL", "disabled" if cfg.reg_max == 0
              else f"reg_max={cfg.reg_max} (bin={cfg.reg_max + 1})")

    if cfg.mode == 'obb':
        if cfg.obb_reg_loss == 'kld':
            _kv("OBB Loss", f"kld (w={cfg.kld_weight}, tau={cfg.kld_tau}, "
                            f"有效权重={cfg.reg_weight * cfg.kld_weight})")
        else:
            _kv("OBB Loss", "probiou (CRIoU + 0.5 x ProbIoU)")

    _title("Training")
    _kv("Device / Precision", f"{device} / {actual_precision.upper()}")
    if scaler is not None:
        _kv("GradScaler", f"init_scale={scaler.get_scale():.0f}")
    _kv("Batch / Epochs", f"{cfg.batch_size} / {cfg.epochs}")
    _kv("Warmup", f"{cfg.warmup_epochs} epochs")
    _kv("LR (neck/head)", f"{cfg.lr:.2e}")
    _kv("LR (backbone)", f"{cfg.lr * cfg.backbone_lr_scale:.2e} "
                         f"(scale={cfg.backbone_lr_scale})")
    _kv("Optimizer / Scheduler", f"{cfg.optimizer} / {cfg.scheduler}")
    _kv("Grad Clip", f"{cfg.grad_clip_norm}")

    loss_line = f"cls={cfg.cls_weight}, reg={cfg.reg_weight}"
    if cfg.reg_max > 0:
        loss_line += f", dfl={cfg.dfl_weight}"
    if cfg.mode == 'obb':
        loss_line += f", angle={cfg.angle_weight}"
    _kv("Loss Weights", loss_line)

    _kv("EMA", f"enabled (decay={cfg.ema_decay})" if cfg.use_ema else "disabled")
    _kv("TAL", f"topk={cfg.tal_topk}, alpha={cfg.tal_alpha}, beta={cfg.tal_beta}")

    _title("Augmentation")
    mosaic_disable = cfg.mosaic_disable_epoch if cfg.mosaic_disable_epoch is not None else "never"
    _kv("Mosaic", f"prob={cfg.mosaic_prob}, disable_epoch={mosaic_disable}")
    _kv("Geometry",
        f"degrees={cfg.degrees}, translate={cfg.translate}, "
        f"scale={cfg.scale}, shear={cfg.shear}, perspective={cfg.perspective}")
    _kv("HSV", f"h={cfg.hsv_h}, s={cfg.hsv_s}, v={cfg.hsv_v}")
    _kv("Color Extra",
        f"blur={cfg.color_blur_prob}, median={cfg.color_median_blur_prob}, "
        f"gray={cfg.color_to_gray_prob}, clahe={cfg.color_clahe_prob}")
    _kv("Flip", f"use={_bool_en(cfg.use_flip)}, "
                f"lr={cfg.fliplr}, ud={cfg.flipud}")
    _kv("MixUp", f"prob={cfg.mixup_prob}, alpha={cfg.mixup_alpha}")
    _kv("Copy-Paste", f"prob={cfg.copy_paste_prob}, "
                      f"max_paste={cfg.copy_paste_max_paste}")

    _title("Evaluation")
    _kv("mAP Start Epoch", cfg.eval_map_start_epoch)
    _kv("Eval Interval", f"{cfg.eval_interval} epoch")
    _kv("Eval Conf Threshold", f"{cfg.eval_conf_threshold}")
    _kv("Eval Precision", "FP32" if args.eval_fp32 else f"{actual_precision.upper()}")

    _title("Output")
    save_interval_str = (f"{cfg.save_interval} epoch"
                         if cfg.save_interval and cfg.save_interval > 0
                         else "disabled (only best/last)")
    _kv("Save Interval", save_interval_str)
    _kv("Patience", f"{cfg.patience} (0 = disabled)" if cfg.patience else "disabled")
    _kv("Plot", "enabled" if not args.no_plot else "disabled")

    print()
    print(LINE_EQ)

    # ---------- EMA 初始化 ----------
    ema_model = None
    if cfg.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # ---------- 恢复检查点 ----------
    start_epoch = 0
    best_fitness = -1.0
    best_fitness_epoch = 0
    best_ap50 = 0.0
    best_ap95 = 0.0
    best_ar100 = 0.0
    global_step = 0
    stopped_early = False

    if args.resume is not None and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError) as e:
            print(f"  警告: 优化器状态恢复失败: {e}")

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception:
                pass

        if ema_model is not None:
            if 'ema_state_dict' in checkpoint:
                try:
                    ema_model.load_state_dict(checkpoint['ema_state_dict'])
                except Exception:
                    ema_model.load_state_dict(model.state_dict())
            else:
                ema_model.load_state_dict(model.state_dict())

        start_epoch = checkpoint['epoch']
        best_fitness = checkpoint.get('best_fitness', -1.0)
        best_fitness_epoch = checkpoint.get('best_fitness_epoch', 0)
        best_ap50 = checkpoint.get('best_ap50', 0.0)
        best_ap95 = checkpoint.get('best_ap95', 0.0)
        best_ar100 = checkpoint.get('best_ar100', 0.0)
        global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)

        train_dataset.set_epoch(start_epoch)

        try:
            if scheduler_type == 'epoch':
                for _ in range(start_epoch):
                    scheduler.step()
            else:
                for _ in range(start_epoch * steps_per_epoch):
                    scheduler.step()
        except Exception:
            pass

        try:
            target_lrs = scheduler.get_last_lr()
            for i, pg in enumerate(optimizer.param_groups):
                if i < len(target_lrs):
                    pg['lr'] = target_lrs[i]
        except Exception:
            pass

        if cfg.freeze_epochs > 0 and start_epoch >= cfg.freeze_epochs:
            model.set_freeze_backbone(False)

        print(f"  从检查点恢复: {args.resume}, 从 epoch {start_epoch} 继续")
        print(f"  最佳 fitness: {best_fitness:.4f} (epoch {best_fitness_epoch}, "
              f"mAP50={best_ap50:.4f}, mAP50-95={best_ap95:.4f}, AR100={best_ar100:.4f})")
        print()

    # ---------- 输出目录 ----------
    base_dir = cfg.save_dir if cfg.save_dir is not None else './output/checkpoints'
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = []
    for item in os.listdir(base_dir):
        if item.startswith('run') and os.path.isdir(os.path.join(base_dir, item)):
            try:
                num = int(item[3:])
                existing_runs.append(num)
            except ValueError:
                continue
    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = os.path.join(base_dir, f'run{next_run_num}')
    os.makedirs(run_dir, exist_ok=True)
    cfg.save_dir = run_dir

    print(f"  Output Directory: {cfg.save_dir}")
    print(LINE)

    log_path = os.path.join(cfg.save_dir, 'training_log.csv')
    config_path = os.path.join(cfg.save_dir, 'config.json')

    save_config_json(cfg, config_path)
    if not os.path.exists(log_path):
        init_csv_log(log_path)

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(start_epoch, cfg.epochs):
        train_dataset.set_epoch(epoch)
        epoch_start_time = time.time()

        check_and_update_freeze(model, epoch, cfg.freeze_epochs)

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg,
            scaler, cfg.grad_clip_norm,
            lr_scheduler=scheduler if scheduler_type == 'step' else None,
            amp_dtype=amp_dtype, use_amp=use_amp,
            ema_model=ema_model,
            ema_decay=cfg.ema_decay,
            global_step=global_step,
        )

        if scheduler_type == 'epoch':
            scheduler.step()

        lr_bb = optimizer.param_groups[0]['lr']
        lr_nh = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else lr_bb

        do_eval = ((epoch + 1) % cfg.eval_interval == 0) or (epoch == cfg.epochs - 1)
        if do_eval:
            compute_map = (epoch + 1) >= cfg.eval_map_start_epoch
            eval_model = ema_model if ema_model is not None else model

            val_loss, ap50, ap95, ar100 = validate_one_epoch(
                eval_model, val_loader, device, epoch, cfg,
                compute_map=compute_map,
                amp_dtype=amp_dtype, use_amp=use_amp,
                eval_use_fp32=args.eval_fp32,
            )
        else:
            val_loss = None
            ap50 = ap95 = ar100 = 0.0
            compute_map = False

        epoch_time = time.time() - epoch_start_time

        current_fitness = None
        if val_loss is not None:
            if compute_map:
                current_fitness = compute_fitness(ap50, ap95)

            append_csv_log(log_path, epoch+1, train_loss, val_loss, ap50, ap95, ar100,
                           epoch_time, lr_bb, lr_nh, current_fitness)

            if compute_map and current_fitness is not None and current_fitness > best_fitness:
                best_fitness = current_fitness
                best_fitness_epoch = epoch + 1
                best_ap50 = ap50
                best_ap95 = ap95
                best_ar100 = ar100

                save_model = ema_model if ema_model is not None else model
                torch.save({
                    'model_state_dict': save_model.state_dict(),
                    'model_meta': model_meta,
                    'epoch': epoch + 1,
                    'fitness': best_fitness,
                    'mAP50': best_ap50,
                    'mAP50_95': best_ap95,
                    'AR100': best_ar100,
                }, os.path.join(cfg.save_dir, 'best.pth'))
        else:
            append_csv_log(log_path, epoch+1, train_loss, '', '', '', '',
                           epoch_time, lr_bb, lr_nh, None)

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_fitness': best_fitness,
            'best_fitness_epoch': best_fitness_epoch,
            'best_ap50': best_ap50,
            'best_ap95': best_ap95,
            'best_ar100': best_ar100,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else None,
            'precision': actual_precision,
            'model_meta': model_meta,
            'current_lr': lr_bb,
            'global_step': global_step,
        }
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        if ema_model is not None:
            ckpt['ema_state_dict'] = ema_model.state_dict()

        torch.save(ckpt, os.path.join(cfg.save_dir, 'last.pth'))

        save_interval = getattr(cfg, 'save_interval', 10)
        if save_interval and save_interval > 0 and (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(cfg.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)

        patience = getattr(cfg, 'patience', 0) or 0
        if patience > 0 and compute_map:
            no_improve = (epoch + 1) - best_fitness_epoch
            if no_improve >= patience:
                print()
                print(LINE_EQ)
                print("  Early Stopping Triggered")
                print(LINE_EQ)
                _kv("Patience", patience, indent=2, total_width=32)
                _kv("Best epoch", best_fitness_epoch, indent=2, total_width=32)
                _kv("Current epoch", epoch + 1, indent=2, total_width=32)
                _kv("No improve", no_improve, indent=2, total_width=32)
                _kv("Best fitness", f"{best_fitness:.4f}", indent=2, total_width=32)
                print(LINE_EQ)
                stopped_early = True
                break

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_meta': model_meta,
    }, os.path.join(cfg.save_dir, 'final.pth'))

    if ema_model is not None:
        torch.save({
            'model_state_dict': ema_model.state_dict(),
            'model_meta': model_meta,
        }, os.path.join(cfg.save_dir, 'final_ema.pth'))

    print()
    print(LINE_EQ)
    print("  Training Complete" + (" (early stopped)" if stopped_early else ""))
    print(LINE_EQ)
    _kv("Best fitness", f"{best_fitness:.4f}", indent=2, total_width=32)
    _kv("Best epoch", best_fitness_epoch, indent=2, total_width=32)
    _kv("Best mAP@0.5", f"{best_ap50:.4f}", indent=2, total_width=32)
    _kv("Best mAP@0.5:0.95", f"{best_ap95:.4f}", indent=2, total_width=32)
    _kv("Best AR@100", f"{best_ar100:.4f}", indent=2, total_width=32)
    _kv("Total epochs", f"{epoch + 1} / {cfg.epochs}", indent=2, total_width=32)
    _kv("Output", cfg.save_dir, indent=2, total_width=32)
    print(LINE_EQ)

    if not args.no_plot:
        plot_training_curves(log_path, cfg.save_dir)


if __name__ == '__main__':
    main()