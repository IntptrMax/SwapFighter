import os
import time
import json
import csv
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from config import Config
from dataset import YOLODataset
from model import DetectFCOS

from utils import (
    compute_fcos_loss, evaluate_mAP, evaluate_map_obb_rotated,
    post_process, post_process_obb
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将无法生成训练曲线图。请安装: pip install matplotlib")


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg,
                    scaler=None, grad_clip_norm=5.0, lr_scheduler=None,
                    amp_dtype=torch.float16, use_amp=True):
    """
    训练一个 epoch。

    参数:
        amp_dtype: autocast 使用的 dtype（fp32 -> torch.float32; fp16 -> float16; bf16 -> bfloat16）
        use_amp:   是否启用 autocast（fp32 时为 False）
    """
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')

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

        # ---------- 前向 + 损失（显式指定 dtype 与 enabled）----------
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            if cfg.mode == 'detect':
                cls_logits_list, reg_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list,
                    targets, strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    mode='detect'
                )
            else:  # obb
                cls_logits_list, reg_preds_list, angle_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list, targets,
                    strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    angle_weight=cfg.angle_weight,
                    mode='obb', angle_format=cfg.angle_format,
                    angle_preds_list=angle_preds_list
                )

        # ---------- 反向传播 ----------
        # 只有 fp16 才使用 GradScaler；bf16 / fp32 直接 backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # 检测 inf/nan 并跳过 optimizer.step（GradScaler 内部也会自动跳）
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # 调度器 step（部分调度器需要在每个 batch 后更新，如 OneCycleLR）
        if lr_scheduler is not None and cfg.scheduler == 'onecycle':
            lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch, cfg, compute_map=True,
                       amp_dtype=torch.float16, use_amp=True,
                       eval_use_fp32=False):
    """
    验证一个 epoch。

    参数:
        amp_dtype / use_amp: 与训练一致的精度策略（用于计算 val loss）
        eval_use_fp32: 若为 True，则 val loss 也强制 FP32，更稳。
                       mAP 评估始终以 FP32 进行。
    """
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')

    # 是否强制 FP32 验证（若 eval_use_fp32，则关闭 autocast）
    val_use_amp = use_amp and not eval_use_fp32
    val_amp_dtype = amp_dtype if not eval_use_fp32 else torch.float32

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
                    mode='detect'
                )
            else:
                cls_logits_list, reg_preds_list, angle_preds_list = model(images)
                loss = compute_fcos_loss(
                    cls_logits_list, reg_preds_list, targets,
                    strides=cfg.strides, device=device, img_size=cfg.img_size,
                    cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                    angle_weight=cfg.angle_weight,
                    mode='obb', angle_format=cfg.angle_format,
                    angle_preds_list=angle_preds_list
                )
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / max(len(dataloader), 1)

    mAP_50 = mAP_50_95 = mar_100 = 0.0
    if compute_map:
        if cfg.mode == 'obb':
            # 使用旋转框 mAP 评估
            mAP_50, mAP_50_95 = evaluate_map_obb_rotated(
                model, dataloader, device, cfg,
                conf_threshold=0.01,
                nms_threshold=cfg.obb_nms_threshold,
                max_det=300,
            )
            mar_100 = 0.0   # 该函数不返回 AR，如需可自行扩展
        else:
            mAP_50, mAP_50_95, mar_100 = evaluate_mAP(model, dataloader, device, cfg)
        if isinstance(mAP_50, torch.Tensor):
            mAP_50 = mAP_50.item()
        if isinstance(mAP_50_95, torch.Tensor):
            mAP_50_95 = mAP_50_95.item()
        if isinstance(mar_100, torch.Tensor):
            mar_100 = mar_100.item()
        print(f'  mAP@0.5: {mAP_50:.4f}, mAP@0.5:0.95: {mAP_50_95:.4f}')

    return avg_loss, mAP_50, mAP_50_95, mar_100


def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter 训练脚本 (支持 Detect 和 OBB)')

    # 模式与数据
    parser.add_argument('--mode', type=str, default=None, choices=['detect', 'obb'],
                        help='检测模式: detect(水平框) 或 obb(旋转框)')
    parser.add_argument('--angle-format', type=str, default=None, choices=['sin_cos', 'direct'],
                        help='OBB 角度编码方式: sin_cos 或 direct')
    parser.add_argument('--data-root', type=str, default=None, help='数据集根目录')
    parser.add_argument('--train-split', type=str, default=None, help='训练集子目录')
    parser.add_argument('--val-split', type=str, default=None, help='验证集子目录')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数量')

    # 模型
    parser.add_argument('--backbone', type=str, default=None, help='Backbone 名称')
    parser.add_argument('--pretrained', action='store_true', default=None, help='是否使用预训练')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='不使用预训练')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像尺寸')
    parser.add_argument('--neck-out-channels', type=int, default=None,
                        help='Neck 输出通道数')
    parser.add_argument('--header-num-convs', type=int, default=None,
                        help='检测头卷积层数')
    parser.add_argument('--use-extra-conv', action='store_true', default=None,
                        help='Neck 中使用额外卷积')
    parser.add_argument('--no-extra-conv', dest='use_extra_conv', action='store_false',
                        help='Neck 中不使用额外卷积')

    # 训练
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--num-workers', type=int, default=None, help='DataLoader 工作线程数')
    parser.add_argument('--grad-clip', type=float, default=None, help='梯度裁剪阈值')
    parser.add_argument('--freeze-epochs', type=int, default=None, help='冻结 Backbone 的轮数')
    parser.add_argument('--unfreeze-lr', type=float, default=None, help='解冻后的学习率')

    # 损失权重
    parser.add_argument('--cls-weight', type=float, default=None, help='分类损失权重')
    parser.add_argument('--reg-weight', type=float, default=None, help='回归损失权重')
    parser.add_argument('--angle-weight', type=float, default=None, help='角度损失权重')

    # 保存与验证
    parser.add_argument('--save-dir', type=str, default=None, help='保存目录')
    parser.add_argument('--save-interval', type=int, default=None, help='保存检查点间隔')
    parser.add_argument('--eval-interval', type=int, default=None, help='验证间隔')
    parser.add_argument('--eval-map-start-epoch', type=int, default=None,
                        help='从该轮开始计算 mAP（之前的轮次跳过 mAP，默认 0 表示第一轮开始）')
    parser.add_argument('--eval-fp32', action='store_true',
                        help='验证阶段强制 FP32（更稳，但略慢）')
    parser.add_argument('--no-plot', action='store_true', help='禁用训练曲线绘制')

    # 精度
    parser.add_argument('--precision', type=str, default=None,
                        choices=['fp32', 'fp16', 'bf16'],
                        help='训练精度: fp32(全精度) / fp16(混合精度+梯度缩放) / bf16(混合精度，Ampere+)')
    parser.add_argument('--scaler-init-scale', type=float, default=2. ** 12,
                        help='GradScaler 初始 scale（FP16 建议 4096，避免前期频繁 inf 跳过 step）')

    # 优化器与调度器
    parser.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'adamw'],
                        help='优化器类型 (默认 adamw)')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['lambda', 'step', 'multistep', 'cosine', 'onecycle'],
                        help='调度器类型 (默认 cosine)')
    parser.add_argument('--scheduler-gamma', type=float, default=None,
                        help='调度器衰减因子 (用于 lambda, step, multistep)')
    parser.add_argument('--scheduler-step-size', type=int, default=None,
                        help='StepLR 的步长')
    parser.add_argument('--scheduler-milestones', type=str, default=None,
                        help='MultiStepLR 的里程碑，逗号分隔，例如 "30,60,80"')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='学习率 warmup 轮数（仅对 cosine/lambda/step/multistep 生效）')

    # 数据增强
    parser.add_argument('--mosaic-prob', type=float, default=None, help='Mosaic 概率')
    parser.add_argument('--mosaic-disable-epoch', type=int, default=None, help='禁用 Mosaic 的轮数')

    # 其他
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复')

    return parser.parse_args()


def update_config_from_args(cfg, args):
    # 模式与数据
    if args.mode is not None:
        cfg.mode = args.mode
    if args.angle_format is not None:
        cfg.angle_format = args.angle_format
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes

    # 模型
    if args.backbone is not None:
        cfg.backbone_name = args.backbone
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.neck_out_channels is not None:
        cfg.neck_out_channels = args.neck_out_channels
    if args.header_num_convs is not None:
        cfg.head_num_convs = args.header_num_convs
    if args.use_extra_conv is not None:
        cfg.use_extra_conv = args.use_extra_conv

    # 训练
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.grad_clip is not None:
        cfg.grad_clip_norm = args.grad_clip
    if args.freeze_epochs is not None:
        cfg.freeze_epochs = args.freeze_epochs
    if args.unfreeze_lr is not None:
        cfg.unfreeze_lr = args.unfreeze_lr

    # 损失权重
    if args.cls_weight is not None:
        cfg.cls_weight = args.cls_weight
    if args.reg_weight is not None:
        cfg.reg_weight = args.reg_weight
    if args.angle_weight is not None:
        cfg.angle_weight = args.angle_weight

    # 保存与验证
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.eval_map_start_epoch is not None:
        cfg.eval_map_start_epoch = args.eval_map_start_epoch

    # 精度
    if args.precision is not None:
        cfg.precision = args.precision

    # 优化器与调度器
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.scheduler is not None:
        cfg.scheduler = args.scheduler
    if args.scheduler_gamma is not None:
        cfg.scheduler_gamma = args.scheduler_gamma
    if args.scheduler_step_size is not None:
        cfg.scheduler_step_size = args.scheduler_step_size
    if args.scheduler_milestones is not None:
        milestones = [int(x.strip()) for x in args.scheduler_milestones.split(',') if x.strip().isdigit()]
        cfg.scheduler_milestones = milestones
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs

    # 数据增强
    if args.mosaic_prob is not None:
        cfg.mosaic_prob = args.mosaic_prob
    if args.mosaic_disable_epoch is not None:
        cfg.mosaic_disable_epoch = args.mosaic_disable_epoch

    # 其他
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    cfg.validate_and_setup()
    return cfg


def resolve_precision(cfg, device, scaler_init_scale=2. ** 12):
    """
    根据 cfg.precision 与硬件情况解析出最终使用的精度配置。

    Returns:
        (amp_dtype, use_amp, scaler, precision_str)
        - amp_dtype: torch.float32 / torch.float16 / torch.bfloat16
        - use_amp:   是否开启 autocast
        - scaler:    GradScaler 或 None（只有 fp16 + CUDA 才需要）
        - precision_str: 最终实际使用的精度字符串（可能因回退而与 cfg.precision 不同）
    """
    precision = getattr(cfg, 'precision', None)
    if precision is None:
        precision = 'fp16' if getattr(cfg, 'use_amp', True) else 'fp32'

    # 非 CUDA 设备只能 fp32
    if device.type != 'cuda':
        if precision != 'fp32':
            print(f"警告: 设备 {device.type} 不支持混合精度，自动回退到 fp32（原设置: {precision}）")
        return torch.float32, False, None, 'fp32'

    if precision == 'fp32':
        return torch.float32, False, None, 'fp32'

    if precision == 'fp16':
        if not torch.cuda.is_available():
            print("警告: CUDA 不可用，FP16 回退到 FP32")
            return torch.float32, False, None, 'fp32'
        # 使用更保守的初始 scale（默认 65536 对检测任务往往过大）
        # 若 scaler_init_scale 为 0 或负值，则使用 PyTorch 默认值
        if scaler_init_scale and scaler_init_scale > 0:
            scaler = torch.amp.GradScaler(
                'cuda',
                init_scale=float(scaler_init_scale),
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=1000,
            )
        else:
            scaler = torch.amp.GradScaler('cuda')
        return torch.float16, True, scaler, 'fp16'

    if precision == 'bf16':
        # 检查硬件是否支持 BF16（Ampere 及更新架构）
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except AttributeError:
            try:
                major = torch.cuda.get_device_properties(0).major
                bf16_ok = major >= 8
            except Exception:
                bf16_ok = False

        if not bf16_ok:
            print("警告: 当前 GPU 不支持 BF16，自动回退到 FP16")
            if scaler_init_scale and scaler_init_scale > 0:
                scaler = torch.amp.GradScaler(
                    'cuda',
                    init_scale=float(scaler_init_scale),
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=1000,
                )
            else:
                scaler = torch.amp.GradScaler('cuda')
            return torch.float16, True, scaler, 'fp16'
        # BF16 与 FP32 动态范围一致，无需 GradScaler
        return torch.bfloat16, True, None, 'bf16'

    raise ValueError(f"未知精度: {precision}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """
    构建学习率调度器，支持 warmup + 多种衰减策略。

    说明：
        - cosine / lambda / step / multistep 使用 epoch 级别的 warmup + 衰减。
        - onecycle 自身包含 warmup，直接使用 OneCycleLR（step 级别更新）。

    Returns:
        scheduler, scheduler_type
        scheduler_type: 'epoch' 或 'step'，指示主循环里如何调用 scheduler.step()
    """
    warmup_epochs = getattr(cfg, 'warmup_epochs', 0) or 0
    total_epochs = cfg.epochs

    if cfg.scheduler == 'onecycle':
        total_steps = max(steps_per_epoch * cfg.epochs, 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=max(0.05, warmup_epochs / max(total_epochs, 1)) if warmup_epochs > 0 else 0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        return scheduler, 'step'

    # ---- warmup + 主衰减的 lambda 组合 ----
    def make_lambda(base_lambda):
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                # 线性 warmup：从 1/warmup 到 1
                return float(epoch + 1) / float(warmup_epochs)
            # warmup 结束后，把 epoch 映射到 [0, total_epochs - warmup_epochs)
            adjusted_epoch = epoch - warmup_epochs
            adjusted_total = max(total_epochs - warmup_epochs, 1)
            return base_lambda(adjusted_epoch, adjusted_total)
        return lr_lambda

    if cfg.scheduler == 'cosine':
        def base_lambda(epoch, total):
            return 0.5 * (1.0 + math.cos(math.pi * epoch / total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_lambda(base_lambda)
        )
        return scheduler, 'epoch'

    if cfg.scheduler == 'lambda':
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            return gamma ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_lambda(base_lambda)
        )
        return scheduler, 'epoch'

    if cfg.scheduler == 'step':
        # StepLR 不支持包裹 warmup，简单做法：手动构造 lambda
        step_size = max(cfg.scheduler_step_size, 1)
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            return gamma ** (epoch // step_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_lambda(base_lambda)
        )
        return scheduler, 'epoch'

    if cfg.scheduler == 'multistep':
        milestones = sorted(cfg.scheduler_milestones or [])
        gamma = cfg.scheduler_gamma
        def base_lambda(epoch, total):
            count = sum(1 for m in milestones if epoch >= m)
            return gamma ** count
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_lambda(base_lambda)
        )
        return scheduler, 'epoch'

    raise ValueError(f"不支持的调度器: {cfg.scheduler}")


def check_and_update_freeze(model, epoch, freeze_epochs, freeze_optimizer=None, unfreeze_lr=None):
    if freeze_epochs > 0 and epoch >= freeze_epochs and model.get_freeze_status():
        print(f"\n{'='*60}")
        print(f"达到第 {freeze_epochs} 轮，解冻 Backbone")
        print(f"{'='*60}")
        model.set_freeze_backbone(False)
        if freeze_optimizer is not None and unfreeze_lr is not None:
            print(f"调整学习率为: {unfreeze_lr}")
            for param_group in freeze_optimizer.param_groups:
                param_group['lr'] = unfreeze_lr


def save_config_json(cfg, save_path):
    config_dict = cfg.to_dict()
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"参数表已保存至: {save_path}")


def init_csv_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'AP50', 'AP95', 'AR100', 'time_sec'])


def append_csv_log(log_path, epoch, train_loss, val_loss, ap50, ap95, ar100, time_sec):
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
            int(epoch),
            to_float(train_loss),
            to_float(val_loss),
            to_float(ap50),
            to_float(ap95),
            to_float(ar100),
            to_float(time_sec)
        ])


def plot_training_curves(log_path, save_dir):
    if not HAS_MATPLOTLIB:
        print("跳过绘图：matplotlib 未安装")
        return

    if not os.path.exists(log_path):
        print("日志文件不存在，无法绘图")
        return

    epochs = []
    train_losses = []
    val_losses = []
    ap50s = []
    ap95s = []
    ar100s = []
    times = []

    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("日志文件为空，无法绘图")
            return
        for row in reader:
            if len(row) < 7:
                continue
            try:
                epoch = int(row[0])
                train_loss = float(row[1]) if row[1] else None
                val_loss = float(row[2]) if row[2] else None
                ap50 = float(row[3]) if row[3] else None
                ap95 = float(row[4]) if row[4] else None
                ar100 = float(row[5]) if row[5] else None
                time_sec = float(row[6]) if row[6] else None
            except (ValueError, IndexError):
                continue
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            ap50s.append(ap50)
            ap95s.append(ap95)
            ar100s.append(ar100)
            times.append(time_sec)

    if not epochs:
        print("日志无有效数据，无法绘图")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves', fontsize=16)

    ax1 = axes[0, 0]
    if train_losses:
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[0, 1]
    if ap50s:
        ax2.plot(epochs, ap50s, 'g-', label='AP@0.5')
    if ap95s:
        ax2.plot(epochs, ap95s, 'm-', label='AP@0.5:0.95')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AP')
    ax2.set_title('Average Precision')
    ax2.legend()
    ax2.grid(True)

    ax3 = axes[0, 2]
    if ar100s:
        ax3.plot(epochs, ar100s, 'c-', label='AR@100')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AR')
    ax3.set_title('Average Recall')
    ax3.legend()
    ax3.grid(True)

    ax4 = axes[1, 0]
    if times:
        ax4.plot(epochs, times, 'orange', label='Time per epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Training Time')
    ax4.legend()
    ax4.grid(True)

    ax5 = axes[1, 1]
    if train_losses and ap50s:
        ax5.plot(epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
        ax5.plot(epochs, ap50s, 'g-', alpha=0.7, label='AP@0.5')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Value')
    ax5.set_title('Loss vs AP')
    ax5.legend()
    ax5.grid(True)

    ax6 = axes[1, 2]
    if ap50s and ar100s:
        ax6.plot(ap50s, ar100s, 'r-', marker='o', markersize=3)
        ax6.set_xlabel('AP@0.5')
        ax6.set_ylabel('AR@100')
        ax6.set_title('AP vs AR')
        ax6.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"训练曲线图已保存至: {plot_path}")


def main():
    args = parse_args()

    cfg = Config()
    cfg = update_config_from_args(cfg, args)

    # 确保兼容原有属性
    if not hasattr(cfg, 'mosaic_prob'):
        cfg.mosaic_prob = 0.0
    if not hasattr(cfg, 'mosaic_disable_epoch'):
        cfg.mosaic_disable_epoch = None
    if not hasattr(cfg, 'grad_clip_norm'):
        cfg.grad_clip_norm = 5.0
    if not hasattr(cfg, 'cls_weight'):
        cfg.cls_weight = 1.0
    if not hasattr(cfg, 'reg_weight'):
        cfg.reg_weight = 2.0
    if not hasattr(cfg, 'angle_weight'):
        cfg.angle_weight = 1.0
    if not hasattr(cfg, 'freeze_epochs'):
        cfg.freeze_epochs = 0
    if not hasattr(cfg, 'unfreeze_lr'):
        cfg.unfreeze_lr = cfg.lr * 0.1
    if not hasattr(cfg, 'mean'):
        cfg.mean = [0.485, 0.456, 0.406]
    if not hasattr(cfg, 'std'):
        cfg.std = [0.229, 0.224, 0.225]
    # 精度相关默认值（若旧版 Config 未提供）
    if not hasattr(cfg, 'precision') or cfg.precision is None:
        cfg.precision = 'fp16' if getattr(cfg, 'use_amp', True) else 'fp32'
    # 新增属性默认值
    if not hasattr(cfg, 'optimizer'):
        cfg.optimizer = 'adamw'
    if not hasattr(cfg, 'scheduler'):
        cfg.scheduler = 'cosine'
    if not hasattr(cfg, 'scheduler_gamma'):
        cfg.scheduler_gamma = 0.95
    if not hasattr(cfg, 'scheduler_step_size'):
        cfg.scheduler_step_size = 1
    if not hasattr(cfg, 'scheduler_milestones'):
        cfg.scheduler_milestones = [30, 60]
    if not hasattr(cfg, 'warmup_epochs'):
        cfg.warmup_epochs = 5
    if not hasattr(cfg, 'neck_out_channels'):
        cfg.neck_out_channels = 128
    if not hasattr(cfg, 'head_num_convs'):
        cfg.head_num_convs = 2
    if not hasattr(cfg, 'use_extra_conv'):
        cfg.use_extra_conv = True
    # mAP 起始轮次
    if not hasattr(cfg, 'eval_map_start_epoch'):
        cfg.eval_map_start_epoch = 0

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device = torch.device('cpu')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # ---------- 解析精度配置 ----------
    amp_dtype, use_amp, scaler, actual_precision = resolve_precision(
        cfg, device, scaler_init_scale=args.scaler_init_scale
    )

    print("=" * 60)
    print("SWAP-Fighter 训练配置:")
    print(f"  模式: {cfg.mode}")
    if cfg.mode == 'obb':
        print(f"  角度编码: {cfg.angle_format}")
    print(f"  数据集: {cfg.data_root}")
    print(f"  类别数: {cfg.num_classes}")
    print(f"  Backbone: {cfg.backbone_name}")
    print(f"  图像尺寸: {cfg.img_size}")
    print(f"  批次大小: {cfg.batch_size}")
    print(f"  学习率: {cfg.lr}")
    print(f"  解冻学习率: {cfg.unfreeze_lr}")
    print(f"  训练轮数: {cfg.epochs}")
    print(f"  设备: {device}")
    # 训练精度 + GradScaler 信息合并到一行
    precision_line = f"  训练精度: {actual_precision.upper()} (请求: {cfg.precision})"
    if scaler is not None:
        precision_line += f", GradScaler init_scale={scaler.get_scale():.0f}"
    print(precision_line)
    print(f"  验证强制 FP32: {args.eval_fp32}")
    print(f"  冻结前 {cfg.freeze_epochs} 轮")
    print(f"  优化器: {cfg.optimizer}")
    print(f"  调度器: {cfg.scheduler}")
    if cfg.scheduler in ['lambda', 'step', 'multistep', 'cosine']:
        print(f"  Warmup 轮数: {cfg.warmup_epochs}")
    if cfg.scheduler in ['lambda', 'step', 'multistep']:
        print(f"  调度器 gamma: {cfg.scheduler_gamma}")
    if cfg.scheduler == 'step':
        print(f"  调度器 step_size: {cfg.scheduler_step_size}")
    if cfg.scheduler == 'multistep':
        print(f"  调度器 milestones: {cfg.scheduler_milestones}")
    # 打印模型参数
    print(f"  Neck 输出通道: {cfg.neck_out_channels}")
    print(f"  检测头卷积层数: {cfg.head_num_convs}")
    print(f"  Neck 额外卷积: {cfg.use_extra_conv}")
    print(f"  梯度裁剪阈值: {cfg.grad_clip_norm}")
    print(f"  损失权重 - cls: {cfg.cls_weight}, reg: {cfg.reg_weight}"
          + (f", angle: {cfg.angle_weight}" if cfg.mode == 'obb' else ""))
    print(f"  mAP 计算起始轮次: {cfg.eval_map_start_epoch} (0 表示第一轮开始)")
    print(f"  Mosaic 概率: {cfg.mosaic_prob}")
    print(f"  Mosaic 禁用轮数: {cfg.mosaic_disable_epoch if cfg.mosaic_disable_epoch is not None else '从不'}")
    print("=" * 60)

    # 创建数据集
    train_dataset = YOLODataset(
        root=cfg.data_root,
        img_size=cfg.img_size,
        split=cfg.train_split,
        use_flip=cfg.use_flip,
        use_color_jitter=cfg.use_color_jitter,
        color_jitter_params=cfg.color_jitter_params,
        mean=cfg.mean,
        std=cfg.std,
        mode=cfg.mode,
        angle_range=cfg.angle_range if cfg.mode == 'obb' else None,
        mosaic_prob=cfg.mosaic_prob,
        mosaic_disable_epoch=cfg.mosaic_disable_epoch,
    )
    val_dataset = YOLODataset(
        root=cfg.data_root,
        img_size=cfg.img_size,
        split=cfg.val_split,
        use_flip=False,
        use_color_jitter=False,
        mean=cfg.mean,
        std=cfg.std,
        mode=cfg.mode,
        angle_range=cfg.angle_range if cfg.mode == 'obb' else None,
        debug_vis=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None
    )

    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')

    # 创建模型
    model = DetectFCOS(
        num_classes=cfg.num_classes,
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        with_angle=cfg.with_angle,
        angle_format=cfg.angle_format,
        neck_out_channels=cfg.neck_out_channels,
        head_num_convs=cfg.head_num_convs,
        use_extra_conv=cfg.use_extra_conv
    ).to(device)

    if cfg.freeze_epochs > 0:
        model.set_freeze_backbone(True)
    else:
        model.set_freeze_backbone(False)

    cfg.strides = model.strides
    print(f"Backbone 各层步长: {cfg.strides}")

    # ---------- 优化器 ----------
    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                              weight_decay=cfg.weight_decay, nesterov=False)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer}")

    # ---------- 调度器（含 warmup） ----------
    steps_per_epoch = max(len(train_loader), 1)
    scheduler, scheduler_type = build_scheduler(optimizer, cfg, steps_per_epoch)
    print(f"使用 {cfg.scheduler} 调度器，更新方式: {scheduler_type}, warmup={cfg.warmup_epochs} epochs")

    # ---------- 恢复检查点 ----------
    start_epoch = 0
    best_mAP = 0.0
    if args.resume is not None:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 若 checkpoint 中保存了 scaler，则恢复（FP16 训练时建议恢复）
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print(f"已恢复 GradScaler 状态 (scale={scaler.get_scale():.0f})")
                except Exception as e:
                    print(f"警告: GradScaler 状态恢复失败: {e}")
            start_epoch = checkpoint['epoch']
            best_mAP = checkpoint.get('best_mAP', 0.0)
            print(f"从检查点恢复训练: {args.resume}, 从 epoch {start_epoch} 继续")
            train_dataset.set_epoch(start_epoch)
            # 快进 scheduler 到当前 epoch
            try:
                if scheduler_type == 'epoch':
                    for _ in range(start_epoch):
                        scheduler.step()
                else:
                    for _ in range(start_epoch * steps_per_epoch):
                        scheduler.step()
            except Exception as e:
                print(f"警告: scheduler 快进失败: {e}")
            print(f"恢复的最佳 mAP: {best_mAP:.4f}")
            if cfg.freeze_epochs > 0 and start_epoch >= cfg.freeze_epochs:
                model.set_freeze_backbone(False)
        else:
            print(f"警告: 检查点文件不存在: {args.resume}")

    # 保存目录
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
    print(f"本次训练结果将保存至: {cfg.save_dir}")

    log_path = os.path.join(cfg.save_dir, 'training_log.csv')
    config_path = os.path.join(cfg.save_dir, 'config.json')

    save_config_json(cfg, config_path)
    if not os.path.exists(log_path):
        init_csv_log(log_path)
    else:
        print(f"日志文件已存在，将追加数据: {log_path}")

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, cfg.epochs):
        train_dataset.set_epoch(epoch)

        epoch_start_time = time.time()

        check_and_update_freeze(
            model, epoch, cfg.freeze_epochs,
            freeze_optimizer=optimizer,
            unfreeze_lr=cfg.unfreeze_lr
        )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg,
            scaler, cfg.grad_clip_norm,
            lr_scheduler=scheduler if scheduler_type == 'step' else None,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        print(f'Epoch {epoch+1}, 训练损失: {train_loss:.4f}, '
              f'当前 lr: {optimizer.param_groups[0]["lr"]:.3e}')

        if scheduler_type == 'epoch':
            scheduler.step()

        do_eval = ((epoch + 1) % cfg.eval_interval == 0) or (epoch == cfg.epochs - 1)
        if do_eval:
            # 根据起始轮次决定是否计算 mAP
            compute_map = (epoch + 1) >= cfg.eval_map_start_epoch
            val_loss, ap50, ap95, ar100 = validate_one_epoch(
                model, val_loader, device, epoch, cfg,
                compute_map=compute_map,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                eval_use_fp32=args.eval_fp32,
            )
            print(f'Epoch {epoch+1}, 验证损失: {val_loss:.4f}, '
                  f'AP@0.5: {ap50:.4f}, AP@0.5:0.95: {ap95:.4f}, AR@100: {ar100:.4f}')
        else:
            val_loss = None
            ap50 = 0.0
            ap95 = 0.0
            ar100 = 0.0

        epoch_time = time.time() - epoch_start_time

        if val_loss is not None:
            append_csv_log(log_path, epoch+1, train_loss, val_loss, ap50, ap95, ar100, epoch_time)
            if ap50 > best_mAP:
                best_mAP = ap50
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'best.pth'))
                print(f'保存最佳模型 (AP@0.5: {ap50:.4f})')
        else:
            append_csv_log(log_path, epoch+1, train_loss, '', '', '', '', epoch_time)

        # 保存 checkpoint（含 scaler 状态，便于 FP16 恢复）
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mAP': best_mAP,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else None,
            'precision': actual_precision,
        }
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()

        torch.save(ckpt, os.path.join(cfg.save_dir, 'last.pth'))

        if (epoch + 1) % cfg.save_interval == 0:
            torch.save(ckpt, os.path.join(cfg.save_dir, f'epoch_{epoch+1}.pth'))

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'final.pth'))
    print(f"训练完成！最佳 AP@0.5: {best_mAP:.4f}")

    if not args.no_plot:
        plot_training_curves(log_path, cfg.save_dir)
    else:
        print("训练曲线绘制已禁用 (--no-plot)")


if __name__ == '__main__':
    main()