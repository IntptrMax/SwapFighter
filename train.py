import os
import time
import json
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from config import Config
from dataset import YOLODataset
from model import DetectFCOS
from utils import compute_fcos_loss, evaluate_mAP

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


def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg, scaler=None, grad_clip_norm=1.0):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for images, targets in pbar:
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            cls_logits_list, reg_preds_list = model(images)
            loss = compute_fcos_loss(
                cls_logits_list, reg_preds_list,
                targets, strides=cfg.strides, device=device, img_size=cfg.img_size,
                cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight
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
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch, cfg, compute_map=True):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    for images, targets in pbar:
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        cls_logits_list, reg_preds_list = model(images)
        loss = compute_fcos_loss(
            cls_logits_list, reg_preds_list,
            targets, strides=cfg.strides, device=device, img_size=cfg.img_size,
            cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight
        )
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)

    if compute_map:
        mAP_50, mAP_50_95, mar_100 = evaluate_mAP(model, dataloader, device, cfg)
        # 确保是 Python float
        if isinstance(mAP_50, torch.Tensor):
            mAP_50 = mAP_50.item()
        if isinstance(mAP_50_95, torch.Tensor):
            mAP_50_95 = mAP_50_95.item()
        if isinstance(mar_100, torch.Tensor):
            mar_100 = mar_100.item()
        print(f'  mAP@0.5: {mAP_50:.4f}, mAP@0.5:0.95: {mAP_50_95:.4f}, AR@100: {mar_100:.4f}')
    else:
        mAP_50, mAP_50_95, mar_100 = 0.0, 0.0, 0.0

    return avg_loss, mAP_50, mAP_50_95, mar_100


def parse_args():
    parser = argparse.ArgumentParser(description='ConvNeXtV2-FCOS 训练脚本')
    
    # ---------- 数据相关 ----------
    parser.add_argument('--data-root', type=str, default=None,
                        help='数据集根目录（覆盖 config）')
    parser.add_argument('--train-split', type=str, default=None,
                        help='训练集子目录名')
    parser.add_argument('--val-split', type=str, default=None,
                        help='验证集子目录名')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数量')
    
    # ---------- 模型相关 ----------
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone 名称')
    parser.add_argument('--pretrained', action='store_true', default=None,
                        help='是否使用预训练权重')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='不使用预训练权重')
    parser.add_argument('--img-size', type=int, default=None,
                        help='输入图像尺寸')
    
    # ---------- 训练相关 ----------
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Warmup 轮数')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader 工作线程数')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='模型保存目录（基础目录，每次运行自动创建 runX 子文件夹）')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='保存检查点间隔（轮数）')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='验证间隔（轮数）')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='梯度裁剪阈值')
    
    # ---------- 损失权重 ----------
    parser.add_argument('--cls-weight', type=float, default=None,
                        help='分类损失权重')
    parser.add_argument('--reg-weight', type=float, default=None,
                        help='回归损失权重')
    
    # ---------- 冻结 Backbone 相关 ----------
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='前 N 轮冻结 Backbone，之后解冻（0 表示不冻结）')
    parser.add_argument('--unfreeze-lr', type=float, default=None,
                        help='解冻 Backbone 后的学习率（默认：原学习率 * 0.1）')
    
    # ---------- 开关参数 ----------
    parser.add_argument('--use-amp', action='store_true', default=None,
                        help='启用混合精度训练')
    
    # ---------- 设备 ----------
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'], help='训练设备')
    
    # ---------- 其他 ----------
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--no-plot', action='store_true',
                        help='禁用训练曲线绘制')
    
    return parser.parse_args()


def update_config_from_args(cfg, args):
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    if args.backbone is not None:
        cfg.backbone_name = args.backbone
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.grad_clip is not None:
        cfg.grad_clip_norm = args.grad_clip
    if args.cls_weight is not None:
        cfg.cls_weight = args.cls_weight
    if args.reg_weight is not None:
        cfg.reg_weight = args.reg_weight
    if args.use_amp is not None:
        cfg.use_amp = args.use_amp
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.freeze_epochs is not None:
        cfg.freeze_epochs = args.freeze_epochs
    if args.unfreeze_lr is not None:
        cfg.unfreeze_lr = args.unfreeze_lr
    return cfg


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
    config_dict = {
        'backbone_name': cfg.backbone_name,
        'num_classes': cfg.num_classes,
        'img_size': cfg.img_size,
        'strides': cfg.strides,
        'mean': getattr(cfg, 'mean', [0.485, 0.456, 0.406]),
        'std': getattr(cfg, 'std', [0.229, 0.224, 0.225]),
        'conf_threshold': getattr(cfg, 'conf_threshold', 0.5),
        'nms_threshold': getattr(cfg, 'nms_threshold', 0.5),
        'freeze_epochs': cfg.freeze_epochs,
        'unfreeze_lr': cfg.unfreeze_lr,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'warmup_epochs': cfg.warmup_epochs,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'use_amp': cfg.use_amp,
        'seed': cfg.seed,
        'data_root': cfg.data_root,
    }
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"参数表已保存至: {save_path}")


def init_csv_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'AP50', 'AP95', 'AR100', 'time_sec'])


def append_csv_log(log_path, epoch, train_loss, val_loss, ap50, ap95, ar100, time_sec):
    """将数据追加到 CSV，所有数值转为 Python 原生类型"""
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
    
    # 1. Loss
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
    
    # 2. AP50, AP95
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
    
    # 3. AR100
    ax3 = axes[0, 2]
    if ar100s:
        ax3.plot(epochs, ar100s, 'c-', label='AR@100')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AR')
    ax3.set_title('Average Recall')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Time
    ax4 = axes[1, 0]
    if times:
        ax4.plot(epochs, times, 'orange', label='Time per epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Training Time')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Loss vs AP
    ax5 = axes[1, 1]
    if train_losses and ap50s:
        ax5.plot(epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
        ax5.plot(epochs, ap50s, 'g-', alpha=0.7, label='AP@0.5')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Value')
    ax5.set_title('Loss vs AP')
    ax5.legend()
    ax5.grid(True)
    
    # 6. AP vs AR
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
    # ---------- 解析参数 ----------
    args = parse_args()
    
    # ---------- 加载配置 ----------
    cfg = Config()
    cfg = update_config_from_args(cfg, args)
    
    # 默认值
    if not hasattr(cfg, 'grad_clip_norm'):
        cfg.grad_clip_norm = 1.0
    if not hasattr(cfg, 'cls_weight'):
        cfg.cls_weight = 1.0
    if not hasattr(cfg, 'reg_weight'):
        cfg.reg_weight = 2.0
    if not hasattr(cfg, 'freeze_epochs'):
        cfg.freeze_epochs = 0
    if not hasattr(cfg, 'unfreeze_lr'):
        cfg.unfreeze_lr = cfg.lr * 0.1
    if not hasattr(cfg, 'use_amp'):
        cfg.use_amp = True
    if not hasattr(cfg, 'mean'):
        cfg.mean = [0.485, 0.456, 0.406]
    if not hasattr(cfg, 'std'):
        cfg.std = [0.229, 0.224, 0.225]
    
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device = torch.device('cpu')
    
    # ---------- 设置种子 ----------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # ---------- 打印配置 ----------
    print("=" * 60)
    print("训练配置:")
    print(f"  数据集: {cfg.data_root}")
    print(f"  类别数: {cfg.num_classes}")
    print(f"  Backbone: {cfg.backbone_name}")
    print(f"  图像尺寸: {cfg.img_size}")
    print(f"  批次大小: {cfg.batch_size}")
    print(f"  学习率: {cfg.lr}")
    print(f"  解冻学习率: {cfg.unfreeze_lr}")
    print(f"  训练轮数: {cfg.epochs}")
    print(f"  设备: {device}")
    print(f"  混合精度: {cfg.use_amp}")
    print(f"  冻结前 {cfg.freeze_epochs} 轮")
    print("=" * 60)
    
    # ---------- 数据集 ----------
    train_dataset = YOLODataset(
        root=cfg.data_root,
        img_size=cfg.img_size,
        split=cfg.train_split,
        use_flip=cfg.use_flip,
        use_color_jitter=cfg.use_color_jitter,
        color_jitter_params=cfg.color_jitter_params,
        mean=cfg.mean,
        std=cfg.std,
    )
    val_dataset = YOLODataset(
        root=cfg.data_root,
        img_size=cfg.img_size,
        split=cfg.val_split,
        use_flip=False,
        use_color_jitter=False,
        mean=cfg.mean,
        std=cfg.std,
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

    # ---------- 模型 ----------
    model = DetectFCOS(
        num_classes=cfg.num_classes,
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained
    ).to(device)
    
    if cfg.freeze_epochs > 0:
        model.set_freeze_backbone(True)
    else:
        model.set_freeze_backbone(False)
    
    cfg.strides = model.strides
    print(f"Backbone 各层步长: {cfg.strides}")

    # ---------- 优化器 ----------
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def warmup_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs
    )

    def combined_scheduler(epoch):
        if epoch < cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

    # ---------- 混合精度 ----------
    scaler = torch.amp.GradScaler('cuda') if (cfg.use_amp and device.type == 'cuda') else None
    if scaler is not None:
        print("启用混合精度训练 (AMP)")
    else:
        print("禁用混合精度训练")

    # ---------- 恢复训练 ----------
    start_epoch = 0
    best_mAP = 0.0
    if args.resume is not None:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_mAP = checkpoint.get('best_mAP', 0.0)
            print(f"从检查点恢复训练: {args.resume}, 从 epoch {start_epoch} 继续")
            print(f"恢复的最佳 mAP: {best_mAP:.4f}")
            if cfg.freeze_epochs > 0 and start_epoch >= cfg.freeze_epochs:
                model.set_freeze_backbone(False)
        else:
            print(f"警告: 检查点文件不存在: {args.resume}")

    # ================== 自动创建 run 文件夹 ==================
    base_dir = cfg.save_dir if cfg.save_dir is not None else './output/checkpoints'
    os.makedirs(base_dir, exist_ok=True)

    # 查找已有 run 文件夹
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
    # ========================================================

    # ---------- 准备保存目录 ----------
    log_path = os.path.join(cfg.save_dir, 'training_log.csv')
    config_path = os.path.join(cfg.save_dir, 'config.json')
    
    save_config_json(cfg, config_path)
    if not os.path.exists(log_path):
        init_csv_log(log_path)
    else:
        print(f"日志文件已存在，将追加数据: {log_path}")

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start_time = time.time()
        
        check_and_update_freeze(
            model, epoch, cfg.freeze_epochs, 
            freeze_optimizer=optimizer, 
            unfreeze_lr=cfg.unfreeze_lr
        )
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg, 
            scaler, cfg.grad_clip_norm
        )
        print(f'Epoch {epoch+1}, 训练损失: {train_loss:.4f}')

        do_eval = ((epoch + 1) % cfg.eval_interval == 0) or (epoch == cfg.epochs - 1)
        if do_eval:
            val_loss, ap50, ap95, ar100 = validate_one_epoch(
                model, val_loader, device, epoch, cfg, compute_map=True
            )
            print(f'Epoch {epoch+1}, 验证损失: {val_loss:.4f}, AP@0.5: {ap50:.4f}, AP@0.5:0.95: {ap95:.4f}, AR@100: {ar100:.4f}')
        else:
            val_loss = None
            ap50 = 0.0
            ap95 = 0.0
            ar100 = 0.0
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录日志
        if val_loss is not None:
            append_csv_log(log_path, epoch+1, train_loss, val_loss, ap50, ap95, ar100, epoch_time)
            if ap50 > best_mAP:
                best_mAP = ap50
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'best.pth'))
                print(f'保存最佳模型 (AP@0.5: {ap50:.4f})')
        else:
            append_csv_log(log_path, epoch+1, train_loss, '', '', '', '', epoch_time)
        
        # 保存 last.pth
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mAP': best_mAP,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else None,
        }, os.path.join(cfg.save_dir, 'last.pth'))
        
        if (epoch + 1) % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mAP': best_mAP,
                'train_loss': train_loss,
                'val_loss': val_loss if val_loss is not None else None,
            }, os.path.join(cfg.save_dir, f'epoch_{epoch+1}.pth'))

        combined_scheduler(epoch)

    # ---------- 保存最终模型 ----------
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'final.pth'))
    print(f"训练完成！最佳 AP@0.5: {best_mAP:.4f}")

    # ---------- 绘制训练曲线 ----------
    if not args.no_plot:
        plot_training_curves(log_path, cfg.save_dir)
    else:
        print("训练曲线绘制已禁用 (--no-plot)")


if __name__ == '__main__':
    main()