import os
import logging

# 屏蔽 huggingface_hub 的匿名访问警告
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

import csv
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..base_trainer import BaseTrainer, suppress_stdout
from ...utils.logger import LINE, LINE_EQ, _title, _kv, _bool_en
from ...datasets.cls_dataset import (MultiLabelClassificationDataset, HFMultiLabelClassificationDataset, IMAGENET_MEAN, IMAGENET_STD)


# ============================================================
# 表头 / 描述行（YOLOv8 风格）
# ============================================================
def _make_train_header():
    return "  ".join([
        f"{'Epoch':>11}",
        f"{'GPU_mem':>7}",
        f"{'loss':>9}",
        f"{'lr_bb':>9}",
        f"{'lr_hd':>9}",
        f"{'Instances':>9}",
        f"{'Size':>4}",
    ])


def _make_train_desc(epoch, epochs, gpu_mem, loss, lr_bb, lr_hd, bs, img_size):
    return "  ".join([
        f"{epoch:>5d}/{epochs:<5d}",
        f"{gpu_mem:>7}",
        f"{loss:>9.4f}",
        f"{lr_bb:>9.2e}",
        f"{lr_hd:>9.2e}",
        f"{bs:>9d}",
        f"{img_size:>4d}",
    ])


TABLE_VAL_HEADER = (
    f"{'Class':>17}  {'Images':>6}  "
    f"{'mAP':>9}  {'CF1':>9}  {'OP':>9}  {'OR':>9}  "
    f"{'OF1':>9}  {'EMR':>9}"
)


# ============================================================
# ASL 损失
# ============================================================
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification (ICCV 2021)"""
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-4):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        x_sigmoid = x_sigmoid.clamp(self.eps, 1.0 - self.eps)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos)
        los_neg = (1 - targets) * torch.log(xs_neg)
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        return loss


# ============================================================
# 指标
# ============================================================
def _compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def compute_multilabel_metrics(all_scores, all_targets, threshold=0.5):
    N, C = all_scores.shape
    preds = (all_scores > threshold).astype(np.float32)
    tgt = all_targets.astype(np.float32)

    APs = []
    for c in range(C):
        y = tgt[:, c]
        s = all_scores[:, c]
        n_pos = int(y.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-s)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        recall = tp / (n_pos + 1e-16)
        precision = tp / (tp + fp + 1e-16)
        APs.append(_compute_ap(recall, precision))
    mAP = float(np.mean(APs)) if APs else 0.0

    tp = (preds * tgt).sum(axis=0)
    fp = (preds * (1 - tgt)).sum(axis=0)
    fn = ((1 - preds) * tgt).sum(axis=0)

    precision_c = tp / (tp + fp + 1e-10)
    recall_c = tp / (tp + fn + 1e-10)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-10)
    CF1 = float(f1_c.mean())

    TP = float(tp.sum()); FP = float(fp.sum()); FN = float(fn.sum())
    OP = TP / (TP + FP + 1e-10)
    OR = TP / (TP + FN + 1e-10)
    OF1 = 2 * OP * OR / (OP + OR + 1e-10)

    EMR = float((preds == tgt).all(axis=1).mean())

    return dict(mAP=mAP, CF1=CF1, OP=OP, OR=OR, OF1=OF1, EMR=EMR)


# ============================================================
# 训练器
# ============================================================
class MultiLabelTrainer(BaseTrainer):

    # ------------------------------------------------------------------
    def build_model(self):
        from ...tasks import get_task
        with suppress_stdout():
            model = get_task(self.cfg.task)['model'](self.cfg)
        return model

    # ------------------------------------------------------------------
    def build_dataloaders(self):
        from torch.utils.data import DataLoader

        cfg = self.cfg
        class_names = self._resolve_class_names()

        with suppress_stdout():
            if cfg.hf_dataset:
                from datasets import load_dataset
                hf_dict = load_dataset(cfg.hf_dataset, cfg.hf_config,
                                        cache_dir=cfg.hf_cache_dir)
                train_key = cfg.hf_train_split
                val_key = cfg.hf_val_split
                if train_key not in hf_dict:
                    only = list(hf_dict.keys())[0]
                    tmp = hf_dict[only].train_test_split(test_size=0.1, seed=cfg.seed)
                    hf_train, hf_val = tmp['train'], tmp['test']
                else:
                    hf_train = hf_dict[train_key]
                    hf_val = hf_dict.get(val_key)
                    if hf_val is None:
                        tmp = hf_train.train_test_split(test_size=0.05, seed=cfg.seed)
                        hf_train, hf_val = tmp['train'], tmp['test']

                train_ds = HFMultiLabelClassificationDataset(
                    hf_train, num_classes=cfg.num_classes,
                    class_names=class_names,
                    img_size=cfg.img_size, split='train',
                    use_flip=cfg.use_flip, fliplr=cfg.fliplr,
                    use_color_aug=cfg.use_color_aug,
                    hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
                    use_rrc=cfg.use_rrc, rrc_scale=(cfg.rrc_min_scale, 1.0),
                    use_mixup=cfg.mixup_prob > 0,
                    mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha)
                cfg.num_classes = train_ds.num_classes
                self.class_names = list(train_ds.class_names)

                val_ds = HFMultiLabelClassificationDataset(
                    hf_val, num_classes=cfg.num_classes,
                    class_names=self.class_names,
                    img_size=cfg.img_size, split='val',
                    use_flip=False, use_color_aug=False, use_rrc=False,
                    use_mixup=False)
            else:
                train_ds = MultiLabelClassificationDataset(
                    root=cfg.data_root, label_file=cfg.train_labels,
                    img_subdir=cfg.train_img_subdir, num_classes=cfg.num_classes,
                    img_size=cfg.img_size, split='train',
                    class_names=class_names,
                    use_flip=cfg.use_flip, fliplr=cfg.fliplr,
                    use_color_aug=cfg.use_color_aug,
                    hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
                    use_rrc=cfg.use_rrc, rrc_scale=(cfg.rrc_min_scale, 1.0),
                    use_mixup=cfg.mixup_prob > 0,
                    mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha)
                val_ds = MultiLabelClassificationDataset(
                    root=cfg.data_root, label_file=cfg.val_labels,
                    img_subdir=cfg.val_img_subdir, num_classes=cfg.num_classes,
                    img_size=cfg.img_size, split='val',
                    class_names=class_names,
                    use_flip=False, use_color_aug=False, use_rrc=False,
                    use_mixup=False)
                self.class_names = list(train_ds.class_names)

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=cfg.num_workers > 0)
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=cfg.num_workers > 0)

        with open(os.path.join(self.run_dir, 'class_names.txt'),
                  'w', encoding='utf-8') as f:
            for n in self.class_names:
                f.write(n + '\n')

    # ------------------------------------------------------------------
    def _resolve_class_names(self):
        cfg = self.cfg

        # ---------- 1. 显式类名文件（最高优先级） ----------
        path = cfg.class_names_file
        if not path and cfg.data_root:
            default_path = os.path.join(cfg.data_root, 'class_names.txt')
            if os.path.isfile(default_path):
                path = default_path
        if not path:
            path = cfg.hf_class_names_file

        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                names = [ln.strip() for ln in f if ln.strip()]
            print(f'[MultiLabelTrainer] 从文件读取类名: {path} '
                  f'({len(names)} 个)')
            return names

        # ---------- 2. HF 数据集：交给数据集自己扫 ----------
        if cfg.hf_dataset:
            print(f'[MultiLabelTrainer] HF 数据集场景，类名交给 '
                  f'HFMultiLabelClassificationDataset 自动推断')
            return None

        # ---------- 3. 本地 txt 场景：占位兜底 ----------
        print(f'[MultiLabelTrainer] 使用占位类名 '
              f'cls_0 ~ cls_{cfg.num_classes - 1}')
        return [f'cls_{i}' for i in range(cfg.num_classes)]

    # ------------------------------------------------------------------
    def build_optimizer(self):
        bb, hd = [], []
        for n, p in self.model.named_parameters():
            (bb if n.startswith('backbone.') else hd).append(p)
        groups = [
            {'params': bb, 'lr': self.cfg.lr * self.cfg.backbone_lr_scale,
             'name': 'backbone'},
            {'params': hd, 'lr': self.cfg.lr, 'name': 'head'},
        ]
        if self.cfg.optimizer == 'adamw':
            return optim.AdamW(groups, lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        return optim.SGD(groups, lr=self.cfg.lr, momentum=0.9,
                         weight_decay=self.cfg.weight_decay, nesterov=True)

    def build_criterion(self):
        if self.cfg.loss == 'asl':
            return AsymmetricLoss(
                gamma_neg=self.cfg.asl_gamma_neg,
                gamma_pos=self.cfg.asl_gamma_pos,
                clip=self.cfg.asl_clip)
        return nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # 完整配置块
    # ------------------------------------------------------------------
    def _print_header(self):
        cfg = self.cfg

        print()
        print(LINE_EQ)
        print("  SWAP-Fighter Training Configuration")
        print(LINE_EQ)

        _title("Project")
        _kv("Name", "SwapFighter")
        _kv("URL", "https://github.com/IntptrMax/SwapFighter")
        _kv("Author", "IntptrMax")
        _kv("Description", "SWAP-Fighter: multi-label classification")

        _title("Dataset")
        _kv("Source", "HuggingFace" if cfg.hf_dataset else "local txt")
        if cfg.hf_dataset:
            _kv("HF Dataset", cfg.hf_dataset)
            if cfg.hf_config:
                _kv("HF Config", cfg.hf_config)
            _kv("HF Train Split", cfg.hf_train_split)
            _kv("HF Val Split", cfg.hf_val_split)
        else:
            _kv("Root", cfg.data_root)
            _kv("Train / Val Labels",
                f"{cfg.train_labels} / {cfg.val_labels}")
            _kv("Train / Val Img",
                f"{cfg.train_img_subdir} / {cfg.val_img_subdir}")
        _kv("Class Names File",
            cfg.class_names_file
            if cfg.class_names_file
            else f"{{data_root}}/class_names.txt (默认)")
        _kv("Num Classes", cfg.num_classes)
        train_len = (len(self.train_loader.dataset)
                     if hasattr(self, 'train_loader') else '?')
        val_len = (len(self.val_loader.dataset)
                   if hasattr(self, 'val_loader') else '?')
        _kv("Train / Val", f"{train_len} / {val_len}")
        _kv("Image Size", cfg.img_size)
        names = getattr(self, 'class_names', None)
        if names:
            preview = names[:5]
            _kv("Class Names",
                f"{preview}{' ...' if len(names) > 5 else ''}")

        _title("Model")
        _kv("Backbone", cfg.backbone_name)
        _kv("Pretrained",
            f"{cfg.pretrained} ({cfg.backbone_pretrained_source})")
        if hasattr(self.model, 'num_features'):
            _kv("Num Features", self.model.num_features)
        _kv("Pool Type", cfg.pool_type)
        _kv("Head Type", cfg.head_type)
        _kv("Multi Scale", _bool_en(cfg.multi_scale))
        _kv("Dropout", cfg.dropout)
        _kv("Hidden Dim",
            cfg.hidden_dim if cfg.hidden_dim > 0 else "=num_features")
        _kv("Drop Path", cfg.drop_path_rate)
        _kv("Grad Checkpointing", _bool_en(cfg.grad_checkpointing))

        _title("Training")
        _kv("Device / Precision",
            f"{self.device} / {self.actual_precision.upper()}")
        _kv("Batch / Accum / Effective",
            f"{cfg.batch_size} / {cfg.accum_steps} / "
            f"{cfg.batch_size * cfg.accum_steps}")
        _kv("Epochs", cfg.epochs)
        _kv("LR (head / backbone)",
            f"{cfg.lr:.2e} / {cfg.lr * cfg.backbone_lr_scale:.2e}")
        _kv("Weight Decay", cfg.weight_decay)
        _kv("Optimizer / Scheduler",
            f"{cfg.optimizer} / {cfg.scheduler}")
        _kv("Warmup", f"{cfg.warmup_epochs} epochs")
        _kv("Grad Clip", cfg.grad_clip_norm)
        _kv("Loss", cfg.loss)
        if cfg.loss == 'asl':
            _kv("ASL Params",
                f"gamma_neg={cfg.asl_gamma_neg}, "
                f"gamma_pos={cfg.asl_gamma_pos}, clip={cfg.asl_clip}")
        _kv("EMA",
            f"enabled (decay={cfg.ema_decay})" if cfg.use_ema else "disabled")
        _kv("Freeze Epochs", cfg.freeze_epochs)
        _kv("Patience",
            f"{cfg.patience} (0 = disabled)" if cfg.patience else "disabled")

        _title("Augmentation")
        _kv("Flip", f"use={_bool_en(cfg.use_flip)}, fliplr={cfg.fliplr}")
        _kv("Color",
            f"use={_bool_en(cfg.use_color_aug)}, "
            f"hsv=({cfg.hsv_h},{cfg.hsv_s},{cfg.hsv_v})")
        _kv("RRC",
            f"use={_bool_en(cfg.use_rrc)}, min_scale={cfg.rrc_min_scale}")
        _kv("Mixup",
            f"prob={cfg.mixup_prob}, alpha={cfg.mixup_alpha}")

        _title("Evaluation")
        _kv("Threshold", cfg.eval_threshold)
        _kv("TTA", _bool_en(cfg.eval_tta))
        _kv("Eval Interval", cfg.eval_interval)

        _title("Output")
        _kv("Save Dir", self.run_dir)

        print()
        print(LINE_EQ)

    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_mem_str():
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.memory_reserved() / (1024 ** 3)
                return f"{mem:.2f}G"
            except Exception:
                return "0.00G"
        return "cpu"

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()
        total, n = 0.0, 0
        accum = cfg.accum_steps

        print()
        print(_make_train_header())

        pbar = tqdm(
            self.train_loader,
            bar_format='{desc}: {percentage:3.0f}% {bar:30} '
                       '{n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
            leave=True,
            ncols=160,
            dynamic_ncols=False,
        )

        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels) / accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            is_update = ((step + 1) % accum == 0) or \
                        (step + 1 == len(self.train_loader))
            if is_update:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema_model is not None:
                    self._update_ema()

            bs = images.size(0)
            total += loss.item() * bs * accum
            n += bs

            lr_bb = self.optimizer.param_groups[0]['lr']
            lr_hd = (self.optimizer.param_groups[1]['lr']
                     if len(self.optimizer.param_groups) > 1 else lr_bb)
            gpu_mem = self._gpu_mem_str()
            desc = _make_train_desc(
                epoch + 1, cfg.epochs, gpu_mem,
                loss.item() * accum, lr_bb, lr_hd, bs, cfg.img_size)
            pbar.set_description_str(desc, refresh=True)

        pbar.close()
        return {'train_loss': total / max(n, 1)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        all_scores, all_targets = [], []

        pbar = tqdm(
            self.val_loader,
            desc=TABLE_VAL_HEADER,
            bar_format='{desc}: {percentage:3.0f}% {bar:30} '
                       '{n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
            leave=True,
            ncols=160,
            dynamic_ncols=False,
        )
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                logits = eval_model(images)
                probs = torch.sigmoid(logits.float())
                if self.cfg.eval_tta:
                    logits_flip = eval_model(torch.flip(images, dims=[3]))
                    probs_flip = torch.sigmoid(logits_flip.float())
                    probs = (probs + probs_flip) / 2.0
            all_scores.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
        pbar.close()

        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        m = compute_multilabel_metrics(
            all_scores, all_targets, threshold=self.cfg.eval_threshold)

        n_img = len(all_scores)
        print(f"{'all':>17}  {n_img:>6d}  "
              f"{m['mAP']:>9.4f}  {m['CF1']:>9.4f}  "
              f"{m['OP']:>9.4f}  {m['OR']:>9.4f}  "
              f"{m['OF1']:>9.4f}  {m['EMR']:>9.4f}")

        return m

    # ------------------------------------------------------------------
    def save_checkpoint(self, path, **extra):
        save_model = self.ema_model if self.ema_model is not None else self.model
        meta = self._build_meta()
        torch.save({
            'model_state_dict': save_model.state_dict(),
            'model_meta': meta,
            **extra,
        }, path)

    def _build_meta(self):
        cfg = self.cfg
        d = cfg.to_dict()
        d.update({
            'project_name': 'SwapFighter',
            'project_url': 'https://github.com/IntptrMax/SwapFighter',
            'project_author': 'IntptrMax',
            'task': 'multilabel',
            'mode': 'multilabel',
            'num_classes': cfg.num_classes,
            'backbone_name': cfg.backbone_name,
            'class_names': getattr(self, 'class_names', None),
            'mean': list(IMAGENET_MEAN),
            'std': list(IMAGENET_STD),
        })
        return d

    def _update_ema(self):
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.cfg.ema_decay).add_(
                        msd[k].detach(), alpha=1.0 - self.cfg.ema_decay)
                else:
                    v.copy_(msd[k])