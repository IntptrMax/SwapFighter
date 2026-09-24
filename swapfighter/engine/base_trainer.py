import os
import sys
import io
import csv
import json
import copy
import math
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import torch

from ..utils.logger import LINE, LINE_EQ, _title, _kv


@contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


class BaseTrainer(ABC):
    def __init__(self, cfg_dict):
        from ..cfg import Config
        task = cfg_dict.pop('task', 'detect')
        device_pref = cfg_dict.pop('device', 'cuda')

        self.cfg = Config.from_kwargs(task, **cfg_dict)

        if device_pref == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._set_seed()
        self._setup_amp()
        self._setup_run_dir()

        self.model = None
        self.ema_model = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_type = 'epoch'
        self.scaler = None
        self.start_epoch = 0
        self.global_step = 0
        self.best_fitness = -1.0
        self.best_epoch = 0

    # ------------------------------------------------------------------
    @abstractmethod
    def build_model(self): ...
    @abstractmethod
    def build_dataloaders(self): ...
    @abstractmethod
    def build_optimizer(self): ...
    @abstractmethod
    def build_criterion(self): ...
    @abstractmethod
    def train_one_epoch(self, epoch): ...
    @abstractmethod
    def validate(self, epoch): ...
    @abstractmethod
    def save_checkpoint(self, path, **extra): ...

    # ------------------------------------------------------------------
    def build_ema(self):
        if not self.cfg.use_ema:
            return
        self.ema_model = copy.deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def build_scheduler(self):
        from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
        cfg = self.cfg
        epochs = cfg.epochs
        warmup = cfg.warmup_epochs
        steps_per_epoch = max(len(self.train_loader), 1)

        if cfg.scheduler == 'onecycle':
            total_steps = epochs * steps_per_epoch
            max_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=max_lrs, total_steps=total_steps,
                pct_start=max(0.05, warmup / max(epochs, 1)),
                anneal_strategy='cos')
            self.scheduler_type = 'step'
            return

        def make_lambda(base_fn):
            def lr_lambda(epoch):
                if warmup > 0 and epoch < warmup:
                    return float(epoch + 1) / warmup
                return base_fn(epoch - warmup, max(epochs - warmup, 1))
            return lr_lambda

        if cfg.scheduler == 'cosine':
            self.scheduler = LambdaLR(self.optimizer, make_lambda(
                lambda e, t: 0.5 * (1.0 + math.cos(math.pi * e / t))))
        elif cfg.scheduler == 'lambda':
            self.scheduler = LambdaLR(self.optimizer, make_lambda(
                lambda e, t: cfg.scheduler_gamma ** e))
        elif cfg.scheduler == 'step':
            self.scheduler = LambdaLR(self.optimizer, make_lambda(
                lambda e, t: cfg.scheduler_gamma ** (e // max(cfg.scheduler_step_size, 1))))
        elif cfg.scheduler == 'multistep':
            ms = sorted(cfg.scheduler_milestones or [])
            self.scheduler = LambdaLR(self.optimizer, make_lambda(
                lambda e, t: cfg.scheduler_gamma ** sum(1 for m in ms if e >= m)))
        else:
            raise ValueError(f"不支持的 scheduler: {cfg.scheduler}")
        self.scheduler_type = 'epoch'

    # ------------------------------------------------------------------
    def _set_seed(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.backends.cudnn.benchmark = True

    def _setup_amp(self):
        cfg = self.cfg
        if cfg.precision == 'fp32' or self.device.type != 'cuda':
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.scaler = None
        elif cfg.precision == 'bf16':
            self.amp_dtype = torch.bfloat16
            self.use_amp = True
            self.scaler = None
        else:
            self.amp_dtype = torch.float16
            self.use_amp = True
            self.scaler = torch.amp.GradScaler('cuda')
        self.actual_precision = cfg.precision if self.use_amp else 'fp32'

    def _setup_run_dir(self):
        cfg = self.cfg
        base = cfg.save_dir
        os.makedirs(base, exist_ok=True)

        existing = [int(d[3:]) for d in os.listdir(base)
                    if d.startswith('run') and d[3:].isdigit()]
        nxt = max(existing) + 1 if existing else 1
        self.run_dir = os.path.join(base, f'run{nxt}')
        os.makedirs(self.run_dir, exist_ok=True)
        cfg.save_dir = self.run_dir

        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(cfg.to_dict(), f, indent=4, ensure_ascii=False, default=str)

        # ★ 初始化 CSV 日志
        self.log_path = os.path.join(self.run_dir, 'training_log.csv')
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'AP50', 'AP95', 'AR100',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ])

    # ------------------------------------------------------------------
    def _append_log(self, epoch, train_loss, val_loss,
                    ap50, ap95, ar100, dt, lr_bb, lr_nh, fitness):
        """把一行写到 CSV。数值转成 Python float / int 兼容 None。"""
        def _fmt(v):
            if v is None or v == '':
                return ''
            if isinstance(v, torch.Tensor):
                return float(v.item())
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                int(epoch), _fmt(train_loss), _fmt(val_loss),
                _fmt(ap50), _fmt(ap95), _fmt(ar100),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ])

    # ------------------------------------------------------------------
    def run(self):
        self.model = self.build_model().to(self.device)
        self.build_dataloaders()
        self.build_ema()
        self.optimizer = self.build_optimizer()
        self.build_scheduler()
        self.criterion = self.build_criterion()

        self._print_header()

        total_epochs = self.cfg.epochs
        for epoch in range(self.start_epoch, total_epochs):
            t0 = time.time()
            self.model.train()

            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)

            train_metrics = self.train_one_epoch(epoch)

            if self.scheduler_type == 'epoch':
                self.scheduler.step()

            do_eval = ((epoch + 1) % self.cfg.eval_interval == 0) or \
                      (epoch == total_epochs - 1)
            val_metrics = self.validate(epoch) if do_eval else {}

            fitness = self._compute_fitness(val_metrics) if val_metrics else 0.0
            dt = time.time() - t0

            # ★ 写日志
            lr_bb = self.optimizer.param_groups[0]['lr']
            lr_nh = (self.optimizer.param_groups[1]['lr']
                     if len(self.optimizer.param_groups) > 1 else lr_bb)
            self._append_log(
                epoch=epoch + 1,
                train_loss=train_metrics.get('train_loss'),
                val_loss=val_metrics.get('val_loss', '') if val_metrics else '',
                ap50=val_metrics.get('mAP50', '') if val_metrics else '',
                ap95=val_metrics.get('mAP50_95', '') if val_metrics else '',
                ar100=val_metrics.get('AR100', '') if val_metrics else '',
                dt=dt,
                lr_bb=lr_bb,
                lr_nh=lr_nh,
                fitness=fitness if val_metrics else '',
            )

            self._on_epoch_end(epoch, train_metrics, val_metrics, fitness, dt)

            if val_metrics and self._is_best(fitness):
                self._update_best(epoch, fitness)
                self.save_checkpoint(
                    os.path.join(self.run_dir, 'best.pth'),
                    epoch=epoch + 1, metrics=val_metrics, fitness=fitness)

            self.save_checkpoint(
                os.path.join(self.run_dir, 'last.pth'),
                epoch=epoch + 1, metrics=val_metrics, fitness=fitness)

            if self._should_stop(epoch):
                print(f"\n  早停：{self.cfg.patience} epochs 无提升")
                break

        self._on_train_end()
        print(f"\n  训练完成。输出目录: {self.run_dir}")

        # ★ 绘制训练曲线
        self._plot_training_curves()

    # ------------------------------------------------------------------
    def _plot_training_curves(self):
        """从 training_log.csv 读取数据，画 2×3 六联图并保存。"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  警告: matplotlib 未安装，跳过绘制训练曲线。"
                  "请执行 pip install matplotlib")
            return

        if not os.path.exists(self.log_path):
            return

        # ---------- 读 CSV ----------
        epochs, train_loss, val_loss = [], [], []
        ap50, ap95, ar100 = [], [], []
        lr_bb, lr_nh = [], []

        with open(self.log_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 8:
                    continue
                try:
                    epochs.append(int(row[0]))
                    train_loss.append(float(row[1]) if row[1] else np.nan)
                    val_loss.append(float(row[2]) if row[2] else np.nan)
                    ap50.append(float(row[3]) if row[3] else np.nan)
                    ap95.append(float(row[4]) if row[4] else np.nan)
                    ar100.append(float(row[5]) if row[5] else np.nan)
                    lr_bb.append(float(row[7]) if row[7] else np.nan)
                    lr_nh.append(float(row[8]) if len(row) > 8 and row[8] else np.nan)
                except (ValueError, IndexError):
                    continue

        if not epochs:
            print("  日志无有效数据，跳过绘制训练曲线。")
            return

        epochs = np.array(epochs)
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        ap50 = np.array(ap50)
        ap95 = np.array(ap95)
        ar100 = np.array(ar100)
        lr_bb = np.array(lr_bb)
        lr_nh = np.array(lr_nh)

        def _smooth(y, f=0.05):
            y = np.asarray(y, dtype=np.float64)
            n = len(y)
            if n < 3:
                return y.copy()
            nf = round(n * f * 2) // 2 + 1
            if nf % 2 == 0:
                nf += 1
            if nf < 3 or nf > n:
                return y.copy()
            p = np.ones(nf // 2)
            yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
            out = np.convolve(yp, np.ones(nf) / nf, mode='valid')
            if len(out) > n:
                out = out[:n]
            elif len(out) < n:
                out = np.concatenate([out, [out[-1]] * (n - len(out))])
            return out

        # ---------- 六联图 ----------
        panels = [
            ('train/loss', 'train loss', train_loss),
            ('val/loss', 'val loss', val_loss),
            ('metrics/lr_bb', 'backbone lr', lr_bb),
            ('metrics/mAP50', 'mAP@0.5', ap50),
            ('metrics/mAP50-95', 'mAP@0.5:0.95', ap95),
            ('metrics/AR100', 'AR@100', ar100),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, (title, ylabel, y) in enumerate(panels):
            ax = axes[i]
            mask = ~np.isnan(y)
            x_v = epochs[mask]
            y_v = y[mask]

            if len(y_v) > 0:
                ax.plot(x_v, y_v, 'o-', color='#1f77b4',
                        markersize=5, linewidth=1.2, label='results')
                if len(y_v) > 1:
                    ys = _smooth(y_v, f=0.05)
                    xs = x_v[:len(ys)]
                    ax.plot(xs, ys, '--', color='#ff7f0e',
                            linewidth=2.2, label='smooth')

            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

            if i == 0:
                ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, 'results.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  训练曲线已保存至: {plot_path}")

    # ------------------------------------------------------------------
    def _compute_fitness(self, metrics: dict) -> float:
        if 'mAP50' in metrics and 'mAP50_95' in metrics:
            return 0.1 * metrics['mAP50'] + 0.9 * metrics['mAP50_95']
        return metrics.get('mAP', 0.0)

    def _is_best(self, fitness: float) -> bool:
        return fitness > self.best_fitness

    def _update_best(self, epoch: int, fitness: float):
        self.best_fitness = fitness
        self.best_epoch = epoch + 1

    def _should_stop(self, epoch: int) -> bool:
        if self.cfg.patience <= 0:
            return False
        return (epoch + 1) - self.best_epoch >= self.cfg.patience

    def _on_epoch_end(self, epoch, train_m, val_m, fitness, dt):
        pass

    def _on_train_end(self):
        pass

    def _print_header(self):
        print()
        print(LINE_EQ)
        print(f"  SWAP-Fighter Training — {self.cfg.task.upper()}")
        print(LINE_EQ)
        _title("Config")
        _kv("Task", self.cfg.task)
        _kv("Backbone", self.cfg.backbone_name)
        _kv("Image Size", self.cfg.img_size)
        _kv("Device / Precision", f"{self.device} / {self.actual_precision}")
        _kv("Epochs / Batch", f"{self.cfg.epochs} / {self.cfg.batch_size}")
        _kv("Save Dir", self.run_dir)
        print(LINE_EQ)

    def validate_only(self):
        self.model = self.build_model().to(self.device)
        self.build_dataloaders()
        best = os.path.join(self.cfg.save_dir, 'best.pth')
        if os.path.exists(best):
            ckpt = torch.load(best, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model.eval()
        return self.validate(0)