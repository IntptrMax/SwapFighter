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

    def build_ema(self):
        if not self.cfg.use_ema:
            return

        if self.cfg.ema_decay is None:
            auto_decay = self._auto_ema_decay()
            self.cfg.ema_decay = auto_decay

        accum = max(1, getattr(self.cfg, 'accum_steps', 1))
        steps_per_epoch = max(1, len(self.train_loader) // accum)
        total_steps = steps_per_epoch * self.cfg.epochs
        self.ema_tau = max(100, int(total_steps * 0.05))

        self.ema_model = copy.deepcopy(self.model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def _auto_ema_decay(self):
        accum = max(1, getattr(self.cfg, 'accum_steps', 1))
        steps_per_epoch = max(1, len(self.train_loader) // accum)
        total_steps = steps_per_epoch * self.cfg.epochs

        target_window = int(total_steps / 3)
        target_window = max(50, min(50000, target_window))

        decay = 1.0 - 1.0 / target_window
        decay = max(0.90, min(0.99998, decay))
        return decay

    def _current_ema_decay(self, global_step):
        base = self.cfg.ema_decay
        tau = getattr(self, 'ema_tau', 2000)
        return base * (1.0 - math.exp(-global_step / tau))

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
                lambda e, t: cfg.scheduler_gamma
                ** (e // max(cfg.scheduler_step_size, 1))))
        elif cfg.scheduler == 'multistep':
            ms = sorted(cfg.scheduler_milestones or [])
            self.scheduler = LambdaLR(self.optimizer, make_lambda(
                lambda e, t: cfg.scheduler_gamma
                ** sum(1 for m in ms if e >= m)))
        else:
            raise ValueError(f"不支持的 scheduler: {cfg.scheduler}")
        self.scheduler_type = 'epoch'

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
            json.dump(cfg.to_dict(), f, indent=4,
                      ensure_ascii=False, default=str)

        self.log_path = os.path.join(self.run_dir, 'training_log.csv')

        task = cfg.task
        if task in ('multilabel', 'linear_probe'):
            header = [
                'epoch', 'train_loss', 'val_loss',
                'mAP', 'CF1', 'OP', 'OR', 'OF1', 'EMR',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]
        elif task == 'detect':
            header = [
                'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
                'val_loss',
                'AP50', 'AP95', 'AR100',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]
        elif task == 'obb':
            header = [
                'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
                'angle_loss',
                'val_loss',
                'AP50', 'AP95', 'AR100',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]
        elif task == 'segment':
            header = [
                'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
                'mask_loss',
                'val_loss',
                'AP50', 'AP95', 'AR100',
                'mask_AP50', 'mask_AP95',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]
        elif task == 'keypoint':
            header = [
                'epoch', 'train_loss', 'box_loss', 'cls_loss', 'dfl_loss',
                'kpt_loss',
                'val_loss',
                'AP50', 'AP95', 'AR100',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]
        else:
            header = [
                'epoch', 'train_loss', 'val_loss',
                'time_sec', 'lr_bb', 'lr_nh', 'fitness',
            ]

        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    def _append_log(self, epoch, train_loss, val_loss,
                    dt, lr_bb, lr_nh, fitness,

                    box_loss=None, cls_loss=None, dfl_loss=None,
                    angle_loss=None, mask_loss=None, kpt_loss=None,
                    ap50=None, ap95=None, ar100=None,
                    mask_ap50=None, mask_ap95=None,

                    mAP=None, CF1=None, OP=None, OR=None, OF1=None, EMR=None):
        def _fmt(v):
            if v is None or v == '':
                return ''
            if isinstance(v, torch.Tensor):
                return float(v.item())
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        task = self.cfg.task

        if task in ('multilabel', 'linear_probe'):
            row = [
                int(epoch), _fmt(train_loss), _fmt(val_loss),
                _fmt(mAP), _fmt(CF1), _fmt(OP), _fmt(OR),
                _fmt(OF1), _fmt(EMR),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]
        elif task == 'detect':
            row = [
                int(epoch), _fmt(train_loss),
                _fmt(box_loss), _fmt(cls_loss), _fmt(dfl_loss),
                _fmt(val_loss),
                _fmt(ap50), _fmt(ap95), _fmt(ar100),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]
        elif task == 'obb':
            row = [
                int(epoch), _fmt(train_loss),
                _fmt(box_loss), _fmt(cls_loss), _fmt(dfl_loss),
                _fmt(angle_loss),
                _fmt(val_loss),
                _fmt(ap50), _fmt(ap95), _fmt(ar100),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]
        elif task == 'segment':
            row = [
                int(epoch), _fmt(train_loss),
                _fmt(box_loss), _fmt(cls_loss), _fmt(dfl_loss),
                _fmt(mask_loss),
                _fmt(val_loss),
                _fmt(ap50), _fmt(ap95), _fmt(ar100),
                _fmt(mask_ap50), _fmt(mask_ap95),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]
        elif task == 'keypoint':
            row = [
                int(epoch), _fmt(train_loss),
                _fmt(box_loss), _fmt(cls_loss), _fmt(dfl_loss),
                _fmt(kpt_loss),
                _fmt(val_loss),
                _fmt(ap50), _fmt(ap95), _fmt(ar100),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]
        else:
            row = [
                int(epoch), _fmt(train_loss), _fmt(val_loss),
                _fmt(dt), _fmt(lr_bb), _fmt(lr_nh), _fmt(fitness),
            ]

        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

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

            lr_bb = self.optimizer.param_groups[0]['lr']
            lr_nh = (self.optimizer.param_groups[1]['lr']
                     if len(self.optimizer.param_groups) > 1 else lr_bb)

            v = val_metrics or {}
            self._append_log(
                epoch=epoch + 1,
                train_loss=train_metrics.get('train_loss'),
                val_loss=v.get('val_loss', ''),
                dt=dt, lr_bb=lr_bb, lr_nh=lr_nh,
                fitness=fitness if val_metrics else '',

                box_loss=train_metrics.get('train_box_loss'),
                cls_loss=train_metrics.get('train_cls_loss'),
                dfl_loss=train_metrics.get('train_dfl_loss'),
                angle_loss=train_metrics.get('train_angle_loss'),
                mask_loss=train_metrics.get('train_mask_loss'),
                kpt_loss=train_metrics.get('train_kpt_loss'),
                ap50=v.get('mAP50', ''),
                ap95=v.get('mAP50_95', ''),
                ar100=v.get('AR100', ''),
                mask_ap50=v.get('mask_mAP50', ''),
                mask_ap95=v.get('mask_mAP50_95', ''),

                mAP=v.get('mAP', ''),
                CF1=v.get('CF1', ''),
                OP=v.get('OP', ''),
                OR=v.get('OR', ''),
                OF1=v.get('OF1', ''),
                EMR=v.get('EMR', ''),
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

        self._plot_training_curves()

    def _plot_training_curves(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  警告: matplotlib 未安装，跳过绘制训练曲线。")
            return

        if not os.path.exists(self.log_path):
            return

        data = {}
        with open(self.log_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return
            for h in header:
                data[h] = []
            for row in reader:
                if len(row) < len(header):
                    continue
                for i, h in enumerate(header):
                    v = row[i] if i < len(row) else ''
                    data[h].append(v)

        if len(data.get('epoch', [])) == 0:
            print("  日志无有效数据，跳过绘制训练曲线。")
            return

        def _to_float(vals):
            out = []
            for v in vals:
                if v is None or v == '':
                    out.append(np.nan)
                else:
                    try:
                        out.append(float(v))
                    except (ValueError, TypeError):
                        out.append(np.nan)
            return np.array(out, dtype=np.float64)

        def _has_valid(arr):
            return len(arr) > 0 and not np.all(np.isnan(arr))

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

        epochs = _to_float(data.get('epoch', []))

        task = self.cfg.task

        def _get(key):
            return _to_float(data.get(key, []))

        if task in ('multilabel', 'linear_probe'):
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
                ('metrics/mAP', 'mAP', _get('mAP')),
                ('metrics/CF1', 'CF1', _get('CF1')),
                ('metrics/OP', 'OP', _get('OP')),
                ('metrics/OR', 'OR', _get('OR')),
                ('metrics/OF1', 'OF1', _get('OF1')),
                ('metrics/EMR', 'EMR', _get('EMR')),
            ]
        elif task == 'detect':
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('train/box_loss', 'box loss', _get('box_loss')),
                ('train/cls_loss', 'cls loss', _get('cls_loss')),
                ('train/dfl_loss', 'dfl loss', _get('dfl_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
                ('metrics/mAP50', 'mAP@0.5', _get('AP50')),
                ('metrics/mAP50-95', 'mAP@0.5:0.95', _get('AP95')),
                ('metrics/AR100', 'AR@100', _get('AR100')),
            ]
        elif task == 'obb':
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('train/box_loss', 'box loss', _get('box_loss')),
                ('train/cls_loss', 'cls loss', _get('cls_loss')),
                ('train/dfl_loss', 'dfl loss', _get('dfl_loss')),
                ('train/angle_loss', 'angle loss', _get('angle_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
                ('metrics/mAP50', 'mAP@0.5', _get('AP50')),
                ('metrics/mAP50-95', 'mAP@0.5:0.95', _get('AP95')),
                ('metrics/AR100', 'AR@100', _get('AR100')),
            ]
        elif task == 'segment':
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('train/box_loss', 'box loss', _get('box_loss')),
                ('train/cls_loss', 'cls loss', _get('cls_loss')),
                ('train/dfl_loss', 'dfl loss', _get('dfl_loss')),
                ('train/mask_loss', 'mask loss', _get('mask_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
                ('metrics/mAP50', 'mAP@0.5', _get('AP50')),
                ('metrics/mAP50-95', 'mAP@0.5:0.95', _get('AP95')),
                ('metrics/AR100', 'AR@100', _get('AR100')),
                ('metrics/mask_mAP50', 'mask mAP@0.5', _get('mask_AP50')),
                ('metrics/mask_mAP50-95', 'mask mAP@0.5:0.95',
                 _get('mask_AP95')),
            ]
        elif task == 'keypoint':
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('train/box_loss', 'box loss', _get('box_loss')),
                ('train/cls_loss', 'cls loss', _get('cls_loss')),
                ('train/dfl_loss', 'dfl loss', _get('dfl_loss')),
                ('train/kpt_loss', 'kpt loss', _get('kpt_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
                ('metrics/mAP50', 'mAP@0.5', _get('AP50')),
                ('metrics/mAP50-95', 'mAP@0.5:0.95', _get('AP95')),
                ('metrics/AR100', 'AR@100', _get('AR100')),
            ]
        else:
            candidates = [
                ('train/loss', 'train loss', _get('train_loss')),
                ('val/loss', 'val loss', _get('val_loss')),
                ('metrics/lr_bb', 'backbone lr', _get('lr_bb')),
            ]

        panels = [(t, yl, y) for (t, yl, y) in candidates
                  if _has_valid(y)]

        if len(panels) == 0:
            print("  日志无有效数据，跳过绘制训练曲线。")
            return

        n = len(panels)
        if n <= 6:
            nrows, ncols = 2, 3
        elif n <= 9:
            nrows, ncols = 3, 3
        elif n <= 12:
            nrows, ncols = 3, 4
        else:
            nrows = (n + 3) // 4
            ncols = 4

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6 * ncols, 5 * nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for i in range(n, len(axes)):
            axes[i].axis('off')

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

    def _compute_fitness(self, metrics: dict) -> float:
        task = self.cfg.task

        if task in ('multilabel', 'linear_probe'):
            return float(metrics.get('mAP', 0.0))

        if task == 'segment':
            mask_ap50 = metrics.get('mask_mAP50')
            mask_ap95 = metrics.get('mask_mAP50_95')
            if mask_ap50 is not None and mask_ap95 is not None:
                return 0.1 * float(mask_ap50) + 0.9 * float(mask_ap95)

            return (0.1 * float(metrics.get('mAP50', 0.0))
                    + 0.9 * float(metrics.get('mAP50_95', 0.0)))

        if task == 'keypoint':
            bmAP50 = float(metrics.get('mAP50', 0.0))
            bmAP95 = float(metrics.get('mAP50_95', 0.0))
            pmAP50 = float(metrics.get('pose_mAP50', 0.0))
            pmAP95 = float(metrics.get('pose_mAP50_95', 0.0))

            return (0.4 * (0.1 * bmAP50 + 0.9 * bmAP95)
                    + 0.6 * (0.1 * pmAP50 + 0.9 * pmAP95))

        if 'mAP50' in metrics and 'mAP50_95' in metrics:
            return 0.1 * float(metrics['mAP50']) + \
                   0.9 * float(metrics['mAP50_95'])
        return float(metrics.get('mAP50_95', 0.0))

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
        _kv("Device / Precision",
            f"{self.device} / {self.actual_precision}")
        _kv("Epochs / Batch",
            f"{self.cfg.epochs} / {self.cfg.batch_size}")
        _kv("Save Dir", self.run_dir)
        print(LINE_EQ)

    def validate_only(self):
        self.model = self.build_model().to(self.device)
        self.build_dataloaders()
        best = os.path.join(self.cfg.save_dir, 'best.pth')
        if os.path.exists(best):
            ckpt = torch.load(best, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        self.model.eval()
        return self.validate(0)