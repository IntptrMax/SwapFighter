import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ..base_trainer import BaseTrainer, suppress_stdout
from ...utils.logger import LINE, LINE_EQ, _title, _kv, _bool_en
from ...utils.losses import compute_fcos_loss
from ...utils.metrics import evaluate_mAP, evaluate_map_obb_rotated


# ============================================================
# 模块级 collate_fn（多进程必须可 pickle）
# ============================================================
def _detect_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    return imgs, tgts


# ============================================================
# YOLOv8 风格表头
# ============================================================
TABLE_VAL_HEADER = (
    f"{'Class':>17}  {'Images':>6}  {'Instances':>9}  "
    f"{'Box(P':>9}  {'R':>9}  {'mAP50':>9}  {'mAP50-95)':>11}"
)


def _make_train_header(cfg):
    cols = [f"{'Epoch':>11}", f"{'GPU_mem':>7}",
            f"{'box_loss':>9}", f"{'cls_loss':>9}"]
    if cfg.reg_max > 0:
        cols.append(f"{'dfl_loss':>9}")
    if cfg.mode == 'obb':
        cols.append(f"{'angle_loss':>10}")
    cols += [f"{'Instances':>9}", f"{'Size':>5}"]
    return "  ".join(cols)


def _make_train_desc(epoch, epochs, gpu_mem, box_v, cls_v, dfl_v, ang_v,
                     n_inst, img_size, cfg):
    cols = [f"{epoch:>5d}/{epochs:<5d}", f"{gpu_mem:>7}",
            f"{box_v:>9.3f}", f"{cls_v:>9.3f}"]
    if cfg.reg_max > 0:
        cols.append(f"{dfl_v:>9.3f}")
    if cfg.mode == 'obb':
        cols.append(f"{ang_v:>10.3f}")
    cols += [f"{n_inst:>9d}", f"{img_size:>5d}"]
    return "  ".join(cols)


class DetectTrainer(BaseTrainer):

    # ------------------------------------------------------------------
    def build_model(self):
        from ...tasks import get_task
        with suppress_stdout():
            model = get_task(self.cfg.task)['model'](self.cfg)
        return model

    # ------------------------------------------------------------------
    def build_dataloaders(self):
        from torch.utils.data import DataLoader
        from ...tasks import get_task
        ds_cls = get_task(self.cfg.task)['dataset_cls']

        with suppress_stdout():
            train_ds = ds_cls(
                root=self.cfg.data_root, img_size=self.cfg.img_size,
                split=self.cfg.train_split, mode=self.cfg.mode,
                **_ds_kwargs(self.cfg))
            val_ds = ds_cls(
                root=self.cfg.data_root, img_size=self.cfg.img_size,
                split=self.cfg.val_split, mode=self.cfg.mode,
                use_flip=False, use_color_aug=False)

        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=_detect_collate_fn,
            pin_memory=True, drop_last=True,
            persistent_workers=self.cfg.num_workers > 0)
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=_detect_collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0)

        self.cfg.strides = self.model.strides

        self._val_n_images = len(val_ds)
        self._val_n_instances = self._count_val_instances(val_ds)

    @staticmethod
    def _count_val_instances(val_ds):
        n = 0
        label_dir = val_ds.label_dir
        for base in val_ds.img_files:
            p = os.path.join(label_dir, f'{base}.txt')
            if os.path.isfile(p):
                with open(p, 'r') as f:
                    n += sum(1 for line in f if line.strip())
        return n

    # ------------------------------------------------------------------
    def build_optimizer(self):
        bb, other = [], []
        for n, p in self.model.named_parameters():
            (bb if n.startswith('backbone.') else other).append(p)
        groups = [
            {'params': bb,
             'lr': self.cfg.lr * self.cfg.backbone_lr_scale,
             'name': 'backbone'},
            {'params': other,
             'lr': self.cfg.lr,
             'name': 'neck_head'},
        ]
        if self.cfg.optimizer == 'sgd':
            return optim.SGD(groups, lr=self.cfg.lr, momentum=0.937,
                             weight_decay=self.cfg.weight_decay)
        return optim.AdamW(groups, lr=self.cfg.lr,
                           weight_decay=self.cfg.weight_decay)

    def build_criterion(self):
        return None

    # ------------------------------------------------------------------
    # 配置块
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
        _kv("Description", "SWAP-Fighter: Anchor-free detector")

        _title("Dataset")
        _kv("Root", cfg.data_root)
        _kv("Classes", cfg.num_classes)
        _kv("Image Size", f"{cfg.img_size} x {cfg.img_size}")
        train_len = len(self.train_loader.dataset) if hasattr(self, 'train_loader') else '?'
        val_len = len(self.val_loader.dataset) if hasattr(self, 'val_loader') else '?'
        _kv("Train / Val", f"{train_len} / {val_len}")

        _title("Model")
        mode_str = ("OBB (rotated bbox, angle=sin_cos)"
                    if cfg.mode == 'obb'
                    else "Detect (axis-aligned bbox)")
        _kv("Mode", mode_str)

        if hasattr(self.model, 'backbone'):
            bb = self.model.backbone
            _kv("Backbone",
                f"{cfg.backbone_name} ({getattr(bb, 'resolved_source', cfg.backbone_pretrained_source)})")
            _kv("Feature Indices", str(getattr(bb, 'feature_indices', 'N/A')))
            _kv("Feature Channels", str(getattr(bb, 'out_channels', 'N/A')))
            _kv("Backbone Strides", str(getattr(bb, 'strides', cfg.strides)))
            if getattr(bb, 'need_permute', False):
                _kv("NHWC → NCHW", "yes")
        else:
            _kv("Backbone", f"{cfg.backbone_name} ({cfg.backbone_pretrained_source})")
            _kv("Strides", str(cfg.strides))

        neck_extra = _bool_en(cfg.use_extra_conv)
        neck_line = f"{cfg.neck_type} (out={cfg.neck_out_channels}, extra_conv={neck_extra})"
        if cfg.neck_type == 'bifpn':
            neck_line += f", layers={cfg.neck_num_layers}"
        _kv("Neck", neck_line)

        _kv("Head", f"{cfg.head_type} (convs={cfg.head_num_convs}, "
                    f"norm={cfg.head_norm_layer}, act={cfg.head_act_layer})")

        cls_c = cfg.head_cls_num_convs if cfg.head_cls_num_convs is not None else cfg.head_num_convs
        reg_c = cfg.head_reg_num_convs if cfg.head_reg_num_convs is not None else cfg.head_num_convs
        ang_c = cfg.head_angle_num_convs if cfg.head_angle_num_convs is not None else cfg.head_num_convs
        _kv("Head Branch Convs", f"cls={cls_c}, reg={reg_c}, angle={ang_c}")

        if cfg.mode == 'obb':
            _kv("Task", f"cls={cfg.num_classes}, reg=4, angle=2")
        else:
            _kv("Task", f"cls={cfg.num_classes}, reg=4")

        _kv("DFL", "disabled" if cfg.reg_max == 0
                  else f"reg_max={cfg.reg_max} (bin={cfg.reg_max + 1})")

        if cfg.mode == 'obb':
            if cfg.obb_reg_loss == 'kld':
                _kv("OBB Loss", f"kld (w={cfg.kld_weight}, tau={cfg.kld_tau})")
            else:
                _kv("OBB Loss", "probiou (CRIoU + 0.5 x ProbIoU)")

        _title("Training")
        _kv("Device / Precision", f"{self.device} / {self.actual_precision.upper()}")
        if self.scaler is not None:
            try:
                _kv("GradScaler", f"init_scale={self.scaler.get_scale():.0f}")
            except Exception:
                pass
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
        mosaic_disable = (cfg.mosaic_disable_epoch
                          if cfg.mosaic_disable_epoch is not None else "never")
        _kv("Mosaic", f"prob={cfg.mosaic_prob}, disable_epoch={mosaic_disable}")
        _kv("Geometry", f"degrees={cfg.degrees}, translate={cfg.translate}, "
                        f"scale={cfg.scale}, shear={cfg.shear}, "
                        f"perspective={cfg.perspective}")
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

        _title("Output")
        save_interval_str = (f"{cfg.save_interval} epoch"
                             if getattr(cfg, 'save_interval', 0) and cfg.save_interval > 0
                             else "disabled (only best/last)")
        _kv("Save Interval", save_interval_str)
        _kv("Patience", f"{cfg.patience} (0 = disabled)" if cfg.patience else "disabled")
        _kv("Save Dir", self.run_dir)

        print()
        print(LINE_EQ)

    # ------------------------------------------------------------------
    def _compute_loss_parts(self, images, targets, model=None):
        model = model if model is not None else self.model
        cfg = self.cfg
        if cfg.mode == 'detect':
            cls_l, reg_l = model(images)
            return compute_fcos_loss(
                cls_l, reg_l, targets,
                strides=cfg.strides, device=self.device, img_size=cfg.img_size,
                cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                dfl_weight=cfg.dfl_weight, mode='detect',
                reg_max=cfg.reg_max,
                return_parts=True)
        cls_l, reg_l, ang_l = model(images)
        return compute_fcos_loss(
            cls_l, reg_l, targets,
            strides=cfg.strides, device=self.device, img_size=cfg.img_size,
            cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
            angle_weight=cfg.angle_weight, dfl_weight=cfg.dfl_weight,
            mode='obb', angle_format=cfg.angle_format, angle_preds_list=ang_l,
            tal_topk=cfg.tal_topk, tal_alpha=cfg.tal_alpha, tal_beta=cfg.tal_beta,
            reg_max=cfg.reg_max,
            kld_weight=cfg.kld_weight, kld_tau=cfg.kld_tau, kld_fun=cfg.kld_fun,
            obb_reg_loss=cfg.obb_reg_loss,
            return_parts=True)
    
    @torch.no_grad()
    def _compute_val_loss(self, eval_model):
        """遍历 val_loader 算平均 loss（静默，不显示进度条）。"""
        cfg = self.cfg
        total = 0.0
        n = 0
        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            for t in targets:
                for k in ('boxes', 'obboxes', 'cxcywh', 'angles', 'labels'):
                    if k in t and t[k] is not None:
                        t[k] = t[k].to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                parts = self._compute_loss_parts(images, targets,
                                                 model=eval_model)
            total += float(parts['total'].detach())
            n += 1
        return total / max(n, 1)

    @staticmethod
    def _gpu_mem_str():
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.memory_reserved() / (1024 ** 3)
                return f"{mem:.2f}G"
            except Exception:
                return "0.00G"
        return "cpu"

    @staticmethod
    def _count_instances(targets):
        n = 0
        for t in targets:
            if 'labels' in t and t['labels'] is not None:
                n += int(t['labels'].shape[0])
        return n

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()
        total_loss = 0.0

        print()
        print(_make_train_header(cfg))

        pbar = tqdm(
            self.train_loader,
            bar_format='{desc}: {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
            leave=True,
            ncols=160,
            dynamic_ncols=False,
        )

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            for t in targets:
                for k in ('boxes', 'obboxes', 'cxcywh', 'angles', 'labels'):
                    if k in t and t[k] is not None:
                        t[k] = t[k].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                parts = self._compute_loss_parts(images, targets)
                loss = parts['total']

            # ★ 在 backward 前先抓取标量（避免 inplace 影响）
            box_v = float(parts['box'].detach())
            cls_v = float(parts['cls'].detach())
            dfl_v = float(parts['dfl'].detach())
            ang_v = float(parts['angle'].detach())
            total_loss += float(loss.detach())

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip_norm)
                self.optimizer.step()

            if self.scheduler_type == 'step':
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.ema_model is not None:
                self._update_ema()

            # ---------- 更新进度条 desc（数值行） ----------
            n_inst = self._count_instances(targets)
            gpu_mem = self._gpu_mem_str()
            desc = _make_train_desc(
                epoch + 1, cfg.epochs, gpu_mem,
                box_v, cls_v, dfl_v, ang_v, n_inst, cfg.img_size, cfg,
            )
            pbar.set_description_str(desc, refresh=True)

        pbar.close()
        return {'train_loss': total_loss / max(len(self.train_loader), 1)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        cfg = self.cfg
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        if (epoch + 1) < cfg.eval_map_start_epoch:
            return {}

        # ---------- 1. 先算 val_loss ----------
        val_loss = self._compute_val_loss(eval_model)

        # ---------- 2. 再算 mAP ----------
        if cfg.mode == 'obb':
            res = evaluate_map_obb_rotated(
                eval_model, self.val_loader, self.device, cfg,
                conf_threshold=cfg.eval_conf_threshold, nms_threshold=0.5,
                max_det=1000, pre_nms_max_det=1000,
                progress=True, desc=TABLE_VAL_HEADER)
        else:
            res = evaluate_mAP(
                eval_model, self.val_loader, self.device, cfg,
                conf_threshold=cfg.eval_conf_threshold,
                progress=True, desc=TABLE_VAL_HEADER)

        if len(res) == 5:
            p, r, ap50, ap95, ar = res
        else:
            ap50, ap95, ar = res
            p = r = ap50

        n_inst_total = getattr(self, '_val_n_instances', 0)
        n_img = getattr(self, '_val_n_images', len(self.val_loader.dataset))
        print(f"{'all':>17}  {n_img:>6d}  {n_inst_total:>9d}  "
              f"{float(p):>9.4f}  {float(r):>9.4f}  "
              f"{float(ap50):>9.4f}  {float(ap95):>11.4f}")

        # ★ 返回里带上 val_loss
        return dict(
            val_loss=float(val_loss),
            mAP50=float(ap50), mAP50_95=float(ap95),
            AR100=float(ar), precision=float(p), recall=float(r),
        )

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
            'task': cfg.task,
            'mode': cfg.mode,
            'strides': list(cfg.strides) if cfg.strides else None,
            'num_classes': cfg.num_classes,
            'backbone_name': cfg.backbone_name,
            'class_names': getattr(cfg, 'class_names', None),
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


def _ds_kwargs(cfg):
    return dict(
        use_flip=cfg.use_flip, use_color_aug=cfg.use_color_aug,
        mean=cfg.mean, std=cfg.std,
        angle_range=cfg.angle_range if cfg.mode == 'obb' else None,
        mosaic_prob=cfg.mosaic_prob,
        mosaic_disable_epoch=cfg.mosaic_disable_epoch,
        degrees=cfg.degrees, translate=cfg.translate, scale=cfg.scale,
        shear=cfg.shear, perspective=cfg.perspective,
        hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        color_blur_prob=cfg.color_blur_prob,
        color_median_blur_prob=cfg.color_median_blur_prob,
        color_to_gray_prob=cfg.color_to_gray_prob,
        color_clahe_prob=cfg.color_clahe_prob,
        fliplr=cfg.fliplr, flipud=cfg.flipud,
        mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha,
        copy_paste_prob=cfg.copy_paste_prob,
        copy_paste_max_paste=cfg.copy_paste_max_paste,
    )