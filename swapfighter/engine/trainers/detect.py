import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..base_trainer import BaseTrainer, suppress_stdout
from ...utils.logger import LINE, LINE_EQ, _title, _kv, _bool_en
from ...utils.losses import compute_fcos_loss
from ...utils.metrics import evaluate_mAP, evaluate_map_obb_rotated
from ...utils.metrics import format_metric_table
from ...tasks import get_task

def _detect_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    return imgs, tgts

def _make_train_header(cfg):
    cols = [f"{'Epoch':>9}", f"{'GPU':>6}"]

    if cfg.task == 'keypoint':

        cols.append(f"{'box':>7}")
        cols.append(f"{'pose':>7}")
        cols.append(f"{'cls':>7}")
    else:
        cols.append(f"{'box':>7}")
        cols.append(f"{'cls':>7}")

    if cfg.reg_max > 0:
        cols.append(f"{'dfl':>7}")
    if cfg.task == 'obb':
        cols.append(f"{'angle':>7}")
    if cfg.task == 'segment':
        cols.append(f"{'mask':>7}")

    cols.append(f"{'Inst':>5}")
    cols.append(f"{'Size':>4}")
    return "  ".join(cols)

def _make_train_desc(epoch, epochs, gpu_mem, box_v, cls_v, dfl_v, ang_v,
                     n_inst, img_size, cfg, mask_v=0.0, kpt_v=0.0):
    cols = [f"{epoch:>4d}/{epochs:<4d}", f"{gpu_mem:>6}"]

    if cfg.task == 'keypoint':

        cols.append(f"{box_v:>7.3f}")
        cols.append(f"{kpt_v:>7.3f}")
        cols.append(f"{cls_v:>7.3f}")
    else:
        cols.append(f"{box_v:>7.3f}")
        cols.append(f"{cls_v:>7.3f}")

    if cfg.reg_max > 0:
        cols.append(f"{dfl_v:>7.3f}")
    if cfg.task == 'obb':
        cols.append(f"{ang_v:>7.3f}")
    if cfg.task == 'segment':
        cols.append(f"{mask_v:>7.3f}")

    cols.append(f"{n_inst:>5d}")
    cols.append(f"{img_size:>4d}")
    return "  ".join(cols)

TABLE_VAL_HEADER = (
    f"{'Class':>17}  {'Images':>6}  {'Instances':>9}  "
    f"{'Box(P':>9}  {'R':>9}  {'mAP50':>9}  {'mAP50-95)':>11}"
)

class DetectTrainer(BaseTrainer):

    def build_model(self):
        from ...tasks import get_task
        with suppress_stdout():
            model = get_task(self.cfg.task)['model'](self.cfg)
        return model

    def build_dataloaders(self):

        ds_cls = get_task(self.cfg.task)['dataset_cls']

        val_kwargs = {}
        if self.cfg.task == 'segment':
            val_kwargs['proto_stride'] = getattr(
                self.cfg, 'proto_stride', 4)
        elif self.cfg.task == 'keypoint':        
            val_kwargs['num_keypoints'] = getattr(
                self.cfg, 'num_keypoints', 17)
            val_kwargs['keypoint_visibility'] = getattr(
                self.cfg, 'keypoint_visibility', 'auto')
            val_kwargs['flip_idx'] = getattr(self.cfg, 'flip_idx', None)

        with suppress_stdout():
            train_ds = ds_cls(
                root=self.cfg.data_root, img_size=self.cfg.img_size,
                split=self.cfg.train_split, task=self.cfg.task,
                **_ds_kwargs(self.cfg))
            val_ds = ds_cls(
                root=self.cfg.data_root, img_size=self.cfg.img_size,
                split=self.cfg.val_split, task=self.cfg.task,
                use_flip=False, use_color_aug=False,
                **val_kwargs)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=_detect_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=_detect_collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

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

    def build_optimizer(self):
        bb, other = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (bb if n.startswith('backbone.') else other).append(p)

        groups = []
        if bb:
            groups.append({
                'params': bb,
                'lr': self.cfg.lr * self.cfg.backbone_lr_scale,
                'name': 'backbone',
            })
        if other:
            groups.append({
                'params': other,
                'lr': self.cfg.lr,
                'name': 'neck_head',
            })

        if not groups:
            raise RuntimeError(
                "[DetectTrainer] 所有参数都被冻结，无法训练。"
                "请检查 freeze_stage 设置")

        if self.cfg.optimizer == 'sgd':
            return optim.SGD(groups, lr=self.cfg.lr, momentum=0.937,
                             weight_decay=self.cfg.weight_decay)
        return optim.AdamW(groups, lr=self.cfg.lr,
                           weight_decay=self.cfg.weight_decay)

    def build_criterion(self):
        return None

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
        train_len = (len(self.train_loader.dataset)
                     if hasattr(self, 'train_loader') else '?')
        val_len = (len(self.val_loader.dataset)
                   if hasattr(self, 'val_loader') else '?')
        _kv("Train / Val", f"{train_len} / {val_len}")

        _title("Model")

        if cfg.task == 'obb':
            mode_str = "OBB (rotated bbox, angle=sin_cos)"
        elif cfg.task == 'segment':
            mode_str = "Segment (instance segmentation)"
        elif cfg.task == 'keypoint':            
            mode_str = "Keypoint (pose estimation)"
        else:
            mode_str = "Detect (axis-aligned bbox)"
        _kv("Mode", mode_str)

        if hasattr(self.model, 'backbone'):
            bb = self.model.backbone
            src = getattr(bb, 'resolved_source',
                          cfg.backbone_pretrained_source)
            _kv("Backbone", f"{cfg.backbone_name} ({src})")
            _kv("Feature Indices",
                str(getattr(bb, 'feature_indices', 'N/A')))
            _kv("Feature Channels",
                str(getattr(bb, 'out_channels', 'N/A')))
            _kv("Backbone Strides",
                str(getattr(bb, 'strides', cfg.strides)))
            if getattr(bb, 'need_permute', False):
                _kv("NHWC → NCHW", "yes")
        else:
            _kv("Backbone",
                f"{cfg.backbone_name} ({cfg.backbone_pretrained_source})")
            _kv("Strides", str(cfg.strides))

        neck_extra = _bool_en(cfg.use_extra_conv)
        neck_out = getattr(self.model, 'neck_out_channels_list',
                           cfg.neck_out_channels)
        neck_line = f"{cfg.neck_type} (out={neck_out}, " \
                    f"extra_conv={neck_extra})"
        if cfg.neck_type == 'bifpn':
            neck_line += f", layers={cfg.neck_num_layers}"
        _kv("Neck", neck_line)

        _kv("Head", f"{cfg.head_type} (convs={cfg.head_num_convs}, "
                    f"norm={cfg.head_norm_layer}, "
                    f"act={cfg.head_act_layer})")

        head_channels = getattr(self.model, 'head_channels', None)
        if head_channels is not None:
            _kv("Head Shared Channels", head_channels)

        cls_c = (cfg.head_cls_num_convs
                 if cfg.head_cls_num_convs is not None
                 else cfg.head_num_convs)
        reg_c = (cfg.head_reg_num_convs
                 if cfg.head_reg_num_convs is not None
                 else cfg.head_num_convs)
        ang_c = (cfg.head_angle_num_convs
                 if cfg.head_angle_num_convs is not None
                 else cfg.head_num_convs)
        kpt_c = (getattr(cfg, 'head_kpt_num_convs', None)
                 if getattr(cfg, 'head_kpt_num_convs', None) is not None
                 else cfg.head_num_convs)

        if cfg.task == 'keypoint':              
            _kv("Head Tower Depth",
                f"cls×{cls_c}, reg×{reg_c}, kpt×{kpt_c}")
        else:
            _kv("Head Tower Depth",
                f"cls×{cls_c}, reg×{reg_c}, angle×{ang_c}")
        
        reg_dim = (4 * (cfg.reg_max + 1)) if cfg.reg_max > 0 else 4
        if cfg.task == 'obb':
            angle_dim = 2 if cfg.angle_format == 'sin_cos' else 1
            _kv("Head Output Dim",
                f"cls={cfg.num_classes}, reg={reg_dim}, "
                f"angle={angle_dim} ({cfg.angle_format})")
        elif cfg.task == 'keypoint':           
            K = getattr(cfg, 'num_keypoints', 17)
            _kv("Head Output Dim",
                f"cls={cfg.num_classes}, reg={reg_dim}, "
                f"kpt={K * 2} (K={K})")
        else:
            _kv("Head Output Dim",
                f"cls={cfg.num_classes}, reg={reg_dim}")

        _kv("DFL", "disabled" if cfg.reg_max == 0
                  else f"reg_max={cfg.reg_max} (bin={cfg.reg_max + 1})")

        if cfg.task == 'obb':
            _kv("Angle Format",
                f"{cfg.angle_format} (range={cfg.angle_range})")
            if cfg.obb_reg_loss == 'kld':
                _kv("OBB Reg Loss",
                    f"kld (weight={cfg.kld_weight}, "
                    f"tau={cfg.kld_tau}, fun={cfg.kld_fun})")
            else:
                _kv("OBB Reg Loss",
                    "probiou (CRIoU + 0.5 × ProbIoU)")
            _kv("OBB NMS Threshold",
                f"{cfg.obb_nms_threshold}")
            

        if cfg.task == 'keypoint':              
            _kv("Num Keypoints", getattr(cfg, 'num_keypoints', 17))
            _kv("Keypoint Visibility",
                getattr(cfg, 'keypoint_visibility', 'auto'))
            _kv("Keypoint Weight",
                getattr(cfg, 'keypoint_weight', 1.0))

        if cfg.task == 'segment':
            proto_dim = getattr(self.model, 'proto_dim',
                                getattr(cfg, 'proto_dim', 'N/A'))
            proto_stride = getattr(self.model, 'proto_stride',
                                   getattr(cfg, 'proto_stride', 'N/A'))
            _kv("Proto Dim", proto_dim)
            _kv("Proto Stride", proto_stride)
            _kv("Num Mask Coef",
                getattr(self.model, 'num_mask_coef', proto_dim))
            _kv("Mask Weight",
                getattr(cfg, 'mask_weight', 1.0))

        freeze_stage = getattr(cfg, 'freeze_stage', 'none')
        if freeze_stage != 'none':
            freeze_desc = {
                'head_only': 'head_only (backbone frozen)',
                'neck_head': 'neck_head (backbone frozen)',
                'last_n': f"last_n (unfreeze last "
                          f"{getattr(cfg, 'freeze_last_n', 2)} blocks)",
            }.get(freeze_stage, freeze_stage)
            _kv("Freeze Stage", freeze_desc)

        _title("Training")
        _kv("Device / Precision",
            f"{self.device} / {self.actual_precision.upper()}")
        if self.scaler is not None:
            try:
                _kv("GradScaler",
                    f"init_scale={self.scaler.get_scale():.0f}")
            except Exception:
                pass
        eff = cfg.batch_size * max(1, getattr(cfg, 'accum_steps', 1))
        _kv("Batch / Accum / Effective",
            f"{cfg.batch_size} / {getattr(cfg, 'accum_steps', 1)} / {eff}")
        _kv("Epochs", cfg.epochs)
        _kv("Warmup", f"{cfg.warmup_epochs} epochs")
        _kv("LR (neck/head)", f"{cfg.lr:.2e}")
        _kv("LR (backbone)", f"{cfg.lr * cfg.backbone_lr_scale:.2e} "
                             f"(scale={cfg.backbone_lr_scale})")
        _kv("Optimizer / Scheduler", f"{cfg.optimizer} / {cfg.scheduler}")
        _kv("Grad Clip", f"{cfg.grad_clip_norm}")

        loss_line = f"cls={cfg.cls_weight}, reg={cfg.reg_weight}"
        if cfg.reg_max > 0:
            loss_line += f", dfl={cfg.dfl_weight}"
        if cfg.task == 'obb':
            loss_line += f", angle={cfg.angle_weight}"
        if cfg.task == 'segment':
            loss_line += f", mask={getattr(cfg, 'mask_weight', 1.0)}"
        if cfg.task == 'keypoint':             
            loss_line += (f", kpt="
                          f"{getattr(cfg, 'keypoint_weight', 1.0)}")
        _kv("Loss Weights", loss_line)

        _kv("EMA", f"enabled (decay={cfg.ema_decay})"
                   if cfg.use_ema else "disabled")
        _kv("TAL", f"topk={cfg.tal_topk}, alpha={cfg.tal_alpha}, "
                   f"beta={cfg.tal_beta}")

        _title("Augmentation")
        mosaic_disable = (cfg.mosaic_disable_epoch
                          if cfg.mosaic_disable_epoch is not None
                          else "never")
        _kv("Mosaic", f"prob={cfg.mosaic_prob}, "
                      f"disable_epoch={mosaic_disable}")
        _kv("Geometry",
            f"degrees={cfg.degrees}, translate={cfg.translate}, "
            f"scale={cfg.scale}, shear={cfg.shear}, "
            f"perspective={cfg.perspective}")
        _kv("HSV", f"h={cfg.hsv_h}, s={cfg.hsv_s}, v={cfg.hsv_v}")
        _kv("Color Extra",
            f"blur={cfg.color_blur_prob}, "
            f"median={cfg.color_median_blur_prob}, "
            f"gray={cfg.color_to_gray_prob}, "
            f"clahe={cfg.color_clahe_prob}")
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
                             if getattr(cfg, 'save_interval', 0) and
                                cfg.save_interval > 0
                             else "disabled (only best/last)")
        _kv("Save Interval", save_interval_str)
        _kv("Patience",
            f"{cfg.patience} (0 = disabled)"
            if cfg.patience else "disabled")
        _kv("Save Dir", self.run_dir)

        print()
        print(LINE_EQ)

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

    @staticmethod
    def _has_bad_grad(model):
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def _is_model_frozen(self):
        return getattr(self.model, 'get_freeze_status',
                       lambda: False)()

    def _compute_loss_from_preds(self, cls_l, reg_l, ang_l, targets):
        cfg = self.cfg
        if cfg.task == 'detect':
            return compute_fcos_loss(
                cls_l, reg_l, targets,
                strides=cfg.strides, device=self.device,
                img_size=cfg.img_size,
                cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
                dfl_weight=cfg.dfl_weight, task='detect',
                reg_max=cfg.reg_max,
                return_parts=True)
        return compute_fcos_loss(
            cls_l, reg_l, targets,
            strides=cfg.strides, device=self.device,
            img_size=cfg.img_size,
            cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight,
            angle_weight=cfg.angle_weight, dfl_weight=cfg.dfl_weight,
            task='obb', angle_format=cfg.angle_format,
            angle_preds_list=ang_l,
            tal_topk=cfg.tal_topk, tal_alpha=cfg.tal_alpha,
            tal_beta=cfg.tal_beta,
            reg_max=cfg.reg_max,
            kld_weight=cfg.kld_weight, kld_tau=cfg.kld_tau,
            kld_fun=cfg.kld_fun,
            obb_reg_loss=cfg.obb_reg_loss,
            return_parts=True)

    def _compute_loss_parts(self, images, targets, model=None):
        model = model if model is not None else self.model
        cfg = self.cfg
        if cfg.task == 'detect':
            cls_l, reg_l = model(images)
            ang_l = None
        else:
            cls_l, reg_l, ang_l = model(images)
        return self._compute_loss_from_preds(cls_l, reg_l, ang_l, targets)

    def _compute_loss_frozen(self, images, targets, model=None):
        model = model if model is not None else self.model

        with torch.no_grad():
            feats = model.backbone(images)

        neck_out = model.neck(feats)

        if model.with_angle:
            cls_l, reg_l, ang_l = model._forward_heads(neck_out)
        else:
            cls_l, reg_l = model._forward_heads(neck_out)
            ang_l = None

        return self._compute_loss_from_preds(cls_l, reg_l, ang_l, targets)

    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()

        if self._is_model_frozen():
            if hasattr(self.model, 'backbone'):
                self.model.backbone.eval()

        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        total_angle_loss = 0.0
        total_mask_loss = 0.0
        total_kpt_loss = 0.0      
        accum = max(1, getattr(cfg, 'accum_steps', 1))
        accum_count = 0

        print(_make_train_header(cfg))

        pbar = tqdm(
            self.train_loader,
            bar_format='{desc} {percentage:3.0f}% {bar:12} '
                       '{n_fmt}/{total_fmt} {rate_fmt} {elapsed}',
            leave=True,
            ncols=120,                  
            dynamic_ncols=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            for t in targets:
                for k in ('boxes', 'obboxes', 'cxcywh', 'angles', 'labels',
                          'mask_per_gt'):  
                    if k in t and t[k] is not None:
                        t[k] = t[k].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                if self._is_model_frozen():

                    parts = self._compute_loss_frozen(images, targets)
                else:
                    parts = self._compute_loss_parts(images, targets)
                loss = parts['total'] / accum

            if not torch.isfinite(loss) or not loss.requires_grad:
                print(f"  [WARN] Epoch {epoch+1} step {step} 跳过异常 batch "
                      f"(finite={torch.isfinite(loss).item()}, "
                      f"requires_grad={loss.requires_grad})")
                del parts, loss
                continue

            box_v = float(parts['box'].detach())
            cls_v = float(parts['cls'].detach())
            dfl_v = float(parts['dfl'].detach()) if 'dfl' in parts else 0.0
            ang_v = float(parts['angle'].detach()) if 'angle' in parts else 0.0
            mask_v = float(parts['mask'].detach()) if 'mask' in parts else 0.0
            kpt_v = float(parts['kpt'].detach()) if 'kpt' in parts else 0.0

            total_loss += float(loss.detach()) * accum
            total_box_loss += box_v * accum
            total_cls_loss += cls_v * accum
            total_dfl_loss += dfl_v * accum
            total_angle_loss += ang_v * accum
            total_mask_loss += mask_v * accum
            total_kpt_loss += kpt_v * accum      

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1

            is_update = (accum_count >= accum) or \
                        (step + 1 == len(self.train_loader))

            if is_update:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                    if self._has_bad_grad(self.model):
                        print(f"  [WARN] Epoch {epoch+1} 梯度含 NaN/Inf，"
                              f"跳过 step")
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                        del parts, loss
                        continue
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self._has_bad_grad(self.model):
                        print(f"  [WARN] Epoch {epoch+1} 梯度含 NaN/Inf，"
                              f"跳过 step")
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_count = 0
                        del parts, loss
                        continue
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip_norm)
                    self.optimizer.step()

                if self.scheduler_type == 'step':
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0

                if self.ema_model is not None:
                    self._update_ema()
                    self.global_step += 1

            n_inst = self._count_instances(targets)
            gpu_mem = self._gpu_mem_str()
            desc = _make_train_desc(
                epoch + 1, cfg.epochs, gpu_mem,
                box_v, cls_v, dfl_v, ang_v, n_inst, cfg.img_size, cfg,
                mask_v=mask_v, kpt_v=kpt_v)
            pbar.set_description_str(desc, refresh=True)
            pbar.set_description_str(desc, refresh=True)

            del parts, loss

        pbar.close()
        n = max(len(self.train_loader), 1)
        return {
            'train_loss': total_loss / n,
            'train_box_loss': total_box_loss / n,
            'train_cls_loss': total_cls_loss / n,
            'train_dfl_loss': total_dfl_loss / n,
            'train_angle_loss': total_angle_loss / n,
            'train_mask_loss': total_mask_loss / n,
            'train_kpt_loss': total_kpt_loss / n,
        }

    @torch.no_grad()
    def _compute_val_loss(self, eval_model, max_batches=15):
        total = 0.0
        n = 0
        frozen = getattr(eval_model, 'get_freeze_status',
                         lambda: False)()

        for i, (images, targets) in enumerate(self.val_loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            for t in targets:
                for k in ('boxes', 'obboxes', 'cxcywh', 'angles', 'labels',
                          'mask_per_gt'):
                    if k in t and t[k] is not None:
                        t[k] = t[k].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                if frozen:
                    parts = self._compute_loss_frozen(
                        images, targets, model=eval_model)
                else:
                    parts = self._compute_loss_parts(
                        images, targets, model=eval_model)

            total += float(parts['total'].detach())
            n += 1
            del images, targets, parts

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, epoch):
        cfg = self.cfg
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        if (epoch + 1) < cfg.eval_map_start_epoch:
            return {}

        from ...utils.metrics import evaluate_mAP, evaluate_map_obb_rotated

        val_loss = self._compute_val_loss(eval_model)

        n_img = getattr(self, '_val_n_images', len(self.val_loader.dataset))
        n_inst_total = getattr(self, '_val_n_instances', 0)
        class_names = getattr(cfg, 'class_names', None)
        verbose = bool(getattr(cfg, 'verbose_metrics', False))

        det_header = (
            f"{'Class':>8}  {'Images':>6}  {'Inst':>5}  "
            f"{'Box(P':>7}  {'R':>6}  {'mAP50':>6}  {'mAP50-95)':>9}"
        )

        if cfg.task == 'obb':
            res = evaluate_map_obb_rotated(
                eval_model, self.val_loader, self.device, cfg,
                conf_threshold=cfg.eval_conf_threshold, nms_threshold=0.5,
                max_det=1000, pre_nms_max_det=1000,
                progress=True, desc=det_header,
                return_per_class=True)
        else:
            res = evaluate_mAP(
                eval_model, self.val_loader, self.device, cfg,
                conf_threshold=cfg.eval_conf_threshold,
                progress=True, desc=det_header,
                return_per_class=True)

        if len(res['classes']) > 0:
            p = float(res['precision'].mean())
            r = float(res['recall'].mean())
            ap50 = float(res['ap'][:, 0].mean())
            ap95 = float(res['ap'].mean())
        else:
            p = r = ap50 = ap95 = 0.0

        print(f"{'all':>8}  {n_img:>6d}  {n_inst_total:>5d}  "
              f"{p:>7.3f}  {r:>6.3f}  {ap50:>6.3f}  {ap95:>9.3f}")

        if verbose and len(res['classes']) > 0:
            for i, cls_id in enumerate(res['classes']):
                if class_names and int(cls_id) < len(class_names):
                    name = class_names[int(cls_id)][:8]
                else:
                    name = f'c{int(cls_id)}'
                pi = float(res['precision'][i])
                ri = float(res['recall'][i])
                a50i = float(res['ap'][i, 0])
                a95i = float(res['ap'][i].mean())
                print(f"{name:>8}  {n_img:>6d}  "
                      f"{int(res['n_gt'][i]):>5d}  "
                      f"{pi:>7.3f}  {ri:>6.3f}  {a50i:>6.3f}  {a95i:>9.3f}")

        print()

        return dict(
            val_loss=float(val_loss),
            mAP50=ap50, mAP50_95=ap95,
            AR100=r, precision=p, recall=r,
        )

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
            'strides': list(cfg.strides) if cfg.strides else None,
            'num_classes': cfg.num_classes,
            'backbone_name': cfg.backbone_name,
            'head_channels': getattr(self.model, 'head_channels', None),
            'class_names': getattr(cfg, 'class_names', None),
        })
        return d

    def _update_ema(self):
        decay = self._current_ema_decay(self.global_step)
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(decay).add_(msd[k].detach(), alpha=1.0 - decay)
                else:
                    v.copy_(msd[k])

def _ds_kwargs(cfg):
    kwargs = dict(
        use_flip=cfg.use_flip, use_color_aug=cfg.use_color_aug,
        mean=cfg.mean, std=cfg.std,
        angle_range=cfg.angle_range if cfg.task == 'obb' else None,
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

    if cfg.task == 'segment':
        kwargs['proto_stride'] = getattr(cfg, 'proto_stride', 4)

    if cfg.task == 'keypoint':
        kwargs['num_keypoints'] = getattr(cfg, 'num_keypoints', 17)
        kwargs['keypoint_visibility'] = getattr(
            cfg, 'keypoint_visibility', 'auto')
        kwargs['flip_idx'] = getattr(cfg, 'flip_idx', None)
    return kwargs