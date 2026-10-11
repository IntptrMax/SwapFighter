import numpy as np
import torch
import torch.nn.functional as F

from .detect import DetectTrainer

class SegTrainer(DetectTrainer):

    def _compute_loss_parts(self, images, targets, model=None):
        model = model if model is not None else self.model
        cfg = self.cfg

        if cfg.with_angle:
            cls_l, reg_l, coef_l, ang_l, proto = model(images)
        else:
            cls_l, reg_l, coef_l, proto = model(images)
            ang_l = None

        detect_parts = self._compute_detect_loss_from_preds(
            cls_l, reg_l, ang_l, targets)
        mask_loss = self._compute_seg_mask_loss(
            coef_l, proto, targets, detect_parts)

        total = detect_parts['total'] + cfg.mask_weight * mask_loss

        return {
            'total': total,
            'cls': detect_parts['cls'],
            'box': detect_parts['box'],
            'dfl': detect_parts['dfl'],
            'angle': detect_parts['angle'],
            'mask': mask_loss.detach(),
            '_pos_batch': detect_parts.get('_pos_batch'),
            '_pos_point': detect_parts.get('_pos_point'),
            '_pos_gt': detect_parts.get('_pos_gt'),
        }

    def _compute_loss_frozen(self, images, targets, model=None):
        model = model if model is not None else self.model

        with torch.no_grad():
            feats = model.backbone(images)

        neck_out = model.neck(feats)
        proto = model.proto_branch(neck_out)

        cls_l, reg_l, coef_l, ang_l = [], [], [], []
        for lvl, feat in enumerate(neck_out):
            feat_proj = model.head_proj[lvl](feat)
            preds = model.head_shared(feat_proj)
            cls_l.append(preds['cls'])
            reg_l.append(preds['reg'])
            coef_l.append(preds['coef'])
            if model.with_angle:
                ang_l.append(preds['angle'])

        if not model.with_angle:
            ang_l = None

        detect_parts = self._compute_detect_loss_from_preds(
            cls_l, reg_l, ang_l, targets)
        mask_loss = self._compute_seg_mask_loss(
            coef_l, proto, targets, detect_parts)

        total = detect_parts['total'] + self.cfg.mask_weight * mask_loss
        return {
            'total': total,
            'cls': detect_parts['cls'],
            'box': detect_parts['box'],
            'dfl': detect_parts['dfl'],
            'angle': detect_parts['angle'],
            'mask': mask_loss.detach(),
            '_pos_batch': detect_parts.get('_pos_batch'),
            '_pos_point': detect_parts.get('_pos_point'),
            '_pos_gt': detect_parts.get('_pos_gt'),
        }

    def _compute_detect_loss_from_preds(self, cls_l, reg_l, ang_l, targets):
        cfg = self.cfg
        from ...utils.losses import compute_fcos_loss

        if cfg.task in ('detect', 'segment'):
            return compute_fcos_loss(
                cls_l, reg_l, targets,
                strides=cfg.strides, device=self.device,
                img_size=cfg.img_size,
                cls_weight=cfg.cls_weight,
                reg_weight=cfg.reg_weight,
                dfl_weight=cfg.dfl_weight,
                task='detect',
                reg_max=cfg.reg_max,
                return_parts=True,
                return_pos_idx=True,
            )
        else:
            return compute_fcos_loss(
                cls_l, reg_l, targets,
                strides=cfg.strides, device=self.device,
                img_size=cfg.img_size,
                cls_weight=cfg.cls_weight,
                reg_weight=cfg.reg_weight,
                angle_weight=cfg.angle_weight,
                dfl_weight=cfg.dfl_weight,
                task='obb', angle_format=cfg.angle_format,
                angle_preds_list=ang_l,
                tal_topk=cfg.tal_topk, tal_alpha=cfg.tal_alpha,
                tal_beta=cfg.tal_beta,
                reg_max=cfg.reg_max,
                kld_weight=cfg.kld_weight, kld_tau=cfg.kld_tau,
                kld_fun=cfg.kld_fun,
                obb_reg_loss=cfg.obb_reg_loss,
                return_parts=True,
                return_pos_idx=True,
            )

    def _compute_seg_mask_loss(self, coef_l, proto, targets, detect_parts):
        pos_batch = detect_parts.get('_pos_batch')
        pos_point = detect_parts.get('_pos_point')
        pos_gt = detect_parts.get('_pos_gt')

        if pos_batch is None or pos_batch.numel() == 0:
            anchor = coef_l[0] if len(coef_l) > 0 else proto
            return anchor.sum() * 0.0

        B = coef_l[0].shape[0]
        C = coef_l[0].shape[1]
        assert C == proto.shape[1], (
            f"[SegTrainer] coef 通道数 ({C}) != proto 通道数 ({proto.shape[1]})，"
            f"请检查 proto_dim 和 num_mask_coef 是否一致"
        )
        ph, pw = proto.shape[2], proto.shape[3]

        all_coef = []
        for coef in coef_l:
            b, c, h, w = coef.shape
            all_coef.append(coef.permute(0, 2, 3, 1).reshape(b, -1, c))
        all_coef = torch.cat(all_coef, dim=1)

        pos_coef = all_coef[pos_batch, pos_point]

        device = pos_coef.device
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_count = 0

        for b in range(B):
            mask_b = (pos_batch == b)
            if not mask_b.any():
                continue

            idx_b = mask_b.nonzero(as_tuple=True)[0]
            coef_b = pos_coef[idx_b]

            proto_b = proto[b].reshape(C, -1)

            pred_masks_b = torch.einsum('nc,cp->np', coef_b, proto_b)

            if 'mask_per_gt' not in targets[b]:
                continue

            gt_idx_b = pos_gt[idx_b]
            gt_mask_b_3d = targets[b]['mask_per_gt'][gt_idx_b]
            ph_gt, pw_gt = gt_mask_b_3d.shape[1], gt_mask_b_3d.shape[2]

            assert pred_masks_b.shape[1] == ph_gt * pw_gt, (
                f"[SegTrainer] mask 尺寸不匹配："
                f"预测 {ph}×{pw} ({pred_masks_b.shape[1]})，"
                f"GT {ph_gt}×{pw_gt} ({ph_gt * pw_gt})。"
                f"请检查 proto_stride 是否在模型和数据集两侧一致："
                f"模型侧 proto={tuple(proto.shape)}，"
                f"数据集侧 mask_per_gt={tuple(targets[b]['mask_per_gt'].shape)}"
            )

            gt_mask_b = gt_mask_b_3d.reshape(len(idx_b), -1)

            loss_pixel = F.binary_cross_entropy_with_logits(
                pred_masks_b.float(),
                gt_mask_b.float(),
                reduction='none',
            )
            loss_per_sample = loss_pixel.mean(dim=1)
            loss_b = loss_per_sample.mean()

            total_loss = total_loss + loss_b
            total_count += 1

        if total_count == 0:
            return pos_coef.sum() * 0.0

        loss = total_loss / total_count
        if not torch.isfinite(loss):
            return pos_coef.sum() * 0.0
        return loss

    @torch.no_grad()
    def validate(self, epoch):
        cfg = self.cfg
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        if (epoch + 1) < cfg.eval_map_start_epoch:
            return {}

        from ...utils.metrics import evaluate_seg

        val_loss = self._compute_val_loss(eval_model)

        n_img = getattr(self, '_val_n_images', len(self.val_loader.dataset))
        n_inst_total = getattr(self, '_val_n_instances', 0)
        class_names = getattr(cfg, 'class_names', None)
        verbose = bool(getattr(cfg, 'verbose_metrics', False))

        seg_header = (
            f"{'Class':>8}  {'Images':>6}  {'Inst':>5}  "
            f"{'Box(P':>7}  {'R':>6}  {'mAP50':>6}  {'mAP50-95)':>9}  "
            f"{'Mask(P':>7}  {'R':>6}  {'mAP50':>6}  {'mAP50-95)':>9}"
        )

        bbox_res, mask_res = evaluate_seg(
            eval_model, self.val_loader, self.device, cfg,
            conf_threshold=cfg.eval_conf_threshold,
            progress=True, desc=seg_header,
            return_per_class=True)

        def _means(r):
            if r is None or len(r['classes']) == 0:
                return 0.0, 0.0, 0.0, 0.0
            return (float(r['precision'].mean()),
                    float(r['recall'].mean()),
                    float(r['ap'][:, 0].mean()),
                    float(r['ap'].mean()))

        bp, br, b50, b95 = _means(bbox_res)
        mp, mr, m50, m95 = _means(mask_res)

        print(f"{'all':>8}  {n_img:>6d}  {n_inst_total:>5d}  "
              f"{bp:>7.3f}  {br:>6.3f}  {b50:>6.3f}  {b95:>9.3f}  "
              f"{mp:>7.3f}  {mr:>6.3f}  {m50:>6.3f}  {m95:>9.3f}")

        if verbose and bbox_res is not None:
            classes = bbox_res['classes']
            for i, cls_id in enumerate(classes):
                if class_names and int(cls_id) < len(class_names):
                    name = class_names[int(cls_id)][:8]
                else:
                    name = f'c{int(cls_id)}'
                bpi = float(bbox_res['precision'][i])
                bri = float(bbox_res['recall'][i])
                b50i = float(bbox_res['ap'][i, 0])
                b95i = float(bbox_res['ap'][i].mean())
                if mask_res and len(mask_res['classes']) > 0:
                    mpi = float(mask_res['precision'][i])
                    mri = float(mask_res['recall'][i])
                    m50i = float(mask_res['ap'][i, 0])
                    m95i = float(mask_res['ap'][i].mean())
                else:
                    mpi = mri = m50i = m95i = 0.0
                print(f"{name:>8}  {n_img:>6d}  "
                      f"{int(bbox_res['n_gt'][i]):>5d}  "
                      f"{bpi:>7.3f}  {bri:>6.3f}  {b50i:>6.3f}  {b95i:>9.3f}  "
                      f"{mpi:>7.3f}  {mri:>6.3f}  {m50i:>6.3f}  {m95i:>9.3f}")

        print()

        return dict(
            val_loss=float(val_loss),
            mAP50=b50, mAP50_95=b95,
            AR100=br, precision=bp, recall=br,
            mask_mAP50=m50, mask_mAP50_95=m95,
        )

    def _compute_fitness(self, metrics):
        if 'mask_mAP50' in metrics and 'mask_mAP50_95' in metrics:
            return 0.1 * metrics['mask_mAP50'] + 0.9 * metrics['mask_mAP50_95']
        return super()._compute_fitness(metrics)

__all__ = ['SegTrainer']