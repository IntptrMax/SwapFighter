import numpy as np
import torch

from .detect import DetectTrainer
from ...utils.keypoint_loss import compute_keypoint_loss

class KeypointTrainer(DetectTrainer):

    def _compute_loss_parts(self, images, targets, model=None):
        model = model if model is not None else self.model
        cfg = self.cfg

        cls_l, reg_l, kpt_l = model(images)
        detect_parts = self._compute_detect_loss_from_preds(
            cls_l, reg_l, None, targets)
        kpt_loss = compute_keypoint_loss(
            kpt_l, targets,
            pos_batch=detect_parts.get('_pos_batch'),
            pos_point=detect_parts.get('_pos_point'),
            pos_gt=detect_parts.get('_pos_gt'),
            strides=cfg.strides,
            img_size=cfg.img_size,
            num_keypoints=getattr(cfg, 'num_keypoints', 17),
            coord_weight=1.0,
        )

        total = detect_parts['total'] + cfg.keypoint_weight * kpt_loss
        return {
            'total': total,
            'cls': detect_parts['cls'],
            'box': detect_parts['box'],
            'dfl': detect_parts['dfl'],
            'angle': detect_parts['angle'],
            'kpt': kpt_loss.detach(),
            '_pos_batch': detect_parts.get('_pos_batch'),
            '_pos_point': detect_parts.get('_pos_point'),
            '_pos_gt': detect_parts.get('_pos_gt'),
        }

    def _compute_loss_frozen(self, images, targets, model=None):
        model = model if model is not None else self.model
        cfg = self.cfg

        with torch.no_grad():
            feats = model.backbone(images)
        neck_out = model.neck(feats)

        cls_l, reg_l, kpt_l = [], [], []
        for lvl, feat in enumerate(neck_out):
            feat_proj = model.head_proj[lvl](feat)
            preds = model.head_shared(feat_proj)
            cls_l.append(preds['cls'])
            reg_l.append(preds['reg'])
            kpt_l.append(preds['kpt'])

        detect_parts = self._compute_detect_loss_from_preds(
            cls_l, reg_l, None, targets)
        kpt_loss = compute_keypoint_loss(
            kpt_l, targets,
            pos_batch=detect_parts.get('_pos_batch'),
            pos_point=detect_parts.get('_pos_point'),
            pos_gt=detect_parts.get('_pos_gt'),
            strides=cfg.strides,
            img_size=cfg.img_size,
            num_keypoints=getattr(cfg, 'num_keypoints', 17),
        )

        total = detect_parts['total'] + cfg.keypoint_weight * kpt_loss
        return {
            'total': total,
            'cls': detect_parts['cls'],
            'box': detect_parts['box'],
            'dfl': detect_parts['dfl'],
            'angle': detect_parts['angle'],
            'kpt': kpt_loss.detach(),
            '_pos_batch': detect_parts.get('_pos_batch'),
            '_pos_point': detect_parts.get('_pos_point'),
            '_pos_gt': detect_parts.get('_pos_gt'),
        }

    def _compute_detect_loss_from_preds(self, cls_l, reg_l, ang_l, targets):
        cfg = self.cfg
        from ...utils.losses import compute_fcos_loss
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

    @torch.no_grad()
    def validate(self, epoch):
        cfg = self.cfg
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        if (epoch + 1) < cfg.eval_map_start_epoch:
            return {}

        from ...utils.metrics import evaluate_keypoint

        val_loss = self._compute_val_loss(eval_model)

        n_img = getattr(self, '_val_n_images', len(self.val_loader.dataset))
        class_names = getattr(cfg, 'class_names', None)
        verbose = bool(getattr(cfg, 'verbose_metrics', False))

        kpt_header = (
            f"{'Class':>10}{'Images':>7}{'Instances':>10}"
            f"{'Box(P':>8}{'R':>7}{'mAP50':>7}{'mAP50-95)':>10}"
            f"{'Pose(P':>8}{'R':>7}{'mAP50':>7}{'mAP50-95)':>10}"
        )

        bbox_res, pose_res = evaluate_keypoint(
            eval_model, self.val_loader, self.device, cfg,
            conf_threshold=cfg.eval_conf_threshold,
            progress=True, desc=kpt_header,
            return_per_class=True)

        def _means(r):
            if r is None or len(r['classes']) == 0:
                return 0.0, 0.0, 0.0, 0.0
            return (float(r['precision'].mean()),
                    float(r['recall'].mean()),
                    float(r['ap'][:, 0].mean()),
                    float(r['ap'].mean()))

        bp, br, b50, b95 = _means(bbox_res)
        pp, pr, p50, p95 = _means(pose_res)

        n_inst = 0
        if bbox_res and len(bbox_res['classes']) > 0:
            n_inst = int(bbox_res['n_gt'].sum())

        print(f"{'all':>10}{n_img:>7d}{n_inst:>10d}"
              f"{bp:>8.3f}{br:>7.3f}{b50:>7.3f}{b95:>10.3f}"
              f"{pp:>8.3f}{pr:>7.3f}{p50:>7.3f}{p95:>10.3f}")

        if verbose and bbox_res and len(bbox_res['classes']) > 0:
            classes = bbox_res['classes']
            for i, cls_id in enumerate(classes):
                if class_names and int(cls_id) < len(class_names):
                    name = class_names[int(cls_id)][:10]
                else:
                    name = f'c{int(cls_id)}'
                bp_i = float(bbox_res['precision'][i])
                br_i = float(bbox_res['recall'][i])
                b50_i = float(bbox_res['ap'][i, 0])
                b95_i = float(bbox_res['ap'][i].mean())
                if pose_res and i < len(pose_res['classes']):
                    pp_i = float(pose_res['precision'][i])
                    pr_i = float(pose_res['recall'][i])
                    p50_i = float(pose_res['ap'][i, 0])
                    p95_i = float(pose_res['ap'][i].mean())
                else:
                    pp_i = pr_i = p50_i = p95_i = 0.0
                print(f"{name:>10}{n_img:>7d}"
                      f"{int(bbox_res['n_gt'][i]):>10d}"
                      f"{bp_i:>8.3f}{br_i:>7.3f}{b50_i:>7.3f}{b95_i:>10.3f}"
                      f"{pp_i:>8.3f}{pr_i:>7.3f}{p50_i:>7.3f}{p95_i:>10.3f}")

        print()

        return dict(
            val_loss=float(val_loss),
            mAP50=b50, mAP50_95=b95,
            AR100=br, precision=bp, recall=br,
            pose_mAP50=p50, pose_mAP50_95=p95,
        )

__all__ = ['KeypointTrainer']