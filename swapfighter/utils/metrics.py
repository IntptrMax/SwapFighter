import numpy as np
import torch
from tqdm import tqdm

from .box_ops import obb_xy8_to_cxcywh_angle_np
from .postprocess import (
    post_process, post_process_obb, _batch_probiou_np,
)

_PBAR_FORMAT = (
    '{desc}  {percentage:3.0f}% {bar:12} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}'
)

def _wrap_loader(loader, progress, desc):
    if not progress:
        return loader
    return tqdm(
        loader,
        desc=desc or "Validating",
        bar_format=_PBAR_FORMAT,
        leave=True,
        ncols=150,
        dynamic_ncols=False,
    )

def _compute_ap_yolov8(recalls, precisions):
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recalls, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions, dtype=np.float64), [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))

def ap_per_class_yolov8(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    T = tp.shape[1]

    ap = np.zeros((nc, T))
    x = np.linspace(0, 1, 101)
    p_curve = np.zeros((nc, 101))
    r_curve = np.zeros((nc, 101))

    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_l = int(nt[ci])
        n_p = int(mask.sum())
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)

        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[mask], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[mask], precision[:, 0], left=1)

        for j in range(T):
            ap[ci, j] = _compute_ap_yolov8(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i_best = f1_curve.argmax(axis=1)
    p_best = p_curve[np.arange(nc), i_best]
    r_best = r_curve[np.arange(nc), i_best]

    return p_best, r_best, ap, unique_classes

def _iou_matrix_axis_aligned(pred_boxes, gt_boxes, eps=1e-6):
    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / (union + eps)

def _compute_rotated_iou_matrix(preds_obboxes, gts_obboxes):
    n_pred = len(preds_obboxes)
    n_gt = len(gts_obboxes)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    preds_obb5 = obb_xy8_to_cxcywh_angle_np(
        np.asarray(preds_obboxes, dtype=np.float32))
    gts_obb5 = obb_xy8_to_cxcywh_angle_np(
        np.asarray(gts_obboxes, dtype=np.float32))
    iou_mat_gt_pred = _batch_probiou_np(gts_obb5, preds_obb5)
    return np.ascontiguousarray(iou_mat_gt_pred.T, dtype=np.float32)

def _compute_tp_matrix(preds, gts, iou_mat, iou_thresholds):
    scores = np.array([p['score'] for p in preds], dtype=np.float64)
    order = np.argsort(-scores)

    n_pred = len(preds)
    n_gt = len(gts)

    if iou_mat.shape[0] != n_pred:
        if iou_mat.shape[1] == n_pred:
            iou_mat = iou_mat.T
        else:
            raise ValueError(
                f"iou_mat 形状不匹配: {iou_mat.shape}, "
                f"期望 (n_pred={n_pred}, n_gt={n_gt})"
            )
    elif iou_mat.shape[1] != n_gt:
        raise ValueError(
            f"iou_mat 形状不匹配: {iou_mat.shape}, "
            f"期望 (n_pred={n_pred}, n_gt={n_gt})"
        )

    iou_sorted = iou_mat[order]
    scores_sorted = scores[order]

    T = len(iou_thresholds)
    tp = np.zeros((n_pred, T), dtype=bool)
    gt_used = np.zeros((T, n_gt), dtype=bool)
    thresholds_arr = np.asarray(iou_thresholds, dtype=np.float64)

    for pi in range(n_pred):
        ious = iou_sorted[pi]
        candidates = np.where(gt_used, -1.0, ious[None, :])
        best_gi = np.argmax(candidates, axis=1)
        best_iou = candidates[np.arange(T), best_gi]

        valid = best_iou >= thresholds_arr
        tp[pi, valid] = True
        gt_used[valid, best_gi[valid]] = True

    return tp, scores_sorted

def _build_per_class_result(all_tp_list, all_conf, all_pred_cls, all_gts,
                             iou_thresholds):
    if len(all_tp_list) == 0:
        return None

    all_tp = np.concatenate(all_tp_list, axis=0)
    all_conf = np.array(all_conf, dtype=np.float64)
    all_pred_cls = np.array(all_pred_cls, dtype=np.int64)
    all_target_cls = np.array([g['label'] for g in all_gts], dtype=np.int64)

    p, r, ap, unique_classes = ap_per_class_yolov8(
        all_tp, all_conf, all_pred_cls, all_target_cls)

    if len(unique_classes) == 0:
        return None

    n_gt_per_class = np.zeros(len(unique_classes), dtype=np.int64)
    for g in all_gts:
        idx = int(np.searchsorted(unique_classes, g['label']))
        if 0 <= idx < len(unique_classes) and unique_classes[idx] == g['label']:
            n_gt_per_class[idx] += 1

    return dict(
        classes=unique_classes,
        precision=p,
        recall=r,
        ap=ap,
        n_gt=n_gt_per_class,
    )

def _empty_per_class_result():
    return dict(
        classes=np.array([], dtype=np.int64),
        precision=np.array([], dtype=np.float64),
        recall=np.array([], dtype=np.float64),
        ap=np.zeros((0, 10), dtype=np.float64),
        n_gt=np.array([], dtype=np.int64),
    )

def _format_metric_result(per_class, return_per_class):
    if return_per_class:
        if per_class is None:
            return _empty_per_class_result()
        return per_class

    if per_class is None or len(per_class['classes']) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    p_mean = float(per_class['precision'].mean())
    r_mean = float(per_class['recall'].mean())
    ap50 = float(per_class['ap'][:, 0].mean())
    ap95 = float(per_class['ap'].mean())
    return p_mean, r_mean, ap50, ap95, r_mean

@torch.no_grad()
def evaluate_mAP(model, dataloader, device, cfg, conf_threshold=0.01,
                 progress=True, desc=None,
                 return_per_class=False):
    model.eval()
    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)

        out = model(images)
        if isinstance(out, (tuple, list)):
            cls_logits_list = out[0]
            reg_preds_list = out[1]
        else:
            raise ValueError(
                f"model(images) 返回类型不支持: {type(out)}，期望 tuple/list")

        B = images.shape[0]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            boxes, scores, labels = post_process(
                [cls_logits_list[l][idx:idx + 1] for l in range(len(cls_logits_list))],
                [reg_preds_list[l][idx:idx + 1] for l in range(len(reg_preds_list))],
                orig_w, orig_h, cfg.img_size,
                conf_threshold=conf_threshold,
                nms_threshold=cfg.nms_threshold,
                strides=cfg.strides,
                scale=t['scale'],
                pad_left=t['pad_left'],
                pad_top=t['pad_top'],
                reg_max=getattr(cfg, 'reg_max', 0),
            )
            for b, s, l in zip(boxes, scores, labels):
                all_preds.append(dict(img_id=img_id,
                                      box=np.asarray(b, dtype=np.float32),
                                      score=float(s),
                                      label=int(l)))

            gt_boxes = t['boxes'].clone().numpy()
            if gt_boxes.shape[0] > 0:
                gt_fill = gt_boxes * cfg.img_size
                gt_abs = gt_fill.copy()
                gt_abs[:, [0, 2]] = (gt_fill[:, [0, 2]] - t['pad_left']) / t['scale']
                gt_abs[:, [1, 3]] = (gt_fill[:, [1, 3]] - t['pad_top']) / t['scale']
            else:
                gt_abs = np.zeros((0, 4), dtype=np.float32)

            gt_labels = t['labels'].numpy()
            for b, l in zip(gt_abs, gt_labels):
                all_gts.append(dict(img_id=img_id,
                                    box=np.asarray(b, dtype=np.float32),
                                    label=int(l)))
            img_id += 1

        del images, out, cls_logits_list, reg_preds_list
        if progress:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if len(all_gts) == 0 or len(all_preds) == 0:
        return _format_metric_result(None, return_per_class)

    preds_grouped, gts_grouped = {}, {}
    for p in all_preds:
        preds_grouped.setdefault((p['img_id'], p['label']), []).append(p)
    for g in all_gts:
        gts_grouped.setdefault((g['img_id'], g['label']), []).append(g)

    iou_cache = {}
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        preds_boxes = np.stack([p['box'] for p in preds], axis=0)
        gts_boxes = np.stack([g['box'] for g in gts], axis=0)
        iou_cache[key] = _iou_matrix_axis_aligned(preds_boxes, gts_boxes)

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    all_tp_list, all_conf, all_pred_cls = [], [], []
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        iou_mat = iou_cache.get(key)
        if iou_mat is None:
            continue
        tp, scores_sorted = _compute_tp_matrix(preds, gts, iou_mat, iou_thresholds)
        all_tp_list.append(tp)
        all_conf.extend(scores_sorted.tolist())
        all_pred_cls.extend([key[1]] * len(preds))

    per_class = _build_per_class_result(
        all_tp_list, all_conf, all_pred_cls, all_gts, iou_thresholds)
    return _format_metric_result(per_class, return_per_class)

@torch.no_grad()
def evaluate_map_obb_rotated(model, dataloader, device, cfg,
                             conf_threshold=0.01, nms_threshold=None,
                             max_det=100, pre_nms_max_det=300,
                             progress=True, desc=None,
                             return_per_class=False):
    if nms_threshold is None:
        nms_threshold = cfg.obb_nms_threshold

    model.eval()
    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        out = model(images)
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            raise ValueError(
                f"evaluate_map_obb_rotated 需要 obb 模型输出 >=3 项")
        cls_logits_list, reg_preds_list, angle_preds_list = out[0], out[1], out[2]
        B = images.shape[0]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            obboxes, scores, labels = post_process_obb(
                [cls_logits_list[l][idx:idx + 1] for l in range(len(cls_logits_list))],
                [reg_preds_list[l][idx:idx + 1] for l in range(len(reg_preds_list))],
                [angle_preds_list[l][idx:idx + 1] for l in range(len(angle_preds_list))],
                orig_w, orig_h, cfg.img_size,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                strides=cfg.strides,
                scale=t['scale'],
                pad_left=t['pad_left'],
                pad_top=t['pad_top'],
                angle_format=cfg.angle_format,
                angle_range=cfg.angle_range,
                max_det=max_det,
                pre_nms_max_det=pre_nms_max_det,
                reg_max=getattr(cfg, 'reg_max', 0),
            )
            for ob, s, l in zip(obboxes, scores, labels):
                all_preds.append(dict(img_id=img_id,
                                      obbox=np.asarray(ob, dtype=np.float32),
                                      score=float(s),
                                      label=int(l)))

            gt_obboxes = t['obboxes'].clone().numpy()
            if gt_obboxes.shape[0] > 0:
                gt_fill = gt_obboxes * cfg.img_size
                gt_abs = gt_fill.copy()
                for i in range(4):
                    gt_abs[:, i * 2] = (gt_fill[:, i * 2] - t['pad_left']) / t['scale']
                    gt_abs[:, i * 2 + 1] = (gt_fill[:, i * 2 + 1] - t['pad_top']) / t['scale']
            else:
                gt_abs = np.zeros((0, 8), dtype=np.float32)

            gt_labels = t['labels'].numpy()
            for ob, l in zip(gt_abs, gt_labels):
                all_gts.append(dict(img_id=img_id,
                                    obbox=np.asarray(ob, dtype=np.float32),
                                    label=int(l)))
            img_id += 1

        del images, out, cls_logits_list, reg_preds_list, angle_preds_list
        if progress:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if len(all_gts) == 0 or len(all_preds) == 0:
        return _format_metric_result(None, return_per_class)

    preds_grouped, gts_grouped = {}, {}
    for p in all_preds:
        preds_grouped.setdefault((p['img_id'], p['label']), []).append(p)
    for g in all_gts:
        gts_grouped.setdefault((g['img_id'], g['label']), []).append(g)

    iou_cache = {}
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        preds_obb = np.stack([p['obbox'] for p in preds], axis=0)
        gts_obb = np.stack([g['obbox'] for g in gts], axis=0)
        iou_cache[key] = _compute_rotated_iou_matrix(preds_obb, gts_obb)

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    all_tp_list, all_conf, all_pred_cls = [], [], []
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        iou_mat = iou_cache.get(key)
        if iou_mat is None:
            continue
        tp, scores_sorted = _compute_tp_matrix(preds, gts, iou_mat, iou_thresholds)
        all_tp_list.append(tp)
        all_conf.extend(scores_sorted.tolist())
        all_pred_cls.extend([key[1]] * len(preds))

    per_class = _build_per_class_result(
        all_tp_list, all_conf, all_pred_cls, all_gts, iou_thresholds)
    return _format_metric_result(per_class, return_per_class)

@torch.no_grad()
def evaluate_mask_mAP(model, dataloader, device, cfg,
                      conf_threshold=0.001,
                      mask_iou_thresholds=None,
                      progress=True, desc=None,
                      return_per_class=False,
                      max_det_for_mask=30):
    if mask_iou_thresholds is None:
        mask_iou_thresholds = np.linspace(0.5, 0.95, 10)

    model.eval()
    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        out = model(images)
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise ValueError(
                "evaluate_mask_mAP 需要 seg 模型输出 4 项 (cls, reg, coef, proto)")
        cls_l, reg_l, coef_l, proto = out[0], out[1], out[2], out[3]
        B = images.shape[0]

        all_coef = []
        for coef in coef_l:
            b, c, h, w = coef.shape
            all_coef.append(coef.permute(0, 2, 3, 1).reshape(b, -1, c))
        all_coef = torch.cat(all_coef, dim=1)

        ph, pw = proto.shape[2], proto.shape[3]
        C = proto.shape[1]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            boxes, scores, labels, origin_idx = post_process(
                [cls_l[l][idx:idx + 1] for l in range(len(cls_l))],
                [reg_l[l][idx:idx + 1] for l in range(len(reg_l))],
                orig_w, orig_h, cfg.img_size,
                conf_threshold=conf_threshold,
                nms_threshold=cfg.nms_threshold,
                strides=cfg.strides,
                scale=t['scale'],
                pad_left=t['pad_left'],
                pad_top=t['pad_top'],
                reg_max=getattr(cfg, 'reg_max', 0),
                return_indices=True,
            )

            if 'mask_per_gt' in t and t['mask_per_gt'].numel() > 0:
                gt_masks = t['mask_per_gt'].cpu().numpy()
                gt_labels = t['labels'].cpu().numpy()

                assert gt_masks.shape[1] == ph and gt_masks.shape[2] == pw, (
                    f"[evaluate_mask_mAP] GT mask 尺寸 {gt_masks.shape[1]}×"
                    f"{gt_masks.shape[2]} != proto 尺寸 {ph}×{pw}，"
                    f"请检查 proto_stride 是否在模型和数据集两侧一致"
                )

                for gm, gl in zip(gt_masks, gt_labels):
                    gm_bin = (gm > 0.5).astype(np.uint8)
                    all_gts.append(dict(img_id=img_id, mask=gm_bin,
                                        label=int(gl)))

            if len(scores) == 0:
                img_id += 1
                continue

            if len(scores) > max_det_for_mask:
                order = np.argsort(-np.asarray(scores))[:max_det_for_mask]
                boxes = np.asarray(boxes)[order]
                scores = np.asarray(scores)[order]
                labels = np.asarray(labels)[order]
                origin_idx = [origin_idx[i] for i in order]

            origin_tensor = torch.tensor(origin_idx, device=device,
                                          dtype=torch.long)
            coef_vecs = all_coef[idx][origin_tensor]
            proto_flat = proto[idx].reshape(C, -1)
            masks_logits = coef_vecs @ proto_flat
            masks_prob = torch.sigmoid(masks_logits).reshape(-1, ph, pw)
            masks_bin = (masks_prob > 0.5).cpu().numpy().astype(np.uint8)

            t_scale = t['scale']
            pad_left = t['pad_left']
            pad_top = t['pad_top']
            proto_sx = pw / cfg.img_size
            proto_sy = ph / cfg.img_size

            for i, bi in enumerate(boxes):

                if t_scale is not None and t_scale > 0:
                    bx1 = (bi[0] - pad_left) / t_scale
                    by1 = (bi[1] - pad_top) / t_scale
                    bx2 = (bi[2] - pad_left) / t_scale
                    by2 = (bi[3] - pad_top) / t_scale
                else:
                    bx1 = bi[0] * cfg.img_size / orig_w
                    by1 = bi[1] * cfg.img_size / orig_h
                    bx2 = bi[2] * cfg.img_size / orig_w
                    by2 = bi[3] * cfg.img_size / orig_h

                x1p = max(0, int(bx1 * proto_sx))
                y1p = max(0, int(by1 * proto_sy))
                x2p = min(pw, int(bx2 * proto_sx))
                y2p = min(ph, int(by2 * proto_sy))

                cropped = np.zeros((ph, pw), dtype=np.uint8)
                if x2p > x1p and y2p > y1p:
                    cropped[y1p:y2p, x1p:x2p] = masks_bin[i, y1p:y2p, x1p:x2p]
                all_preds.append(dict(
                    img_id=img_id, mask=cropped,
                    score=float(scores[i]), label=int(labels[i])))

            img_id += 1

        del images, out, all_coef
        if progress:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if len(all_gts) == 0 or len(all_preds) == 0:
        return _format_metric_result(None, return_per_class)

    preds_grouped, gts_grouped = {}, {}
    for p in all_preds:
        preds_grouped.setdefault((p['img_id'], p['label']), []).append(p)
    for g in all_gts:
        gts_grouped.setdefault((g['img_id'], g['label']), []).append(g)

    def _mask_iou_matrix(pred_masks, gt_masks):
        n_p = len(pred_masks)
        n_g = len(gt_masks)
        if n_p == 0 or n_g == 0:
            return np.zeros((n_p, n_g), dtype=np.float32)

        pm_shape = pred_masks[0].shape
        gm_shape = gt_masks[0].shape
        assert pm_shape == gm_shape, (
            f"[evaluate_mask_mAP] mask 尺寸不匹配："
            f"预测 {pm_shape}，GT {gm_shape}"
        )

        P = np.stack([(m.reshape(-1) > 0) for m in pred_masks],
                     axis=0).astype(np.float32)
        G = np.stack([(m.reshape(-1) > 0) for m in gt_masks],
                     axis=0).astype(np.float32)

        inter = P @ G.T
        area_p = P.sum(axis=1, keepdims=True)
        area_g = G.sum(axis=1, keepdims=True).T
        union = area_p + area_g - inter
        return (inter / (union + 1e-6)).astype(np.float32)

    iou_cache = {}
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        iou_cache[key] = _mask_iou_matrix(
            [p['mask'] for p in preds],
            [g['mask'] for g in gts])

    all_tp_list, all_conf, all_pred_cls = [], [], []
    for key, preds in preds_grouped.items():
        gts = gts_grouped.get(key, [])
        if len(gts) == 0:
            continue
        iou_mat = iou_cache.get(key)
        if iou_mat is None:
            continue
        tp, scores_sorted = _compute_tp_matrix(
            preds, gts, iou_mat, mask_iou_thresholds)
        all_tp_list.append(tp)
        all_conf.extend(scores_sorted.tolist())
        all_pred_cls.extend([key[1]] * len(preds))

    per_class = _build_per_class_result(
        all_tp_list, all_conf, all_pred_cls, all_gts, mask_iou_thresholds)
    return _format_metric_result(per_class, return_per_class)

def format_metric_table(res_list, n_img, class_names=None,
                        verbose=False, indent=''):
    if not res_list:
        return []

    lines = []

    header = f"{'Class':>10}{'Images':>7}{'Instances':>10}"
    for prefix, _ in res_list:
        header += (f"{prefix + '(P':>8}{'R':>7}"
                   f"{'mAP50':>7}{'mAP50-95)':>10}")
    lines.append(indent + header)

    primary_res = res_list[0][1]
    classes = primary_res['classes']

    if verbose:
        for i, cls_id in enumerate(classes):
            if class_names and int(cls_id) < len(class_names):
                name = class_names[int(cls_id)]
            else:
                name = f'cls_{int(cls_id)}'
            name = name[:10]
            row = f"{name:>10}{n_img:>7d}"
            row += f"{int(primary_res['n_gt'][i]):>10d}"
            for _, res in res_list:
                p = float(res['precision'][i]) if i < len(res['precision']) else 0.0
                r = float(res['recall'][i]) if i < len(res['recall']) else 0.0
                ap50 = float(res['ap'][i, 0]) if i < len(res['ap']) else 0.0
                ap95 = float(res['ap'][i].mean()) if i < len(res['ap']) else 0.0
                row += f"{p:>8.3f}{r:>7.3f}{ap50:>7.3f}{ap95:>10.3f}"
            lines.append(indent + row)

    if len(primary_res['classes']) > 0:
        total_inst = int(primary_res['n_gt'].sum())
    else:
        total_inst = 0

    row = f"{'all':>10}{n_img:>7d}{total_inst:>10d}"
    for _, res in res_list:
        if len(res['classes']) > 0:
            p = float(res['precision'].mean())
            r = float(res['recall'].mean())
            ap50 = float(res['ap'][:, 0].mean())
            ap95 = float(res['ap'].mean())
        else:
            p = r = ap50 = ap95 = 0.0
        row += f"{p:>8.3f}{r:>7.3f}{ap50:>7.3f}{ap95:>10.3f}"
    lines.append(indent + row)

    return lines

@torch.no_grad()
def evaluate_seg(model, dataloader, device, cfg,
                 conf_threshold=0.001,
                 progress=True, desc=None,
                 return_per_class=False,
                 max_det_for_mask=30):
    model.eval()

    bbox_preds, bbox_gts = [], []
    mask_preds, mask_gts = [], []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        out = model(images)
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise ValueError("evaluate_seg 需要 seg 模型输出 4 项")
        cls_l, reg_l, coef_l, proto = out[0], out[1], out[2], out[3]
        B = images.shape[0]

        all_coef = []
        for coef in coef_l:
            _, C, H, W = coef.shape
            all_coef.append(coef.permute(0, 2, 3, 1).reshape(B, -1, C))
        all_coef = torch.cat(all_coef, dim=1)

        ph, pw = proto.shape[2], proto.shape[3]
        C = proto.shape[1]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            boxes, scores, labels, origin_idx = post_process(
                [cls_l[l][idx:idx + 1] for l in range(len(cls_l))],
                [reg_l[l][idx:idx + 1] for l in range(len(reg_l))],
                orig_w, orig_h, cfg.img_size,
                conf_threshold=conf_threshold,
                nms_threshold=cfg.nms_threshold,
                strides=cfg.strides,
                scale=t['scale'],
                pad_left=t['pad_left'],
                pad_top=t['pad_top'],
                reg_max=getattr(cfg, 'reg_max', 0),
                return_indices=True,
            )

            gt_boxes = t['boxes'].clone().numpy()
            if gt_boxes.shape[0] > 0:
                gf = gt_boxes * cfg.img_size
                gtb = gf.copy()
                gtb[:, [0, 2]] = (gf[:, [0, 2]] - t['pad_left']) / t['scale']
                gtb[:, [1, 3]] = (gf[:, [1, 3]] - t['pad_top']) / t['scale']
            else:
                gtb = np.zeros((0, 4), dtype=np.float32)
            gt_labels_np = t['labels'].numpy()

            for b, l in zip(gtb, gt_labels_np):
                bbox_gts.append(dict(img_id=img_id,
                                     box=np.asarray(b, dtype=np.float32),
                                     label=int(l)))

            if 'mask_per_gt' in t and t['mask_per_gt'].numel() > 0:
                gm_all = t['mask_per_gt'].cpu().numpy()

                assert gm_all.shape[1] == ph and gm_all.shape[2] == pw, (
                    f"[evaluate_seg] GT mask 尺寸 {gm_all.shape[1]}×"
                    f"{gm_all.shape[2]} != proto 尺寸 {ph}×{pw}，"
                    f"请检查 proto_stride 是否在模型和数据集两侧一致"
                )

                for gm, gl in zip(gm_all, gt_labels_np):
                    gm_bin = (gm > 0.5).astype(np.uint8)
                    mask_gts.append(dict(img_id=img_id, mask=gm_bin,
                                          label=int(gl)))

            if len(scores) == 0:
                img_id += 1
                continue

            for b, s, l in zip(boxes, scores, labels):
                bbox_preds.append(dict(img_id=img_id,
                                       box=np.asarray(b, dtype=np.float32),
                                       score=float(s),
                                       label=int(l)))

            if len(scores) > max_det_for_mask:
                order = np.argsort(-np.asarray(scores))[:max_det_for_mask]
                boxes = np.asarray(boxes)[order]
                scores = np.asarray(scores)[order]
                labels = np.asarray(labels)[order]
                origin_idx = [origin_idx[i] for i in order]

            oi_t = torch.tensor(origin_idx, device=device, dtype=torch.long)
            coef_vecs = all_coef[idx][oi_t]
            proto_flat = proto[idx].reshape(C, -1)
            masks_logits = coef_vecs @ proto_flat
            masks_prob = torch.sigmoid(masks_logits).reshape(-1, ph, pw)
            masks_bin = (masks_prob > 0.5).cpu().numpy().astype(np.uint8)

            t_scale = t['scale']
            pad_left = t['pad_left']
            pad_top = t['pad_top']
            proto_sx = pw / cfg.img_size
            proto_sy = ph / cfg.img_size

            for i, bi in enumerate(boxes):

                if t_scale is not None and t_scale > 0:
                    bx1 = (bi[0] - pad_left) / t_scale
                    by1 = (bi[1] - pad_top) / t_scale
                    bx2 = (bi[2] - pad_left) / t_scale
                    by2 = (bi[3] - pad_top) / t_scale
                else:
                    bx1 = bi[0] * cfg.img_size / orig_w
                    by1 = bi[1] * cfg.img_size / orig_h
                    bx2 = bi[2] * cfg.img_size / orig_w
                    by2 = bi[3] * cfg.img_size / orig_h

                x1p = max(0, int(bx1 * proto_sx))
                y1p = max(0, int(by1 * proto_sy))
                x2p = min(pw, int(bx2 * proto_sx))
                y2p = min(ph, int(by2 * proto_sy))

                cropped = np.zeros((ph, pw), dtype=np.uint8)
                if x2p > x1p and y2p > y1p:
                    cropped[y1p:y2p, x1p:x2p] = \
                        masks_bin[i, y1p:y2p, x1p:x2p]

                mask_preds.append(dict(img_id=img_id, mask=cropped,
                                       score=float(scores[i]),
                                       label=int(labels[i])))

            img_id += 1

        del images, out, all_coef
        if progress:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    bbox_res = _compute_ap_axis_aligned(bbox_preds, bbox_gts)
    mask_res = _compute_ap_mask(mask_preds, mask_gts)

    if return_per_class:
        return bbox_res, mask_res

    def _fmt(r):
        if r is None or len(r['classes']) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return (float(r['precision'].mean()),
                float(r['recall'].mean()),
                float(r['ap'][:, 0].mean()),
                float(r['ap'].mean()),
                float(r['recall'].mean()))

    return _fmt(bbox_res), _fmt(mask_res)

def _oks(pred_kpt, gt_kpt, bbox_wh, kpt_sigmas):
    v = (gt_kpt[:, 2] > 0).astype(np.float32)
    if v.sum() == 0:
        return 0.0

    d2 = (pred_kpt[:, 0] - gt_kpt[:, 0]) ** 2 + \
         (pred_kpt[:, 1] - gt_kpt[:, 1]) ** 2

    s = float(np.sqrt(max(bbox_wh[0] * bbox_wh[1], 1.0)))

    e = d2 / (2.0 * s ** 2 * kpt_sigmas ** 2 + 1e-9)
    oks = np.exp(-e) * v
    return float(oks.sum() / v.sum())

def _compute_ap_pose(pose_preds, pose_gts, K, kpt_sigmas=None):
    if kpt_sigmas is None:

        if K == 17:
            kpt_sigmas = np.array([
                0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
                0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
        else:
            kpt_sigmas = np.full(K, 0.1)

    if len(pose_gts) == 0 or len(pose_preds) == 0:
        return None

    pg, gg = {}, {}
    for p in pose_preds:
        pg.setdefault((p['img_id'], p['label']), []).append(p)
    for g in pose_gts:
        gg.setdefault((g['img_id'], g['label']), []).append(g)

    oks_cache = {}
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        n_p, n_g = len(ps), len(gs)
        oks_mat = np.zeros((n_p, n_g), dtype=np.float32)
        for i, p in enumerate(ps):
            for j, g in enumerate(gs):
                gk = g['kpt']

                x_min, x_max = gk[:, 0].min(), gk[:, 0].max()
                y_min, y_max = gk[:, 1].min(), gk[:, 1].max()
                bw = max(x_max - x_min, 1.0)
                bh = max(y_max - y_min, 1.0)
                oks_mat[i, j] = _oks(p['kpt'], gk, (bw, bh), kpt_sigmas)
        oks_cache[k] = oks_mat

    thr = np.linspace(0.5, 0.95, 10)
    tp_list, conf_list, cls_list = [], [], []
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        im = oks_cache.get(k)
        if im is None:
            continue
        tp, sc = _compute_tp_matrix(ps, gs, im, thr)
        tp_list.append(tp)
        conf_list.extend(sc.tolist())
        cls_list.extend([k[1]] * len(ps))

    return _build_per_class_result(
        tp_list, conf_list, cls_list, pose_gts, thr)

@torch.no_grad()
def evaluate_keypoint(model, dataloader, device, cfg,
                      conf_threshold=0.001,
                      progress=True, desc=None,
                      return_per_class=False,
                      max_det_for_kpt=30):
    K = getattr(cfg, 'num_keypoints', 17)

    model.eval()

    bbox_preds, bbox_gts = [], []
    pose_preds, pose_gts = [], []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        out = model(images)
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            raise ValueError(
                "evaluate_keypoint 需要 keypoint 模型 3 项输出")
        cls_l, reg_l, kpt_l = out[0], out[1], out[2]
        B = images.shape[0]

        all_kpt = []
        for kpt in kpt_l:
            b, c, h, w = kpt.shape
            all_kpt.append(kpt.reshape(b, c, -1))
        all_kpt = torch.cat(all_kpt, dim=2)

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            boxes, scores, labels, origin_idx = post_process(
                [cls_l[l][idx:idx + 1] for l in range(len(cls_l))],
                [reg_l[l][idx:idx + 1] for l in range(len(reg_l))],
                orig_w, orig_h, cfg.img_size,
                conf_threshold=conf_threshold,
                nms_threshold=cfg.nms_threshold,
                strides=cfg.strides,
                scale=t['scale'],
                pad_left=t['pad_left'],
                pad_top=t['pad_top'],
                reg_max=getattr(cfg, 'reg_max', 0),
                return_indices=True,
            )

            gt_boxes = t['boxes'].clone().numpy()
            if gt_boxes.shape[0] > 0:
                gf = gt_boxes * cfg.img_size
                gtb = gf.copy()
                gtb[:, [0, 2]] = (gf[:, [0, 2]] - t['pad_left']) / t['scale']
                gtb[:, [1, 3]] = (gf[:, [1, 3]] - t['pad_top']) / t['scale']
            else:
                gtb = np.zeros((0, 4), dtype=np.float32)
            gt_labels_np = t['labels'].numpy()

            for b, l in zip(gtb, gt_labels_np):
                bbox_gts.append(dict(img_id=img_id,
                                     box=np.asarray(b, dtype=np.float32),
                                     label=int(l)))

            if 'keypoints' in t and t['keypoints'].numel() > 0:
                gt_kpts = t['keypoints'].cpu().numpy()
                for kp, l in zip(gt_kpts, gt_labels_np):
                    kp = kp.copy()
                    if t['scale'] > 0:
                        kp[:, 0] = (kp[:, 0] - t['pad_left']) / t['scale']
                        kp[:, 1] = (kp[:, 1] - t['pad_top']) / t['scale']
                    pose_gts.append(dict(
                        img_id=img_id,
                        kpt=np.asarray(kp, dtype=np.float32),
                        label=int(l)))

            if len(scores) == 0:
                img_id += 1
                continue

            for b, s, l in zip(boxes, scores, labels):
                bbox_preds.append(dict(img_id=img_id,
                                       box=np.asarray(b, dtype=np.float32),
                                       score=float(s),
                                       label=int(l)))

            if len(scores) > max_det_for_kpt:
                order = np.argsort(-np.asarray(scores))[:max_det_for_kpt]
                boxes = np.asarray(boxes)[order]
                scores = np.asarray(scores)[order]
                labels = np.asarray(labels)[order]
                origin_idx = [origin_idx[i] for i in order]

            for b, s, l, oi in zip(boxes, scores, labels, origin_idx):
                if oi >= all_kpt.shape[2]:
                    continue
                kpt_norm = all_kpt[idx, :, oi].cpu().numpy().reshape(K, 2)
                kpt_xy = kpt_norm * cfg.img_size
                if t['scale'] > 0:
                    kpt_xy[:, 0] = (kpt_xy[:, 0] - t['pad_left']) / t['scale']
                    kpt_xy[:, 1] = (kpt_xy[:, 1] - t['pad_top']) / t['scale']
                kpt_xy[:, 0] = np.clip(kpt_xy[:, 0], 0, orig_w)
                kpt_xy[:, 1] = np.clip(kpt_xy[:, 1], 0, orig_h)

                pose_preds.append(dict(
                    img_id=img_id,
                    kpt=kpt_xy.astype(np.float32),
                    score=float(s),
                    label=int(l)))

            img_id += 1

        del images, out, all_kpt

    bbox_res = _compute_ap_axis_aligned(bbox_preds, bbox_gts)
    pose_res = _compute_ap_pose(pose_preds, pose_gts, K)

    if return_per_class:
        return bbox_res, pose_res

    def _fmt(r):
        if r is None or len(r['classes']) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return (float(r['precision'].mean()),
                float(r['recall'].mean()),
                float(r['ap'][:, 0].mean()),
                float(r['ap'].mean()),
                float(r['recall'].mean()))

    return _fmt(bbox_res), _fmt(pose_res)

def _compute_ap_axis_aligned(preds, gts):
    if len(gts) == 0 or len(preds) == 0:
        return None
    pg, gg = {}, {}
    for p in preds:
        pg.setdefault((p['img_id'], p['label']), []).append(p)
    for g in gts:
        gg.setdefault((g['img_id'], g['label']), []).append(g)

    iou_cache = {}
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        pb = np.stack([p['box'] for p in ps], axis=0)
        gb = np.stack([g['box'] for g in gs], axis=0)
        iou_cache[k] = _iou_matrix_axis_aligned(pb, gb)

    thr = np.linspace(0.5, 0.95, 10)
    tp_list, conf_list, cls_list = [], [], []
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        im = iou_cache.get(k)
        if im is None:
            continue
        tp, sc = _compute_tp_matrix(ps, gs, im, thr)
        tp_list.append(tp)
        conf_list.extend(sc.tolist())
        cls_list.extend([k[1]] * len(ps))

    return _build_per_class_result(tp_list, conf_list, cls_list, gts, thr)

def _compute_ap_mask(preds, gts):
    if len(gts) == 0 or len(preds) == 0:
        return None
    pg, gg = {}, {}
    for p in preds:
        pg.setdefault((p['img_id'], p['label']), []).append(p)
    for g in gts:
        gg.setdefault((g['img_id'], g['label']), []).append(g)

    def _miou(pm_list, gm_list):
        n_p, n_g = len(pm_list), len(gm_list)
        if n_p == 0 or n_g == 0:
            return np.zeros((n_p, n_g), dtype=np.float32)

        pm_shape = pm_list[0].shape
        gm_shape = gm_list[0].shape
        assert pm_shape == gm_shape, (
            f"[_compute_ap_mask] mask 尺寸不匹配："
            f"预测 {pm_shape}，GT {gm_shape}。"
            f"请检查 proto_stride 是否在模型和数据集两侧一致"
        )

        P = np.stack([(m.reshape(-1) > 0) for m in pm_list],
                     axis=0).astype(np.float32)
        G = np.stack([(m.reshape(-1) > 0) for m in gm_list],
                     axis=0).astype(np.float32)
        inter = P @ G.T
        ap_ = P.sum(1, keepdims=True)
        ag_ = G.sum(1, keepdims=True).T
        union = ap_ + ag_ - inter
        return (inter / (union + 1e-6)).astype(np.float32)

    iou_cache = {}
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        iou_cache[k] = _miou([p['mask'] for p in ps],
                              [g['mask'] for g in gs])

    thr = np.linspace(0.5, 0.95, 10)
    tp_list, conf_list, cls_list = [], [], []
    for k, ps in pg.items():
        gs = gg.get(k, [])
        if not gs:
            continue
        im = iou_cache.get(k)
        if im is None:
            continue
        tp, sc = _compute_tp_matrix(ps, gs, im, thr)
        tp_list.append(tp)
        conf_list.extend(sc.tolist())
        cls_list.extend([k[1]] * len(ps))

    return _build_per_class_result(tp_list, conf_list, cls_list, gts, thr)

__all__ = [
    'evaluate_mAP',
    'evaluate_map_obb_rotated',
    'evaluate_mask_mAP',
    'evaluate_seg',
    'evaluate_keypoint',
    'format_metric_table',
    'ap_per_class_yolov8',
]