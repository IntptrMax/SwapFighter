import numpy as np
import torch
from tqdm import tqdm

from .box_ops import obb_xy8_to_cxcywh_angle_np
from .postprocess import (
    post_process, post_process_obb, _batch_probiou_np,
)


# ============================================================
# tqdm 进度条样式（与 train.py 验证时保持一致）
# ============================================================
_PBAR_FORMAT = (
    '{desc}: {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}'
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


# ============================================================
# 101 点插值 AP
# ============================================================
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


# ============================================================
# ap_per_class
# ============================================================
def ap_per_class_yolov8(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    与 ultralytics/utils/metrics.py::ap_per_class 一致。
    返回: p, r, ap, unique_classes
    """
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


# ============================================================
# 轴对齐 IoU 矩阵（pred, gt）
# ============================================================
def _iou_matrix_axis_aligned(pred_boxes, gt_boxes, eps=1e-6):
    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / (union + eps)


# ============================================================
# 旋转 IoU 矩阵（明确 (n_pred, n_gt)）
# ============================================================
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


# ============================================================
# TP 矩阵
# ============================================================
def _compute_tp_matrix(preds, gts, iou_mat, iou_thresholds):
    scores = np.array([p['score'] for p in preds], dtype=np.float64)
    order = np.argsort(-scores)

    n_pred = len(preds)
    n_gt = len(gts)

    # 自动纠正朝向
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


# ============================================================
# 水平框 mAP
# ============================================================
@torch.no_grad()
def evaluate_mAP(model, dataloader, device, cfg, conf_threshold=0.01,
                 progress=True, desc=None):
    model.eval()
    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        cls_logits_list, reg_preds_list = model(images)
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

    if len(all_gts) == 0 or len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

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

    if len(all_tp_list) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_tp = np.concatenate(all_tp_list, axis=0)
    all_conf = np.array(all_conf, dtype=np.float64)
    all_pred_cls = np.array(all_pred_cls, dtype=np.int64)
    all_target_cls = np.array([g['label'] for g in all_gts], dtype=np.int64)

    p, r, ap, unique_classes = ap_per_class_yolov8(
        all_tp, all_conf, all_pred_cls, all_target_cls)

    if len(unique_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (float(p.mean()), float(r.mean()),
            float(ap[:, 0].mean()), float(ap.mean()), float(r.mean()))


# ============================================================
# OBB mAP
# ============================================================
@torch.no_grad()
def evaluate_map_obb_rotated(model, dataloader, device, cfg,
                             conf_threshold=0.01, nms_threshold=None,
                             max_det=100, pre_nms_max_det=300,
                             progress=True, desc=None):
    if nms_threshold is None:
        nms_threshold = cfg.obb_nms_threshold

    model.eval()
    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in _wrap_loader(dataloader, progress, desc):
        images = images.to(device)
        cls_logits_list, reg_preds_list, angle_preds_list = model(images)
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

    if len(all_gts) == 0 or len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

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

    if len(all_tp_list) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_tp = np.concatenate(all_tp_list, axis=0)
    all_conf = np.array(all_conf, dtype=np.float64)
    all_pred_cls = np.array(all_pred_cls, dtype=np.int64)
    all_target_cls = np.array([g['label'] for g in all_gts], dtype=np.int64)

    p, r, ap, unique_classes = ap_per_class_yolov8(
        all_tp, all_conf, all_pred_cls, all_target_cls)

    if len(unique_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (float(p.mean()), float(r.mean()),
            float(ap[:, 0].mean()), float(ap.mean()), float(r.mean()))