import math
import torch
import numpy as np
from torchvision.ops import nms

from .box_ops import (
    LOG_SCALE_MIN, LOG_SCALE_MAX,
    cxcywh_angle_to_obb, obb_xy8_to_cxcywh_angle_np, cxcywh_angle_to_obb_np,
)

def _batch_probiou_np(gt_obb, pred_obb, eps=1e-6):
    gt_w = np.maximum(gt_obb[:, 2], eps)
    gt_h = np.maximum(gt_obb[:, 3], eps)
    gc = np.cos(gt_obb[:, 4]); gs = np.sin(gt_obb[:, 4])
    ga = gt_w ** 2 / 12 * gc ** 2 + gt_h ** 2 / 12 * gs ** 2
    gb = gt_w ** 2 / 12 * gs ** 2 + gt_h ** 2 / 12 * gc ** 2
    gcc = (gt_w ** 2 / 12 - gt_h ** 2 / 12) * gc * gs
    gab = np.maximum(ga * gb - gcc ** 2, eps)
    gcx = gt_obb[:, 0:1]; gcy = gt_obb[:, 1:2]

    pw = np.maximum(pred_obb[:, 2], eps)
    ph = np.maximum(pred_obb[:, 3], eps)
    pc = np.cos(pred_obb[:, 4]); ps = np.sin(pred_obb[:, 4])
    pa = pw ** 2 / 12 * pc ** 2 + ph ** 2 / 12 * ps ** 2
    pb = pw ** 2 / 12 * ps ** 2 + ph ** 2 / 12 * pc ** 2
    pcc = (pw ** 2 / 12 - ph ** 2 / 12) * pc * ps
    pab = np.maximum(pa * pb - pcc ** 2, eps)
    pcx = pred_obb[:, 0]; pcy = pred_obb[:, 1]

    a_sum = ga[:, None] + pa[None, :]
    b_sum = gb[:, None] + pb[None, :]
    c_sum = gcc[:, None] + pcc[None, :]
    denom = np.maximum(a_sum * b_sum - c_sum ** 2, eps)

    dx = gcx - pcx[None, :]
    dy = gcy - pcy[None, :]

    t1 = ((a_sum * dy ** 2 + b_sum * dx ** 2) / denom) * 0.25
    t2 = -(c_sum * dx * dy / denom) * 0.5

    log_arg = np.maximum(denom / (4 * np.sqrt(gab[:, None] * pab[None, :]) + eps), eps)
    t3 = np.log(log_arg) * 0.5

    bd = np.clip(t1 + t2 + t3, eps, 20.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd

def rotated_nms(obboxes, scores, iou_threshold=0.5, labels=None):
    if len(obboxes) == 0:
        return np.array([], dtype=np.int64)

    obboxes = np.asarray(obboxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    obb5 = obb_xy8_to_cxcywh_angle_np(obboxes)

    def _nms_single(obb5_sub, scores_sub, global_idx, thr):
        keep_local = []
        order = np.argsort(-scores_sub)
        while order.size > 0:
            i = int(order[0])
            keep_local.append(i)
            if order.size == 1:
                break
            cur = obb5_sub[i:i + 1]
            rest = obb5_sub[order[1:]]
            ious = _batch_probiou_np(cur, rest).squeeze(0)
            remaining = np.where(ious <= thr)[0]
            order = order[remaining + 1]
        return [global_idx[i] for i in keep_local]

    if labels is None:
        keep = _nms_single(obb5, scores, list(range(len(obb5))), iou_threshold)
        return np.array(keep, dtype=np.int64)

    labels = np.asarray(labels)
    keep_all = []
    for cls_id in np.unique(labels):
        cls_idx = np.where(labels == cls_id)[0]
        if len(cls_idx) == 0:
            continue
        keep_local = _nms_single(obb5[cls_idx], scores[cls_idx],
                                 list(cls_idx), iou_threshold)
        keep_all.extend(keep_local)
    keep_all = sorted(keep_all, key=lambda x: -scores[x])
    return np.array(keep_all, dtype=np.int64)

def _classwise_nms(boxes, scores, labels, nms_threshold):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    keep_list = []
    for cls_id in torch.unique(labels):
        cls_mask = (labels == cls_id)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_keep = nms(cls_boxes, cls_scores, nms_threshold)
        global_idx = torch.where(cls_mask)[0][cls_keep]
        keep_list.append(global_idx)
    keep = torch.cat(keep_list)
    order = torch.argsort(scores[keep], descending=True)
    return keep[order]

def post_process(cls_logits_list, reg_preds_list,
                 orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                 strides=None, scale=None, pad_left=0, pad_top=0,
                 reg_max=0,
                 return_indices=False):
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2]
                   for i in range(num_levels)]

    use_dfl = reg_max > 0
    all_boxes, all_scores, all_labels = [], [], []

    origin_point = []
    offset = 0

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx].squeeze(0)
        reg_preds = reg_preds_list[level_idx].squeeze(0)

        C, H, W = cls_logits.shape
        scores = torch.sigmoid(cls_logits)
        max_scores, max_labels = torch.max(scores, dim=0)
        keep = max_scores > conf_threshold
        if not keep.any():
            offset += H * W
            continue

        y_keep, x_keep = torch.where(keep)
        local_idx = y_keep * W + x_keep
        origin_point.extend((offset + local_idx).cpu().tolist())
        offset += H * W

        y_coords = torch.arange(H, device=device).float() * stride + stride / 2
        x_coords = torch.arange(W, device=device).float() * stride + stride / 2
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        y_grid = grid_y[keep]; x_grid = grid_x[keep]
        scores_keep = max_scores[keep]; labels_keep = max_labels[keep]
        reg = reg_preds[:, keep]

        if use_dfl:
            reg = reg.view(4, reg_max + 1, -1)
            probs = reg.softmax(dim=1)
            idx = torch.arange(reg_max + 1, device=device).view(1, -1, 1).float()
            distances = (probs * idx).sum(dim=1)
            l_dist, t_dist, r_dist, b_dist = distances.unbind(0)
            x1 = x_grid - l_dist * stride
            y1 = y_grid - t_dist * stride
            x2 = x_grid + r_dist * stride
            y2 = y_grid + b_dist * stride
        else:
            dx = reg[0]; dy = reg[1]
            dw = reg[2].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            dh = reg[3].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            cx = x_grid + dx * stride; cy = y_grid + dy * stride
            w = torch.exp(dw) * stride; h = torch.exp(dh) * stride
            x1 = cx - w / 2; y1 = cy - h / 2
            x2 = cx + w / 2; y2 = cy + h / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        all_boxes.append(boxes)
        all_scores.append(scores_keep)
        all_labels.append(labels_keep)

    if not all_boxes:
        if return_indices:
            return [], [], [], []
        return [], [], []

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if scale is not None:
        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale
    else:
        boxes[:, [0, 2]] *= orig_w / img_size
        boxes[:, [1, 3]] *= orig_h / img_size

    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, orig_h)

    keep = _classwise_nms(boxes, scores, labels, nms_threshold)
    keep_list = keep.cpu().tolist()

    boxes_np = boxes[keep].cpu().numpy()
    scores_np = scores[keep].cpu().numpy()
    labels_np = labels[keep].cpu().numpy()

    if return_indices:

        origin_kept = [origin_point[i] for i in keep_list]
        return boxes_np, scores_np, labels_np, origin_kept

    return boxes_np, scores_np, labels_np

def post_process_obb(cls_logits_list, reg_preds_list, angle_preds_list,
                     orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                     strides=None, scale=None, pad_left=0, pad_top=0,
                     angle_format='sin_cos', angle_range='pi',
                     max_det=1000, pre_nms_max_det=300,
                     reg_max=16):
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2] for i in range(num_levels)]

    use_dfl = reg_max > 0
    MAX_ALLOWED_SIDE = img_size * 0.8
    MIN_ALLOWED_SIDE = 2.0

    all_obboxes, all_scores, all_labels = [], [], []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx].squeeze(0)
        reg_preds = reg_preds_list[level_idx].squeeze(0)
        angle_preds = angle_preds_list[level_idx].squeeze(0)

        C, H, W = cls_logits.shape
        scores = torch.sigmoid(cls_logits)
        max_scores, max_labels = torch.max(scores, dim=0)
        keep = max_scores > conf_threshold
        if not keep.any():
            continue

        y_coords = torch.arange(H, device=device).float() * stride + stride / 2
        x_coords = torch.arange(W, device=device).float() * stride + stride / 2
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        x_grid = grid_x[keep]; y_grid = grid_y[keep]
        scores_keep = max_scores[keep]; labels_keep = max_labels[keep]
        angle_preds_keep = angle_preds[:, keep]

        if use_dfl:
            reg = reg_preds[:, keep].view(4, reg_max + 1, -1)
            probs = reg.softmax(dim=1)
            idx = torch.arange(reg_max + 1, device=device).view(1, -1, 1).float()
            distances = (probs * idx).sum(dim=1)
            l_dist, t_dist, r_dist, b_dist = distances.unbind(0)
            cx_local_pix = (r_dist - l_dist) * 0.5 * stride
            cy_local_pix = (b_dist - t_dist) * 0.5 * stride
            w_pix = (l_dist + r_dist) * stride
            h_pix = (t_dist + b_dist) * stride
        else:
            reg = reg_preds[:, keep]
            dx, dy, dw, dh = reg[0], reg[1], reg[2], reg[3]
            cx_local_pix = dx * stride
            cy_local_pix = dy * stride
            w_pix = torch.exp(dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
            h_pix = torch.exp(dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
        size_mask = (
            (w_pix < MAX_ALLOWED_SIDE) & (h_pix < MAX_ALLOWED_SIDE) &
            (w_pix > MIN_ALLOWED_SIDE) & (h_pix > MIN_ALLOWED_SIDE)
        )
        if not size_mask.any():
            continue

        x_grid = x_grid[size_mask]
        y_grid = y_grid[size_mask]
        cx_local_pix = cx_local_pix[size_mask]
        cy_local_pix = cy_local_pix[size_mask]
        w_pix = w_pix[size_mask]
        h_pix = h_pix[size_mask]
        scores_keep = scores_keep[size_mask]
        labels_keep = labels_keep[size_mask]
        angle_preds_keep = angle_preds_keep[:, size_mask]

        if angle_format == 'direct':
            angle = angle_preds_keep[0]
        else:
            angle = torch.atan2(angle_preds_keep[0], angle_preds_keep[1])

        cos_a = torch.cos(angle); sin_a = torch.sin(angle)
        cx = x_grid + cx_local_pix * cos_a - cy_local_pix * sin_a
        cy = y_grid + cx_local_pix * sin_a + cy_local_pix * cos_a

        if angle_range == 'pi':
            mask = w_pix < h_pix
            if mask.any():
                w_new = torch.where(mask, h_pix, w_pix)
                h_new = torch.where(mask, w_pix, h_pix)
                w_pix, h_pix = w_new, h_new
                angle = torch.where(mask, (angle + math.pi / 2) % math.pi, angle)
            angle = angle % math.pi
        else:
            mask = w_pix < h_pix
            if mask.any():
                w_new = torch.where(mask, h_pix, w_pix)
                h_new = torch.where(mask, w_pix, h_pix)
                w_pix, h_pix = w_new, h_new
                angle = torch.where(mask, angle + math.pi / 2, angle)
            angle = torch.atan2(torch.sin(angle), torch.cos(angle))
            angle = torch.clamp(angle, -math.pi / 2, math.pi / 2)

        cx = cx / img_size; cy = cy / img_size
        w_n = w_pix / img_size; h_n = h_pix / img_size

        obboxes = cxcywh_angle_to_obb(cx, cy, w_n, h_n, angle)
        all_obboxes.append(obboxes)
        all_scores.append(scores_keep)
        all_labels.append(labels_keep)

    if not all_obboxes:
        return [], [], []

    obboxes = torch.cat(all_obboxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if len(obboxes) > pre_nms_max_det:
        idx = torch.argsort(scores, descending=True)[:pre_nms_max_det]
        obboxes = obboxes[idx]; scores = scores[idx]; labels = labels[idx]

    if scale is not None and scale > 0:
        obboxes_fill = obboxes * img_size
        obboxes_orig = obboxes_fill.clone()
        for i in range(4):
            obboxes_orig[:, i * 2] = (obboxes_fill[:, i * 2] - pad_left) / scale
            obboxes_orig[:, i * 2 + 1] = (obboxes_fill[:, i * 2 + 1] - pad_top) / scale
        obboxes_abs = obboxes_orig
    else:
        obboxes_abs = obboxes.clone()
        obboxes_abs[:, 0::2] *= orig_w
        obboxes_abs[:, 1::2] *= orig_h

    for i in range(4):
        obboxes_abs[:, i * 2] = torch.clamp(obboxes_abs[:, i * 2], 0, orig_w)
        obboxes_abs[:, i * 2 + 1] = torch.clamp(obboxes_abs[:, i * 2 + 1], 0, orig_h)

    keep_indices = rotated_nms(
        obboxes_abs.cpu().numpy(),
        scores.cpu().numpy(),
        iou_threshold=nms_threshold,
        labels=labels.cpu().numpy(),
    )
    obboxes_abs = obboxes_abs[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    if len(obboxes_abs) > max_det:
        idx = torch.argsort(scores, descending=True)[:max_det]
        obboxes_abs = obboxes_abs[idx]; scores = scores[idx]; labels = labels[idx]

    return obboxes_abs.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()