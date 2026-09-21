import torch
import torch.nn.functional as F
from torchvision.ops import nms
import cv2
import numpy as np
import math


# ============================================================
# 常量
# ============================================================
LOG_SCALE_MIN = -4.0
LOG_SCALE_MAX = 4.0
MIN_POS_PER_GT = 4


# ============================================================
# DFL（Distribution Focal Loss）
# ============================================================
def dfl_decode_distances(reg_logits, reg_max):
    probs = reg_logits.softmax(dim=-1)
    idx = torch.arange(reg_max + 1, device=reg_logits.device, dtype=probs.dtype)
    return (probs * idx).sum(dim=-1)


def dfl_loss(pred_logits, target_distances, reg_max):
    if pred_logits.numel() == 0:
        return torch.tensor(0.0, device=pred_logits.device, dtype=torch.float32)

    target_distances = target_distances.clamp(0.0, reg_max - 0.01)
    tl = target_distances.floor().long().clamp(0, reg_max - 1)
    tr = (tl + 1).clamp(1, reg_max)
    wl = (tr.float() - target_distances)
    wr = (target_distances - tl.float())

    pred_flat = pred_logits.reshape(-1, reg_max + 1)
    tl_flat = tl.reshape(-1)
    tr_flat = tr.reshape(-1)
    wl_flat = wl.reshape(-1).detach()
    wr_flat = wr.reshape(-1).detach()

    loss = (
        F.cross_entropy(pred_flat, tl_flat, reduction='none') * wl_flat +
        F.cross_entropy(pred_flat, tr_flat, reduction='none') * wr_flat
    )
    return loss.mean()


# ============================================================
# CIoU Loss
# ============================================================
def ciou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    pred_xyxy = pred_xyxy.float()
    gt_xyxy = gt_xyxy.float()

    if pred_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return torch.tensor(0.0, device=pred_xyxy.device, dtype=torch.float32)

    px_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    px_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    gx_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
    gy_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
    center_dist2 = (px_cx - gx_cx) ** 2 + (px_cy - gy_cy) ** 2

    enclose_x1 = torch.min(pred_xyxy[:, 0], gt_xyxy[:, 0])
    enclose_y1 = torch.min(pred_xyxy[:, 1], gt_xyxy[:, 1])
    enclose_x2 = torch.max(pred_xyxy[:, 2], gt_xyxy[:, 2])
    enclose_y2 = torch.max(pred_xyxy[:, 3], gt_xyxy[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    lt = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)
    gt_w = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=eps)
    gt_h = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=eps)

    area_pred = pred_w * pred_h
    area_gt = gt_w * gt_h
    union = area_pred + area_gt - inter + eps
    iou = inter / union

    v = (4 / (math.pi ** 2)) * (torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)).pow(2)
    alpha = v / (v - iou + 1 + eps)

    ciou = iou - center_dist2 / c2 - alpha * v
    loss = (1.0 - ciou).clamp(min=0).mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred_xyxy.device, dtype=torch.float32)
    return loss


# ============================================================
# Focal Loss
# ============================================================
def focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt).pow(gamma) * bce
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal


# ============================================================
# Letterbox
# ============================================================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top), shape


# ============================================================
# 角度损失
# ============================================================
def periodic_l1_loss(pred_angle, target_angle, angle_range='pi'):
    pred_angle = pred_angle.float()
    target_angle = target_angle.float()
    diff = pred_angle - target_angle
    diff = diff % math.pi
    diff = torch.min(diff, math.pi - diff)
    return diff.abs().mean()


def cosine_angle_loss(pred_sin_cos, target_sin_cos):
    pred_norm = F.normalize(pred_sin_cos, dim=1)
    target_norm = F.normalize(target_sin_cos, dim=1)
    return (1.0 - (pred_norm * target_norm).sum(dim=1)).mean()


# ============================================================
# ProbIoU（旋转框，torch 版本）
# ============================================================
def _get_covariance_matrix(boxes):
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2, sin2 = cos.pow(2), sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, eps=1e-6):
    obb1 = obb1.float()
    obb2 = obb2.float()

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    a1 = a1.clamp(min=eps); b1 = b1.clamp(min=eps)
    a2 = a2.clamp(min=eps); b2 = b2.clamp(min=eps)

    denom = ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)).clamp(min=eps)

    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / (denom + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / (denom + eps)) * 0.5

    log_arg = denom / (
        4 * ((a1 * b1 - c1.pow(2)).clamp_(min=eps) *
             (a2 * b2 - c2.pow(2)).clamp_(min=eps)).sqrt() + eps
    )
    log_arg = torch.clamp(log_arg, min=eps)
    t3 = log_arg.log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 20.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def probiou_loss(pred_obb, target_obb):
    iou = probiou(pred_obb, target_obb)
    return (1.0 - iou).clamp(min=0).mean()


def center_l1_loss(pred_obb, target_obb):
    return (pred_obb[:, :2] - target_obb[:, :2]).abs().mean()


# ============================================================
# CRIoU Loss
# ============================================================
def criou_loss(pred_bbox, target_bbox, eps=1e-7):
    pred_bbox = pred_bbox.float()
    target_bbox = target_bbox.float()

    if torch.isnan(pred_bbox).any() or torch.isnan(target_bbox).any():
        return torch.tensor(0.0, device=pred_bbox.device, dtype=torch.float32)

    pred_cx, pred_cy, pred_w, pred_h, pred_angle = pred_bbox.unbind(dim=1)
    target_cx, target_cy, target_w, target_h, target_angle = target_bbox.unbind(dim=1)

    pred_w = torch.clamp(pred_w, min=eps)
    pred_h = torch.clamp(pred_h, min=eps)
    target_w = torch.clamp(target_w, min=eps)
    target_h = torch.clamp(target_h, min=eps)

    pred_x1 = pred_cx - pred_w / 2; pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2; pred_y2 = pred_cy + pred_h / 2
    target_x1 = target_cx - target_w / 2; target_y1 = target_cy - target_h / 2
    target_x2 = target_cx + target_w / 2; target_y2 = target_cy + target_h / 2

    iou_r = probiou(pred_bbox, target_bbox)

    center_dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
    min_x = torch.min(pred_x1, target_x1); min_y = torch.min(pred_y1, target_y1)
    max_x = torch.max(pred_x2, target_x2); max_y = torch.max(pred_y2, target_y2)
    c = torch.sqrt((max_x - min_x).pow(2) + (max_y - min_y).pow(2) + eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)).pow(2)
    alpha = v / (v - iou_r + (1 + eps))
    criou = iou_r - (center_dist_sq / (c.pow(2) + eps)) - alpha * v

    loss = (1.0 - criou).clamp(min=0).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred_bbox.device, dtype=torch.float32)
    return loss


# ============================================================
# KLD Loss
# ============================================================
def _xywha_to_gaussian(cx, cy, w, h, angle, eps=1e-7):
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    R = torch.stack([
        torch.stack([cos_a, -sin_a], dim=1),
        torch.stack([sin_a, cos_a], dim=1),
    ], dim=1)
    half_w2 = (w * 0.5).pow(2).clamp(min=eps)
    half_h2 = (h * 0.5).pow(2).clamp(min=eps)
    S = torch.zeros_like(R)
    S[:, 0, 0] = half_w2
    S[:, 1, 1] = half_h2
    sigma = R @ S @ R.transpose(1, 2)
    mu = torch.stack([cx, cy], dim=1)
    return mu, sigma


def kld_loss(pred_obb, target_obb, tau=1.0, fun='log1p', sqrt=False, eps=1e-7):
    if pred_obb.numel() == 0 or target_obb.numel() == 0:
        return torch.tensor(0.0, device=pred_obb.device, dtype=torch.float32)

    pred_obb = pred_obb.float()
    target_obb = target_obb.float()

    mu_p, sigma_p = _xywha_to_gaussian(
        pred_obb[:, 0], pred_obb[:, 1], pred_obb[:, 2], pred_obb[:, 3], pred_obb[:, 4])
    mu_t, sigma_t = _xywha_to_gaussian(
        target_obb[:, 0], target_obb[:, 1], target_obb[:, 2], target_obb[:, 3], target_obb[:, 4])

    eye = eps * torch.eye(2, device=sigma_t.device).unsqueeze(0)
    sigma_t_inv = torch.inverse(sigma_t + eye)

    diff = (mu_p - mu_t).unsqueeze(-1)
    term_center = (diff.transpose(1, 2) @ sigma_t_inv @ diff).squeeze(-1).squeeze(-1)
    term_trace = (sigma_t_inv @ sigma_p).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    logdet_t = torch.logdet(sigma_t + eye)
    logdet_p = torch.logdet(sigma_p + eye)
    term_logdet = logdet_t - logdet_p

    kld = 0.5 * (term_center + term_trace + term_logdet - 2.0)
    kld = kld.clamp(min=0.0)

    if sqrt:
        kld = (kld + eps).sqrt()

    if fun == 'log1p':
        kld = torch.log1p(kld / tau)
    elif fun == 'sqrt':
        kld = (kld / tau + eps).sqrt()
    elif fun is None:
        kld = kld / tau
    else:
        raise ValueError(f"不支持的 fun: {fun}")

    loss = kld.mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred_obb.device, dtype=torch.float32)
    return loss


# ============================================================
# 批量 IoU（训练时用，torch 张量）
# ============================================================
def _batch_probiou(gt_obb, pred_obb, eps=1e-6, chunk_size=128):
    """gt_obb: (N_gt, 5), pred_obb: (P, 5) → (N_gt, P)"""
    device = gt_obb.device
    N_gt = gt_obb.shape[0]
    P = pred_obb.shape[0]
    iou = torch.empty((N_gt, P), device=device, dtype=torch.float32)

    gt_w = gt_obb[:, 2].clamp(min=eps); gt_h = gt_obb[:, 3].clamp(min=eps)
    gc = torch.cos(gt_obb[:, 4]); gs = torch.sin(gt_obb[:, 4])
    ga = gt_w.pow(2) / 12 * gc.pow(2) + gt_h.pow(2) / 12 * gs.pow(2)
    gb = gt_w.pow(2) / 12 * gs.pow(2) + gt_h.pow(2) / 12 * gc.pow(2)
    gcc = (gt_w.pow(2) / 12 - gt_h.pow(2) / 12) * gc * gs
    gab = (ga * gb - gcc.pow(2)).clamp(min=eps)
    gcx = gt_obb[:, 0:1]; gcy = gt_obb[:, 1:2]

    pw = pred_obb[:, 2].clamp(min=eps); ph = pred_obb[:, 3].clamp(min=eps)
    pc = torch.cos(pred_obb[:, 4]); ps = torch.sin(pred_obb[:, 4])
    pa = pw.pow(2) / 12 * pc.pow(2) + ph.pow(2) / 12 * ps.pow(2)
    pb = pw.pow(2) / 12 * ps.pow(2) + ph.pow(2) / 12 * pc.pow(2)
    pcc = (pw.pow(2) / 12 - ph.pow(2) / 12) * pc * ps
    pab = (pa * pb - pcc.pow(2)).clamp(min=eps)
    pcx = pred_obb[:, 0]; pcy = pred_obb[:, 1]

    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        sl = slice(start, end)

        a_sum = ga[:, None] + pa[None, sl]
        b_sum = gb[:, None] + pb[None, sl]
        c_sum = gcc[:, None] + pcc[None, sl]
        denom = (a_sum * b_sum - c_sum.pow(2)).clamp(min=eps)

        dx = gcx - pcx[None, sl]
        dy = gcy - pcy[None, sl]

        t1 = ((a_sum * dy.pow(2) + b_sum * dx.pow(2)) / denom) * 0.25
        t2 = -(c_sum * dx * dy / denom) * 0.5

        log_arg = (denom / (4 * (gab[:, None] * pab[None, sl]).sqrt() + eps)).clamp(min=eps)
        t3 = log_arg.log() * 0.5

        bd = (t1 + t2 + t3).clamp(min=eps, max=20.0)
        hd = (1.0 - (-bd).exp() + eps).sqrt()
        iou[:, sl] = 1.0 - hd

    return iou


def _batch_bbox_iou(gt_boxes, pred_boxes, eps=1e-6):
    lt = torch.max(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    rb = torch.min(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    a2 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + eps)


# ============================================================
# 纯 numpy ProbIoU（NMS 和评估用）
# ============================================================
def _obb_xy8_to_cxcywh_angle_np(obboxes_8):
    """(N, 8) → (N, 5) [cx, cy, w, h, angle]，纯 numpy"""
    pts = obboxes_8.reshape(-1, 4, 2).astype(np.float32)
    cx = pts[:, :, 0].mean(axis=1)
    cy = pts[:, :, 1].mean(axis=1)

    v1 = 0.5 * ((pts[:, 1] - pts[:, 0]) + (pts[:, 2] - pts[:, 3]))
    v2 = 0.5 * ((pts[:, 2] - pts[:, 1]) + (pts[:, 3] - pts[:, 0]))
    w1 = np.linalg.norm(v1, axis=1)
    h1 = np.linalg.norm(v2, axis=1)
    mask = w1 >= h1
    w = np.where(mask, w1, h1)
    h = np.where(mask, h1, w1)
    angle = np.where(mask,
                     np.arctan2(v1[:, 1], v1[:, 0]),
                     np.arctan2(v2[:, 1], v2[:, 0]))
    return np.stack([cx, cy, w, h, angle], axis=1).astype(np.float32)


def _batch_probiou_np(gt_obb, pred_obb, eps=1e-6):
    """
    纯 numpy ProbIoU
    gt_obb:   (N_gt, 5)
    pred_obb: (P, 5)
    返回:     (N_gt, P)  ★ 注意：第一个维度是 gt
    """
    N_gt = gt_obb.shape[0]
    P = pred_obb.shape[0]

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


# ============================================================
# 纯 CPU numpy 旋转 NMS
# ============================================================
def rotated_nms(obboxes, scores, iou_threshold=0.5, labels=None):
    """
    纯 CPU numpy 旋转 NMS（避免 GPU-CPU 同步）
    obboxes: (N, 8) numpy
    scores:  (N,) numpy
    labels:  (N,) numpy 或 None
    """
    if len(obboxes) == 0:
        return np.array([], dtype=np.int64)

    obboxes = np.asarray(obboxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    obb5 = _obb_xy8_to_cxcywh_angle_np(obboxes)

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


# ============================================================
# compute_fcos_loss
# ============================================================
def compute_fcos_loss(cls_logits_list, reg_preds_list, targets, strides, device, img_size,
                      cls_weight=1.0, reg_weight=1.0, angle_weight=1.0,
                      dfl_weight=1.5,
                      mode='detect', angle_format='sin_cos', angle_preds_list=None,
                      angle_range='pi',
                      tal_topk=13, tal_alpha=1.0, tal_beta=6.0,
                      reg_max=16,
                      kld_weight=1.0, kld_tau=1.0, kld_fun='log1p',
                      obb_reg_loss='probiou'):
    if mode == 'obb' and angle_preds_list is None:
        raise ValueError("OBB mode requires angle_preds_list")

    use_dfl = (reg_max > 0)
    reg_channels = 4 * (reg_max + 1) if use_dfl else 4

    cls_logits_list = [x.float() for x in cls_logits_list]
    reg_preds_list = [x.float() for x in reg_preds_list]
    if angle_preds_list is not None:
        angle_preds_list = [x.float() for x in angle_preds_list]

    num_levels = len(cls_logits_list)
    B = cls_logits_list[0].shape[0]
    C = cls_logits_list[0].shape[1]

    # ==================== 拼接所有 level 的预测 ====================
    all_grids, all_cls, all_regs, all_angles, all_strides = [], [], [], [], []
    offset = 0

    for l in range(num_levels):
        stride = strides[l]
        cls_l = cls_logits_list[l]
        reg_l = reg_preds_list[l]
        _, _, H, W = cls_l.shape
        P_l = H * W

        y = torch.arange(H, device=device).float() * stride + stride / 2
        x = torch.arange(W, device=device).float() * stride + stride / 2
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        grid_l = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

        all_grids.append(grid_l)
        all_cls.append(cls_l.permute(0, 2, 3, 1).reshape(B, P_l, C))
        all_regs.append(reg_l.permute(0, 2, 3, 1).reshape(B, P_l, reg_channels))

        if angle_preds_list is not None:
            A = angle_preds_list[l].shape[1]
            all_angles.append(angle_preds_list[l].permute(0, 2, 3, 1).reshape(B, P_l, A))

        all_strides.append(torch.full((P_l,), float(stride), device=device))
        offset += P_l

    P_total = offset
    all_grids = torch.cat(all_grids, dim=0)
    all_cls = torch.cat(all_cls, dim=1)
    all_regs = torch.cat(all_regs, dim=1)
    all_strides = torch.cat(all_strides, dim=0)
    if len(all_angles) > 0:
        all_angles = torch.cat(all_angles, dim=1)

    # ==================== GT 拼接 ====================
    gt_lens_t = torch.tensor([targets[b]['labels'].shape[0] for b in range(B)],
                             device=device, dtype=torch.long)
    gt_offsets = torch.zeros(B, device=device, dtype=torch.long)
    if B > 1:
        gt_offsets[1:] = torch.cumsum(gt_lens_t[:-1], dim=0)

    all_gt_labels = torch.cat(
        [targets[b]['labels'].long() for b in range(B)], dim=0
    ) if gt_lens_t.sum() > 0 else torch.zeros(0, dtype=torch.long, device=device)

    if mode == 'detect':
        all_gt_boxes = torch.cat(
            [targets[b]['boxes'].float() * img_size for b in range(B)], dim=0
        ) if gt_lens_t.sum() > 0 else torch.zeros(0, 4, device=device)
    else:
        all_gt_cxcywh = torch.cat(
            [targets[b]['cxcywh'].float() * img_size for b in range(B)], dim=0
        ) if gt_lens_t.sum() > 0 else torch.zeros(0, 4, device=device)
        all_gt_angle = torch.cat(
            [targets[b]['angles'].float() for b in range(B)], dim=0
        ) if gt_lens_t.sum() > 0 else torch.zeros(0, device=device)

    # ==================== 解码所有预测 ====================
    with torch.no_grad():
        grid_x_1d = all_grids[:, 0][None, :]
        grid_y_1d = all_grids[:, 1][None, :]
        strides_1d = all_strides[None, :]

        if use_dfl:
            all_regs_4d = all_regs.view(B, P_total, 4, reg_max + 1)
            distances = dfl_decode_distances(all_regs_4d, reg_max)
            l_d, t_d, r_d, b_d = distances.unbind(-1)
            cx_local_stride = (r_d - l_d) * 0.5
            cy_local_stride = (b_d - t_d) * 0.5
            w_stride = l_d + r_d
            h_stride = t_d + b_d

            if mode == 'obb':
                pred_angle = (torch.atan2(all_angles[..., 0], all_angles[..., 1])
                              if angle_format == 'sin_cos' else all_angles[..., 0])
                cos_a = torch.cos(pred_angle); sin_a = torch.sin(pred_angle)
                cx_local_pix = cx_local_stride * strides_1d
                cy_local_pix = cy_local_stride * strides_1d
                pred_cx = grid_x_1d + cx_local_pix * cos_a - cy_local_pix * sin_a
                pred_cy = grid_y_1d + cx_local_pix * sin_a + cy_local_pix * cos_a
                pred_w = w_stride * strides_1d
                pred_h = h_stride * strides_1d
                pred_obb_all = torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_angle], dim=-1)
            else:
                pred_cx = grid_x_1d + cx_local_stride * strides_1d
                pred_cy = grid_y_1d + cy_local_stride * strides_1d
                pred_w = w_stride * strides_1d
                pred_h = h_stride * strides_1d
                pred_xyxy_all = torch.stack([
                    pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                    pred_cx + pred_w / 2, pred_cy + pred_h / 2], dim=-1)
        else:
            dx = all_regs[..., 0]; dy = all_regs[..., 1]
            dw = all_regs[..., 2].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            dh = all_regs[..., 3].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            pred_cx = grid_x_1d + dx * strides_1d
            pred_cy = grid_y_1d + dy * strides_1d
            pred_w = torch.exp(dw) * strides_1d
            pred_h = torch.exp(dh) * strides_1d

            if mode == 'obb':
                pred_angle = (torch.atan2(all_angles[..., 0], all_angles[..., 1])
                              if angle_format == 'sin_cos' else all_angles[..., 0])
                pred_obb_all = torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_angle], dim=-1)
            else:
                pred_xyxy_all = torch.stack([
                    pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                    pred_cx + pred_w / 2, pred_cy + pred_h / 2], dim=-1)

        cls_scores_all = torch.sigmoid(all_cls)

        # ==================== TAL 分配 ====================
        pos_batch_list, pos_point_list, pos_gt_list = [], [], []

        for batch_idx in range(B):
            n_gt = int(gt_lens_t[batch_idx].item())
            if n_gt == 0:
                continue
            b_start = int(gt_offsets[batch_idx].item())
            b_end = b_start + n_gt

            if mode == 'obb':
                gt_cxcywh_b = all_gt_cxcywh[b_start:b_end]
                gt_angle_b = all_gt_angle[b_start:b_end]
                dx_ = all_grids[:, 0][None, :] - gt_cxcywh_b[:, 0:1]
                dy_ = all_grids[:, 1][None, :] - gt_cxcywh_b[:, 1:2]
                gc = torch.cos(gt_angle_b)[:, None]; gs = torch.sin(gt_angle_b)[:, None]
                u = dx_ * gc + dy_ * gs
                v = -dx_ * gs + dy_ * gc
                half_w = gt_cxcywh_b[:, 2:3] * 0.5
                half_h = gt_cxcywh_b[:, 3:4] * 0.5
                inside_all = (u.abs() <= half_w) & (v.abs() <= half_h)
                gt_obb_b = torch.cat([gt_cxcywh_b, gt_angle_b[:, None]], dim=1)
                cur_pred_obb = pred_obb_all[batch_idx]
                iou_all = _batch_probiou(gt_obb_b, cur_pred_obb, chunk_size=128)
            else:
                gt_boxes_b = all_gt_boxes[b_start:b_end]
                gx1 = gt_boxes_b[:, 0:1]; gy1 = gt_boxes_b[:, 1:2]
                gx2 = gt_boxes_b[:, 2:3]; gy2 = gt_boxes_b[:, 3:4]
                px = all_grids[:, 0][None, :]; py = all_grids[:, 1][None, :]
                inside_all = (px >= gx1) & (px <= gx2) & (py >= gy1) & (py <= gy2)
                cur_pred_xyxy = pred_xyxy_all[batch_idx]
                iou_all = _batch_bbox_iou(gt_boxes_b, cur_pred_xyxy)

            gt_labels_b = all_gt_labels[b_start:b_end]
            cur_scores = cls_scores_all[batch_idx]
            cls_per_gt = cur_scores[:, gt_labels_b].transpose(0, 1)

            score_all = cls_per_gt.pow(tal_alpha) * (iou_all + 1e-7).pow(tal_beta)
            score_all = score_all * inside_all.float()

            k = min(tal_topk, P_total)
            topk_vals, topk_idx = score_all.topk(k, dim=1)

            gt_ids_flat = torch.arange(n_gt, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            point_ids_flat = topk_idx.reshape(-1)
            scores_flat = topk_vals.reshape(-1)

            best_per_point = torch.full((P_total,), -float('inf'),
                                        device=device, dtype=torch.float32)
            best_per_point.scatter_reduce_(0, point_ids_flat, scores_flat,
                                           reduce='amax', include_self=True)

            is_winner = scores_flat == best_per_point[point_ids_flat]
            valid_gt = gt_ids_flat[is_winner]
            valid_point = point_ids_flat[is_winner]

            if valid_gt.numel() == 0:
                continue

            pos_batch_list.append(torch.full_like(valid_gt, batch_idx, dtype=torch.long))
            pos_point_list.append(valid_point)
            pos_gt_list.append(valid_gt)

        if len(pos_point_list) > 0:
            pos_batch = torch.cat(pos_batch_list, dim=0)
            pos_point = torch.cat(pos_point_list, dim=0)
            pos_gt = torch.cat(pos_gt_list, dim=0)
            N_pos = pos_point.numel()
        else:
            pos_batch = torch.zeros(0, dtype=torch.long, device=device)
            pos_point = torch.zeros(0, dtype=torch.long, device=device)
            pos_gt = torch.zeros(0, dtype=torch.long, device=device)
            N_pos = 0

        cls_targets_all = torch.zeros((B, P_total, C), device=device)
        if N_pos > 0:
            global_gt_idx = gt_offsets[pos_batch] + pos_gt
            pos_labels = all_gt_labels[global_gt_idx]
            cls_targets_all[pos_batch, pos_point, pos_labels] = 1.0

    # ==================== 分类损失 ====================
    cls_logits_flat = torch.clamp(all_cls.reshape(-1, C), min=-20, max=20)
    cls_targets_flat = cls_targets_all.reshape(-1, C)

    bce = F.binary_cross_entropy_with_logits(cls_logits_flat, cls_targets_flat, reduction='none')
    target_scores_sum = cls_targets_flat.sum().clamp(min=1.0)
    cls_loss = bce.sum() / target_scores_sum

    if torch.isnan(cls_loss) or torch.isinf(cls_loss):
        cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    # ==================== 回归 / 角度损失 ====================
    reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    angle_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    if N_pos > 0:
        pos_stride = all_strides[pos_point]
        pos_grid_x = all_grids[pos_point, 0]
        pos_grid_y = all_grids[pos_point, 1]
        global_gt_idx = gt_offsets[pos_batch] + pos_gt

        if use_dfl:
            pos_reg_logits = all_regs.view(B, P_total, 4, reg_max + 1)[pos_batch, pos_point]

            if mode == 'obb':
                pos_gt_cxcywh = all_gt_cxcywh[global_gt_idx]
                pos_gt_angle = all_gt_angle[global_gt_idx]
                gt_cx, gt_cy, gt_w, gt_h = pos_gt_cxcywh.unbind(-1)

                ddx = pos_grid_x - gt_cx
                ddy = pos_grid_y - gt_cy
                c_a = torch.cos(pos_gt_angle); s_a = torch.sin(pos_gt_angle)
                cx_local = ddx * c_a + ddy * s_a
                cy_local = -ddx * s_a + ddy * c_a

                l_gt = (gt_w / 2 - cx_local) / pos_stride
                t_gt = (gt_h / 2 - cy_local) / pos_stride
                r_gt = (gt_w / 2 + cx_local) / pos_stride
                b_gt = (gt_h / 2 + cy_local) / pos_stride
                target_distances = torch.stack([l_gt, t_gt, r_gt, b_gt], dim=-1).clamp(0.0, float(reg_max))

                l_dfl = dfl_loss(pos_reg_logits, target_distances, reg_max)

                pos_distances = dfl_decode_distances(pos_reg_logits, reg_max)
                l_p, t_p, r_p, b_p = pos_distances.unbind(-1)
                cx_local_pix = (r_p - l_p) * 0.5 * pos_stride
                cy_local_pix = (b_p - t_p) * 0.5 * pos_stride
                w_pix = (l_p + r_p) * pos_stride
                h_pix = (t_p + b_p) * pos_stride

                pos_angle_reg = all_angles[pos_batch, pos_point]
                pos_pred_angle = (torch.atan2(pos_angle_reg[:, 0], pos_angle_reg[:, 1])
                                  if angle_format == 'sin_cos' else pos_angle_reg[:, 0])

                cos_pa = torch.cos(pos_pred_angle); sin_pa = torch.sin(pos_pred_angle)
                pred_cx = pos_grid_x + cx_local_pix * cos_pa - cy_local_pix * sin_pa
                pred_cy = pos_grid_y + cx_local_pix * sin_pa + cy_local_pix * cos_pa

                pred_obb_pos = torch.stack([pred_cx, pred_cy, w_pix, h_pix, pos_pred_angle], dim=1)
                gt_obb_pos = torch.cat([pos_gt_cxcywh, pos_gt_angle.unsqueeze(1)], dim=1)

                if obb_reg_loss == 'kld':
                    l_kld = kld_loss(pred_obb_pos, gt_obb_pos, tau=kld_tau, fun=kld_fun)
                    reg_loss = kld_weight * l_kld + dfl_weight * l_dfl
                else:
                    l_criou = criou_loss(pred_obb_pos, gt_obb_pos)
                    l_probiou = probiou_loss(pred_obb_pos, gt_obb_pos)
                    reg_loss = l_criou + 0.5 * l_probiou + dfl_weight * l_dfl

                if angle_format == 'sin_cos':
                    tgt_sc = torch.stack([torch.sin(pos_gt_angle), torch.cos(pos_gt_angle)], dim=1)
                    cos_loss = cosine_angle_loss(pos_angle_reg, tgt_sc)
                    aux = periodic_l1_loss(pos_pred_angle, pos_gt_angle, angle_range)
                    angle_loss = cos_loss + 0.5 * aux
                else:
                    angle_loss = periodic_l1_loss(pos_pred_angle, pos_gt_angle, angle_range)
            else:
                pos_gt_boxes = all_gt_boxes[global_gt_idx]
                gt_x1 = pos_gt_boxes[:, 0]; gt_y1 = pos_gt_boxes[:, 1]
                gt_x2 = pos_gt_boxes[:, 2]; gt_y2 = pos_gt_boxes[:, 3]

                l_target = ((pos_grid_x - gt_x1) / pos_stride).clamp(0.0, float(reg_max))
                t_target = ((pos_grid_y - gt_y1) / pos_stride).clamp(0.0, float(reg_max))
                r_target = ((gt_x2 - pos_grid_x) / pos_stride).clamp(0.0, float(reg_max))
                b_target = ((gt_y2 - pos_grid_y) / pos_stride).clamp(0.0, float(reg_max))
                target_distances = torch.stack([l_target, t_target, r_target, b_target], dim=-1)

                l_dfl = dfl_loss(pos_reg_logits, target_distances, reg_max)

                pos_distances = dfl_decode_distances(pos_reg_logits, reg_max)
                l_pred, t_pred, r_pred, b_pred = pos_distances.unbind(-1)
                px1 = pos_grid_x - l_pred * pos_stride
                py1 = pos_grid_y - t_pred * pos_stride
                px2 = pos_grid_x + r_pred * pos_stride
                py2 = pos_grid_y + b_pred * pos_stride
                pred_xyxy_pos = torch.stack([px1, py1, px2, py2], dim=1)

                l_ciou = ciou_loss(pred_xyxy_pos, pos_gt_boxes)
                reg_loss = l_ciou + dfl_weight * l_dfl
        else:
            pos_reg = all_regs[pos_batch, pos_point]
            pos_cx = pos_grid_x + pos_reg[:, 0] * pos_stride
            pos_cy = pos_grid_y + pos_reg[:, 1] * pos_stride
            pos_w = torch.exp(pos_reg[:, 2].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * pos_stride
            pos_h = torch.exp(pos_reg[:, 3].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * pos_stride

            if mode == 'obb':
                pos_gt_cxcywh = all_gt_cxcywh[global_gt_idx]
                pos_gt_angle = all_gt_angle[global_gt_idx]
                pos_angle_reg = all_angles[pos_batch, pos_point]
                pos_pred_angle = (torch.atan2(pos_angle_reg[:, 0], pos_angle_reg[:, 1])
                                  if angle_format == 'sin_cos' else pos_angle_reg[:, 0])

                pred_obb_pos = torch.stack([pos_cx, pos_cy, pos_w, pos_h, pos_pred_angle], dim=1)
                gt_obb_pos = torch.cat([pos_gt_cxcywh, pos_gt_angle.unsqueeze(1)], dim=1)

                if obb_reg_loss == 'kld':
                    l_kld = kld_loss(pred_obb_pos, gt_obb_pos, tau=kld_tau, fun=kld_fun)
                    reg_loss = kld_weight * l_kld
                else:
                    l_criou = criou_loss(pred_obb_pos, gt_obb_pos)
                    l_probiou = probiou_loss(pred_obb_pos, gt_obb_pos)
                    reg_loss = l_criou + 0.5 * l_probiou

                if angle_format == 'sin_cos':
                    tgt_sc = torch.stack([torch.sin(pos_gt_angle), torch.cos(pos_gt_angle)], dim=1)
                    cos_loss = cosine_angle_loss(pos_angle_reg, tgt_sc)
                    aux = periodic_l1_loss(pos_pred_angle, pos_gt_angle, angle_range)
                    angle_loss = cos_loss + 0.5 * aux
                else:
                    angle_loss = periodic_l1_loss(pos_pred_angle, pos_gt_angle, angle_range)
            else:
                pred_xyxy_pos = torch.stack([
                    pos_cx - pos_w / 2, pos_cy - pos_h / 2,
                    pos_cx + pos_w / 2, pos_cy + pos_h / 2], dim=1)
                pos_gt_boxes = all_gt_boxes[global_gt_idx]
                reg_loss = ciou_loss(pred_xyxy_pos, pos_gt_boxes)

        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if torch.isnan(angle_loss) or torch.isinf(angle_loss):
            angle_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    total_loss = cls_weight * cls_loss + reg_weight * reg_loss + angle_weight * angle_loss
    return total_loss


# ============================================================
# 按类别分组 NMS（水平框）
# ============================================================
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


# ============================================================
# 水平框后处理
# ============================================================
def post_process(cls_logits_list, reg_preds_list,
                 orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                 strides=None, scale=None, pad_left=0, pad_top=0,
                 reg_max=0):
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2] for i in range(num_levels)]

    use_dfl = reg_max > 0

    all_boxes, all_scores, all_labels = [], [], []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx].squeeze(0)
        reg_preds = reg_preds_list[level_idx].squeeze(0)

        C, H, W = cls_logits.shape
        scores = torch.sigmoid(cls_logits)
        max_scores, max_labels = torch.max(scores, dim=0)
        keep = max_scores > conf_threshold
        if not keep.any():
            continue

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
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    return boxes, scores, labels


# ============================================================
# 旋转框后处理
# ============================================================
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


# ============================================================
# cxcywh+angle → 8 顶点
# ============================================================
def cxcywh_angle_to_obb(cx, cy, w, h, angle):
    N = cx.shape[0]
    device = cx.device
    half_w = w / 2; half_h = h / 2

    corners = torch.zeros((N, 4, 2), device=device)
    corners[:, 0, 0] = half_w; corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w; corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w; corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w; corners[:, 3, 1] = -half_h

    cos_a = torch.cos(angle); sin_a = torch.sin(angle)
    rot_matrix = torch.stack([
        torch.stack([cos_a, sin_a], dim=1),
        torch.stack([-sin_a, cos_a], dim=1)
    ], dim=1)

    corners_rot = torch.matmul(corners, rot_matrix)
    center = torch.stack([cx, cy], dim=1).unsqueeze(1)
    corners_final = corners_rot + center

    obboxes = corners_final.reshape(N, 8)
    return torch.clamp(obboxes, 0, 1)


# ============================================================
# AP 计算（101 点插值 + 梯形积分）
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
    ap = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))
    return ap


# ============================================================
# YOLOv8 兼容的 ap_per_class
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
# 轴对齐 IoU 矩阵
# ============================================================
def _iou_matrix_axis_aligned(pred_boxes, gt_boxes, eps=1e-6):
    """pred: (n_pred, 4), gt: (n_gt, 4) → (n_pred, n_gt)"""
    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / (union + eps)


# ============================================================
# ★ TP 矩阵（自动纠正 iou_mat 朝向）
# ============================================================
def _compute_tp_matrix(preds, gts, iou_mat, iou_thresholds):
    """
    返回 (tp: (n_pred, T) bool, scores_sorted: (n_pred,))
    自动兼容 iou_mat 的两种朝向 (n_pred, n_gt) 或 (n_gt, n_pred)
    """
    scores = np.array([p['score'] for p in preds], dtype=np.float64)
    order = np.argsort(-scores)

    n_pred = len(preds)
    n_gt = len(gts)

    # ★ 自动纠正朝向
    if iou_mat.shape[0] != n_pred:
        if iou_mat.shape[1] == n_pred:
            iou_mat = iou_mat.T
        else:
            raise ValueError(
                f"iou_mat 形状不匹配: {iou_mat.shape}, "
                f"期望 (n_pred={n_pred}, n_gt={n_gt}) 或 (n_gt={n_gt}, n_pred={n_pred})"
            )
    elif iou_mat.shape[1] != n_gt:
        raise ValueError(
            f"iou_mat 形状不匹配: {iou_mat.shape}, "
            f"期望 (n_pred={n_pred}, n_gt={n_gt})"
        )

    iou_sorted = iou_mat[order]                    # (n_pred, n_gt)
    scores_sorted = scores[order]

    T = len(iou_thresholds)
    tp = np.zeros((n_pred, T), dtype=bool)
    gt_used = np.zeros((T, n_gt), dtype=bool)

    thresholds_arr = np.asarray(iou_thresholds, dtype=np.float64)

    for pi in range(n_pred):
        ious = iou_sorted[pi]                      # (n_gt,)
        candidates = np.where(gt_used, -1.0, ious[None, :])   # (T, n_gt)
        best_gi = np.argmax(candidates, axis=1)               # (T,)
        best_iou = candidates[np.arange(T), best_gi]          # (T,)

        valid = best_iou >= thresholds_arr
        tp[pi, valid] = True
        gt_used[valid, best_gi[valid]] = True

    return tp, scores_sorted


# ============================================================
# ★ 旋转 IoU 矩阵（明确返回 (n_pred, n_gt)）
# ============================================================
def _compute_rotated_iou_matrix(preds_obboxes, gts_obboxes):
    """
    ★ 明确返回 (n_pred, n_gt) 的 IoU 矩阵
    """
    n_pred = len(preds_obboxes)
    n_gt = len(gts_obboxes)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    preds_obb5 = _obb_xy8_to_cxcywh_angle_np(
        np.asarray(preds_obboxes, dtype=np.float32)
    )
    gts_obb5 = _obb_xy8_to_cxcywh_angle_np(
        np.asarray(gts_obboxes, dtype=np.float32)
    )

    # _batch_probiou_np(gt, pred) → (n_gt, n_pred)
    iou_mat_gt_pred = _batch_probiou_np(gts_obb5, preds_obb5)

    # ★ 转置为 (n_pred, n_gt)
    iou_mat = iou_mat_gt_pred.T

    return np.ascontiguousarray(iou_mat, dtype=np.float32)


# ============================================================
# 水平框评估
# ============================================================
@torch.no_grad()
def evaluate_mAP(model, dataloader, device, cfg, conf_threshold=0.01):
    """
    返回 (precision, recall, mAP50, mAP50_95, AR100)
    """
    model.eval()

    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in dataloader:
        images = images.to(device)
        cls_logits_list, reg_preds_list = model(images)
        B = images.shape[0]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            boxes, scores, labels = post_process(
                [cls_logits_list[level][idx:idx + 1] for level in range(len(cls_logits_list))],
                [reg_preds_list[level][idx:idx + 1] for level in range(len(reg_preds_list))],
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

    all_tp_list = []
    all_conf = []
    all_pred_cls = []

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
        all_tp, all_conf, all_pred_cls, all_target_cls
    )

    if len(unique_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    precision_mean = float(p.mean())
    recall_mean = float(r.mean())
    mAP50 = float(ap[:, 0].mean())
    mAP50_95 = float(ap.mean())
    AR100 = recall_mean

    return precision_mean, recall_mean, mAP50, mAP50_95, AR100


# ============================================================
# OBB 评估
# ============================================================
@torch.no_grad()
def evaluate_map_obb_rotated(model, dataloader, device, cfg,
                             conf_threshold=0.01, nms_threshold=None,
                             max_det=100, pre_nms_max_det=300):
    """
    返回 (precision, recall, mAP50, mAP50_95, AR100)
    """
    if nms_threshold is None:
        nms_threshold = cfg.obb_nms_threshold

    model.eval()

    all_preds = []
    all_gts = []

    img_id = 0
    for images, targets in dataloader:
        images = images.to(device)
        cls_logits_list, reg_preds_list, angle_preds_list = model(images)
        B = images.shape[0]

        for idx in range(B):
            t = targets[idx]
            orig_w, orig_h = t['orig_size']

            obboxes, scores, labels = post_process_obb(
                [cls_logits_list[level][idx:idx + 1] for level in range(len(cls_logits_list))],
                [reg_preds_list[level][idx:idx + 1] for level in range(len(reg_preds_list))],
                [angle_preds_list[level][idx:idx + 1] for level in range(len(angle_preds_list))],
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

    all_tp_list = []
    all_conf = []
    all_pred_cls = []

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
        all_tp, all_conf, all_pred_cls, all_target_cls
    )

    if len(unique_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    precision_mean = float(p.mean())
    recall_mean = float(r.mean())
    mAP50 = float(ap[:, 0].mean())
    mAP50_95 = float(ap.mean())
    AR100 = recall_mean

    return precision_mean, recall_mean, mAP50, mAP50_95, AR100