import torch
import torch.nn.functional as F
from torchvision.ops import nms, generalized_box_iou_loss
import cv2
import numpy as np
import math

# ---------- OBB 回归对数尺度 clamp（训练/推理共用） ----------
LOG_SCALE_MIN = -4.0
LOG_SCALE_MAX = 6.0

# ---------- 每个 GT 的最少正样本数 ----------
MIN_POS_PER_GT = 5


# ---------- Letterbox ----------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
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


# ---------- 水平框 IoU ----------
def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    if bboxes1.numel() == 0 or bboxes2.numel() == 0:
        return torch.zeros((bboxes1.shape[0], bboxes2.shape[0]), device=bboxes1.device)
    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + eps)
    return iou


# ---------- 角度损失（周期性 L1，全程 FP32） ----------
def periodic_l1_loss(pred_angle, target_angle, angle_range='pi'):
    pred_angle = pred_angle.float()
    target_angle = target_angle.float()
    diff = pred_angle - target_angle
    if angle_range == 'pi':
        diff = diff % math.pi
        diff = torch.min(diff, math.pi - diff)
    else:
        diff = diff % (2 * math.pi)
        diff = torch.min(diff, 2 * math.pi - diff)
    return diff.abs().mean()


# ---------- 余弦角度损失（对 sin_cos 编码，天然处理周期） ----------
def cosine_angle_loss(pred_sin_cos, target_sin_cos):
    pred_norm = F.normalize(pred_sin_cos, dim=1)
    target_norm = F.normalize(target_sin_cos, dim=1)
    return (1.0 - (pred_norm * target_norm).sum(dim=1)).mean()


# ---------- 分类 Focal Loss（缓解正负样本不平衡） ----------
def focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    pred_logits: (N, C) 未经过 sigmoid 的 logits
    targets:     (N, C) 0/1 标签
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt).pow(gamma) * bce
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal


# ---------- probiou / probiou_loss / center_l1_loss ----------
def _get_covariance_matrix(boxes):
    """boxes: (N, 5) -> [cx, cy, w, h, angle]"""
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2, sin2 = cos.pow(2), sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, eps=1e-6):
    """
    旋转框 probiou，输入 (N, 5) -> [cx, cy, w, h, angle]。
    返回 (N,) 的 IoU 值。
    """
    obb1 = obb1.float()
    obb2 = obb2.float()

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    a1 = a1.clamp(min=eps)
    b1 = b1.clamp(min=eps)
    a2 = a2.clamp(min=eps)
    b2 = b2.clamp(min=eps)

    denom = ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
    denom = denom.clamp(min=eps)

    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) /
          (denom + eps)) * 0.25
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
    """(N,5) OBB 的 probiou 直接损失"""
    iou = probiou(pred_obb, target_obb)
    return (1.0 - iou).clamp(min=0).mean()


def center_l1_loss(pred_obb, target_obb):
    """中心点 L1 辅助损失"""
    return (pred_obb[:, :2] - target_obb[:, :2]).abs().mean()


# ---------- CRIoU Loss（FP32 稳定） ----------
def criou_loss(pred_bbox, target_bbox, eps=1e-7):
    """
    pred_bbox / target_bbox: (N, 5) -> [cx, cy, w, h, angle]
    """
    pred_bbox = pred_bbox.float()
    target_bbox = target_bbox.float()

    if torch.isnan(pred_bbox).any() or torch.isnan(target_bbox).any():
        print("Warning: NaN detected in criou_loss inputs, returning 0")
        return torch.tensor(0.0, device=pred_bbox.device, dtype=torch.float32)

    pred_cx, pred_cy, pred_w, pred_h, pred_angle = pred_bbox.unbind(dim=1)
    target_cx, target_cy, target_w, target_h, target_angle = target_bbox.unbind(dim=1)

    pred_w = torch.clamp(pred_w, min=eps)
    pred_h = torch.clamp(pred_h, min=eps)
    target_w = torch.clamp(target_w, min=eps)
    target_h = torch.clamp(target_h, min=eps)

    # 水平外接框 IoU 用于中心点距离项的尺度参考
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2
    target_x1 = target_cx - target_w / 2
    target_y1 = target_cy - target_h / 2
    target_x2 = target_cx + target_w / 2
    target_y2 = target_cy + target_h / 2

    iou_r = probiou(pred_bbox, target_bbox)

    # 中心点距离
    center_dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
    min_x = torch.min(pred_x1, target_x1)
    min_y = torch.min(pred_y1, target_y1)
    max_x = torch.max(pred_x2, target_x2)
    max_y = torch.max(pred_y2, target_y2)
    c = torch.sqrt((max_x - min_x).pow(2) + (max_y - min_y).pow(2) + eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)).pow(2)
    alpha = v / (v - iou_r + (1 + eps))
    criou = iou_r - (center_dist_sq / (c.pow(2) + eps)) - alpha * v

    loss = (1.0 - criou).clamp(min=0).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: criou_loss produced NaN/Inf, returning 0")
        return torch.tensor(0.0, device=pred_bbox.device, dtype=torch.float32)
    return loss


# ---------- 主损失函数 ----------
def compute_fcos_loss(cls_logits_list, reg_preds_list, targets, strides, device, img_size,
                      cls_weight=1.0, reg_weight=1.0, angle_weight=20.0,
                      mode='detect', angle_format='direct', angle_preds_list=None,
                      angle_range='pi', use_focal=True,
                      focal_alpha=0.25, focal_gamma=2.0):
    """
    主要改进：
      1. OBB 正样本采样：旋转框局部坐标系 + 中心采样 + 每 GT 最少正样本数保证
      2. OBB 回归解码：直接回归旋转框 cxcywh（dx/dy 平移，dw/dh 对数尺度）
      3. OBB 角度损失：余弦损失 + 周期性 L1 辅助损失
      4. OBB 回归损失：criou + 0.5 * probiou + 0.1 * center_l1
      5. Detect 回归解码：对 l/t/r/b 应用 ReLU，保证预测框宽高非负
      6. Detect 分类损失：默认使用 Focal Loss 缓解正负样本不平衡
      7. Detect 正样本采样：加入中心采样，提升正样本质量
    """
    if mode == 'obb' and angle_preds_list is None:
        raise ValueError("OBB mode requires angle_preds_list")

    # 强制 FP32
    cls_logits_list = [x.float() for x in cls_logits_list]
    reg_preds_list = [x.float() for x in reg_preds_list]
    if angle_preds_list is not None:
        angle_preds_list = [x.float() for x in angle_preds_list]

    num_levels = len(cls_logits_list)

    all_cls_logits = []
    all_cls_targets = []
    all_reg_pred = []
    all_reg_gt = []
    all_angle_preds = []
    all_angle_targets = []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx]
        reg_preds = reg_preds_list[level_idx]

        B, C, H, W = cls_logits.shape
        y_coords = torch.arange(H, device=device).float() * stride + stride / 2
        x_coords = torch.arange(W, device=device).float() * stride + stride / 2
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

        for batch_idx in range(B):
            if mode == 'detect':
                gt_boxes = targets[batch_idx]['boxes'].float() * img_size
                gt_labels = targets[batch_idx]['labels']
            else:
                gt_cxcywh = targets[batch_idx]['cxcywh'].float() * img_size
                gt_labels = targets[batch_idx]['labels']
                gt_angles = targets[batch_idx]['angles'].float()

            N = gt_boxes.shape[0] if mode == 'detect' else gt_cxcywh.shape[0]
            if N == 0:
                cls_logits_flat = cls_logits[batch_idx].permute(1, 2, 0).reshape(-1, C)
                cls_logits_flat = torch.clamp(cls_logits_flat, min=-20, max=20)
                cls_target_flat = torch.zeros((H * W, C), dtype=torch.float32, device=device)
                all_cls_logits.append(cls_logits_flat)
                all_cls_targets.append(cls_target_flat)
                continue

            # ---------------- 正样本采样 ----------------
            grid_points = grid_xy.view(2, -1).T
            gx = grid_points[:, 0:1]                # (P, 1)
            gy = grid_points[:, 1:2]                # (P, 1)

            if mode == 'detect':
                x1 = gt_boxes[:, 0].unsqueeze(0)
                y1 = gt_boxes[:, 1].unsqueeze(0)
                x2 = gt_boxes[:, 2].unsqueeze(0)
                y2 = gt_boxes[:, 3].unsqueeze(0)
                inside_bbox = (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)

                # ---- 中心采样：只保留靠近 GT 中心的点 ----
                gt_cx = (x1 + x2) / 2
                gt_cy = (y1 + y2) / 2
                gt_w = (x2 - x1).clamp(min=1e-6)
                gt_h = (y2 - y1).clamp(min=1e-6)
                radius = torch.clamp(
                    torch.sqrt(gt_w * gt_h) * 0.25,
                    min=1.0,
                    max=float(stride * 3)
                )
                dist_sq = (gx - gt_cx).pow(2) + (gy - gt_cy).pow(2)
                inside_center = dist_sq <= radius.pow(2)
                inside = inside_bbox & inside_center

                # 若中心采样后某 GT 无点，退回完整框内点
                center_count = inside.sum(dim=0)
                empty_gt = center_count == 0
                if empty_gt.any():
                    inside[:, empty_gt] = inside_bbox[:, empty_gt]

                areas = (x2 - x1) * (y2 - y1)
            else:
                gt_cx = gt_cxcywh[:, 0]
                gt_cy = gt_cxcywh[:, 1]
                gt_w  = gt_cxcywh[:, 2]
                gt_h  = gt_cxcywh[:, 3]
                cos_a = torch.cos(gt_angles)
                sin_a = torch.sin(gt_angles)

                dx = gx - gt_cx.unsqueeze(0)         # (P, N)
                dy = gy - gt_cy.unsqueeze(0)

                # 与 cxcywh_angle_to_obb 中 [[cos, sin], [-sin, cos]] 对应的逆变换
                u = dx * cos_a.unsqueeze(0) + dy * sin_a.unsqueeze(0)
                v = -dx * sin_a.unsqueeze(0) + dy * cos_a.unsqueeze(0)

                half_w = (gt_w * 0.5).unsqueeze(0)
                half_h = (gt_h * 0.5).unsqueeze(0)
                inside_obb = (u.abs() <= half_w) & (v.abs() <= half_h)

                # 中心采样：只保留靠近旋转框中心的点
                radius = torch.clamp(
                    torch.sqrt(gt_w * gt_h).unsqueeze(0) * 0.25,
                    min=1.0,
                    max=float(stride * 3)
                )
                inside_center = (u.pow(2) + v.pow(2)) <= radius.pow(2)
                inside = inside_obb & inside_center

                # 若中心采样后某 GT 无点，退回完整旋转框内点
                center_count = inside.sum(dim=0)  # (N,)
                empty_gt = center_count == 0
                if empty_gt.any():
                    inside[:, empty_gt] = inside_obb[:, empty_gt]

                areas = (gt_w * gt_h).unsqueeze(0)

            # 面积优先分配
            gt_idx_per_point = torch.full((H * W,), -1, dtype=torch.long, device=device)
            min_area = torch.full((H * W,), float('inf'), device=device)
            for i in range(N):
                covered = inside[:, i]
                area = areas[0, i]
                update = covered & (area < min_area)
                gt_idx_per_point[update] = i
                min_area[update] = area

            # 每个 GT 至少 MIN_POS_PER_GT 个正样本
            assigned_count = torch.bincount(
                gt_idx_per_point[gt_idx_per_point != -1], minlength=N
            )
            for i in range(N):
                deficit = MIN_POS_PER_GT - int(assigned_count[i].item())
                if deficit <= 0:
                    continue
                if mode == 'detect':
                    cx_i = (gt_boxes[i, 0] + gt_boxes[i, 2]) / 2
                    cy_i = (gt_boxes[i, 1] + gt_boxes[i, 3]) / 2
                else:
                    cx_i = gt_cxcywh[i, 0]
                    cy_i = gt_cxcywh[i, 1]
                dist = (grid_points[:, 0] - cx_i).pow(2) + (grid_points[:, 1] - cy_i).pow(2)
                # 排除已经属于该 GT 的点
                dist = dist.masked_fill(gt_idx_per_point == i, float('inf'))
                k = min(deficit, int((gt_idx_per_point != i).sum().item()))
                if k <= 0:
                    continue
                _, nearest_idx = dist.topk(k, largest=False)
                gt_idx_per_point[nearest_idx] = i

            # ---------------- 分类目标 ----------------
            cls_target = torch.zeros((C, H, W), dtype=torch.float32, device=device)
            gt_idx_map = gt_idx_per_point.view(H, W)
            valid_mask_2d = gt_idx_map != -1
            rows, cols = valid_mask_2d.nonzero(as_tuple=True)
            if rows.numel() > 0:
                gt_indices = gt_idx_map[rows, cols].to(torch.long)
                labels = gt_labels[gt_indices]
                cls_target[labels, rows, cols] = 1.0

            cls_logits_flat = cls_logits[batch_idx].permute(1, 2, 0).reshape(-1, C)
            cls_logits_flat = torch.clamp(cls_logits_flat, min=-20, max=20)
            cls_target_flat = cls_target.permute(1, 2, 0).reshape(-1, C)
            all_cls_logits.append(cls_logits_flat)
            all_cls_targets.append(cls_target_flat)

            # ---------------- 回归 / 角度 ----------------
            if rows.numel() > 0:
                grid_x_pts = grid_x[rows, cols]
                grid_y_pts = grid_y[rows, cols]

                if mode == 'detect':
                    selected_gt = gt_boxes[gt_indices]
                    reg_preds_local = reg_preds[batch_idx][:, rows, cols]
                    # 对 l/t/r/b 应用 ReLU，保证预测距离非负
                    pred_l = F.relu(reg_preds_local[0])
                    pred_t = F.relu(reg_preds_local[1])
                    pred_r = F.relu(reg_preds_local[2])
                    pred_b = F.relu(reg_preds_local[3])
                    pred_x1 = grid_x_pts - pred_l
                    pred_y1 = grid_y_pts - pred_t
                    pred_x2 = grid_x_pts + pred_r
                    pred_y2 = grid_y_pts + pred_b
                    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                    all_reg_pred.append(pred_boxes)
                    all_reg_gt.append(selected_gt)
                else:
                    selected_gt_cxcywh = gt_cxcywh[gt_indices]
                    selected_gt_angles = gt_angles[gt_indices]

                    reg_preds_local = reg_preds[batch_idx][:, rows, cols]

                    # ---- 直接解码旋转框 cxcywh ----
                    pred_dx = reg_preds_local[0]
                    pred_dy = reg_preds_local[1]
                    pred_dw = reg_preds_local[2]
                    pred_dh = reg_preds_local[3]

                    pred_cx = grid_x_pts + pred_dx * stride
                    pred_cy = grid_y_pts + pred_dy * stride
                    pred_w  = torch.exp(pred_dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                    pred_h  = torch.exp(pred_dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

                    # ---- 角度预测 ----
                    angle_preds_local = angle_preds_list[level_idx][batch_idx][:, rows, cols]
                    if angle_format == 'direct':
                        pred_angle = angle_preds_local[0]
                        all_angle_preds.append(pred_angle)
                        all_angle_targets.append(selected_gt_angles)
                        pred_angle_for_reg = pred_angle
                    else:  # sin_cos
                        pred_sin = angle_preds_local[0]
                        pred_cos = angle_preds_local[1]
                        target_sin = torch.sin(selected_gt_angles)
                        target_cos = torch.cos(selected_gt_angles)
                        all_angle_preds.append(torch.stack([pred_sin, pred_cos], dim=1))
                        all_angle_targets.append(torch.stack([target_sin, target_cos], dim=1))
                        pred_angle_for_reg = torch.atan2(pred_sin, pred_cos)

                    pred_obb = torch.stack(
                        [pred_cx, pred_cy, pred_w, pred_h, pred_angle_for_reg], dim=1
                    )
                    gt_obb = torch.cat(
                        [selected_gt_cxcywh, selected_gt_angles.unsqueeze(1)], dim=1
                    )
                    all_reg_pred.append(pred_obb)
                    all_reg_gt.append(gt_obb)

    # ---- 分类损失 ----
    if len(all_cls_logits) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    cls_logits_all = torch.cat(all_cls_logits, dim=0)
    cls_targets_all = torch.cat(all_cls_targets, dim=0)
    cls_logits_all = torch.clamp(cls_logits_all, min=-20, max=20)

    if use_focal:
        cls_loss = focal_loss(
            cls_logits_all, cls_targets_all,
            alpha=focal_alpha, gamma=focal_gamma, reduction='mean'
        )
    else:
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits_all, cls_targets_all, reduction='mean'
        )

    if torch.isnan(cls_loss) or torch.isinf(cls_loss):
        print("Warning: cls_loss is NaN/Inf, setting to 0")
        cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    # ---- 回归损失 ----
    reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if len(all_reg_pred) > 0:
        pred_boxes = torch.cat(all_reg_pred, dim=0)
        gt_boxes = torch.cat(all_reg_gt, dim=0)
        if torch.isnan(pred_boxes).any() or torch.isnan(gt_boxes).any():
            print("Warning: NaN in regression pred or gt, skipping reg_loss")
        else:
            if mode == 'detect':
                reg_loss = generalized_box_iou_loss(pred_boxes, gt_boxes, reduction='mean')
                reg_loss = torch.clamp(reg_loss, min=0.0)
            else:
                # OBB：criou + 0.5 * probiou + 0.1 * center_l1
                l_criou = criou_loss(pred_boxes, gt_boxes)
                l_probiou = probiou_loss(pred_boxes, gt_boxes)
                l_center = center_l1_loss(pred_boxes, gt_boxes)
                reg_loss = l_criou + 0.5 * l_probiou + 0.1 * l_center
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            print("Warning: reg_loss is NaN/Inf, setting to 0")
            reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    # ---- 角度损失 ----
    angle_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if mode == 'obb' and len(all_angle_preds) > 0:
        if angle_format == 'direct':
            pred_angles = torch.cat(all_angle_preds, dim=0)
            target_angles = torch.cat(all_angle_targets, dim=0)
            if torch.isnan(pred_angles).any() or torch.isnan(target_angles).any():
                print("Warning: NaN in direct angle pred/target, skipping angle_loss")
            else:
                angle_loss = periodic_l1_loss(pred_angles, target_angles, angle_range)
        else:  # sin_cos
            pred_sin_cos = torch.cat(all_angle_preds, dim=0)
            target_sin_cos = torch.cat(all_angle_targets, dim=0)
            if torch.isnan(pred_sin_cos).any() or torch.isnan(target_sin_cos).any():
                print("Warning: NaN in sin_cos angle pred/target, skipping angle_loss")
            else:
                # 余弦损失（主要）
                cos_loss = cosine_angle_loss(pred_sin_cos, target_sin_cos)
                # 辅助周期 L1（对接近对齐的角度更敏感）
                pred_angle = torch.atan2(pred_sin_cos[:, 0], pred_sin_cos[:, 1])
                target_angle = torch.atan2(target_sin_cos[:, 0], target_sin_cos[:, 1])
                aux_loss = periodic_l1_loss(pred_angle, target_angle, angle_range)
                angle_loss = cos_loss + 0.5 * aux_loss
        if torch.isnan(angle_loss) or torch.isinf(angle_loss):
            print("Warning: angle_loss is NaN/Inf, setting to 0")
            angle_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    total_loss = cls_weight * cls_loss + reg_weight * reg_loss + angle_weight * angle_loss

    return total_loss


# ---------- 后处理（水平框） ----------
def post_process(cls_logits_list, reg_preds_list,
                 orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                 strides=None, scale=None, pad_left=0, pad_top=0):
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2] for i in range(num_levels)]

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

        y_grid = grid_y[keep]
        x_grid = grid_x[keep]
        scores_keep = max_scores[keep]
        labels_keep = max_labels[keep]
        reg = reg_preds[:, keep]

        # 对 l/t/r/b 应用 ReLU，保证预测距离非负（与训练解码保持一致）
        l = torch.relu(reg[0])
        t = torch.relu(reg[1])
        r = torch.relu(reg[2])
        b = torch.relu(reg[3])
        x1 = x_grid - l
        y1 = y_grid - t
        x2 = x_grid + r
        y2 = y_grid + b
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

    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    return boxes, scores, labels


# ---------- 旋转框 NMS ----------
def rotated_nms(obboxes, scores, iou_threshold=0.5):
    if len(obboxes) == 0:
        return np.array([])
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("警告: shapely 未安装，旋转 NMS 将使用水平框 NMS 替代")
        boxes_xyxy = np.zeros((len(obboxes), 4), dtype=np.float32)
        for i, obbox in enumerate(obboxes):
            pts = obbox.reshape(4, 2)
            boxes_xyxy[i] = [np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])]
        scores = scores.astype(np.float32)
        keep = nms(torch.from_numpy(boxes_xyxy).float(), torch.from_numpy(scores).float(), iou_threshold).numpy()
        return keep

    def obbox_to_polygon(obbox):
        pts = obbox.reshape(4, 2)
        return Polygon(pts)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        poly_i = obbox_to_polygon(obboxes[i])
        rest = order[1:]
        if rest.size == 0:
            break
        ious = []
        for j in rest:
            poly_j = obbox_to_polygon(obboxes[j])
            if not poly_i.is_valid or not poly_j.is_valid or poly_i.area == 0 or poly_j.area == 0:
                ious.append(0.0)
                continue
            inter = poly_i.intersection(poly_j).area
            union = poly_i.area + poly_j.area - inter
            iou = inter / (union + 1e-6)
            ious.append(iou)
        ious = np.array(ious)
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


# ---------- OBB 后处理 ----------
def post_process_obb(cls_logits_list, reg_preds_list, angle_preds_list,
                     orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                     strides=None, scale=None, pad_left=0, pad_top=0,
                     angle_format='direct', angle_range='pi', max_det=1000):
    """
    与训练解码保持一致：reg 输出 [dx, dy, dw, dh]
      cx = grid_x + dx * stride
      cy = grid_y + dy * stride
      w  = exp(dw) * stride
      h  = exp(dh) * stride
    max_det 从 300 提到 1000，避免小目标被截断。
    """
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2] for i in range(num_levels)]

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

        y_grid = grid_y[keep]
        x_grid = grid_x[keep]
        scores_keep = max_scores[keep]
        labels_keep = max_labels[keep]
        reg = reg_preds[:, keep]

        dx, dy, dw, dh = reg[0], reg[1], reg[2], reg[3]
        cx = x_grid + dx * stride
        cy = y_grid + dy * stride
        w  = torch.exp(dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
        h  = torch.exp(dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

        cx = torch.clamp(cx, 0, img_size)
        cy = torch.clamp(cy, 0, img_size)
        w = torch.clamp(w, 0, img_size)
        h = torch.clamp(h, 0, img_size)

        if angle_format == 'direct':
            angle = angle_preds[0][keep]
        else:
            sin_val = angle_preds[0][keep]
            cos_val = angle_preds[1][keep]
            angle = torch.atan2(sin_val, cos_val)

        # 强制 w >= h
        if angle_range == 'pi':
            mask = w < h
            if mask.any():
                w[mask], h[mask] = h[mask], w[mask]
                angle[mask] = angle[mask] + math.pi / 2
                angle[mask] = angle[mask] % math.pi
        else:
            mask = w < h
            if mask.any():
                w[mask], h[mask] = h[mask], w[mask]
                angle[mask] = angle[mask] + math.pi / 2
                angle[mask] = torch.atan2(torch.sin(angle[mask]), torch.cos(angle[mask]))
                angle[mask] = torch.clamp(angle[mask], -math.pi / 2, math.pi / 2)

        if angle_range == 'pi':
            angle = angle % math.pi
        else:
            angle = torch.atan2(torch.sin(angle), torch.cos(angle))
            angle = torch.clamp(angle, -math.pi / 2, math.pi / 2)

        cx = cx / img_size
        cy = cy / img_size
        w = w / img_size
        h = h / img_size

        obboxes = cxcywh_angle_to_obb(cx, cy, w, h, angle)
        all_obboxes.append(obboxes)
        all_scores.append(scores_keep)
        all_labels.append(labels_keep)

    if not all_obboxes:
        return [], [], []

    obboxes = torch.cat(all_obboxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if len(obboxes) > max_det:
        idx = torch.argsort(scores, descending=True)[:max_det]
        obboxes = obboxes[idx]
        scores = scores[idx]
        labels = labels[idx]

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

    obboxes_abs = torch.clamp(obboxes_abs, 0, max(orig_w, orig_h))

    keep_indices = rotated_nms(obboxes_abs.cpu().numpy(), scores.cpu().numpy(), nms_threshold)
    obboxes_abs = obboxes_abs[keep_indices].cpu().numpy()
    scores = scores[keep_indices].cpu().numpy()
    labels = labels[keep_indices].cpu().numpy()

    return obboxes_abs, scores, labels


# ---------- 辅助：cxcywh+angle → 8顶点 ----------
def cxcywh_angle_to_obb(cx, cy, w, h, angle):
    N = cx.shape[0]
    device = cx.device
    half_w = w / 2
    half_h = h / 2

    corners = torch.zeros((N, 4, 2), device=device)
    corners[:, 0, 0] = half_w
    corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w
    corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w
    corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w
    corners[:, 3, 1] = -half_h

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rot_matrix = torch.stack([
        torch.stack([cos_a, sin_a], dim=1),
        torch.stack([-sin_a, cos_a], dim=1)
    ], dim=1)

    corners_rot = torch.matmul(corners, rot_matrix)
    center = torch.stack([cx, cy], dim=1).unsqueeze(1)
    corners_final = corners_rot + center

    obboxes = corners_final.reshape(N, 8)
    return torch.clamp(obboxes, 0, 1)


# ---------- 旋转框 mAP 评估 ----------
def evaluate_map_obb_rotated(model, dataloader, device, cfg,
                             conf_threshold=0.01, nms_threshold=None, max_det=300):
    """
    使用 shapely 计算旋转框 IoU，输出 VOC 风格的 mAP@0.5 / mAP@0.5:0.95。
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("警告: shapely 未安装，无法做旋转框 mAP 评估，返回 0")
        return 0.0, 0.0

    if nms_threshold is None:
        nms_threshold = cfg.obb_nms_threshold

    all_preds = []
    all_gts = []

    model.eval()
    img_id = 0
    with torch.no_grad():
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
                )
                for ob, sc, lb in zip(obboxes, scores, labels):
                    all_preds.append(dict(obbox=ob.astype(np.float32),
                                          score=float(sc),
                                          label=int(lb),
                                          img_id=img_id))
                # GT
                gt_obboxes = t['obboxes'].clone().numpy()
                if gt_obboxes.shape[0] > 0:
                    gt_fill = gt_obboxes * cfg.img_size
                    gt_abs = gt_fill.copy()
                    for i in range(4):
                        gt_abs[:, i * 2] = (gt_fill[:, i * 2] - t['pad_left']) / t['scale']
                        gt_abs[:, i * 2 + 1] = (gt_fill[:, i * 2 + 1] - t['pad_top']) / t['scale']
                    gt_abs = np.clip(gt_abs, 0, max(orig_w, orig_h))
                else:
                    gt_abs = np.zeros((0, 8), dtype=np.float32)
                gt_labels = t['labels'].numpy()
                for ob, lb in zip(gt_abs, gt_labels):
                    all_gts.append(dict(obbox=ob.astype(np.float32),
                                        label=int(lb),
                                        img_id=img_id,
                                        used=False))
                img_id += 1

    if len(all_preds) == 0 or len(all_gts) == 0:
        return 0.0, 0.0

    def poly_from_obbox(obbox):
        return Polygon(obbox.reshape(4, 2))

    def rot_iou(ob1, ob2):
        p1 = poly_from_obbox(ob1)
        p2 = poly_from_obbox(ob2)
        if p1.area == 0 or p2.area == 0 or not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / (union + 1e-6)

    num_classes = cfg.num_classes
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    ap_per_class_per_iou = np.zeros((num_classes, len(iou_thresholds)), dtype=np.float32)

    for cls_id in range(num_classes):
        preds_cls = [p for p in all_preds if p['label'] == cls_id]
        gts_cls = [g for g in all_gts if g['label'] == cls_id]
        n_gt = len(gts_cls)
        if n_gt == 0 or len(preds_cls) == 0:
            continue

        preds_cls = sorted(preds_cls, key=lambda x: x['score'], reverse=True)

        for ti, iou_thr in enumerate(iou_thresholds):
            gt_used = [False] * n_gt
            tp = np.zeros(len(preds_cls), dtype=np.float32)
            fp = np.zeros(len(preds_cls), dtype=np.float32)

            for pi, p in enumerate(preds_cls):
                best_iou = 0.0
                best_gt = -1
                for gi, g in enumerate(gts_cls):
                    if gt_used[gi]:
                        continue
                    if g['img_id'] != p['img_id']:
                        continue
                    iou = rot_iou(p['obbox'], g['obbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi
                if best_iou >= iou_thr and best_gt >= 0:
                    tp[pi] = 1.0
                    gt_used[best_gt] = True
                else:
                    fp[pi] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / (n_gt + 1e-9)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
            ap_per_class_per_iou[cls_id, ti] = ap

    valid_classes = [c for c in range(num_classes)
                     if any(g['label'] == c for g in all_gts)]
    if not valid_classes:
        return 0.0, 0.0
    map_per_iou = ap_per_class_per_iou[valid_classes].mean(axis=0)
    mAP50 = float(map_per_iou[0])
    mAP50_95 = float(map_per_iou.mean())
    return mAP50, mAP50_95


# ---------- mAP 评估（水平框版本） ----------
def evaluate_mAP(model, dataloader, device, cfg):
    """原版本，用水平外接框计算 mAP（会虚高 mAP50，低估 mAP50:95）"""
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        print("警告: torchmetrics 未安装，mAP 评估将返回 0。请运行: pip install torchmetrics")
        return 0.0, 0.0, 0.0

    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=False)
    metric.warn_on_many_detections = False

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            if cfg.mode == 'detect':
                cls_logits_list, reg_preds_list = model(images)
                for idx, t in enumerate(targets):
                    orig_w, orig_h = t['orig_size']
                    boxes, scores, labels = post_process(
                        [cls_logits_list[level][idx:idx + 1] for level in range(len(cls_logits_list))],
                        [reg_preds_list[level][idx:idx + 1] for level in range(len(reg_preds_list))],
                        orig_w, orig_h, cfg.img_size,
                        conf_threshold=0.01,
                        nms_threshold=cfg.nms_threshold,
                        strides=cfg.strides,
                        scale=t['scale'],
                        pad_left=t['pad_left'],
                        pad_top=t['pad_top']
                    )
                    gt_boxes = t['boxes'].clone()
                    scale = t['scale']
                    pad_left = t['pad_left']
                    pad_top = t['pad_top']
                    gt_abs = gt_boxes * cfg.img_size
                    gt_abs[:, [0, 2]] = (gt_abs[:, [0, 2]] - pad_left) / scale
                    gt_abs[:, [1, 3]] = (gt_abs[:, [1, 3]] - pad_top) / scale
                    gt_abs = torch.clamp(gt_abs, 0, max(orig_w, orig_h))
                    gt_abs[:, 0] = torch.clamp(gt_abs[:, 0], 0, orig_w)
                    gt_abs[:, 1] = torch.clamp(gt_abs[:, 1], 0, orig_h)
                    gt_abs[:, 2] = torch.clamp(gt_abs[:, 2], 0, orig_w)
                    gt_abs[:, 3] = torch.clamp(gt_abs[:, 3], 0, orig_h)

                    if len(boxes) > 100:
                        idx_sorted = np.argsort(scores)[::-1][:100]
                        boxes = boxes[idx_sorted]
                        scores = scores[idx_sorted]
                        labels = labels[idx_sorted]

                    preds = [dict(
                        boxes=torch.from_numpy(boxes).to(device) if len(boxes) > 0 else torch.zeros((0, 4), device=device),
                        scores=torch.from_numpy(scores).to(device) if len(scores) > 0 else torch.zeros(0, device=device),
                        labels=torch.from_numpy(labels).to(device) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64, device=device)
                    )]
                    gts = [dict(boxes=gt_abs.to(device), labels=t['labels'].to(device))]
                    metric.update(preds, gts)
            else:
                cls_logits_list, reg_preds_list, angle_preds_list = model(images)
                for idx, t in enumerate(targets):
                    orig_w, orig_h = t['orig_size']
                    obboxes, scores, labels = post_process_obb(
                        [cls_logits_list[level][idx:idx + 1] for level in range(len(cls_logits_list))],
                        [reg_preds_list[level][idx:idx + 1] for level in range(len(reg_preds_list))],
                        [angle_preds_list[level][idx:idx + 1] for level in range(len(angle_preds_list))],
                        orig_w, orig_h, cfg.img_size,
                        conf_threshold=0.001,
                        nms_threshold=cfg.obb_nms_threshold,
                        strides=cfg.strides,
                        scale=t['scale'],
                        pad_left=t['pad_left'],
                        pad_top=t['pad_top'],
                        angle_format=cfg.angle_format,
                        angle_range=cfg.angle_range
                    )
                    if len(obboxes) > 100:
                        idx_sorted = np.argsort(scores)[::-1][:100]
                        obboxes = obboxes[idx_sorted]
                        scores = scores[idx_sorted]
                        labels = labels[idx_sorted]

                    if len(obboxes) > 0:
                        boxes_xyxy = np.zeros((len(obboxes), 4))
                        for i, obbox in enumerate(obboxes):
                            pts = obbox.reshape(4, 2)
                            boxes_xyxy[i] = [np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])]
                    else:
                        boxes_xyxy = np.zeros((0, 4))

                    gt_obboxes = t['obboxes'].clone()
                    if gt_obboxes.shape[0] > 0:
                        gt_fill = gt_obboxes * cfg.img_size
                        gt_abs = gt_fill.clone()
                        for i in range(4):
                            gt_abs[:, i * 2] = (gt_fill[:, i * 2] - t['pad_left']) / t['scale']
                            gt_abs[:, i * 2 + 1] = (gt_fill[:, i * 2 + 1] - t['pad_top']) / t['scale']
                        gt_abs = torch.clamp(gt_abs, 0, max(orig_w, orig_h))
                        gt_boxes_xyxy = torch.zeros((gt_abs.shape[0], 4), device=device)
                        for i in range(gt_abs.shape[0]):
                            pts = gt_abs[i].reshape(4, 2)
                            gt_boxes_xyxy[i] = torch.tensor([
                                torch.min(pts[:, 0]), torch.min(pts[:, 1]),
                                torch.max(pts[:, 0]), torch.max(pts[:, 1])
                            ])
                    else:
                        gt_boxes_xyxy = torch.zeros((0, 4), device=device)

                    preds = [dict(
                        boxes=torch.from_numpy(boxes_xyxy).to(device) if len(boxes_xyxy) > 0 else torch.zeros((0, 4), device=device),
                        scores=torch.from_numpy(scores).to(device) if len(scores) > 0 else torch.zeros(0, device=device),
                        labels=torch.from_numpy(labels).to(device) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64, device=device)
                    )]
                    gts = [dict(boxes=gt_boxes_xyxy.to(device), labels=t['labels'].to(device))]
                    metric.update(preds, gts)

    results = metric.compute()
    return results['map_50'], results['map'], results['mar_100']