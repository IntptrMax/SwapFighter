import torch
import torch.nn.functional as F
from torchvision.ops import nms, generalized_box_iou_loss
import cv2
import numpy as np

# ---------- Letterbox 函数 ----------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    """
    将图像缩放并填充至 new_shape，保持宽高比。
    返回: 缩放后的图像 (RGB), 缩放因子, 填充的 (左, 上), 原始尺寸 (height, width)
    """
    shape = img.shape[:2]  # [height, width]
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


# ---------- 辅助函数：计算 IoU 矩阵 ----------
def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    """
    计算两个边界框集合之间的 IoU 矩阵。
    bboxes1: (N, 4), bboxes2: (M, 4)  格式 [x1, y1, x2, y2]
    返回: (N, M) IoU 矩阵
    """
    if bboxes1.numel() == 0 or bboxes2.numel() == 0:
        return torch.zeros((bboxes1.shape[0], bboxes2.shape[0]), device=bboxes1.device)
    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])   # (N, M, 2)
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])   # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                            # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]                      # (N, M)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # (N,)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # (M,)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + eps)
    return iou


# ---------- Varifocal Loss ----------
def varifocal_loss(pred, target, alpha=0.75, gamma=2.0):
    """
    Varifocal Loss (稳定版)
    pred: (N, C) 预测 logits
    target: (N, C) 软标签，正样本位置为 IoU (0~1)，负样本为 0
    返回: 标量损失（非负）
    """
    # 分离正负样本
    pos_mask = target > 0
    neg_mask = target == 0

    # 如果没有正样本，直接返回 0（避免负样本单独贡献）
    if not pos_mask.any():
        return torch.tensor(0.0, device=pred.device)

    pred_sigmoid = pred.sigmoid()
    loss = torch.tensor(0.0, device=pred.device)

    # 正样本损失
    pos_pred = pred[pos_mask]
    pos_target = target[pos_mask]
    p_t = pos_pred.sigmoid() * pos_target + (1 - pos_pred.sigmoid()) * (1 - pos_target)
    pos_weight = (1 - p_t) ** gamma
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, pos_target, reduction='none')
    loss += (pos_loss * pos_weight * alpha).sum()

    # 负样本损失（仅当负样本存在时）
    if neg_mask.any():
        neg_pred = pred[neg_mask]
        neg_target = target[neg_mask]  # 全 0
        neg_weight = pred_sigmoid[neg_mask] ** gamma
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, neg_target, reduction='none')
        loss += (neg_loss * neg_weight * (1 - alpha)).sum()

    # 按正样本数量平均
    pos_count = pos_mask.sum().float()
    loss = loss / pos_count
    # 数值截断，防止极小负值（理论上不会，但为安全）
    return torch.clamp(loss, min=0.0)


# ---------- FCOS 损失 (VFL + GIoU) ----------
def compute_fcos_loss(cls_logits_list, reg_preds_list,
                      targets, strides, device, img_size,
                      cls_weight=1.0, reg_weight=2.0):
    """
    使用 Varifocal Loss 和 GIoU Loss 计算总损失。
    cls_logits_list: list of (B, C, H, W)   每个FPN层的分类logits
    reg_preds_list:  list of (B, 4, H, W)   每个FPN层的回归预测 (l,t,r,b)
    targets: list of dict，包含 'boxes' (归一化) 和 'labels'
    strides: list of int，每层的下采样步长
    device: torch.device
    img_size: int，输入图像尺寸（用于将归一化坐标转为绝对坐标）
    cls_weight, reg_weight: 损失权重
    返回: 标量损失
    """
    num_levels = len(cls_logits_list)
    batch_size = cls_logits_list[0].shape[0]

    all_cls_logits = []
    all_cls_targets = []      # (num_points, C) 每点一个目标向量
    all_reg_preds = []
    all_reg_targets = []
    all_giou_pred_boxes = []
    all_giou_gt_boxes = []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx]   # (B, C, H, W)
        reg_preds = reg_preds_list[level_idx]     # (B, 4, H, W)

        B, C, H, W = cls_logits.shape
        y_coords = torch.arange(H, device=device).float() * stride + stride / 2
        x_coords = torch.arange(W, device=device).float() * stride + stride / 2
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([grid_x, grid_y], dim=0)   # (2, H, W)

        for batch_idx in range(B):
            gt_boxes = targets[batch_idx]['boxes'] * img_size   # (N,4) 绝对坐标
            gt_labels = targets[batch_idx]['labels']            # (N,)
            N = gt_boxes.shape[0]

            cls_target = torch.zeros((C, H, W), dtype=torch.float32, device=device)
            reg_target = torch.zeros((4, H, W), dtype=torch.float32, device=device)
            pos_pred_boxes = []
            pos_gt_boxes = []

            if N > 0:
                grid_points = grid_xy.view(2, -1).T  # (M, 2)
                x1 = gt_boxes[:, 0].unsqueeze(0)      # (1, N)
                y1 = gt_boxes[:, 1].unsqueeze(0)
                x2 = gt_boxes[:, 2].unsqueeze(0)
                y2 = gt_boxes[:, 3].unsqueeze(0)
                gx = grid_points[:, 0].unsqueeze(1)   # (M, 1)
                gy = grid_points[:, 1].unsqueeze(1)

                inside = (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)  # (M, N)
                areas = (x2 - x1) * (y2 - y1)          # (1, N)

                gt_idx_per_point = torch.full((H * W,), -1, dtype=torch.long, device=device)
                min_area = torch.full((H * W,), float('inf'), device=device)
                for i in range(N):
                    covered = inside[:, i]
                    area = areas[0, i]
                    update = covered & (area < min_area)
                    gt_idx_per_point[update] = i
                    min_area[update] = area

                gt_idx_map = gt_idx_per_point.view(H, W)
                valid_mask = gt_idx_map != -1
                if valid_mask.any():
                    rows, cols = valid_mask.nonzero(as_tuple=True)
                    gt_indices = gt_idx_map[rows, cols].to(torch.long)
                    selected_gt = gt_boxes[gt_indices]       # (M', 4)
                    selected_labels = gt_labels[gt_indices]  # (M',)
                    grid_x_pts = grid_x[rows, cols]
                    grid_y_pts = grid_y[rows, cols]

                    l = grid_x_pts - selected_gt[:, 0]
                    t = grid_y_pts - selected_gt[:, 1]
                    r = selected_gt[:, 2] - grid_x_pts
                    b = selected_gt[:, 3] - grid_y_pts
                    reg_targets = torch.stack([l, t, r, b], dim=1)   # (M', 4)

                    reg_preds_local = reg_preds[batch_idx][:, rows, cols]  # (4, M')
                    pred_l = reg_preds_local[0]
                    pred_t = reg_preds_local[1]
                    pred_r = reg_preds_local[2]
                    pred_b = reg_preds_local[3]
                    pred_x1 = grid_x_pts - pred_l
                    pred_y1 = grid_y_pts - pred_t
                    pred_x2 = grid_x_pts + pred_r
                    pred_y2 = grid_y_pts + pred_b
                    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (M', 4)

                    ious_matrix = bbox_overlaps(pred_boxes, selected_gt)  # (M', M')
                    ious = ious_matrix.diag()  # (M',)

                    cls_target[selected_labels, rows, cols] = ious
                    reg_target[:, rows, cols] = reg_targets.T
                    pos_pred_boxes.append(pred_boxes)
                    pos_gt_boxes.append(selected_gt)

            # 展平收集
            cls_logits_flat = cls_logits[batch_idx].permute(1, 2, 0).reshape(-1, C)
            cls_target_flat = cls_target.permute(1, 2, 0).reshape(-1, C)
            reg_preds_flat = reg_preds[batch_idx].permute(1, 2, 0).reshape(-1, 4)
            reg_target_flat = reg_target.permute(1, 2, 0).reshape(-1, 4)

            all_cls_logits.append(cls_logits_flat)
            all_cls_targets.append(cls_target_flat)
            all_reg_preds.append(reg_preds_flat)
            all_reg_targets.append(reg_target_flat)

            if len(pos_pred_boxes) > 0:
                all_giou_pred_boxes.append(torch.cat(pos_pred_boxes, dim=0))
                all_giou_gt_boxes.append(torch.cat(pos_gt_boxes, dim=0))

    # 拼接
    cls_logits_all = torch.cat(all_cls_logits, dim=0)
    cls_targets_all = torch.cat(all_cls_targets, dim=0)

    # 分类损失
    cls_loss = varifocal_loss(cls_logits_all, cls_targets_all, alpha=0.75, gamma=2.0)

    # 回归损失 (GIoU)
    reg_loss = torch.tensor(0.0, device=device)
    if len(all_giou_pred_boxes) > 0:
        pred_boxes_all = torch.cat(all_giou_pred_boxes, dim=0)
        gt_boxes_all = torch.cat(all_giou_gt_boxes, dim=0)
        # 过滤无效框（宽高必须为正）
        valid = (pred_boxes_all[:, 2] > pred_boxes_all[:, 0]) & (pred_boxes_all[:, 3] > pred_boxes_all[:, 1]) & \
                (gt_boxes_all[:, 2] > gt_boxes_all[:, 0]) & (gt_boxes_all[:, 3] > gt_boxes_all[:, 1])
        if valid.any():
            pred_boxes_valid = pred_boxes_all[valid]
            gt_boxes_valid = gt_boxes_all[valid]
            reg_loss = generalized_box_iou_loss(pred_boxes_valid, gt_boxes_valid, reduction='mean')
            reg_loss = torch.clamp(reg_loss, min=0.0)

    total_loss = cls_weight * cls_loss + reg_weight * reg_loss
    return total_loss


# ---------- 后处理 (去除 centerness) ----------
def post_process(cls_logits_list, reg_preds_list,
                 orig_w, orig_h, img_size, conf_threshold=0.5, nms_threshold=0.5,
                 strides=None, scale=None, pad_left=0, pad_top=0):
    """
    对单张图片的模型输出进行后处理，得到最终检测框。
    cls_logits_list: list of (1, C, H, W) 每个FPN层的分类logits
    reg_preds_list:  list of (1, 4, H, W) 每个FPN层的回归预测 (l,t,r,b)
    其他参数同前。
    返回: boxes, scores, labels (numpy arrays)
    """
    device = cls_logits_list[0].device
    num_levels = len(cls_logits_list)
    if strides is None:
        strides = [img_size // cls_logits_list[i].shape[2] for i in range(num_levels)]

    all_boxes, all_scores, all_labels = [], [], []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        cls_logits = cls_logits_list[level_idx].squeeze(0)  # (C, H, W)
        reg_preds = reg_preds_list[level_idx].squeeze(0)    # (4, H, W)

        C, H, W = cls_logits.shape
        scores = torch.sigmoid(cls_logits)   # (C, H, W)

        max_scores, max_labels = torch.max(scores, dim=0)  # (H, W)
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
        reg = reg_preds[:, keep]   # (4, N)

        l, t, r, b = reg[0], reg[1], reg[2], reg[3]
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


def evaluate_mAP(model, dataloader, device, cfg):
    """
    使用 torchmetrics 计算验证集上的 mAP@0.5, mAP@0.5:0.95, AR@100。
    需要安装 torchmetrics: pip install torchmetrics
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        print("警告: torchmetrics 未安装，mAP 评估将返回 0。请运行: pip install torchmetrics")
        return 0.0, 0.0, 0.0

    # 创建 metric 实例，使用正确的参数名 max_detection
    metric = MeanAveragePrecision(
        iou_type='bbox',
        class_metrics=False,
        # max_detection_thresholds=[1,10,100],
    )
    # 关闭“检测框过多”警告（可选）
    metric.warn_on_many_detections = False

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            cls_logits_list, reg_preds_list = model(images)
            for idx, t in enumerate(targets):
                orig_w, orig_h = t['orig_size']
                boxes, scores, labels = post_process(
                    [cls_logits_list[level][idx:idx+1] for level in range(len(cls_logits_list))],
                    [reg_preds_list[level][idx:idx+1] for level in range(len(reg_preds_list))],
                    orig_w, orig_h, cfg.img_size,
                    conf_threshold=0.01,
                    nms_threshold=cfg.nms_threshold,
                    strides=cfg.strides,
                    scale=t['scale'],
                    pad_left=t['pad_left'],
                    pad_top=t['pad_top']
                )
                # 准备 GT
                gt_boxes = t['boxes'].clone()
                scale = t['scale']
                pad_left = t['pad_left']
                pad_top = t['pad_top']
                gt_abs = gt_boxes * cfg.img_size
                gt_abs[:, [0,2]] = (gt_abs[:, [0,2]] - pad_left) / scale
                gt_abs[:, [1,3]] = (gt_abs[:, [1,3]] - pad_top) / scale
                gt_abs = torch.clamp(gt_abs, 0, max(orig_w, orig_h))
                gt_abs[:, 0] = torch.clamp(gt_abs[:, 0], 0, orig_w)
                gt_abs[:, 1] = torch.clamp(gt_abs[:, 1], 0, orig_h)
                gt_abs[:, 2] = torch.clamp(gt_abs[:, 2], 0, orig_w)
                gt_abs[:, 3] = torch.clamp(gt_abs[:, 3], 0, orig_h)

                preds = [
                    dict(
                        boxes=torch.from_numpy(boxes).to(device) if len(boxes) > 0 else torch.zeros((0,4), device=device),
                        scores=torch.from_numpy(scores).to(device) if len(scores) > 0 else torch.zeros(0, device=device),
                        labels=torch.from_numpy(labels).to(device) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64, device=device)
                    )
                ]
                gts = [
                    dict(
                        boxes=gt_abs.to(device),
                        labels=t['labels'].to(device)
                    )
                ]
                metric.update(preds, gts)

    results = metric.compute()
    # 返回 map50, map, mar_100
    return results['map_50'], results['map'], results['mar_100']