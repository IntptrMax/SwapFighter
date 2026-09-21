import os
import sys
import json
import math
import argparse

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from config import Config
from utils import letterbox


# ============================================================
# 日志辅助
# ============================================================
LINE = "─" * 64
LINE_EQ = "═" * 64


def _title(text):
    print(f"\n  [{text}]")


def _kv(key, value, indent=4, total_width=40):
    prefix = " " * indent
    label = f"{key} "
    pad_len = max(2, total_width - len(label))
    dots = " " + "." * pad_len + " "
    print(f"{prefix}{label}{dots} {value}")


# ============================================================
# 类别颜色
# ============================================================
CLASS_COLORS = [
    (0, 200, 0), (255, 100, 0), (0, 100, 255), (200, 200, 0),
    (200, 0, 200), (0, 200, 200), (100, 100, 255), (255, 0, 100),
    (100, 255, 100), (255, 200, 100),
]


def _get_class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]


# ============================================================
# 水平框 NMS
# ============================================================
def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# ============================================================
# 旋转框 NMS（ProbIoU）
# ============================================================
def _obb_xy8_to_cxcywh_angle_np(obboxes_8):
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
    return np.stack([cx, cy, w, h, angle], axis=1)


def _batch_probiou_np(gt_obb, pred_obb, eps=1e-6):
    gt_w = np.maximum(gt_obb[:, 2], eps); gt_h = np.maximum(gt_obb[:, 3], eps)
    gc = np.cos(gt_obb[:, 4]); gs = np.sin(gt_obb[:, 4])
    ga = gt_w ** 2 / 12 * gc ** 2 + gt_h ** 2 / 12 * gs ** 2
    gb = gt_w ** 2 / 12 * gs ** 2 + gt_h ** 2 / 12 * gc ** 2
    gcc = (gt_w ** 2 / 12 - gt_h ** 2 / 12) * gc * gs
    gab = np.maximum(ga * gb - gcc ** 2, eps)
    gcx = gt_obb[:, 0:1]; gcy = gt_obb[:, 1:2]

    pw = np.maximum(pred_obb[:, 2], eps); ph = np.maximum(pred_obb[:, 3], eps)
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


def cxcywh_angle_to_obb_np(cx, cy, w, h, angle):
    N = cx.shape[0]
    half_w = w / 2
    half_h = h / 2

    corners = np.zeros((N, 4, 2), dtype=np.float32)
    corners[:, 0, 0] = half_w; corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w; corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w; corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w; corners[:, 3, 1] = -half_h

    cos_a = np.cos(angle); sin_a = np.sin(angle)
    rot = np.stack([
        np.stack([cos_a, sin_a], axis=1),
        np.stack([-sin_a, cos_a], axis=1)
    ], axis=1)

    corners_rot = np.einsum('nij,nkj->nki', rot, corners)
    center = np.stack([cx, cy], axis=1)[:, None, :]
    corners_final = corners_rot + center
    return corners_final.reshape(N, 8).astype(np.float32)


# ============================================================
# detect 后处理
# ============================================================
def postprocess_detect(cls_probs, boxes, img_size, strides,
                       conf_threshold=0.5, nms_threshold=0.5,
                       scale=1.0, pad_left=0, pad_top=0,
                       orig_w=0, orig_h=0):
    cls_probs = cls_probs[0]
    box_norm = boxes[0]

    all_boxes, all_scores, all_labels = [], [], []
    idx = 0
    for stride in strides:
        h = img_size // stride
        w = img_size // stride
        points = h * w
        end = idx + points

        cls_slice = cls_probs[idx:end, :]
        box_slice = box_norm[idx:end, :]
        idx = end

        max_scores = np.max(cls_slice, axis=1)
        max_labels = np.argmax(cls_slice, axis=1)
        keep = max_scores > conf_threshold
        if not np.any(keep):
            continue

        bf = box_slice[keep]
        sf = max_scores[keep]
        lf = max_labels[keep]

        cx = bf[:, 0] * img_size
        cy = bf[:, 1] * img_size
        w_ = bf[:, 2] * img_size
        h_ = bf[:, 3] * img_size
        x1 = cx - w_ / 2; y1 = cy - h_ / 2
        x2 = cx + w_ / 2; y2 = cy + h_ / 2

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(sf)
        all_labels.append(lf)

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    boxes_xyxy = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if scale is not None and scale > 0:
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale
    else:
        boxes_xyxy[:, [0, 2]] *= orig_w / img_size
        boxes_xyxy[:, [1, 3]] *= orig_h / img_size

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    keep_indices = nms(boxes_xyxy, scores, nms_threshold)
    return boxes_xyxy[keep_indices], scores[keep_indices], labels[keep_indices]


# ============================================================
# OBB 后处理
# ============================================================
def postprocess_obb(cls_probs, boxes, angles, img_size, strides,
                    angle_format='sin_cos', angle_range='pi',
                    conf_threshold=0.5, nms_threshold=0.5,
                    scale=1.0, pad_left=0, pad_top=0,
                    orig_w=0, orig_h=0):
    cls_probs = cls_probs[0]
    box_norm = boxes[0]
    angle_all = angles[0]

    all_cxcywh, all_angle, all_scores, all_labels = [], [], [], []

    idx = 0
    for stride in strides:
        h = img_size // stride
        w = img_size // stride
        points = h * w
        end = idx + points

        cls_slice = cls_probs[idx:end, :]
        box_slice = box_norm[idx:end, :]
        angle_slice = angle_all[idx:end, :]
        idx = end

        max_scores = np.max(cls_slice, axis=1)
        max_labels = np.argmax(cls_slice, axis=1)
        keep = max_scores > conf_threshold
        if not np.any(keep):
            continue

        bf = box_slice[keep]
        af = angle_slice[keep]
        sf = max_scores[keep]
        lf = max_labels[keep]

        if angle_format == 'sin_cos':
            angle = np.arctan2(af[:, 0], af[:, 1])
        else:
            angle = af[:, 0]

        all_cxcywh.append(bf)
        all_angle.append(angle)
        all_scores.append(sf)
        all_labels.append(lf)

    if not all_cxcywh:
        return np.array([]), np.array([]), np.array([])

    cxcywh = np.concatenate(all_cxcywh, axis=0)
    angle = np.concatenate(all_angle, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    cx, cy, w_, h_ = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]

    # 强制 w >= h
    swap = w_ < h_
    if np.any(swap):
        w_new = np.where(swap, h_, w_)
        h_new = np.where(swap, w_, h_)
        w_, h_ = w_new, h_new
        angle[swap] = angle[swap] + math.pi / 2

    if angle_range == 'pi':
        angle = angle % math.pi
    else:
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        angle = np.clip(angle, -math.pi / 2, math.pi / 2)

    cx_pix = cx * img_size
    cy_pix = cy * img_size
    w_pix = w_ * img_size
    h_pix = h_ * img_size

    if scale is not None and scale > 0:
        cx_pix = (cx_pix - pad_left) / scale
        cy_pix = (cy_pix - pad_top) / scale
        w_pix = w_pix / scale
        h_pix = h_pix / scale
    else:
        cx_pix = cx_pix * orig_w / img_size
        cy_pix = cy_pix * orig_h / img_size
        w_pix = w_pix * orig_w / img_size
        h_pix = h_pix * orig_h / img_size

    obboxes = cxcywh_angle_to_obb_np(cx_pix, cy_pix, w_pix, h_pix, angle)

    keep_indices = rotated_nms(obboxes, scores, nms_threshold, labels)
    return obboxes[keep_indices], scores[keep_indices], labels[keep_indices]


# ============================================================
# 从 ONNX 读取 metadata
# ============================================================
def load_onnx_metadata(sess):
    meta = {}
    try:
        meta_props = sess.get_modelmeta().custom_metadata_map
        for k, v in meta_props.items():
            if v.startswith('[') or v.startswith('{'):
                try:
                    meta[k] = json.loads(v)
                    continue
                except json.JSONDecodeError:
                    pass
            if v in ('true', 'True'):
                meta[k] = True
                continue
            if v in ('false', 'False'):
                meta[k] = False
                continue
            try:
                if '.' in v or 'e' in v.lower():
                    meta[k] = float(v)
                else:
                    meta[k] = int(v)
                continue
            except ValueError:
                pass
            meta[k] = v
    except Exception as e:
        print(f"  [警告] 读取 ONNX metadata 失败: {e}")
    return meta


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter ONNX 推理')
    parser.add_argument('--model', default='model.onnx', help='ONNX 模型路径')
    parser.add_argument('--image', default='test.jpg', help='输入图片')
    parser.add_argument('--output', default='output.jpg', help='输出图片')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--class-names', nargs='+', default=None, help='类别名称')

    parser.add_argument('--mode', default=None, choices=['detect', 'obb'])
    parser.add_argument('--angle-format', default=None, choices=['sin_cos', 'direct'])
    parser.add_argument('--angle-range', default=None, choices=['pi', 'pi2'])
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--conf-threshold', type=float, default=None)
    parser.add_argument('--nms-threshold', type=float, default=None)
    parser.add_argument('--strides', type=int, nargs='+', default=None)

    parser.add_argument('--show', action='store_true', help='显示结果窗口')
    parser.add_argument('--top-k', type=int, default=20, help='结果表格显示前 K 个')
    return parser.parse_args()


# ============================================================
# 可视化
# ============================================================
def draw_obb(img, obbox, color, thickness=2):
    pts = obbox.reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_text(img, text, pos, font_scale, thickness, bg_color, text_color):
    x, y = pos
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(img, (x, y - text_h - 8), (x + text_w + 4, y + 4),
                  bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def _print_config(model_path, image_path, orig_w, orig_h, meta, conf_thr, nms_thr):
    print()
    print(LINE_EQ)
    print("  SWAP-Fighter ONNX Inference")
    print(LINE_EQ)

    _title("Input")
    _kv("Image", image_path)
    _kv("Image Size", f"{orig_w} x {orig_h}")

    _title("Model")
    _kv("ONNX Path", model_path)
    mode_str = ("OBB (rotated bbox)" if meta['mode'] == 'obb'
                else "Detect (axis-aligned bbox)")
    _kv("Mode", mode_str)
    _kv("Backbone", meta.get('backbone_name', 'N/A'))
    _kv("Image Size (Model)", meta['img_size'])
    _kv("Strides", str(meta['strides']))
    _kv("Task", f"cls={meta['num_classes']}, reg=4"
                + (f", angle=2 ({meta['angle_format']}, {meta['angle_range']})"
                   if meta['mode'] == 'obb' else ""))
    _kv("DFL", "disabled" if meta.get('reg_max', 0) == 0
              else f"reg_max={meta['reg_max']}")

    _title("Inference")
    _kv("Conf Threshold", f"{conf_thr:.3f}")
    _kv("NMS Threshold", f"{nms_thr:.3f}")
    _kv("Mean", str(meta.get('mean', [0, 0, 0])))
    _kv("Std", str(meta.get('std', [1, 1, 1])))

    print()
    print(LINE_EQ)


def _print_result(mode, boxes, obboxes, scores, labels, class_names, top_k=20):
    n_total = len(scores)
    print()
    print(LINE_EQ)
    print(f"  Result  ({n_total} detections)")
    print(LINE_EQ)

    if n_total == 0:
        print("  (no objects detected)")
        print()
        print(LINE_EQ)
        return

    if class_names:
        print("\n  Class Breakdown:")
        from collections import Counter
        counts = Counter(int(l) for l in labels)
        for cls_id in sorted(counts.keys()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            cnt = counts[cls_id]
            bar = "█" * min(int(cnt / max(counts.values()) * 20), 20)
            print(f"    {name:<20}  {cnt:>4d}  {bar}")

    order = np.argsort(-np.array(scores))[:top_k]
    print()
    if mode == 'detect':
        print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  {'Geometry (x1,y1,x2,y2)'}")
        print("  " + "─" * 76)
        for i, idx in enumerate(order, 1):
            name = class_names[int(labels[idx])] if class_names and int(labels[idx]) < len(class_names) else str(int(labels[idx]))
            box = boxes[idx]
            geom = f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
            print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom}")
    else:
        print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  {'Geometry (cx,cy,w,h,angle)'}")
        print("  " + "─" * 76)
        for i, idx in enumerate(order, 1):
            name = class_names[int(labels[idx])] if class_names and int(labels[idx]) < len(class_names) else str(int(labels[idx]))
            pts = obboxes[idx].reshape(4, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            v1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
            v2 = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))
            w = float(np.linalg.norm(v1))
            h = float(np.linalg.norm(v2))
            angle = float(np.degrees(np.arctan2(v1[1], v1[0])))
            geom = f"({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f}, {angle:+.1f}°)"
            print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom}")

    if n_total > top_k:
        print(f"  {'':>4}  ... ({n_total - top_k} more detections omitted)")

    print()
    print(LINE_EQ)


# ============================================================
# 主入口
# ============================================================
def main():
    args = parse_args()
    cfg = Config()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"ONNX 模型不存在: {args.model}")

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(args.model, providers=providers)

    # ---------- 从 ONNX 读取 metadata ----------
    onnx_meta = load_onnx_metadata(sess)

    # ---------- 参数优先级：CLI > ONNX meta > config ----------
    def get_param(name, cli_value, cfg_attr, default=None):
        if cli_value is not None:
            return cli_value
        if name in onnx_meta:
            return onnx_meta[name]
        if cfg_attr is not None and hasattr(cfg, cfg_attr):
            return getattr(cfg, cfg_attr)
        return default

    mode = get_param('mode', args.mode, 'mode', 'detect')
    angle_format = get_param('angle_format', args.angle_format, 'angle_format', 'sin_cos')
    angle_range = get_param('angle_range', args.angle_range, 'angle_range', 'pi')
    img_size = get_param('img_size', args.img_size, 'img_size', 640)
    reg_max = onnx_meta.get('reg_max', getattr(cfg, 'reg_max', 0))
    num_classes = onnx_meta.get('num_classes', cfg.num_classes)
    mean = onnx_meta.get('mean', getattr(cfg, 'mean', [0, 0, 0]))
    std = onnx_meta.get('std', getattr(cfg, 'std', [1, 1, 1]))
    strides = onnx_meta.get('strides', getattr(cfg, 'strides', None)) or [8, 16, 32]

    if args.strides is not None:
        strides = args.strides

    # 阈值
    if args.conf_threshold is not None:
        conf_thr = args.conf_threshold
    else:
        conf_thr = onnx_meta.get('conf_threshold', cfg.conf_threshold)

    if args.nms_threshold is not None:
        nms_thr = args.nms_threshold
    elif mode == 'obb':
        nms_thr = onnx_meta.get('obb_nms_threshold',
                                getattr(cfg, 'obb_nms_threshold', 0.5))
    else:
        nms_thr = onnx_meta.get('nms_threshold', cfg.nms_threshold)

    class_names = args.class_names or [f"Class_{i}" for i in range(num_classes)]

    # ---------- 读取图片 ----------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # ---------- 打印配置 ----------
    _print_config(
        model_path=args.model,
        image_path=args.image,
        orig_w=orig_w,
        orig_h=orig_h,
        meta={
            'mode': mode,
            'backbone_name': onnx_meta.get('backbone_name', 'N/A'),
            'img_size': img_size,
            'strides': strides,
            'num_classes': num_classes,
            'reg_max': reg_max,
            'angle_format': angle_format,
            'angle_range': angle_range,
            'mean': mean,
            'std': std,
        },
        conf_thr=conf_thr,
        nms_thr=nms_thr,
    )

    # ---------- 预处理 ----------
    img_padded, scale, (pad_left, pad_top), _ = letterbox(
        img_rgb, new_shape=(img_size, img_size), auto=False, stride=32
    )
    img_pil = Image.fromarray(img_padded)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0).numpy()

    # ---------- 推理 ----------
    onnx_input = {sess.get_inputs()[0].name: input_tensor}
    outputs = sess.run(None, onnx_input)

    # ---------- 后处理 ----------
    if mode == 'obb':
        cls_probs, boxes, angles = outputs[0], outputs[1], outputs[2]
        obboxes, scores, labels = postprocess_obb(
            cls_probs, boxes, angles, img_size, strides,
            angle_format=angle_format, angle_range=angle_range,
            conf_threshold=conf_thr, nms_threshold=nms_thr,
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            orig_w=orig_w, orig_h=orig_h,
        )

        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(orig_w, orig_h) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))
        text_thickness = max(1, int(round(font_scale * 1.5)))

        for ob, sc, lb in zip(obboxes, scores, labels):
            color = _get_class_color(lb)
            draw_obb(img_vis, ob, color, thickness)
            pts = ob.reshape(4, 2).astype(np.int32)
            name = class_names[int(lb)] if int(lb) < len(class_names) else str(int(lb))
            _draw_text(img_vis, f"{name}: {sc:.2f}", (pts[0][0], pts[0][1]),
                       font_scale, text_thickness, color, (255, 255, 255))

        cv2.imwrite(args.output, img_vis)
        _print_result('obb', None, obboxes, scores, labels, class_names, args.top_k)

    else:
        cls_probs, boxes = outputs[0], outputs[1]
        boxes_xyxy, scores, labels = postprocess_detect(
            cls_probs, boxes, img_size, strides,
            conf_threshold=conf_thr, nms_threshold=nms_thr,
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            orig_w=orig_w, orig_h=orig_h,
        )

        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(orig_w, orig_h) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))
        text_thickness = max(1, int(round(font_scale * 1.5)))

        for box, sc, lb in zip(boxes_xyxy, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = _get_class_color(lb)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
            name = class_names[int(lb)] if int(lb) < len(class_names) else str(int(lb))
            _draw_text(img_vis, f"{name}: {sc:.2f}", (x1, y1),
                       font_scale, text_thickness, color, (255, 255, 255))

        cv2.imwrite(args.output, img_vis)
        _print_result('detect', boxes_xyxy, None, scores, labels, class_names, args.top_k)

    print(f"\n  检测结果图像已保存至: {args.output}")
    print(f"  检测到 {len(scores)} 个目标\n")

    if args.show:
        cv2.imshow('Detection', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()