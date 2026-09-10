import argparse
import math
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from config import Config
from utils import letterbox


# -------------------- 水平框 NMS --------------------
def nms(boxes, scores, iou_threshold=0.5):
    """boxes: (N,4) xyxy"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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


# -------------------- 旋转框 NMS --------------------
def rotated_nms(obboxes, scores, iou_threshold=0.5):
    """obboxes: (N,8) 顶点坐标 (x1,y1,...,x4,y4)"""
    if len(obboxes) == 0:
        return np.array([], dtype=np.int64)

    try:
        from shapely.geometry import Polygon
    except ImportError:
        # 回退：用水平外接框 NMS
        xyxy = np.zeros((len(obboxes), 4), dtype=np.float32)
        for i, ob in enumerate(obboxes):
            pts = ob.reshape(4, 2)
            xyxy[i] = [pts[:, 0].min(), pts[:, 1].min(),
                       pts[:, 0].max(), pts[:, 1].max()]
        return nms(xyxy, scores, iou_threshold)

    def to_poly(ob):
        return Polygon(ob.reshape(4, 2))

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        poly_i = to_poly(obboxes[i])
        rest = order[1:]
        if rest.size == 0:
            break
        ious = []
        for j in rest:
            poly_j = to_poly(obboxes[j])
            if not poly_i.is_valid or not poly_j.is_valid or poly_i.area == 0 or poly_j.area == 0:
                ious.append(0.0)
                continue
            inter = poly_i.intersection(poly_j).area
            union = poly_i.area + poly_j.area - inter
            ious.append(inter / (union + 1e-6))
        ious = np.array(ious)
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# -------------------- 从 (cx,cy,w,h,angle) 生成 8 顶点 --------------------
def cxcywh_angle_to_obb_np(cx, cy, w, h, angle):
    """
    输入均可为 numpy 数组，返回 (N, 8) 顶点坐标（与 utils.cxcywh_angle_to_obb 一致）
    """
    N = cx.shape[0]
    half_w = w / 2
    half_h = h / 2

    corners = np.zeros((N, 4, 2), dtype=np.float32)
    corners[:, 0, 0] = half_w
    corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w
    corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w
    corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w
    corners[:, 3, 1] = -half_h

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # 旋转矩阵 [[cos, sin], [-sin, cos]]
    rot = np.stack([
        np.stack([cos_a, sin_a], axis=1),
        np.stack([-sin_a, cos_a], axis=1)
    ], axis=1)  # (N,2,2)

    corners_rot = np.einsum('nij,nkj->nki', rot, corners)  # (N,4,2)
    center = np.stack([cx, cy], axis=1)[:, None, :]        # (N,1,2)
    corners_final = corners_rot + center

    return corners_final.reshape(N, 8).astype(np.float32)


# -------------------- detect 后处理 --------------------
def postprocess_detect(cls_probs, boxes, img_size, strides,
                       conf_threshold=0.5, nms_threshold=0.5,
                       scale=1.0, pad_left=0, pad_top=0, orig_w=0, orig_h=0):
    """
    cls_probs: (1, N, C)  已 sigmoid
    boxes:     (1, N, 4)  归一化 (cx,cy,w,h)
    """
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

        max_scores = np.max(cls_slice, axis=1)
        max_labels = np.argmax(cls_slice, axis=1)
        keep = max_scores > conf_threshold
        idx = end
        if not np.any(keep):
            continue

        bf = box_slice[keep]
        sf = max_scores[keep]
        lf = max_labels[keep]

        # (cx,cy,w,h) -> xyxy
        x1 = bf[:, 0] - bf[:, 2] / 2
        y1 = bf[:, 1] - bf[:, 3] / 2
        x2 = bf[:, 0] + bf[:, 2] / 2
        y2 = bf[:, 1] + bf[:, 3] / 2

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        all_boxes.append(xyxy)
        all_scores.append(sf)
        all_labels.append(lf)

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    boxes_xyxy = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # NMS 前先换算回原图坐标，避免归一化带来的误差
    if scale is not None and scale > 0:
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] * img_size - pad_left) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] * img_size - pad_top) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] * img_size - pad_left) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] * img_size - pad_top) / scale
    else:
        boxes_xyxy[:, 0::2] *= orig_w
        boxes_xyxy[:, 1::2] *= orig_h

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    keep_indices = nms(boxes_xyxy, scores, nms_threshold)
    return boxes_xyxy[keep_indices], scores[keep_indices], labels[keep_indices]


# -------------------- OBB 后处理 --------------------
def postprocess_obb(cls_probs, boxes, angles, img_size, strides,
                    angle_format='sin_cos', angle_range='pi',
                    conf_threshold=0.5, nms_threshold=0.5,
                    scale=1.0, pad_left=0, pad_top=0, orig_w=0, orig_h=0):
    """
    cls_probs: (1, N, C)      已 sigmoid
    boxes:     (1, N, 4)      归一化 (cx,cy,w,h)
    angles:    (1, N, 2)      sin_cos: [sin, cos]；direct: [angle, 0]
    """
    cls_probs = cls_probs[0]
    box_norm = boxes[0]
    angle_all = angles[0]

    all_cxcywh = []
    all_angle = []
    all_scores = []
    all_labels = []

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
            sin_val = af[:, 0]
            cos_val = af[:, 1]
            angle = np.arctan2(sin_val, cos_val)
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

    # 强制 w >= h（与训练后处理一致）
    swap = w_ < h_
    if np.any(swap):
        w_[swap], h_[swap] = h_[swap], w_[swap].copy()
        angle[swap] = angle[swap] + math.pi / 2

    if angle_range == 'pi':
        angle = angle % math.pi
    else:
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        angle = np.clip(angle, -math.pi / 2, math.pi / 2)

    # 归一化 -> 像素（含逆 letterbox）
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

    # 旋转 NMS
    keep_indices = rotated_nms(obboxes, scores, nms_threshold)
    return obboxes[keep_indices], scores[keep_indices], labels[keep_indices]


# -------------------- 参数解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter ONNX 推理演示')
    parser.add_argument('--model', default='model.onnx', help='ONNX 模型路径')
    parser.add_argument('--image', default='test.jpg', help='输入图片')
    parser.add_argument('--output', default='output.jpg', help='输出图片')
    parser.add_argument('--mode', default=None, choices=['detect', 'obb'],
                        help='检测模式（覆盖 config）')
    parser.add_argument('--angle-format', default=None, choices=['sin_cos', 'direct'],
                        help='OBB 角度编码（覆盖 config）')
    parser.add_argument('--angle-range', default=None, choices=['pi', 'pi2'],
                        help='OBB 角度范围（覆盖 config）')
    parser.add_argument('--img-size', type=int, default=None, help='模型输入尺寸')
    parser.add_argument('--conf-threshold', type=float, default=None, help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=None, help='NMS 阈值')
    parser.add_argument('--strides', type=int, nargs='+', default=None, help='FPN 步长')
    parser.add_argument('--class-names', nargs='+', default=None, help='类别名称')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--show', action='store_true', help='显示结果')
    return parser.parse_args()


def draw_obb(img, obbox, color, thickness=2):
    pts = obbox.reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def main():
    args = parse_args()
    cfg = Config()

    mode = args.mode or cfg.mode
    angle_format = args.angle_format or cfg.angle_format
    angle_range = args.angle_range or cfg.angle_range

    img_size = args.img_size if args.img_size is not None else cfg.img_size
    if mode == 'obb':
        conf_thr = args.conf_threshold if args.conf_threshold is not None else cfg.conf_threshold
        nms_thr = args.nms_threshold if args.nms_threshold is not None else cfg.obb_nms_threshold
    else:
        conf_thr = args.conf_threshold if args.conf_threshold is not None else cfg.conf_threshold
        nms_thr = args.nms_threshold if args.nms_threshold is not None else cfg.nms_threshold

    if args.strides is not None:
        strides = args.strides
    elif getattr(cfg, 'strides', None) is not None:
        strides = cfg.strides
    else:
        strides = [8, 16, 32]

    num_classes = cfg.num_classes
    class_names = args.class_names or [f"Class_{i}" for i in range(num_classes)]
    mean = getattr(cfg, 'mean', [0.0, 0.0, 0.0])
    std = getattr(cfg, 'std', [1.0, 1.0, 1.0])

    print("=" * 60)
    print("SWAP-Fighter ONNX 推理演示")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"图片: {args.image}")
    print(f"模式: {mode}")
    if mode == 'obb':
        print(f"角度编码: {angle_format}, 角度范围: {angle_range}")
    print(f"输入尺寸: {img_size}")
    print(f"置信度阈值: {conf_thr}, NMS阈值: {nms_thr}")
    print(f"步长: {strides}")
    print(f"均值: {mean}, 标准差: {std}")
    print("=" * 60)

    # 加载 ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' \
        else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=providers)
    print("ONNX 模型加载成功，设备:", sess.get_providers())

    # ---------- 预处理 ----------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_padded, scale, (pad_left, pad_top), _ = letterbox(
        img_rgb, new_shape=(img_size, img_size), auto=False, stride=32
    )

    img_pil = Image.fromarray(img_padded)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    print(f"原始尺寸: {orig_w}x{orig_h}, scale={scale:.4f}, pad_left={pad_left}, pad_top={pad_top}")

    # ---------- 推理 ----------
    onnx_input = {sess.get_inputs()[0].name: input_tensor.numpy()}
    outputs = sess.run(None, onnx_input)

    if mode == 'obb':
        cls_probs, boxes, angles = outputs[0], outputs[1], outputs[2]
        print(f"推理完成，probs: {cls_probs.shape}, boxes: {boxes.shape}, angles: {angles.shape}")

        obboxes, scores, labels = postprocess_obb(
            cls_probs, boxes, angles, img_size, strides,
            angle_format=angle_format, angle_range=angle_range,
            conf_threshold=conf_thr, nms_threshold=nms_thr,
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            orig_w=orig_w, orig_h=orig_h,
        )

        print(f"检测框数量: {len(obboxes)}")
        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if len(obboxes) == 0:
            print("未检测到目标")
            cv2.putText(img_vis, "No detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for i, (ob, sc, lb) in enumerate(zip(obboxes, scores, labels)):
                name = class_names[int(lb)] if int(lb) < len(class_names) else f"Class_{int(lb)}"
                pts = ob.reshape(4, 2).astype(np.int32)
                print(f"  [{i}] {name}: pts={pts.tolist()} conf={sc:.4f}")
                draw_obb(img_vis, ob, (0, 255, 0), 2)
                cv2.putText(img_vis, f"{name}: {sc:.2f}", (int(pts[0][0]), int(pts[0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        cls_probs, boxes = outputs[0], outputs[1]
        print(f"推理完成，probs: {cls_probs.shape}, boxes: {boxes.shape}")

        boxes_xyxy, scores, labels = postprocess_detect(
            cls_probs, boxes, img_size, strides,
            conf_threshold=conf_thr, nms_threshold=nms_thr,
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            orig_w=orig_w, orig_h=orig_h,
        )

        print(f"检测框数量: {len(boxes_xyxy)}")
        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if len(boxes_xyxy) == 0:
            print("未检测到目标")
            cv2.putText(img_vis, "No detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for i, (box, sc, lb) in enumerate(zip(boxes_xyxy, scores, labels)):
                x1, y1, x2, y2 = map(int, box)
                name = class_names[int(lb)] if int(lb) < len(class_names) else f"Class_{int(lb)}"
                print(f"  [{i}] {name}: ({x1},{y1},{x2},{y2}) conf={sc:.4f}")
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_vis, f"{name}: {sc:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(args.output, img_vis)
    print(f"结果保存至: {args.output}")

    if args.show:
        cv2.imshow('Detection', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()