import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from config import Config
from utils import letterbox


def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制"""
    if len(boxes) == 0:
        return np.array([])
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
    return np.array(keep)


def postprocess_onnx(cls_probs, boxes, img_size, strides,
                     conf_threshold=0.8, nms_threshold=0.5,
                     scale=1.0, pad_left=0, pad_top=0, orig_w=0, orig_h=0):
    """
    cls_probs: (1, total_points, num_classes)  已经过 sigmoid，值为概率
    boxes:     (1, total_points, 4)            归一化 (cx, cy, w, h)
    """
    # cls_probs 已经是概率，直接使用
    cls_probs = cls_probs[0]   # (total_points, C)
    box_norm = boxes[0]        # (total_points, 4)

    all_boxes = []
    all_scores = []
    all_labels = []

    idx = 0
    for stride in strides:
        h = img_size // stride
        w = img_size // stride
        points = h * w
        end = idx + points

        cls_slice = cls_probs[idx:end, :]      # (points, C)
        box_slice = box_norm[idx:end, :]       # (points, 4)

        max_scores = np.max(cls_slice, axis=1)
        max_labels = np.argmax(cls_slice, axis=1)

        keep = max_scores > conf_threshold
        if not np.any(keep):
            idx = end
            continue

        box_filtered = box_slice[keep]
        scores_filtered = max_scores[keep]
        labels_filtered = max_labels[keep]

        # (cx,cy,w,h) -> (x1,y1,x2,y2) 归一化
        cx = box_filtered[:, 0]
        cy = box_filtered[:, 1]
        w_ = box_filtered[:, 2]
        h_ = box_filtered[:, 3]
        x1 = cx - w_/2
        y1 = cy - h_/2
        x2 = cx + w_/2
        y2 = cy + h_/2

        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        all_boxes.append(boxes_xyxy)
        all_scores.append(scores_filtered)
        all_labels.append(labels_filtered)

        idx = end

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    boxes_xyxy = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # NMS
    keep_indices = nms(boxes_xyxy, scores, nms_threshold)
    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # 逆变换：从填充后坐标转为原始图像坐标
    if len(boxes_xyxy) > 0:
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] * img_size - pad_left) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] * img_size - pad_top) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] * img_size - pad_left) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] * img_size - pad_top) / scale
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    return boxes_xyxy, scores, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.onnx', help='ONNX 模型路径')
    parser.add_argument('--image', default='test.jpg', help='输入图片')
    parser.add_argument('--output', default='output.jpg', help='输出图片')
    parser.add_argument('--img-size', type=int, default=None, help='模型输入尺寸')
    parser.add_argument('--conf-threshold', type=float, default=None, help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=None, help='NMS 阈值')
    parser.add_argument('--strides', type=int, nargs='+', default=None, help='FPN 步长')
    parser.add_argument('--class-names', nargs='+', default=None, help='类别名称')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--show', action='store_true', help='显示结果')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    img_size = args.img_size if args.img_size is not None else getattr(cfg, 'img_size', 224)
    conf_thr = args.conf_threshold if args.conf_threshold is not None else getattr(cfg, 'conf_threshold', 0.5)
    nms_thr = args.nms_threshold if args.nms_threshold is not None else getattr(cfg, 'nms_threshold', 0.5)

    if args.strides is not None:
        strides = args.strides
    elif hasattr(cfg, 'strides') and cfg.strides is not None:
        strides = cfg.strides
    else:
        strides = [8, 16, 32]

    num_classes = getattr(cfg, 'num_classes', 4)
    class_names = args.class_names or [f"Class_{i}" for i in range(num_classes)]

    # 训练时使用的均值和标准差（必须与导出时一致）
    mean = getattr(cfg, 'mean', [0.0, 0.0, 0.0])
    std = getattr(cfg, 'std', [1.0, 1.0, 1.0])

    print("="*60)
    print("SWAP-Fighter ONNX 推理演示")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"图片: {args.image}")
    print(f"输入尺寸: {img_size}")
    print(f"置信度阈值: {conf_thr}, NMS阈值: {nms_thr}")
    print(f"步长: {strides}")
    print(f"均值: {mean}, 标准差: {std}")
    print("="*60)

    # 加载 ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device=='cuda' else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=providers)
    print("ONNX 模型加载成功，设备:", sess.get_providers())

    # ---------- 预处理 (与训练一致) ----------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_padded, scale, (pad_left, pad_top), _ = letterbox(
        img_rgb,
        new_shape=(img_size, img_size),
        auto=False,
        stride=32
    )

    img_pil = Image.fromarray(img_padded)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0)  # (1,3,H,W)

    print(f"原始尺寸: {orig_w}x{orig_h}, scale={scale:.4f}, pad_left={pad_left}, pad_top={pad_top}")

    # ---------- 推理 ----------
    onnx_input = {sess.get_inputs()[0].name: input_tensor.numpy()}
    outputs = sess.run(None, onnx_input)
    cls_probs = outputs[0]   # (1, total_points, num_classes) 已为概率
    boxes = outputs[1]       # (1, total_points, 4) 归一化 (cx,cy,w,h)

    print(f"推理完成，cls形状: {cls_probs.shape}, box形状: {boxes.shape}")

    # ---------- 后处理 ----------
    boxes_xyxy, scores, labels = postprocess_onnx(
        cls_probs, boxes, img_size, strides,
        conf_thr, nms_thr,
        scale, pad_left, pad_top, orig_w, orig_h
    )

    print(f"检测框数量: {len(boxes_xyxy)}")
    if len(boxes_xyxy) == 0:
        print("未检测到目标")
        img_vis = img_rgb.copy()
        cv2.putText(img_vis, "No detection", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        for i, (box, score, label) in enumerate(zip(boxes_xyxy, scores, labels)):
            x1,y1,x2,y2 = map(int, box)
            name = class_names[int(label)] if int(label) < len(class_names) else f"Class_{int(label)}"
            print(f"  [{i}] {name}: ({x1},{y1},{x2},{y2}) conf={score:.4f}")
        # 绘制
        img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for box, score, label in zip(boxes_xyxy, scores, labels):
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(img_vis, (x1,y1), (x2,y2), (0,255,0), 2)
            name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
            cv2.putText(img_vis, f"{name}: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imwrite(args.output, img_vis)
    print(f"结果保存至: {args.output}")

    if args.show:
        cv2.imshow('Detection', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()