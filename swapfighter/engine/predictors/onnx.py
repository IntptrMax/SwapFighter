import os
import json
import math
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms as tv_nms

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from ..base_predictor import BasePredictor
from ...utils.box_ops import (
    letterbox, cxcywh_angle_to_obb_np,
)
from ...utils.postprocess import rotated_nms
from ...utils.logger import LINE, LINE_EQ, _title, _kv


CLASS_COLORS = [
    (0, 200, 0), (255, 100, 0), (0, 100, 255), (200, 200, 0),
    (200, 0, 200), (0, 200, 200), (100, 100, 255), (255, 0, 100),
    (100, 255, 100), (255, 200, 100),
]

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def _get_class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]


class ONNXPredictor(BasePredictor):
    """ONNX 推理器。"""

    # ==================================================================
    # 初始化：加载 session + 解析 metadata
    # ==================================================================
    def __init__(self, cfg_dict, weights=None):
        if not HAS_ORT:
            raise ImportError(
                "ONNX 推理需要 onnxruntime，请执行: pip install onnxruntime")
        if weights is None:
            raise ValueError("ONNXPredictor 需要 --weights 指向 .onnx 文件")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"ONNX 文件不存在: {weights}")

        # ---------- 1. 加载 session ----------
        device_pref = cfg_dict.get('device', 'cuda')
        if device_pref == 'cuda' and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        try:
            self._session = ort.InferenceSession(weights, providers=providers)
        except Exception as e:
            raise RuntimeError(f"加载 ONNX 失败: {e}")

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # ---------- 2. 判断 output_kind ----------
        onames = set(self._output_names)
        if {'num_dets', 'boxes', 'scores', 'classes'}.issubset(onames):
            self._output_kind = 'end2end'
        else:
            self._output_kind = 'raw'

        # ---------- 3. 解析 metadata ----------
        meta = {}
        try:
            props = self._session.get_modelmeta().custom_metadata_map or {}
            meta = {k: self._parse_meta_value(v) for k, v in props.items()}
        except Exception:
            pass
        self._onnx_meta = meta

        # ---------- 4. 从 metadata 提取模型信息（用户传的优先） ----------
        mode = (cfg_dict.get('mode')
                or meta.get('train_mode')
                or 'detect')

        num_classes = (cfg_dict.get('num_classes')
                       or meta.get('train_num_classes'))

        # img_size 兜底：从 ONNX 输入 shape 读
        in_shape = self._session.get_inputs()[0].shape
        onnx_img_size = None
        if len(in_shape) == 4 and isinstance(in_shape[2], int):
            onnx_img_size = int(in_shape[2])

        img_size = (cfg_dict.get('img_size')
                    or meta.get('train_img_size')
                    or onnx_img_size
                    or 640)

        angle_format = (cfg_dict.get('angle_format')
                        or meta.get('train_angle_format')
                        or 'sin_cos')
        angle_range = (cfg_dict.get('angle_range')
                       or meta.get('train_angle_range')
                       or 'pi2')
        reg_max = (cfg_dict.get('reg_max')
                   if cfg_dict.get('reg_max') is not None
                   else meta.get('train_reg_max', 0))
        with_angle = (cfg_dict.get('with_angle')
                      if cfg_dict.get('with_angle') is not None
                      else meta.get('train_with_angle', mode == 'obb'))
        strides = meta.get('model_strides') or [8, 16, 32]
        mean = meta.get('train_mean') or cfg_dict.get('mean') or [0, 0, 0]
        std = meta.get('train_std') or cfg_dict.get('std') or [1, 1, 1]

        # ---------- 5. 把信息写回 cfg_dict，供 Config.from_kwargs 使用 ----------
        cfg_dict.setdefault('task', mode)
        cfg_dict.setdefault('mode', mode)
        if num_classes is not None:
            cfg_dict.setdefault('num_classes', num_classes)
        cfg_dict.setdefault('img_size', img_size)
        cfg_dict.setdefault('angle_format', angle_format)
        cfg_dict.setdefault('angle_range', angle_range)
        cfg_dict.setdefault('reg_max', reg_max)
        cfg_dict.setdefault('with_angle', with_angle)
        cfg_dict.setdefault('strides', list(strides))
        cfg_dict.setdefault('mean', list(mean))
        cfg_dict.setdefault('std', list(std))

        # ---------- 6. 调父类构造 ----------
        super().__init__(cfg_dict, weights)

        # ---------- 7. 保存供后续使用 ----------
        self._mode = mode
        self._num_classes = num_classes
        self._img_size = int(img_size)
        self._angle_format = angle_format
        self._angle_range = angle_range
        self._reg_max = int(reg_max or 0)
        self._with_angle = bool(with_angle)
        self._strides = list(strides)
        self._mean = list(mean)
        self._std = list(std)

        # 用于打印 header
        self._load_status = 'OK (session loaded)'

    # ==================================================================
    # metadata 值解析：字符串 → int / float / bool / list / dict
    # ==================================================================
    @staticmethod
    def _parse_meta_value(v):
        s = str(v)
        if s.startswith('[') or s.startswith('{'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        low = s.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    # ==================================================================
    # build_model：ONNX 不走 PyTorch 模型
    # ==================================================================
    def build_model(self):
        # session 已在 __init__ 里加载，这里直接返回
        return self._session

    # ==================================================================
    # run：主入口
    # ==================================================================
    def run(self, source):
        if not hasattr(self, '_session'):
            self.build_model()

        # ---------- 收集图片 ----------
        if os.path.isdir(source):
            images = sorted(
                os.path.join(source, f) for f in os.listdir(source)
                if f.lower().endswith(_IMG_EXTS)
            )
            if not images:
                raise FileNotFoundError(f"目录下没有图片: {source}")
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"图片不存在: {source}")
            images = [source]

        # ---------- Header ----------
        if len(images) == 1:
            self._print_header_single(images[0])
        else:
            self._print_header_multi(source, len(images))

        # ---------- 逐图推理 ----------
        results = []
        for img_path in images:
            r = self._predict_one(img_path)
            results.append(r)

            if len(images) == 1:
                self._print_result(r)

        # ---------- 多图汇总 ----------
        if len(images) > 1:
            print()
            print(LINE_EQ)
            print(f"  Summary  ({len(images)} images)")
            print(LINE_EQ)
            for r in results:
                name = os.path.basename(r['image'])
                n = len(r.get('scores', []))
                print(f"    {name:<50s}  {n:>4d} detections")
            if self.output_image:
                print()
                print("  Output Files:")
                _kv("Image", self.output_image, indent=4)
            print()
            print(LINE_EQ)

        return results

    # ==================================================================
    # Header 打印
    # ==================================================================
    def _print_header_single(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        print()
        print(LINE_EQ)
        print("  SWAP-Fighter ONNX Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Image", image_path)
        _kv("Image Size", f"{orig_w} x {orig_h}")
        _kv("Device", str(self.device))

        self._print_model_block()
        self._print_inference_block()
        print()
        print(LINE_EQ)

    def _print_header_multi(self, source, n_images):
        print()
        print(LINE_EQ)
        print("  SWAP-Fighter ONNX Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Source", source)
        _kv("Images", n_images)
        _kv("Device", str(self.device))

        self._print_model_block()
        self._print_inference_block()
        print()
        print(LINE_EQ)

    def _print_model_block(self):
        _title("Model")
        kind_str = ("End2End (with NMS)"
                    if self._output_kind == 'end2end'
                    else "Raw (YOLOv8-style single tensor)")
        _kv("ONNX Path", self.weights)
        _kv("Output Kind", kind_str)
        mode_str = ("OBB (rotated bbox)" if self._mode == 'obb'
                    else "Detect (axis-aligned bbox)")
        _kv("Mode", mode_str)
        _kv("Backbone", self._onnx_meta.get('train_backbone', 'N/A'))
        _kv("Image Size (Model)", self._img_size)
        _kv("Strides", str(self._strides))
        if self._mode == 'obb':
            _kv("Task", f"cls={self._num_classes}, reg=4, angle=2 "
                        f"({self._angle_format}, {self._angle_range})")
        else:
            _kv("Task", f"cls={self._num_classes}, reg=4")
        _kv("DFL", "disabled" if self._reg_max == 0
                  else f"reg_max={self._reg_max}")
        _kv("Mean", str(self._mean))
        _kv("Std", str(self._std))

    def _print_inference_block(self):
        conf = self.cfg.conf_threshold
        if self._mode == 'obb':
            nms_thr = getattr(self.cfg, 'obb_nms_threshold',
                              self.cfg.nms_threshold)
        else:
            nms_thr = self.cfg.nms_threshold

        _title("Inference")
        _kv("Conf Threshold", f"{conf:.3f}")
        _kv("NMS Threshold", f"{nms_thr:.3f}"
            + ("  (已固化)" if self._output_kind == 'end2end' else ""))
        _kv("NMS Location",
            "ONNX (Native)" if self._output_kind == 'end2end'
            else "Python (numpy)")

    # ==================================================================
    # 单图推理
    # ==================================================================
    @torch.no_grad()
    def _predict_one(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # ---------- 预处理 ----------
        img_padded, scale, (pl, pt), _ = letterbox(
            img_rgb, new_shape=(self._img_size, self._img_size),
            auto=False, stride=32)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std),
        ])
        x = transform(Image.fromarray(img_padded)).unsqueeze(0) \
            .numpy().astype(np.float32)

        # ---------- 推理 ----------
        outputs = self._session.run(None, {self._input_name: x})

        # ---------- 后处理 ----------
        if self._output_kind == 'end2end':
            num_dets, boxes, scores, classes = outputs
            if self._mode == 'obb':
                obboxes, scores, labels = self._postprocess_end2end_obb(
                    num_dets, boxes, scores, classes,
                    scale, pl, pt, orig_w, orig_h)
                boxes = None
            else:
                boxes, scores, labels = self._postprocess_end2end_detect(
                    num_dets, boxes, scores, classes,
                    scale, pl, pt, orig_w, orig_h)
                obboxes = None
        else:
            output0 = outputs[0]
            if self._mode == 'obb':
                obboxes, scores, labels = self._postprocess_raw_obb(
                    output0, self.cfg.conf_threshold,
                    getattr(self.cfg, 'obb_nms_threshold',
                            self.cfg.nms_threshold),
                    scale, pl, pt, orig_w, orig_h)
                boxes = None
            else:
                boxes, scores, labels = self._postprocess_raw_detect(
                    output0, self.cfg.conf_threshold,
                    self.cfg.nms_threshold,
                    scale, pl, pt, orig_w, orig_h)
                obboxes = None

        # ---------- 可视化 ----------
        img_vis = self._draw(img_rgb, boxes, obboxes, scores, labels)
        if self.output_image:
            cv2.imwrite(self.output_image, img_vis)

        return {
            'image': image_path,
            'orig_size': (orig_w, orig_h),
            'mode': self._mode,
            'boxes': boxes,
            'obboxes': obboxes,
            'scores': scores,
            'labels': labels,
            'img_vis': img_vis,
        }

    # ==================================================================
    # 后处理：raw 模式
    # ==================================================================
    def _postprocess_raw_detect(self, output0, conf_thr, nms_thr,
                                 scale, pad_l, pad_t, orig_w, orig_h):
        """output0: (1, 4+nc, P)，坐标已归一化到 img_size。"""
        out = output0[0]                      # (4+nc, P)
        boxes_n = out[:4, :]                  # cx, cy, w, h (归一化)
        cls_scores = out[4:, :]               # (nc, P)

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > conf_thr

        if not np.any(keep):
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        cx = boxes_n[0, keep] * self._img_size
        cy = boxes_n[1, keep] * self._img_size
        w_ = boxes_n[2, keep] * self._img_size
        h_ = boxes_n[3, keep] * self._img_size

        x1 = cx - w_ / 2
        y1 = cy - h_ / 2
        x2 = cx + w_ / 2
        y2 = cy + h_ / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = max_scores[keep].astype(np.float32)
        labels = max_labels[keep].astype(np.int64)

        # 逆 letterbox
        if scale is not None and scale > 0:
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_l) / scale
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_t) / scale
        else:
            boxes_xyxy[:, [0, 2]] *= orig_w / self._img_size
            boxes_xyxy[:, [1, 3]] *= orig_h / self._img_size

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        keep_idx = self._axis_aligned_nms(boxes_xyxy, scores, labels, nms_thr)
        return boxes_xyxy[keep_idx], scores[keep_idx], labels[keep_idx]

    def _postprocess_raw_obb(self, output0, conf_thr, nms_thr,
                              scale, pad_l, pad_t, orig_w, orig_h):
        """output0: (1, 5+nc, P)，[cx, cy, w, h, angle(rad), cls...] 归一化。"""
        out = output0[0]                      # (5+nc, P)
        boxes_n = out[:4, :]
        angle_rad = out[4, :].astype(np.float32).copy()
        cls_scores = out[5:, :]

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > conf_thr

        if not np.any(keep):
            return (np.zeros((0, 8), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        cx = boxes_n[0, keep] * self._img_size
        cy = boxes_n[1, keep] * self._img_size
        w_ = boxes_n[2, keep] * self._img_size
        h_ = boxes_n[3, keep] * self._img_size
        a = angle_rad[keep]
        scores = max_scores[keep].astype(np.float32)
        labels = max_labels[keep].astype(np.int64)

        # 强制 w >= h
        swap = w_ < h_
        if np.any(swap):
            w_new = np.where(swap, h_, w_)
            h_new = np.where(swap, w_, h_)
            w_, h_ = w_new, h_new
            a = np.where(swap, a + math.pi / 2, a)

        # 角度归一化
        if self._angle_range == 'pi':
            a = a % math.pi
        else:
            a = np.arctan2(np.sin(a), np.cos(a))
            a = np.clip(a, -math.pi / 2, math.pi / 2)

        # 逆 letterbox
        if scale is not None and scale > 0:
            cx = (cx - pad_l) / scale
            cy = (cy - pad_t) / scale
            w_ = w_ / scale
            h_ = h_ / scale
        else:
            cx = cx * orig_w / self._img_size
            cy = cy * orig_h / self._img_size
            w_ = w_ * orig_w / self._img_size
            h_ = h_ * orig_h / self._img_size

        obboxes = cxcywh_angle_to_obb_np(cx, cy, w_, h_, a)
        keep_idx = rotated_nms(obboxes, scores, nms_thr, labels)
        return obboxes[keep_idx], scores[keep_idx], labels[keep_idx]

    # ==================================================================
    # 后处理：end2end 模式
    # ==================================================================
    def _postprocess_end2end_detect(self, num_dets, boxes, scores, classes,
                                     scale, pad_l, pad_t, orig_w, orig_h):
        n = int(num_dets.reshape(-1)[0])
        if n <= 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        b = boxes[0, :n].astype(np.float32).copy()
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)

        if scale is not None and scale > 0:
            b[:, [0, 2]] = (b[:, [0, 2]] - pad_l) / scale
            b[:, [1, 3]] = (b[:, [1, 3]] - pad_t) / scale
        else:
            b[:, [0, 2]] *= orig_w / self._img_size
            b[:, [1, 3]] *= orig_h / self._img_size

        b[:, 0] = np.clip(b[:, 0], 0, orig_w)
        b[:, 1] = np.clip(b[:, 1], 0, orig_h)
        b[:, 2] = np.clip(b[:, 2], 0, orig_w)
        b[:, 3] = np.clip(b[:, 3], 0, orig_h)

        return b, s, c

    def _postprocess_end2end_obb(self, num_dets, boxes, scores, classes,
                                  scale, pad_l, pad_t, orig_w, orig_h):
        n = int(num_dets.reshape(-1)[0])
        if n <= 0:
            return (np.zeros((0, 8), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        b = boxes[0, :n].astype(np.float32).copy()    # (n, 8)
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)

        if scale is not None and scale > 0:
            for i in range(4):
                b[:, i * 2] = (b[:, i * 2] - pad_l) / scale
                b[:, i * 2 + 1] = (b[:, i * 2 + 1] - pad_t) / scale
        else:
            for i in range(4):
                b[:, i * 2] *= orig_w / self._img_size
                b[:, i * 2 + 1] *= orig_h / self._img_size

        for i in range(4):
            b[:, i * 2] = np.clip(b[:, i * 2], 0, orig_w)
            b[:, i * 2 + 1] = np.clip(b[:, i * 2 + 1], 0, orig_h)

        return b, s, c

    # ==================================================================
    # 轴对齐 NMS（按类别分组）
    # ==================================================================
    @staticmethod
    def _axis_aligned_nms(boxes, scores, labels, iou_threshold=0.5):
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        keep_all = []
        for cls_id in np.unique(labels):
            mask = labels == cls_id
            cls_boxes = torch.from_numpy(boxes[mask]).float()
            cls_scores = torch.from_numpy(scores[mask]).float()
            cls_keep = tv_nms(cls_boxes, cls_scores, iou_threshold).numpy()
            global_idx = np.where(mask)[0][cls_keep]
            keep_all.extend(global_idx.tolist())

        keep_all = np.array(keep_all, dtype=np.int64)
        order = np.argsort(-scores[keep_all])
        return keep_all[order]

    # ==================================================================
    # Result 打印
    # ==================================================================
    def _print_result(self, r):
        boxes = r.get('boxes')
        obboxes = r.get('obboxes')
        scores = r.get('scores')
        labels = r.get('labels')

        n_total = 0 if scores is None else len(scores)

        print()
        print(LINE_EQ)
        print(f"  Result  ({n_total} detections)")
        print(LINE_EQ)

        if n_total == 0:
            print("  (no objects detected)")
            self._print_output_files()
            print()
            print(LINE_EQ)
            return

        # ---------- 类别分布 ----------
        from collections import Counter
        class_names = self._get_class_names()
        print("\n  Class Breakdown:")
        counts = Counter(int(l) for l in labels)
        max_cnt = max(counts.values())
        for cls_id in sorted(counts.keys()):
            name = (class_names[cls_id] if cls_id < len(class_names)
                    else f"cls_{cls_id}")
            cnt = counts[cls_id]
            bar = "█" * min(int(cnt / max_cnt * 20), 20)
            print(f"    {name:<20}  {cnt:>4d}  {bar}")

        # ---------- 检测详情 ----------
        top_k = getattr(self.cfg, 'top_k', 20)
        order = np.argsort(-np.array(scores))[:top_k]
        print()

        if self._mode == 'detect':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (x1,y1,x2,y2)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = (class_names[int(labels[idx])]
                        if int(labels[idx]) < len(class_names)
                        else str(int(labels[idx])))
                box = boxes[idx]
                geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                        f"{box[2]:.0f}, {box[3]:.0f})")
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")
        else:
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (cx,cy,w,h,angle)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = (class_names[int(labels[idx])]
                        if int(labels[idx]) < len(class_names)
                        else str(int(labels[idx])))
                pts = obboxes[idx].reshape(4, 2)
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                v1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
                v2 = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))
                w = float(np.linalg.norm(v1))
                h = float(np.linalg.norm(v2))
                angle = float(np.degrees(np.arctan2(v1[1], v1[0])))
                geom = f"({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f}, {angle:+.1f}°)"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")

        if n_total > top_k:
            print(f"  {'':>4}  ... ({n_total - top_k} more detections omitted)")

        self._print_output_files()
        print()
        print(LINE_EQ)

    def _print_output_files(self):
        print()
        print("  Output Files:")
        if self.output_image:
            _kv("Image", self.output_image, indent=4)

    def _get_class_names(self):
        if self.class_names:
            return list(self.class_names)
        if self._num_classes is None:
            return []
        return [f'cls_{i}' for i in range(self._num_classes)]

    # ==================================================================
    # 可视化
    # ==================================================================
    def _draw(self, img_rgb, boxes, obboxes, scores, labels):
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(vis.shape[:2]) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        class_names = self._get_class_names()

        def _name(lb):
            return (class_names[int(lb)] if int(lb) < len(class_names)
                    else str(int(lb)))

        if boxes is not None:
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, b)
                color = _get_class_color(l)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(vis, f'{_name(l)}: {s:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)
        else:
            for ob, s, l in zip(obboxes, scores, labels):
                color = _get_class_color(l)
                pts = ob.reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, color, thickness)
                cv2.putText(vis, f'{_name(l)}: {s:.2f}',
                            (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)
        return vis