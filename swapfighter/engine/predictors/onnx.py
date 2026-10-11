import os
import math
import json
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

    def __init__(self, cfg_dict, weights=None):
        if not HAS_ORT:
            raise ImportError(
                "ONNX 推理需要 onnxruntime，请执行: pip install onnxruntime")
        if weights is None:
            raise ValueError("ONNXPredictor 需要 --weights 指向 .onnx 文件")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"ONNX 文件不存在: {weights}")

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

        onames = set(self._output_names)
        if {'num_dets', 'boxes', 'scores', 'classes'}.issubset(onames):
            self._output_kind = 'end2end'
        else:
            self._output_kind = 'raw'

        meta = {}
        try:
            props = self._session.get_modelmeta().custom_metadata_map or {}
            meta = {k: self._parse_meta_value(v) for k, v in props.items()}
        except Exception:
            pass
        self._onnx_meta = meta

        task = (cfg_dict.get('task')
                or meta.get('train_task')
                or 'detect')

        num_classes = (cfg_dict.get('num_classes')
                       or meta.get('train_num_classes')
                       or 1)

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
                      else meta.get('train_with_angle', task == 'obb'))
        strides = meta.get('model_strides') or [8, 16, 32]
        mean = meta.get('train_mean') or cfg_dict.get('mean') or [0, 0, 0]
        std = meta.get('train_std') or cfg_dict.get('std') or [1, 1, 1]

        proto_dim = (cfg_dict.get('proto_dim')
                     or meta.get('train_proto_dim')
                     or 32)
        proto_stride = (cfg_dict.get('proto_stride')
                        or meta.get('train_proto_stride')
                        or 4)
        num_keypoints = (cfg_dict.get('num_keypoints')
                         or meta.get('train_num_keypoints')
                         or 17)

        cfg_dict.setdefault('task', task)
        cfg_dict.setdefault('num_classes', num_classes)
        cfg_dict.setdefault('img_size', img_size)
        cfg_dict.setdefault('angle_format', angle_format)
        cfg_dict.setdefault('angle_range', angle_range)
        cfg_dict.setdefault('reg_max', reg_max)
        cfg_dict.setdefault('with_angle', with_angle)
        cfg_dict.setdefault('strides', list(strides))
        cfg_dict.setdefault('mean', list(mean))
        cfg_dict.setdefault('std', list(std))
        cfg_dict.setdefault('proto_dim', proto_dim)
        cfg_dict.setdefault('proto_stride', proto_stride)
        cfg_dict.setdefault('num_keypoints', num_keypoints)

        super().__init__(cfg_dict, weights)

        self._task = task
        self._num_classes = num_classes
        self._img_size = int(img_size)
        self._angle_format = angle_format
        self._angle_range = angle_range
        self._reg_max = int(reg_max or 0)
        self._with_angle = bool(with_angle)
        self._strides = list(strides)
        self._mean = list(mean)
        self._std = list(std)
        self._proto_dim = int(proto_dim)
        self._proto_stride = int(proto_stride)
        self._num_keypoints = int(num_keypoints)

        self._load_status = 'OK (session loaded)'

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

    def build_model(self):
        return self._session

    def run(self, source):
        if not hasattr(self, '_session'):
            self.build_model()

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

        if len(images) == 1:
            self._print_header_single(images[0])
        else:
            self._print_header_multi(source, len(images))

        results = []
        for img_path in images:
            r = self._predict_one(img_path)
            results.append(r)

            if len(images) == 1:
                self._print_result(r)

        if len(images) > 1:
            print()
            print(LINE_EQ)
            print(f"  Summary  ({len(images)} images)")
            print(LINE_EQ)
            for r in results:
                name = os.path.basename(r['image'])
                if self._task in ('multilabel', 'linear_probe'):
                    probs = r.get('probs')
                    n = int((probs >= r.get('threshold', 0.5)).sum()) \
                        if probs is not None else 0
                    print(f"    {name:<50s}  {n:>4d} positives")
                else:
                    n = len(r.get('scores', []))
                    print(f"    {name:<50s}  {n:>4d} detections")
            if self.output_image:
                print()
                print("  Output Files:")
                _kv("Image", self.output_image, indent=4)
            print()
            print(LINE_EQ)

        return results

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
                    else "Raw (Python postprocess)")
        _kv("ONNX Path", self.weights)
        _kv("Output Kind", kind_str)

        if self._task == 'obb':
            mode_str = "OBB (rotated bbox)"
        elif self._task == 'segment':
            mode_str = "Seg (instance segmentation)"
        elif self._task == 'keypoint':
            mode_str = "Keypoint (pose estimation)"
        elif self._task in ('multilabel', 'linear_probe'):
            mode_str = "Multi-label classification"
        else:
            mode_str = "Detect (axis-aligned bbox)"
        _kv("Mode", mode_str)

        _kv("Backbone", self._onnx_meta.get('train_backbone', 'N/A'))
        _kv("Image Size (Model)", self._img_size)
        if self._task not in ('multilabel', 'linear_probe'):
            _kv("Strides", str(self._strides))

        if self._task == 'obb':
            _kv("Task", f"cls={self._num_classes}, reg=4, angle=2 "
                        f"({self._angle_format}, {self._angle_range})")
        elif self._task == 'segment':
            _kv("Task", f"cls={self._num_classes}, reg=4, "
                        f"proto_dim={self._proto_dim}")
            _kv("Proto Dim / Stride",
                f"{self._proto_dim} / {self._proto_stride}")
        elif self._task == 'keypoint':
            _kv("Task", f"cls={self._num_classes}, reg=4, "
                        f"kpt={self._num_keypoints * 2} "
                        f"(K={self._num_keypoints})")
        elif self._task in ('multilabel', 'linear_probe'):
            _kv("Task", f"num_classes={self._num_classes}")

            output_kind = self._onnx_meta.get('output_kind', 'logits')
            kind_desc = ("probs (sigmoid embedded)"
                         if output_kind == 'probs' else "logits")
            _kv("Output Kind", kind_desc)
        else:
            _kv("Task", f"cls={self._num_classes}, reg=4")

        if self._task not in ('multilabel', 'linear_probe'):
            _kv("DFL", "disabled" if self._reg_max == 0
                      else f"reg_max={self._reg_max}")
        _kv("Mean", str(self._mean))
        _kv("Std", str(self._std))

    def _print_inference_block(self):
        _title("Inference")
        if self._task in ('multilabel', 'linear_probe'):
            thr = (getattr(self.cfg, 'eval_threshold', 0.5)
                   if getattr(self.cfg, 'eval_threshold', None) is not None
                   else 0.5)
            _kv("Threshold", f"{thr:.3f}")
            _kv("Top-K", getattr(self.cfg, 'top_k', 5))
            _kv("Postprocess", "sigmoid + topk / threshold")
            return

        conf = self.cfg.conf_threshold
        if self._task == 'obb':
            nms_thr = getattr(self.cfg, 'obb_nms_threshold',
                              self.cfg.nms_threshold)
        else:
            nms_thr = self.cfg.nms_threshold

        _kv("Conf Threshold", f"{conf:.3f}")
        _kv("NMS Threshold", f"{nms_thr:.3f}"
            + ("  (固化在 ONNX)" if self._output_kind == 'end2end' else ""))
        _kv("NMS Location",
            "ONNX (Native)" if self._output_kind == 'end2end'
            else "Python (numpy)")

    @torch.no_grad()
    def _predict_one(self, image_path):

        if self._task in ('multilabel', 'linear_probe'):
            return self._predict_one_multilabel(image_path)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_padded, scale, (pl, pt), _ = letterbox(
            img_rgb, new_shape=(self._img_size, self._img_size),
            auto=False, stride=32)

        x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std),
        ])(Image.fromarray(img_padded)).unsqueeze(0) \
            .numpy().astype(np.float32)

        outputs = self._session.run(None, {self._input_name: x})

        boxes = obboxes = scores = labels = masks = keypoints = None

        if self._output_kind == 'end2end':
            if self._task == 'segment':
                boxes, scores, labels, masks = \
                    self._postprocess_end2end_seg(
                        outputs, scale, pl, pt, orig_w, orig_h)
            elif self._task == 'keypoint':
                boxes, scores, labels, keypoints = \
                    self._postprocess_end2end_keypoint(
                        outputs, scale, pl, pt, orig_w, orig_h)
            elif self._task == 'obb':
                obboxes, scores, labels = self._postprocess_end2end_obb(
                    outputs, scale, pl, pt, orig_w, orig_h)
            else:
                boxes, scores, labels = self._postprocess_end2end_detect(
                    outputs, scale, pl, pt, orig_w, orig_h)
        else:
            if self._task == 'segment':
                boxes, scores, labels, masks = \
                    self._postprocess_raw_seg(
                        outputs, scale, pl, pt, orig_w, orig_h)
            elif self._task == 'keypoint':
                boxes, scores, labels, keypoints = \
                    self._postprocess_raw_keypoint(
                        outputs, scale, pl, pt, orig_w, orig_h)
            elif self._task == 'obb':
                obboxes, scores, labels = self._postprocess_raw_obb(
                    outputs, scale, pl, pt, orig_w, orig_h)
            else:
                boxes, scores, labels = self._postprocess_raw_detect(
                    outputs, scale, pl, pt, orig_w, orig_h)

        img_vis = self._draw(img_rgb, boxes, obboxes, scores, labels,
                              masks=masks, keypoints=keypoints)
        if self.output_image:
            cv2.imwrite(self.output_image, img_vis)

        return {
            'image': image_path,
            'orig_size': (orig_w, orig_h),
            'mode': self._task,
            'boxes': boxes,
            'obboxes': obboxes,
            'scores': scores,
            'labels': labels,
            'masks': masks,
            'keypoints': keypoints,
            'img_vis': img_vis,
        }

    def _predict_one_multilabel(self, image_path):
        img = Image.open(image_path).convert('RGB')

        interp = transforms.InterpolationMode.BICUBIC
        transform = transforms.Compose([
            transforms.Resize(int(self._img_size * 1.14),
                            interpolation=interp),
            transforms.CenterCrop(self._img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std),
        ])
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)

        outputs = self._session.run(None, {self._input_name: x})[0]
        output_kind = self._onnx_meta.get('output_kind', 'logits')

        if output_kind == 'probs':

            probs = outputs[0].astype(np.float32)
        else:

            logits = outputs[0]
            probs = 1.0 / (1.0 + np.exp(-logits))

        thr = getattr(self.cfg, 'eval_threshold', None)
        if thr is None:
            thr = self._onnx_meta.get('train_eval_threshold', 0.5)
        thr = float(thr)

        top_k = getattr(self.cfg, 'top_k', 5)
        class_names = self.class_names or self._onnx_meta.get(
            'class_names') or None

        order = np.argsort(-probs)
        topk = []
        for r, idx in enumerate(order[:top_k], 1):
            cid = int(idx)
            name = (class_names[cid]
                    if class_names and cid < len(class_names)
                    else f'cls_{cid}')
            topk.append({
                'rank': r,
                'class_id': cid,
                'class_name': name,
                'prob': float(probs[cid]),
            })

        positives = []
        for idx in np.where(probs >= thr)[0]:
            cid = int(idx)
            name = (class_names[cid]
                    if class_names and cid < len(class_names)
                    else f'cls_{cid}')
            positives.append({
                'class_id': cid,
                'class_name': name,
                'prob': float(probs[cid]),
            })
        positives.sort(key=lambda v: -v['prob'])

        return {
            'image': image_path,
            'probs': probs,
            'threshold': thr,
            'topk': topk,
            'positives': positives,
            'img_vis': None,
        }

    def _postprocess_raw_detect(self, outputs, scale, pad_l, pad_t,
                                 orig_w, orig_h):
        output0 = outputs[0]
        out = output0[0]
        boxes_n = out[:4, :]
        cls_scores = out[4:, :]

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > self.cfg.conf_threshold

        if not np.any(keep):
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        cx = boxes_n[0, keep] * self._img_size
        cy = boxes_n[1, keep] * self._img_size
        w_ = boxes_n[2, keep] * self._img_size
        h_ = boxes_n[3, keep] * self._img_size

        x1 = cx - w_ / 2; y1 = cy - h_ / 2
        x2 = cx + w_ / 2; y2 = cy + h_ / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = max_scores[keep].astype(np.float32)
        labels = max_labels[keep].astype(np.int64)

        boxes_xyxy = self._invert_letterbox_boxes(
            boxes_xyxy, scale, pad_l, pad_t, orig_w, orig_h)

        keep_idx = self._axis_aligned_nms(boxes_xyxy, scores, labels,
                                           self.cfg.nms_threshold)
        return boxes_xyxy[keep_idx], scores[keep_idx], labels[keep_idx]

    def _postprocess_raw_obb(self, outputs, scale, pad_l, pad_t,
                              orig_w, orig_h):
        output0 = outputs[0]
        out = output0[0]
        boxes_n = out[:4, :]
        angle_rad = out[4, :].astype(np.float32).copy()
        cls_scores = out[5:, :]

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > self.cfg.conf_threshold

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

        MAX_ALLOWED_SIDE = self._img_size * 0.8
        MIN_ALLOWED_SIDE = 2.0
        size_mask = (
            (w_ < MAX_ALLOWED_SIDE) & (h_ < MAX_ALLOWED_SIDE) &
            (w_ > MIN_ALLOWED_SIDE) & (h_ > MIN_ALLOWED_SIDE))
        cx = cx[size_mask]; cy = cy[size_mask]
        w_ = w_[size_mask]; h_ = h_[size_mask]
        a = a[size_mask]
        scores = scores[size_mask]; labels = labels[size_mask]

        if len(cx) == 0:
            return (np.zeros((0, 8), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        swap = w_ < h_
        if np.any(swap):
            w_new = np.where(swap, h_, w_)
            h_new = np.where(swap, w_, h_)
            w_, h_ = w_new, h_new
            a = np.where(swap, a + math.pi / 2, a)

        if self._angle_range == 'pi':
            a = a % math.pi
        else:
            a = np.arctan2(np.sin(a), np.cos(a))
            a = np.clip(a, -math.pi / 2, math.pi / 2)

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
        keep_idx = rotated_nms(obboxes, scores,
                                self.cfg.obb_nms_threshold, labels)
        return obboxes[keep_idx], scores[keep_idx], labels[keep_idx]

    def _postprocess_raw_seg(self, outputs, scale, pad_l, pad_t,
                              orig_w, orig_h):
        output0 = outputs[0]
        proto = outputs[1]

        out = output0[0]
        nc = self._num_classes
        k = self._proto_dim
        ph, pw = proto.shape[2], proto.shape[3]

        boxes_n = out[:4, :]
        cls_scores = out[4:4 + nc, :]
        coefs = out[4 + nc:4 + nc + k, :]

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > self.cfg.conf_threshold

        if not np.any(keep):
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    [])

        keep_idx_before_nms = np.where(keep)[0]

        cx = boxes_n[0, keep] * self._img_size
        cy = boxes_n[1, keep] * self._img_size
        w_ = boxes_n[2, keep] * self._img_size
        h_ = boxes_n[3, keep] * self._img_size
        x1 = cx - w_ / 2; y1 = cy - h_ / 2
        x2 = cx + w_ / 2; y2 = cy + h_ / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = max_scores[keep].astype(np.float32)
        labels = max_labels[keep].astype(np.int64)

        coefs_kept = coefs[:, keep_idx_before_nms].T

        boxes_xyxy = self._invert_letterbox_boxes(
            boxes_xyxy, scale, pad_l, pad_t, orig_w, orig_h)

        keep_idx = self._axis_aligned_nms(boxes_xyxy, scores, labels,
                                           self.cfg.nms_threshold)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        coefs_kept = coefs_kept[keep_idx]

        masks = self._generate_masks(
            coefs_kept, proto[0], boxes_xyxy, orig_w, orig_h)

        return boxes_xyxy, scores, labels, masks

    def _postprocess_raw_keypoint(self, outputs, scale, pad_l, pad_t,
                                   orig_w, orig_h):
        output0 = outputs[0]
        out = output0[0]
        nc = self._num_classes
        K = self._num_keypoints

        boxes_n = out[:4, :]
        cls_scores = out[4:4 + nc, :]
        kpt_norm = out[4 + nc:4 + nc + K * 2, :]

        max_scores = cls_scores.max(axis=0)
        max_labels = cls_scores.argmax(axis=0)
        keep = max_scores > self.cfg.conf_threshold

        if not np.any(keep):
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    [])

        keep_idx_before_nms = np.where(keep)[0]

        cx = boxes_n[0, keep] * self._img_size
        cy = boxes_n[1, keep] * self._img_size
        w_ = boxes_n[2, keep] * self._img_size
        h_ = boxes_n[3, keep] * self._img_size
        x1 = cx - w_ / 2; y1 = cy - h_ / 2
        x2 = cx + w_ / 2; y2 = cy + h_ / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = max_scores[keep].astype(np.float32)
        labels = max_labels[keep].astype(np.int64)

        kpt_kept = kpt_norm[:, keep_idx_before_nms].T

        boxes_xyxy = self._invert_letterbox_boxes(
            boxes_xyxy, scale, pad_l, pad_t, orig_w, orig_h)

        keep_idx = self._axis_aligned_nms(boxes_xyxy, scores, labels,
                                           self.cfg.nms_threshold)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        kpt_kept = kpt_kept[keep_idx]

        keypoints = []
        for i in range(len(kpt_kept)):
            kpt_xy = kpt_kept[i].reshape(K, 2) * self._img_size
            kpt_xy = self._invert_letterbox_points(
                kpt_xy, scale, pad_l, pad_t, orig_w, orig_h)
            kpt3 = np.concatenate(
                [kpt_xy, np.ones((K, 1), dtype=np.float32)], axis=1)
            keypoints.append(kpt3)

        return boxes_xyxy, scores, labels, keypoints

    def _postprocess_end2end_detect(self, outputs, scale, pad_l, pad_t,
                                     orig_w, orig_h):
        num_dets, boxes, scores, classes = outputs[:4]
        n = int(num_dets.reshape(-1)[0])
        if n <= 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        b = boxes[0, :n].astype(np.float32).copy()
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)
        b = self._invert_letterbox_boxes(
            b, scale, pad_l, pad_t, orig_w, orig_h)
        return b, s, c

    def _postprocess_end2end_obb(self, outputs, scale, pad_l, pad_t,
                                  orig_w, orig_h):
        num_dets, boxes, scores, classes = outputs[:4]
        n = int(num_dets.reshape(-1)[0])
        if n <= 0:
            return (np.zeros((0, 8), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        b = boxes[0, :n].astype(np.float32).copy()
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)

        for i in range(4):
            if scale is not None and scale > 0:
                b[:, i * 2] = (b[:, i * 2] - pad_l) / scale
                b[:, i * 2 + 1] = (b[:, i * 2 + 1] - pad_t) / scale
            else:
                b[:, i * 2] *= orig_w / self._img_size
                b[:, i * 2 + 1] *= orig_h / self._img_size
            b[:, i * 2] = np.clip(b[:, i * 2], 0, orig_w)
            b[:, i * 2 + 1] = np.clip(b[:, i * 2 + 1], 0, orig_h)

        return b, s, c

    def _postprocess_end2end_seg(self, outputs, scale, pad_l, pad_t,
                                  orig_w, orig_h):
        num_dets, boxes, scores, classes, mask_coefs, proto = outputs[:6]
        n = int(num_dets.reshape(-1)[0])
        if n <= 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    [])

        b = boxes[0, :n].astype(np.float32).copy()
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)
        coefs = mask_coefs[0, :n].astype(np.float32)

        b = self._invert_letterbox_boxes(
            b, scale, pad_l, pad_t, orig_w, orig_h)

        masks = self._generate_masks(
            coefs, proto[0], b, orig_w, orig_h)

        return b, s, c, masks

    def _postprocess_end2end_keypoint(self, outputs, scale, pad_l, pad_t,
                                       orig_w, orig_h):
        num_dets, boxes, scores, classes, kpts = outputs[:5]
        n = int(num_dets.reshape(-1)[0])
        K = self._num_keypoints

        if n <= 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    [])

        b = boxes[0, :n].astype(np.float32).copy()
        s = scores[0, :n].astype(np.float32)
        c = classes[0, :n].astype(np.int64)
        k = kpts[0, :n].astype(np.float32)

        b = self._invert_letterbox_boxes(
            b, scale, pad_l, pad_t, orig_w, orig_h)

        for i in range(n):
            k[i] = self._invert_letterbox_points(
                k[i], scale, pad_l, pad_t, orig_w, orig_h)

        keypoints = []
        for i in range(n):
            kpt3 = np.concatenate(
                [k[i], np.ones((K, 1), dtype=np.float32)], axis=1)
            keypoints.append(kpt3)

        return b, s, c, keypoints

    @staticmethod
    def _invert_letterbox_boxes(boxes_xyxy, scale, pad_l, pad_t,
                                 orig_w, orig_h):
        b = boxes_xyxy.copy()
        if scale is not None and scale > 0:
            b[:, [0, 2]] = (b[:, [0, 2]] - pad_l) / scale
            b[:, [1, 3]] = (b[:, [1, 3]] - pad_t) / scale
        else:
            b[:, [0, 2]] *= orig_w / (orig_w + 1e-6)
            b[:, [1, 3]] *= orig_h / (orig_h + 1e-6)
        b[:, 0] = np.clip(b[:, 0], 0, orig_w)
        b[:, 1] = np.clip(b[:, 1], 0, orig_h)
        b[:, 2] = np.clip(b[:, 2], 0, orig_w)
        b[:, 3] = np.clip(b[:, 3], 0, orig_h)
        return b

    @staticmethod
    def _invert_letterbox_points(points_xy, scale, pad_l, pad_t,
                                  orig_w, orig_h):
        p = points_xy.copy()
        if scale is not None and scale > 0:
            p[:, 0] = (p[:, 0] - pad_l) / scale
            p[:, 1] = (p[:, 1] - pad_t) / scale
        p[:, 0] = np.clip(p[:, 0], 0, orig_w)
        p[:, 1] = np.clip(p[:, 1], 0, orig_h)
        return p

    def _generate_masks(self, coefs, proto, boxes, orig_w, orig_h):
        N = coefs.shape[0]
        if N == 0:
            return []
        k, ph, pw = proto.shape

        proto_flat = proto.reshape(k, -1)
        masks_logits = coefs @ proto_flat
        masks_prob = 1.0 / (1.0 + np.exp(-masks_logits))
        masks_prob = masks_prob.reshape(N, ph, pw)

        masks_list = []
        for i in range(N):
            m = masks_prob[i]
            m_up = cv2.resize(m, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
            m_bin = (m_up > 0.5).astype(np.uint8)

            x1 = max(0, int(boxes[i][0]))
            y1 = max(0, int(boxes[i][1]))
            x2 = min(orig_w, int(boxes[i][2]))
            y2 = min(orig_h, int(boxes[i][3]))
            cropped = np.zeros_like(m_bin)
            if x2 > x1 and y2 > y1:
                cropped[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
            masks_list.append(cropped)

        return masks_list

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

    def _print_result(self, r):
        if self._task in ('multilabel', 'linear_probe'):
            return self._print_result_multilabel(r)

        boxes = r.get('boxes')
        obboxes = r.get('obboxes')
        scores = r.get('scores')
        labels = r.get('labels')
        masks = r.get('masks')
        keypoints = r.get('keypoints')

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

        top_k = getattr(self.cfg, 'top_k', 20)
        order = np.argsort(-np.array(scores))[:top_k]
        print()

        if self._task == 'obb':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (cx,cy,w,h,angle)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = self._name(labels[idx], class_names)
                pts = obboxes[idx].reshape(4, 2)
                cx = float(pts[:, 0].mean()); cy = float(pts[:, 1].mean())
                v1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
                v2 = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))
                w = float(np.linalg.norm(v1)); h = float(np.linalg.norm(v2))
                angle = float(np.degrees(np.arctan2(v1[1], v1[0])))
                geom = f"({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f}, {angle:+.1f}°)"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")
        elif self._task == 'keypoint':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (x1,y1,x2,y2)':<40}  {'Kpts':>6}")
            print("  " + "─" * 84)
            for i, idx in enumerate(order, 1):
                name = self._name(labels[idx], class_names)
                box = boxes[idx]
                geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                        f"{box[2]:.0f}, {box[3]:.0f})")
                kpt_str = "N/A"
                if keypoints is not None and idx < len(keypoints) \
                        and keypoints[idx] is not None:
                    kpt_str = f"{len(keypoints[idx]):>6d}"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  "
                      f"{geom:<40}  {kpt_str:>6}")
        elif self._task == 'segment':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (x1,y1,x2,y2)':<40}  {'Mask':>8}")
            print("  " + "─" * 88)
            for i, idx in enumerate(order, 1):
                name = self._name(labels[idx], class_names)
                box = boxes[idx]
                geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                        f"{box[2]:.0f}, {box[3]:.0f})")
                mask_str = "N/A"
                if masks is not None and idx < len(masks) \
                        and masks[idx] is not None:
                    mask_str = f"{int(masks[idx].sum()):>5d}px"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  "
                      f"{geom:<40}  {mask_str:>8}")
        else:
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (x1,y1,x2,y2)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = self._name(labels[idx], class_names)
                box = boxes[idx]
                geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                        f"{box[2]:.0f}, {box[3]:.0f})")
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")

        if n_total > top_k:
            print(f"  {'':>4}  ... ({n_total - top_k} more detections omitted)")

        self._print_output_files()
        print()
        print(LINE_EQ)

    def _print_result_multilabel(self, r):
        probs = r.get('probs')
        name = os.path.basename(r['image'])
        thr = r.get('threshold', 0.5)
        n_pos = int((probs >= thr).sum())
        top_k = getattr(self.cfg, 'top_k', 5)

        print()
        print(LINE_EQ)
        print(f"  {name}")
        print(LINE_EQ)

        print(f"\n  预测正类数 (score >= {thr:.2f}): {n_pos}")

        print(f"\n  Top-{top_k}:")
        print(f"  {'#':>4}  {'Class':<24}  {'Score':>9}  {'Hit':^4}")
        print("  " + "─" * 48)
        for item in r['topk']:
            mark = "✓" if item['prob'] >= thr else " "
            print(f"  {item['rank']:>4}  {item['class_name']:<24}  "
                  f"{item['prob']:>9.4f}  {mark:^4}")

        if self.output_json:
            self._save_multilabel_json(r)

        print()
        print(LINE_EQ)

    def _save_multilabel_json(self, r):
        try:
            payload = {
                'image': os.path.basename(r['image']),
                'image_path': os.path.abspath(r['image']),
                'threshold': r['threshold'],
                'topk': r['topk'],
                'positives': r['positives'],
            }
            with open(self.output_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  JSON 保存失败: {e}")

    def _print_output_files(self):
        print()
        print("  Output Files:")
        if self.output_image:
            _kv("Image", self.output_image, indent=4)

    def _get_class_names(self):
        if self.class_names:
            return list(self.class_names)
        meta_names = self._onnx_meta.get('class_names')
        if meta_names and isinstance(meta_names, (list, tuple)):
            return list(meta_names)
        if self._num_classes is None:
            return []
        return [f'cls_{i}' for i in range(self._num_classes)]

    @staticmethod
    def _name(label, class_names):
        i = int(label)
        if class_names and i < len(class_names):
            return class_names[i]
        return str(i)

    def _draw(self, img_rgb, boxes, obboxes, scores, labels,
              masks=None, keypoints=None):
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(vis.shape[:2]) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        class_names = self._get_class_names()

        def _name(lb):
            return (class_names[int(lb)] if int(lb) < len(class_names)
                    else str(int(lb)))

        if masks is not None and len(masks) > 0:
            overlay = vis.copy()
            for m, l in zip(masks, labels):
                color = _get_class_color(l)
                overlay[m > 0] = color
            vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

        if keypoints is not None and len(keypoints) > 0:
            r = max(4, int(round(6 * min(vis.shape[:2]) / 800.0)))
            for kpt in keypoints:
                if kpt is None:
                    continue
                for k in range(len(kpt)):
                    if kpt[k, 2] > 0.3:
                        cx, cy = int(kpt[k, 0]), int(kpt[k, 1])
                        cv2.circle(vis, (cx, cy), r, (0, 0, 255), -1)
                        cv2.circle(vis, (cx, cy), r, (255, 255, 255), 2)

        if boxes is not None:
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, b)
                color = _get_class_color(l)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(vis, f'{_name(l)}: {s:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)
        elif obboxes is not None:
            for ob, s, l in zip(obboxes, scores, labels):
                color = _get_class_color(l)
                pts = ob.reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, color, thickness)
                cv2.putText(vis, f'{_name(l)}: {s:.2f}',
                            (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)

        return vis