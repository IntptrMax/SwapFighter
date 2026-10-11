import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from ..base_predictor import BasePredictor
from ...utils.box_ops import letterbox
from ...utils.postprocess import post_process, post_process_obb
from ...utils.logger import LINE, LINE_EQ, _title, _kv, suppress_stdout

CLASS_COLORS = [
    (0, 200, 0), (255, 100, 0), (0, 100, 255), (200, 200, 0),
    (200, 0, 200), (0, 200, 200), (100, 100, 255), (255, 0, 100),
    (100, 255, 100), (255, 200, 100),
]

def _get_class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]

def _bool_en(b):
    return "yes" if b else "no"

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

class DetectPredictor(BasePredictor):

    def __init__(self, cfg_dict, weights=None):
        super().__init__(cfg_dict, weights)

        self._weights_meta = {}
        self._load_status = 'N/A'
        self._applied_overrides = []
        self._has_meta = False

    def build_model(self):
        from ...tasks import get_task
        task_def = get_task(self.cfg.task)

        with suppress_stdout():
            model = task_def['model'](self.cfg)

        ckpt = torch.load(self.weights, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            meta = ckpt.get('model_meta', {}) or {}
            self._has_meta = bool(meta)
            self._weights_meta = meta
        else:
            sd = ckpt
            self._has_meta = False

        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)

        real_missing = [k for k in missing if 'num_batches_tracked' not in k]
        real_unexpected = [k for k in unexpected if 'num_batches_tracked' not in k]
        if not real_missing and not real_unexpected:
            self._load_status = 'OK (full match)'
        else:
            parts = []
            if real_missing:
                parts.append(f'missing {len(real_missing)}')
            if real_unexpected:
                parts.append(f'unexpected {len(real_unexpected)}')
            self._load_status = 'WARN: ' + ', '.join(parts)

        return model.eval().to(self.device)

    def run(self, source):
        if self.model is None:
            self.model = self.build_model()

        if os.path.isdir(source):
            images = sorted(
                os.path.join(source, f) for f in os.listdir(source)
                if f.lower().endswith(_IMG_EXTS)
            )
            if not images:
                raise FileNotFoundError(f"目录下没有图片: {source}")
        else:
            images = [source]

        if len(images) == 1:
            self._print_header(images[0])
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
                n = len(r.get('scores', []))
                print(f"    {name:<50s}  {n:>4d} detections")
            if self.output_image:
                print()
                print("  Output Files:")
                _kv("Image", self.output_image, indent=4)
            print()
            print(LINE_EQ)

        return results

    def _print_header(self, image_path):
        cfg = self.cfg

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        print()
        print(LINE_EQ)
        print("  SWAP-Fighter Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Image", image_path)
        _kv("Image Size", f"{orig_w} x {orig_h}")
        _kv("Device", str(self.device))

        self._print_weights_block()
        self._print_model_block()
        self._print_inference_block()

        print()
        print(LINE_EQ)

    def _print_header_multi(self, source, n_images):
        print()
        print(LINE_EQ)
        print("  SWAP-Fighter Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Source", source)
        _kv("Images", f"{n_images}")
        _kv("Device", str(self.device))

        self._print_weights_block()
        self._print_model_block()
        self._print_inference_block()

        print()
        print(LINE_EQ)

    def _print_weights_block(self):
        _title("Weights")
        _kv("Path", self.weights)
        if self._has_meta:
            _kv("Model Meta", f"{len(self._weights_meta)} fields")
        else:
            _kv("Model Meta", "not found, using config defaults")
        if self._applied_overrides:
            ov = ', '.join(self._applied_overrides[:5])
            if len(self._applied_overrides) > 5:
                ov += f" (+{len(self._applied_overrides) - 5} more)"
            _kv("CLI Overrides", ov)
        else:
            _kv("CLI Overrides", "none")
        _kv("Load Status", self._load_status)

    def _print_model_block(self):
        cfg = self.cfg
        _title("Model")

        if cfg.task == 'obb':
            task_str = "OBB (rotated bbox)"
        elif cfg.task == 'segment':
            task_str = "Segment (instance segmentation)"
        else:
            task_str = "Detect (axis-aligned bbox)"
        _kv("Task", task_str)

        if hasattr(self.model, 'backbone'):
            bb = self.model.backbone
            src = getattr(bb, 'resolved_source', 'N/A')
            _kv("Backbone", f"{getattr(cfg, 'backbone_name', 'N/A')} ({src})")
            _kv("Feature Indices", str(getattr(bb, 'feature_indices', 'N/A')))
            _kv("Feature Channels", str(getattr(bb, 'out_channels', 'N/A')))
            _kv("Strides", str(getattr(bb, 'strides', cfg.strides)))
            if getattr(bb, 'need_permute', False):
                _kv("NHWC → NCHW", "yes")
        else:
            _kv("Backbone", getattr(cfg, 'backbone_name', 'N/A'))
            _kv("Strides", str(getattr(cfg, 'strides', 'N/A')))

        neck_extra = _bool_en(getattr(cfg, 'use_extra_conv', True))
        neck_type = getattr(cfg, 'neck_type', 'lightfpn')
        neck_line = (f"{neck_type} (out={getattr(cfg, 'neck_out_channels', 128)}, "
                     f"extra_conv={neck_extra})")
        if neck_type == 'bifpn':
            neck_line += f", layers={getattr(cfg, 'neck_num_layers', 2)}"
        _kv("Neck", neck_line)

        _kv("Head", f"{getattr(cfg, 'head_type', 'decoupled')} "
                    f"(convs={getattr(cfg, 'head_num_convs', 3)}, "
                    f"norm={getattr(cfg, 'head_norm_layer', 'group')}, "
                    f"act={getattr(cfg, 'head_act_layer', 'silu')})")

        if cfg.task == 'obb':
            _kv("Task", f"cls={cfg.num_classes}, reg=4, angle=2 "
                        f"({getattr(cfg, 'angle_format', 'sin_cos')})")
        else:
            _kv("Task", f"cls={cfg.num_classes}, reg=4")

        _kv("DFL", "disabled" if getattr(cfg, 'reg_max', 0) == 0
                  else f"reg_max={cfg.reg_max}")

    def _print_inference_block(self):
        cfg = self.cfg
        _title("Inference")
        _kv("Conf Threshold", f"{cfg.conf_threshold:.3f}")
        if cfg.task == 'obb':
            nms = getattr(cfg, 'obb_nms_threshold', None) or cfg.nms_threshold
            _kv("NMS Threshold", f"{nms:.3f}")
            _kv("Angle Range", getattr(cfg, 'angle_range', 'pi'))
        else:
            _kv("NMS Threshold", f"{cfg.nms_threshold:.3f}")

    @torch.no_grad()
    def _predict_one(self, image_path):
        cfg = self.cfg
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_padded, scale, (pl, pt), _ = letterbox(
            img_rgb, new_shape=(cfg.img_size, cfg.img_size),
            auto=False, stride=32)

        x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])(Image.fromarray(img_padded)).unsqueeze(0).to(self.device)

        if cfg.task == 'detect':
            cls_l, reg_l = self.model(x)
            boxes, scores, labels = post_process(
                cls_l, reg_l, orig_w, orig_h, cfg.img_size,
                conf_threshold=cfg.conf_threshold,
                nms_threshold=cfg.nms_threshold,
                strides=cfg.strides, scale=scale,
                pad_left=pl, pad_top=pt,
                reg_max=cfg.reg_max)
            obboxes = None
        else:
            cls_l, reg_l, ang_l = self.model(x)
            obboxes, scores, labels = post_process_obb(
                cls_l, reg_l, ang_l, orig_w, orig_h, cfg.img_size,
                conf_threshold=cfg.conf_threshold,
                nms_threshold=getattr(cfg, 'obb_nms_threshold', cfg.nms_threshold),
                strides=cfg.strides, scale=scale,
                pad_left=pl, pad_top=pt,
                angle_format=cfg.angle_format,
                angle_range=cfg.angle_range,
                reg_max=cfg.reg_max)
            boxes = None

        img_vis = self._draw(img_rgb, boxes, obboxes, scores, labels)

        if self.output_image:
            cv2.imwrite(self.output_image, img_vis)

        return {
            'image': image_path,
            'orig_size': (orig_w, orig_h),
            'mode': cfg.task,
            'boxes': boxes,
            'obboxes': obboxes,
            'scores': scores,
            'labels': labels,
            'img_vis': img_vis,
        }

    def _print_result(self, r):
        cfg = self.cfg
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

        from collections import Counter
        class_names = self.class_names
        if class_names:
            print("\n  Class Breakdown:")
            counts = Counter(int(l) for l in labels)
            max_cnt = max(counts.values())
            for cls_id in sorted(counts.keys()):
                name = (class_names[cls_id] if cls_id < len(class_names)
                        else f"cls_{cls_id}")
                cnt = counts[cls_id]
                bar = "█" * min(int(cnt / max_cnt * 20), 20)
                print(f"    {name:<20}  {cnt:>4d}  {bar}")

        top_k = getattr(cfg, 'top_k', 20)
        order = np.argsort(-np.array(scores))[:top_k]
        print()

        if cfg.task == 'detect':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (x1,y1,x2,y2)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = self._get_label_name(labels[idx])
                box = boxes[idx]
                geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                        f"{box[2]:.0f}, {box[3]:.0f})")
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")
        else:
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
                  f"{'Geometry (cx,cy,w,h,angle)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = self._get_label_name(labels[idx])
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

    def _get_label_name(self, label):
        if self.class_names and int(label) < len(self.class_names):
            return self.class_names[int(label)]
        return str(int(label))

    def _draw(self, img_rgb, boxes, obboxes, scores, labels):
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(vis.shape[:2]) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        if boxes is not None:
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, b)
                color = _get_class_color(l)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                name = self._get_label_name(l)
                cv2.putText(vis, f'{name}: {s:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)
        else:
            for ob, s, l in zip(obboxes, scores, labels):
                color = _get_class_color(l)
                pts = ob.reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, color, thickness)
                name = self._get_label_name(l)
                cv2.putText(vis, f'{name}: {s:.2f}',
                            (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)
        return vis