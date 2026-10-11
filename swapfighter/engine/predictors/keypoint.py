import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from .detect import DetectPredictor, _get_class_color
from ...utils.box_ops import letterbox
from ...utils.logger import LINE_EQ, _kv, _title

class KeypointPredictor(DetectPredictor):
    def __init__(self, cfg_dict, weights=None):

        self.kpt_radius = cfg_dict.pop('kpt_radius', 6)
        self.kpt_thr = cfg_dict.pop('kpt_thr', 0.3)
        self.draw_kpt = cfg_dict.pop('draw_kpt', True)
        super().__init__(cfg_dict, weights)

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

        out = self.model(x)
        cls_l, reg_l, kpt_l = out[0], out[1], out[2]

        from ...utils.postprocess import post_process

        result = post_process(
            cls_l, reg_l,
            orig_w, orig_h, cfg.img_size,
            conf_threshold=cfg.conf_threshold,
            nms_threshold=cfg.nms_threshold,
            strides=cfg.strides,
            scale=scale,
            pad_left=pl, pad_top=pt,
            reg_max=cfg.reg_max,
            return_indices=True,
        )
        boxes, scores, labels, origin_idx = result

        K = getattr(cfg, 'num_keypoints', 17)
        device = kpt_l[0].device
        all_kpt = []
        for kpt in kpt_l:
            _, c, h, w = kpt.shape
            all_kpt.append(kpt[0].reshape(c, -1))
        all_kpt = torch.cat(all_kpt, dim=1)

        keypoints = []
        for oi in origin_idx:
            if oi >= all_kpt.shape[1]:
                keypoints.append(np.zeros((K, 3), dtype=np.float32))
                continue
            kpt_norm = all_kpt[:, oi].cpu().numpy().reshape(K, 2)

            kpt_xy = kpt_norm * cfg.img_size

            if scale is not None and scale > 0:
                kpt_xy[:, 0] = (kpt_xy[:, 0] - pl) / scale
                kpt_xy[:, 1] = (kpt_xy[:, 1] - pt) / scale
            else:
                kpt_xy[:, 0] *= orig_w / cfg.img_size
                kpt_xy[:, 1] *= orig_h / cfg.img_size

            kpt_xy[:, 0] = np.clip(kpt_xy[:, 0], 0, orig_w)
            kpt_xy[:, 1] = np.clip(kpt_xy[:, 1], 0, orig_h)

            kpt3 = np.concatenate(
                [kpt_xy, np.ones((K, 1), dtype=np.float32)], axis=1)
            keypoints.append(kpt3)

        img_vis = self._draw_keypoint(
            img_rgb, boxes, scores, labels, keypoints)

        if self.output_image:
            cv2.imwrite(self.output_image, img_vis)

        return {
            'image': image_path,
            'orig_size': (orig_w, orig_h),
            'mode': 'keypoint',
            'boxes': boxes,
            'obboxes': None,
            'scores': scores,
            'labels': labels,
            'keypoints': keypoints,
            'img_vis': img_vis,
        }

    def _print_model_block(self):
        cfg = self.cfg
        _title("Model")
        _kv("Mode", "Keypoint (pose estimation)")
        _kv("Backbone", getattr(cfg, 'backbone_name', 'N/A'))
        _kv("Image Size", cfg.img_size)
        _kv("Num Keypoints", getattr(cfg, 'num_keypoints', 17))
        _kv("Keypoint Visibility",
            getattr(cfg, 'keypoint_visibility', 'auto'))
        reg_max = getattr(cfg, 'reg_max', 0)
        _kv("DFL", "disabled" if reg_max == 0
                  else f"reg_max={reg_max}")

    def _print_result(self, r):
        from collections import Counter

        cfg = self.cfg
        boxes = r.get('boxes')
        scores = r.get('scores')
        labels = r.get('labels')
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
        print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
              f"{'Geometry (x1,y1,x2,y2)':<40}  {'Kpts':>6}")
        print("  " + "─" * 84)

        for i, idx in enumerate(order, 1):
            name = self._get_label_name(labels[idx])
            box = boxes[idx]
            geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                    f"{box[2]:.0f}, {box[3]:.0f})")
            kpt_str = "N/A"
            if keypoints is not None and idx < len(keypoints):
                k = keypoints[idx]
                if k is not None:
                    kpt_str = f"{len(k):>6d}"
            print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  "
                  f"{geom:<40}  {kpt_str:>6}")

        if n_total > top_k:
            print(f"  {'':>4}  ... "
                  f"({n_total - top_k} more detections omitted)")

        self._print_output_files()
        print()
        print(LINE_EQ)

    def _print_output_files(self):
        print()
        print("  Output Files:")
        if self.output_image:
            _kv("Image", self.output_image, indent=4)

    def _draw_keypoint(self, img_rgb, boxes, scores, labels, keypoints):
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(vis.shape[:2]) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        base_r = self.kpt_radius
        img_short = min(vis.shape[:2])

        kpt_r = max(4, int(round(base_r * img_short / 800.0)))

        for b, s, l, kpt in zip(boxes, scores, labels, keypoints):
            color = _get_class_color(l)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            name = self._get_label_name(l)
            cv2.putText(vis, f'{name}: {s:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color, thickness)

            if not self.draw_kpt or kpt is None:
                continue

            for k in range(len(kpt)):
                if kpt[k, 2] > self.kpt_thr:
                    cx, cy = int(kpt[k, 0]), int(kpt[k, 1])

                    cv2.circle(vis, (cx, cy), kpt_r, color, -1)

                    cv2.circle(vis, (cx, cy), kpt_r, (255, 255, 255), 2)

        return vis

__all__ = ['KeypointPredictor']