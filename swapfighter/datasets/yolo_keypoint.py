import os
import random
import cv2
import numpy as np
import torch
from PIL import Image

from .yolo_det import YOLODataset
from ..utils.box_ops import letterbox

class YOLOKeypointDataset(YOLODataset):
    def __init__(self, root, img_size=640, split='train',
                 num_keypoints=17,
                 keypoint_visibility='auto',
                 flip_idx=None,
                 **kwargs):
        kwargs['task'] = 'keypoint'
        kwargs['mosaic_prob'] = 0.0
        kwargs['copy_paste_prob'] = 0.0
        kwargs['mixup_prob'] = 0.0
        super().__init__(root, img_size, split, **kwargs)

        self.num_keypoints = num_keypoints
        self.visibility_mode = keypoint_visibility
        self.flip_idx = list(flip_idx) if flip_idx is not None else None

        if self.visibility_mode == 'auto':
            self.visibility_mode = self._detect_visibility_mode()

        print(f'[YOLOKeypointDataset] K={num_keypoints}, '
              f'visibility={self.visibility_mode}, '
              f'flip_idx={"none" if flip_idx is None else len(flip_idx)}')

    def _detect_visibility_mode(self):
        for base in self.img_files[:20]:
            label_path = os.path.join(self.label_dir, f'{base}.txt')
            if not os.path.exists(label_path):
                continue
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    n = len(parts) - 5

                    if n == 3 * self.num_keypoints:
                        third = [float(v) for v in parts[7::3]]
                        if third and all(v in (0.0, 1.0, 2.0) for v in third):
                            return 'yes'
                    if n == 2 * self.num_keypoints:
                        return 'no'
        return 'yes'

    def _parse_keypoint_line(self, parts):
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            rest = list(map(float, parts[5:]))
        except ValueError:
            return None

        K = self.num_keypoints
        n_rest = len(rest)
        if n_rest == 3 * K:
            kpt = np.array(rest, dtype=np.float32).reshape(K, 3)
        elif n_rest == 2 * K:
            xy = np.array(rest, dtype=np.float32).reshape(K, 2)
            v = np.ones((K, 1), dtype=np.float32)
            kpt = np.concatenate([xy, v], axis=1)
        else:

            raise ValueError(
                f"关键点数量不匹配: 标签有 {n_rest} 个值, "
                f"num_keypoints={K} 期望 {2*K} (无 visibility) "
                f"或 {3*K} (带 visibility)。"
                f"请重新指定 --num-keypoints。")

        return cls_id, np.array([cx, cy, w, h], dtype=np.float32), kpt

    def _load_keypoint_labels(self, label_path, img_w, img_h):
        boxes, labels, kpts = [], [], []
        if not os.path.exists(label_path):
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0, self.num_keypoints, 3), dtype=np.float32))

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                res = self._parse_keypoint_line(parts)
                if res is None:
                    continue
                cls_id, box, kpt = res

                kpt[:, 0] *= img_w
                kpt[:, 1] *= img_h
                boxes.append(box)
                labels.append(cls_id)
                kpts.append(kpt)

        boxes = np.array(boxes, dtype=np.float32) if boxes \
            else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels \
            else np.zeros((0,), dtype=np.int64)
        kpts = np.stack(kpts, axis=0) if kpts \
            else np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
        return boxes, labels, kpts

    def _load_single_keypoint(self, idx):
        base_name = self.img_files[idx]
        img_path = self._find_image_path(base_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'无法读取图片: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        label_path = os.path.join(self.label_dir, f'{base_name}.txt')
        boxes, labels, keypoints = self._load_keypoint_labels(
            label_path, w0, h0)

        img_padded, lb_scale, (pad_left, pad_top), _ = letterbox(
            img_rgb, new_shape=(self.img_size, self.img_size),
            auto=False, stride=32)

        if len(boxes) > 0:

            boxes_xyxy = boxes.copy()
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

            bx = boxes_xyxy * np.array([w0, h0, w0, h0], dtype=np.float32)
            bx[:, [0, 2]] = bx[:, [0, 2]] * lb_scale + pad_left
            bx[:, [1, 3]] = bx[:, [1, 3]] * lb_scale + pad_top
            boxes = np.clip(bx / self.img_size, 0, 1).astype(np.float32)

        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[..., 0] = keypoints[..., 0] * lb_scale + pad_left
            keypoints[..., 1] = keypoints[..., 1] * lb_scale + pad_top

        targets = {
            'mode': 'keypoint',
            'scale': lb_scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_size': (w0, h0),
            'flip': False, 'flip_dir': None,
            'img_size': (self.img_size, self.img_size),
            'labels': labels,
            'boxes': boxes,
            'keypoints': keypoints,
        }

        if self.fliplr > 0 and random.random() < self.fliplr:
            img_padded = cv2.flip(img_padded, 1)
            targets['flip'] = True
            targets['flip_dir'] = 'h'

            if len(targets['boxes']) > 0:
                targets['boxes'] = self._flip_detect_boxes_h(
                    targets['boxes'])

            if len(keypoints) > 0:
                kp = targets['keypoints'].copy()
                W = self.img_size
                kp[..., 0] = (W - 1) - kp[..., 0]
                if self.flip_idx is not None:
                    kp = kp[:, self.flip_idx, :]
                targets['keypoints'] = kp

        return img_padded, targets

    def __getitem__(self, idx):
        img_padded, targets = self._load_single_keypoint(idx)

        if self.color_aug is not None:
            img_padded = self.color_aug(img_padded)

        img_tensor = self.basic_transform(Image.fromarray(img_padded))

        for k in ('boxes', 'labels', 'keypoints'):
            if k in targets and isinstance(targets[k], np.ndarray):
                targets[k] = torch.from_numpy(targets[k])

        return img_tensor, targets

__all__ = ['YOLOKeypointDataset']