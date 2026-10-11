import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .yolo_det import YOLODataset
from ..utils.box_ops import letterbox
from ..utils.mask_ops import (
    polygons_to_mask,
    polygons_to_masks_per_gt,
)

class YOLOSegDataset(YOLODataset):

    def __init__(self, root, img_size=640, split='train',
                 proto_stride=4,
                 **kwargs):

        kwargs['task'] = 'segment'

        super().__init__(root, img_size, split, **kwargs)

        self.proto_stride = proto_stride
        self.proto_h = img_size // proto_stride
        self.proto_w = img_size // proto_stride

        self.random_perspective = None

        print(f'[YOLOSegDataset] split={split}, proto_size={self.proto_h}x{self.proto_w}, '
              f'samples={len(self.img_files)}')

    def _load_seg_labels(self, label_path, img_w, img_h):
        polygons = []
        labels = []

        if not os.path.exists(label_path):
            return polygons, np.zeros((0,), dtype=np.int64)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = np.array(
                        list(map(float, parts[1:])), dtype=np.float32)
                except ValueError:
                    continue

                if len(coords) % 2 != 0:
                    continue

                pts = coords.reshape(-1, 2).copy()
                pts[:, 0] *= img_w
                pts[:, 1] *= img_h
                polygons.append(pts)
                labels.append(cls_id)

        labels = np.array(labels, dtype=np.int64) if labels \
            else np.zeros((0,), dtype=np.int64)
        return polygons, labels

    @staticmethod
    def _polygons_to_boxes(polygons, img_w, img_h):
        boxes = []
        for poly in polygons:
            if len(poly) < 3:
                boxes.append([0, 0, 0, 0])
                continue
            x1 = poly[:, 0].min() / img_w
            y1 = poly[:, 1].min() / img_h
            x2 = poly[:, 0].max() / img_w
            y2 = poly[:, 1].max() / img_h
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32) if boxes \
            else np.zeros((0, 4), dtype=np.float32)

    def __getitem__(self, idx):
        img_padded, targets = self._load_single_seg(idx)

        if self.color_aug is not None:
            img_padded = self.color_aug(img_padded)

        img_tensor = self.basic_transform(Image.fromarray(img_padded))

        for k in ('boxes', 'labels', 'mask_per_gt'):
            if k in targets and isinstance(targets[k], np.ndarray):
                targets[k] = torch.from_numpy(targets[k])
            elif k in targets and isinstance(targets[k], torch.Tensor):
                pass
            elif k in targets and targets[k] is None:
                if k == 'boxes':
                    targets[k] = torch.zeros((0, 4))
                elif k == 'labels':
                    targets[k] = torch.zeros((0,), dtype=torch.int64)
                else:
                    targets[k] = torch.zeros((0, self.proto_h, self.proto_w))

        return img_tensor, targets

    def _load_single_seg(self, idx):
        base_name = self.img_files[idx]
        img_path = self._find_image_path(base_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'无法读取图片: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        label_path = os.path.join(self.label_dir, f'{base_name}.txt')

        polygons, labels = self._load_seg_labels(label_path, w0, h0)
        boxes = self._polygons_to_boxes(polygons, w0, h0)

        img_padded, lb_scale, (pad_left, pad_top), _ = letterbox(
            img_rgb, new_shape=(self.img_size, self.img_size),
            auto=False, stride=32)

        if len(boxes) > 0:
            boxes_abs = boxes * np.array([w0, h0, w0, h0], dtype=np.float32)
            boxes_abs[:, [0, 2]] = boxes_abs[:, [0, 2]] * lb_scale + pad_left
            boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]] * lb_scale + pad_top
            boxes = np.clip(boxes_abs / self.img_size, 0, 1).astype(np.float32)

        transformed_polys = []
        for poly in polygons:
            new_poly = poly * lb_scale
            new_poly[:, 0] += pad_left
            new_poly[:, 1] += pad_top
            transformed_polys.append(new_poly)

        targets = {
            'mode': 'segment',
            'scale': lb_scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_size': (w0, h0),
            'flip': False,
            'flip_dir': None,
            'img_size': (self.img_size, self.img_size),
            'labels': labels,
            'boxes': boxes,
        }

        img_padded, targets, transformed_polys = self._random_flip_seg(
            img_padded, targets, transformed_polys)

        if len(transformed_polys) > 0:
            mask_per_gt = polygons_to_masks_per_gt(
                transformed_polys,
                self.img_size, self.img_size,
                downsample=self.proto_stride,
            )
        else:
            mask_per_gt = np.zeros((0, self.proto_h, self.proto_w),
                                   dtype=np.float32)

        targets['mask_per_gt'] = mask_per_gt

        return img_padded, targets

    def _random_flip_seg(self, img, targets, polygons):
        if self.fliplr > 0 and random.random() < self.fliplr:
            img = cv2.flip(img, 1)
            targets['flip'] = True
            targets['flip_dir'] = 'h'

            if len(targets['boxes']) > 0:
                targets['boxes'] = self._flip_detect_boxes_h(targets['boxes'])

            W = img.shape[1]
            for poly in polygons:
                poly[:, 0] = (W - 1) - poly[:, 0]

        if self.flipud > 0 and random.random() < self.flipud:
            img = cv2.flip(img, 0)
            targets['flip'] = True
            targets['flip_dir'] = 'v'
            if len(targets['boxes']) > 0:
                targets['boxes'] = self._flip_detect_boxes_v(targets['boxes'])
            H = img.shape[0]
            for poly in polygons:
                poly[:, 1] = (H - 1) - poly[:, 1]

        return img, targets, polygons

__all__ = ['YOLOSegDataset']