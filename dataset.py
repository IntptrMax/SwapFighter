import os
import math
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils import letterbox


# ============================================================
# YOLOv8 风格颜色增强
# ============================================================
class YOLOv8ColorAugment:
    def __init__(self,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 blur_prob=0.01, median_blur_prob=0.01,
                 to_gray_prob=0.01, clahe_prob=0.01):
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.blur_prob = blur_prob
        self.median_blur_prob = median_blur_prob
        self.to_gray_prob = to_gray_prob
        self.clahe_prob = clahe_prob

        # ★ 不在这里创建 CLAHE！延迟到首次调用
        self._clahe_obj = None

    # ---------- pickle 支持 ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_clahe_obj'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe_obj = None

    @property
    def clahe(self):
        if self._clahe_obj is None:
            self._clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        return self._clahe_obj

    def __call__(self, img):
        img = self._augment_hsv(img)

        if self.blur_prob > 0 and random.random() < self.blur_prob:
            img = self._blur(img)
        if self.median_blur_prob > 0 and random.random() < self.median_blur_prob:
            img = self._median_blur(img)
        if self.to_gray_prob > 0 and random.random() < self.to_gray_prob:
            img = self._to_gray(img)
        if self.clahe_prob > 0 and random.random() < self.clahe_prob:
            img = self._clahe(img)

        return img

    def _augment_hsv(self, im):
        if not (self.hsv_h or self.hsv_s or self.hsv_v):
            return im

        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1

        im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(im_hsv)
        dtype = im.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val),
        ))

        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        return im

    @staticmethod
    def _blur(im):
        k = random.choice([3, 5, 7])
        return cv2.blur(im, (k, k))

    @staticmethod
    def _median_blur(im):
        k = random.choice([3, 5])
        return cv2.medianBlur(im, k)

    @staticmethod
    def _to_gray(im):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    def _clahe(self, im):
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ============================================================
# RandomPerspective：旋转 + 平移 + 缩放 + 剪切 + 透视
# ============================================================
class RandomPerspective:
    def __init__(self, degrees=0.0, translate=0.1, scale=0.5,
                 shear=0.0, perspective=0.0, img_size=640,
                 area_thr=0.2, visibility_thr=0.3,
                 min_area_px=16.0,
                 border_value=(114, 114, 114)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.img_size = img_size
        self.area_thr = area_thr
        self.visibility_thr = visibility_thr
        self.min_area_px = min_area_px
        self.border_value = border_value

    def __call__(self, img, targets):
        h, w = img.shape[:2]
        s = self.img_size

        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2.0
        C[1, 2] = -h / 2.0

        if self.perspective:
            C[0, 2] += random.uniform(-self.perspective, self.perspective) * w
            C[1, 2] += random.uniform(-self.perspective, self.perspective) * h

        if self.scale:
            s_val = random.uniform(1 - self.scale, 1 + self.scale)
            C[0, 0] = s_val
            C[1, 1] = s_val

        R = np.eye(3, dtype=np.float32)
        if self.degrees:
            a = random.uniform(-self.degrees, self.degrees)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)

        S = np.eye(3, dtype=np.float32)
        if self.shear:
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180.0)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180.0)

        M = S @ R @ C

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = s / 2.0
        T[1, 2] = s / 2.0
        M = T @ M

        if self.translate:
            M[0, 2] += random.uniform(-self.translate, self.translate) * s
            M[1, 2] += random.uniform(-self.translate, self.translate) * s

        img_out = cv2.warpPerspective(
            img, M, (s, s),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

        mode = targets.get('mode', 'detect')
        labels = targets.get('labels', np.zeros((0,), dtype=np.int64))

        if mode == 'detect':
            new_boxes, new_labels = self._transform_detect(
                targets.get('boxes', None), labels, M, w, h, s
            )
            targets['boxes'] = new_boxes
            targets['labels'] = new_labels
        else:
            new_obboxes, new_labels = self._transform_obb(
                targets.get('obboxes', None), labels, M, w, h, s
            )
            targets['obboxes'] = new_obboxes
            targets['labels'] = new_labels

        targets['scale'] = 1.0
        targets['pad_left'] = 0
        targets['pad_top'] = 0
        targets['orig_size'] = (s, s)

        return img_out, targets

    def _transform_detect(self, boxes, labels, M, w, h, s):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        boxes_pix = boxes * np.array([w, h, w, h], dtype=np.float32)

        x1 = boxes_pix[:, 0]
        y1 = boxes_pix[:, 1]
        x2 = boxes_pix[:, 2]
        y2 = boxes_pix[:, 3]

        corners = np.stack([
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ], axis=1).astype(np.float32)

        orig_area = np.maximum((x2 - x1) * (y2 - y1), 0.0)

        corners_flat = corners.reshape(-1, 2)
        corners_t = cv2.perspectiveTransform(
            corners_flat.reshape(1, -1, 2), M
        ).reshape(-1, 4, 2)

        np.clip(corners_t[..., 0], 0, s, out=corners_t[..., 0])
        np.clip(corners_t[..., 1], 0, s, out=corners_t[..., 1])

        new_x1 = corners_t[..., 0].min(axis=1)
        new_y1 = corners_t[..., 1].min(axis=1)
        new_x2 = corners_t[..., 0].max(axis=1)
        new_y2 = corners_t[..., 1].max(axis=1)

        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        new_area = new_w * new_h

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(orig_area > 1e-6, new_area / np.maximum(orig_area, 1e-6), 0.0)

        valid = (new_w > 2.0) & (new_h > 2.0) & (ratio >= self.area_thr)

        new_boxes = np.stack(
            [new_x1[valid], new_y1[valid], new_x2[valid], new_y2[valid]], axis=1
        )
        new_boxes = np.clip(new_boxes / s, 0, 1).astype(np.float32)
        new_labels = labels[valid].astype(np.int64)

        return new_boxes, new_labels

    def _transform_obb(self, obboxes, labels, M, w, h, s):
        if obboxes is None or len(obboxes) == 0:
            return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        obboxes = np.asarray(obboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        obboxes_pix = obboxes.copy()
        obboxes_pix[:, 0::2] *= w
        obboxes_pix[:, 1::2] *= h

        pts = obboxes_pix.reshape(-1, 4, 2).astype(np.float32)
        orig_area = self._polygon_area(pts)

        pts_flat = pts.reshape(-1, 2)
        pts_t = cv2.perspectiveTransform(
            pts_flat.reshape(1, -1, 2), M
        ).reshape(-1, 4, 2)

        full_area = self._polygon_area(pts_t)

        pts_clipped = pts_t.copy()
        np.clip(pts_clipped[..., 0], 0, s, out=pts_clipped[..., 0])
        np.clip(pts_clipped[..., 1], 0, s, out=pts_clipped[..., 1])
        visible_area = self._polygon_area(pts_clipped)

        with np.errstate(divide='ignore', invalid='ignore'):
            area_ratio = np.where(orig_area > 1e-6,
                                  full_area / np.maximum(orig_area, 1e-6), 0.0)
            vis_ratio = np.where(full_area > 1e-6,
                                 visible_area / np.maximum(full_area, 1e-6), 0.0)

        valid = (full_area > self.min_area_px) & \
                (area_ratio >= self.area_thr) & \
                (vis_ratio >= self.visibility_thr)

        if not np.any(valid):
            return (np.zeros((0, 8), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        pts_out = pts_t[valid]
        new_labels = labels[valid].astype(np.int64)

        obboxes_new = (pts_out / s).reshape(-1, 8).astype(np.float32)
        return obboxes_new, new_labels

    @staticmethod
    def _polygon_area(corners):
        corners = np.asarray(corners, dtype=np.float32)
        if corners.ndim == 2:
            corners = corners[np.newaxis, ...]
        x = corners[..., 0]
        y = corners[..., 1]
        return 0.5 * np.abs(
            x[:, 0] * (y[:, 1] - y[:, 3]) +
            x[:, 1] * (y[:, 2] - y[:, 0]) +
            x[:, 2] * (y[:, 3] - y[:, 1]) +
            x[:, 3] * (y[:, 0] - y[:, 2])
        )


# ============================================================
# YOLO Dataset
# ============================================================
class YOLODataset(Dataset):
    def __init__(self, root, img_size=640, split='train',
                 use_flip=True, use_color_aug=True,
                 mean=[0, 0, 0], std=[1, 1, 1], mode='detect', angle_range='pi',
                 debug_vis=False,
                 mosaic_prob=0.0, mosaic_disable_epoch=None,
                 degrees=0.0, translate=0.1, scale=0.5,
                 shear=0.0, perspective=0.0,
                 area_thr=0.2, visibility_thr=0.3,
                 # ★ YOLOv8 风格 HSV 增强参数
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 color_blur_prob=0.01, color_median_blur_prob=0.01,
                 color_to_gray_prob=0.01, color_clahe_prob=0.01,
                 # ★ YOLOv8 风格翻转参数
                 fliplr=0.5, flipud=0.0,
                 # ★ YOLOv8 风格 MixUp / Copy-Paste
                 mixup_prob=0.0, mixup_alpha=32.0,
                 copy_paste_prob=0.0, copy_paste_max_paste=3):
        self.root = root
        self.img_size = img_size
        self.split = split
        self.use_flip = use_flip and split == 'train'
        self.use_color_aug = use_color_aug and split == 'train'
        self.mode = mode
        self.angle_range = angle_range
        self.debug_vis = debug_vis
        self.mosaic_prob = mosaic_prob if split == 'train' else 0.0
        self.mosaic_disable_epoch = mosaic_disable_epoch
        self.epoch = 0

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.area_thr = area_thr
        self.visibility_thr = visibility_thr

        self.fliplr = fliplr if self.use_flip else 0.0
        self.flipud = flipud if self.use_flip else 0.0

        # ★ MixUp / Copy-Paste（仅 train）
        self.mixup_prob = mixup_prob if split == 'train' else 0.0
        self.mixup_alpha = mixup_alpha
        self.copy_paste_prob = copy_paste_prob if split == 'train' else 0.0
        self.copy_paste_max_paste = copy_paste_max_paste

        if self.mode == 'obb':
            if self.angle_range not in ('pi', 'pi2'):
                raise ValueError(
                    f"angle_range 必须为 'pi' 或 'pi2'，当前为 {self.angle_range}"
                )
        else:
            if self.angle_range is None:
                self.angle_range = 'pi'

        self.img_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)

        self.img_files = []
        for f in os.listdir(self.img_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                self.img_files.append(os.path.splitext(f)[0])
        if not self.img_files:
            raise RuntimeError(f'在 {self.img_dir} 中未找到任何图片')

        # ★ 颜色增强：YOLOv8 风格 HSV
        if self.use_color_aug:
            self.color_aug = YOLOv8ColorAugment(
                hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
                blur_prob=color_blur_prob,
                median_blur_prob=color_median_blur_prob,
                to_gray_prob=color_to_gray_prob,
                clahe_prob=color_clahe_prob,
            )
            print(f"[YOLODataset] 启用 YOLOv8 风格颜色增强: "
                  f"hsv_h={hsv_h}, hsv_s={hsv_s}, hsv_v={hsv_v}, "
                  f"blur={color_blur_prob}, median_blur={color_median_blur_prob}, "
                  f"to_gray={color_to_gray_prob}, clahe={color_clahe_prob}")
        else:
            self.color_aug = None

        if split == 'train' and self.use_flip:
            print(f"[YOLODataset] 启用翻转增强: fliplr={self.fliplr}, flipud={self.flipud}")

        if split == 'train':
            if self.mixup_prob > 0:
                print(f"[YOLODataset] 启用 MixUp: prob={self.mixup_prob}, alpha={self.mixup_alpha}")
            if self.copy_paste_prob > 0:
                print(f"[YOLODataset] 启用 Copy-Paste: prob={self.copy_paste_prob}, "
                      f"max_paste={self.copy_paste_max_paste}")

        self.basic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        if split == 'train':
            self.random_perspective = RandomPerspective(
                degrees=degrees, translate=translate, scale=scale,
                shear=shear, perspective=perspective, img_size=img_size,
                area_thr=area_thr, visibility_thr=visibility_thr,
            )
        else:
            self.random_perspective = None

        if self.debug_vis:
            self.debug_dir = os.path.join('./debug_vis', self.split)
            os.makedirs(self.debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.img_files)

    # ==========================================================
    # 主入口
    #
    # 流程（与 YOLOv8 一致）：
    #   1. mosaic（概率触发）  → 得到 numpy 图像 + targets
    #   2. mixup（概率触发）   → 与另一张图混合
    #   3. copy_paste（概率触发）→ 从另一张图复制目标
    #   4. 颜色增强（numpy）
    #   5. ToTensor + Normalize
    # ==========================================================
    def __getitem__(self, idx):
        use_mosaic = (
            self.split == 'train'
            and self.mosaic_prob > 0
            and (self.mosaic_disable_epoch is None or self.epoch < self.mosaic_disable_epoch)
            and random.random() < self.mosaic_prob
        )

        if use_mosaic:
            img_padded, targets = self._load_mosaic_as_numpy(idx)
        else:
            img_padded, targets = self._load_single_as_numpy(idx)

        # ---------- ★ MixUp ----------
        if self.mixup_prob > 0 and random.random() < self.mixup_prob:
            img_padded, targets = self._apply_mixup(img_padded, targets)

        # ---------- ★ Copy-Paste ----------
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            img_padded, targets = self._apply_copy_paste(img_padded, targets)

        # ---------- 颜色增强 ----------
        if self.color_aug is not None:
            img_padded = self.color_aug(img_padded)

        # ---------- ToTensor + Normalize ----------
        img_pil = Image.fromarray(img_padded)
        img_tensor = self.basic_transform(img_pil)

        # ---------- numpy → torch ----------
        for k in ('boxes', 'obboxes', 'cxcywh', 'angles', 'labels'):
            if k in targets and isinstance(targets[k], np.ndarray):
                targets[k] = torch.from_numpy(targets[k])
            elif k in targets and targets[k] is None:
                if k == 'boxes':
                    targets[k] = torch.zeros((0, 4))
                elif k == 'obboxes':
                    targets[k] = torch.zeros((0, 8))
                elif k == 'cxcywh':
                    targets[k] = torch.zeros((0, 4))
                elif k == 'angles':
                    targets[k] = torch.zeros((0,))
                else:
                    targets[k] = torch.zeros((0,), dtype=torch.int64)

        if self.debug_vis and idx < 5:
            self._debug_visualize(img_padded, self.img_files[idx], targets)

        return img_tensor, targets

    # ==========================================================
    # 加载单张图 + 几何增强（返回 numpy）
    # ==========================================================
    def _load_single_as_numpy(self, idx):
        base_name = self.img_files[idx]
        img_path = self._find_image_path(base_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'无法读取图片: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        label_path = os.path.join(self.label_dir, f'{base_name}.txt')

        if self.mode == 'detect':
            boxes, labels = self._load_detect_labels(label_path)
            obboxes = None
        else:
            obboxes, labels, _, _ = self._load_obb_labels(label_path, w0, h0)
            boxes = None

        # ---------- 1. Letterbox ----------
        img_padded, lb_scale, (pad_left, pad_top), _ = letterbox(
            img_rgb, new_shape=(self.img_size, self.img_size),
            auto=False, stride=32
        )

        # ---------- 2. 标注 → letterbox 归一化坐标 ----------
        if self.mode == 'detect' and boxes is not None and len(boxes) > 0:
            boxes = self._transform_detect_boxes(boxes, w0, h0, lb_scale, pad_left, pad_top)
        elif self.mode == 'obb' and obboxes is not None and len(obboxes) > 0:
            obboxes = self._transform_obb_boxes(obboxes, w0, h0, lb_scale, pad_left, pad_top)

        # ---------- 3. targets ----------
        targets = {
            'mode': self.mode,
            'scale': lb_scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_size': (w0, h0),
            'flip': False,
            'flip_dir': None,
            'labels': labels if labels is not None else np.zeros((0,), dtype=np.int64),
        }
        if self.mode == 'detect':
            targets['boxes'] = boxes if boxes is not None else np.zeros((0, 4), dtype=np.float32)
        else:
            targets['obboxes'] = obboxes if obboxes is not None else np.zeros((0, 8), dtype=np.float32)
            if len(targets['obboxes']) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    targets['obboxes'], img_size=self.img_size
                )
            else:
                cxcywh = np.zeros((0, 4), dtype=np.float32)
                angles = np.zeros((0,), dtype=np.float32)
            targets['cxcywh'] = cxcywh
            targets['angles'] = angles

        # ---------- 4. RandomPerspective ----------
        if self.random_perspective is not None:
            img_padded, targets = self.random_perspective(img_padded, targets)

            if self.mode == 'obb':
                obboxes_new = targets['obboxes']
                if len(obboxes_new) > 0:
                    cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                        obboxes_new, img_size=self.img_size
                    )
                else:
                    cxcywh = np.zeros((0, 4), dtype=np.float32)
                    angles = np.zeros((0,), dtype=np.float32)
                targets['cxcywh'] = cxcywh
                targets['angles'] = angles

        # ---------- 5. 翻转 ----------
        img_padded, targets = self._random_flip(img_padded, targets)

        return img_padded, targets

    # ==========================================================
    # 加载 Mosaic + 几何增强（返回 numpy）
    # ==========================================================
    def _load_mosaic_as_numpy(self, idx):
        s = self.img_size

        xc = int(random.uniform(0, 2 * s))
        yc = int(random.uniform(0, 2 * s))

        indices = [idx] + random.sample(range(len(self.img_files)), 3)
        random.shuffle(indices)

        mosaic_img = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)

        all_boxes = []
        all_obboxes = []
        all_labels = []

        for i, sub_idx in enumerate(indices):
            base_name = self.img_files[sub_idx]
            img_path = self._find_image_path(base_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            r = s / max(h_orig, w_orig)
            new_w = max(1, int(round(w_orig * r)))
            new_h = max(1, int(round(h_orig * r)))
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_r = r

            if i == 0:
                x_place, y_place = xc - new_w, yc - new_h
            elif i == 1:
                x_place, y_place = xc, yc - new_h
            elif i == 2:
                x_place, y_place = xc - new_w, yc
            else:
                x_place, y_place = xc, yc

            x1c = max(0, x_place)
            y1c = max(0, y_place)
            x2c = min(2 * s, x_place + new_w)
            y2c = min(2 * s, y_place + new_h)
            if x2c <= x1c or y2c <= y1c:
                continue

            x1s = max(0, x1c - x_place)
            y1s = max(0, y1c - y_place)
            x2s = min(new_w, x2c - x_place)
            y2s = min(new_h, y2c - y_place)
            if x2s <= x1s or y2s <= y1s:
                continue

            mosaic_img[y1c:y2c, x1c:x2c] = img_resized[y1s:y2s, x1s:x2s]

            label_path = os.path.join(self.label_dir, f'{base_name}.txt')

            if self.mode == 'detect':
                boxes, labels = self._load_detect_labels(label_path)
                if len(boxes) == 0:
                    continue

                boxes_abs = boxes * np.array([w_orig, h_orig, w_orig, h_orig], dtype=np.float32)

                orig_area = (boxes_abs[:, 2] - boxes_abs[:, 0]) * \
                            (boxes_abs[:, 3] - boxes_abs[:, 1]) * (scale_r ** 2)

                boxes_abs[:, [0, 2]] = boxes_abs[:, [0, 2]] * scale_r + x_place
                boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]] * scale_r + y_place

                boxes_abs[:, 0] = np.clip(boxes_abs[:, 0], x1c, x2c)
                boxes_abs[:, 1] = np.clip(boxes_abs[:, 1], y1c, y2c)
                boxes_abs[:, 2] = np.clip(boxes_abs[:, 2], x1c, x2c)
                boxes_abs[:, 3] = np.clip(boxes_abs[:, 3], y1c, y2c)

                new_w_box = boxes_abs[:, 2] - boxes_abs[:, 0]
                new_h_box = boxes_abs[:, 3] - boxes_abs[:, 1]
                new_area = new_w_box * new_h_box

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(orig_area > 1e-6,
                                     new_area / np.maximum(orig_area, 1e-6), 0.0)
                valid = (new_w_box > 2.0) & (new_h_box > 2.0) & (ratio >= self.area_thr)

                boxes_abs = boxes_abs[valid]
                labels = labels[valid]
                if len(boxes_abs) == 0:
                    continue

                boxes_norm = np.clip(boxes_abs / (2 * s), 0, 1).astype(np.float32)
                all_boxes.append(boxes_norm)
                all_labels.append(labels.astype(np.int64))

            else:  # obb
                obboxes, labels, _, _ = self._load_obb_labels(label_path, w_orig, h_orig)
                if len(obboxes) == 0:
                    continue

                obboxes_abs = obboxes * np.array([w_orig, h_orig] * 4, dtype=np.float32)

                orig_pts = obboxes_abs.reshape(-1, 4, 2)
                orig_area = RandomPerspective._polygon_area(orig_pts) * (scale_r ** 2)

                for j in range(4):
                    obboxes_abs[:, 2 * j] = obboxes_abs[:, 2 * j] * scale_r + x_place
                    obboxes_abs[:, 2 * j + 1] = obboxes_abs[:, 2 * j + 1] * scale_r + y_place

                full_pts = obboxes_abs.reshape(-1, 4, 2)
                full_area = RandomPerspective._polygon_area(full_pts)

                clipped = obboxes_abs.reshape(-1, 4, 2).copy()
                np.clip(clipped[..., 0], x1c, x2c, out=clipped[..., 0])
                np.clip(clipped[..., 1], y1c, y2c, out=clipped[..., 1])
                visible_area = RandomPerspective._polygon_area(clipped)

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(orig_area > 1e-6,
                                     full_area / np.maximum(orig_area, 1e-6), 0.0)
                    vis_ratio = np.where(full_area > 1e-6,
                                         visible_area / np.maximum(full_area, 1e-6), 0.0)

                valid = (full_area > 16.0) & \
                        (ratio >= self.area_thr) & \
                        (vis_ratio >= self.visibility_thr)

                if not np.any(valid):
                    continue

                obboxes_abs = obboxes_abs[valid]
                labels = labels[valid]

                obboxes_norm = (obboxes_abs / (2 * s)).astype(np.float32)
                all_obboxes.append(obboxes_norm)
                all_labels.append(labels.astype(np.int64))

        # ---------- 拼接 ----------
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_labels = np.zeros((0,), dtype=np.int64)

        targets = {
            'mode': self.mode,
            'scale': 1.0,
            'pad_left': 0,
            'pad_top': 0,
            'orig_size': (2 * s, 2 * s),
            'flip': False,
            'flip_dir': None,
            'labels': all_labels,
        }

        if self.mode == 'detect':
            if len(all_boxes) > 0:
                targets['boxes'] = np.concatenate(all_boxes, axis=0)
            else:
                targets['boxes'] = np.zeros((0, 4), dtype=np.float32)
        else:
            if len(all_obboxes) > 0:
                targets['obboxes'] = np.concatenate(all_obboxes, axis=0)
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    targets['obboxes'], img_size=2 * s
                )
            else:
                targets['obboxes'] = np.zeros((0, 8), dtype=np.float32)
                cxcywh = np.zeros((0, 4), dtype=np.float32)
                angles = np.zeros((0,), dtype=np.float32)
            targets['cxcywh'] = cxcywh
            targets['angles'] = angles

        # ---------- RandomPerspective：中心裁剪 + 几何变换 ----------
        if self.random_perspective is not None:
            mosaic_img, targets = self.random_perspective(mosaic_img, targets)

            if self.mode == 'obb':
                obboxes_new = targets['obboxes']
                if len(obboxes_new) > 0:
                    cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                        obboxes_new, img_size=self.img_size
                    )
                else:
                    cxcywh = np.zeros((0, 4), dtype=np.float32)
                    angles = np.zeros((0,), dtype=np.float32)
                targets['cxcywh'] = cxcywh
                targets['angles'] = angles
        else:
            mosaic_img = mosaic_img[s // 2: s // 2 + s, s // 2: s // 2 + s]

        # ---------- 翻转 ----------
        mosaic_img, targets = self._random_flip(mosaic_img, targets)

        return mosaic_img, targets

    # ==========================================================
    # ★ MixUp
    #
    # 与另一张随机图像按 lam ~ Beta(alpha, alpha) 加权混合：
    #   img = lam * img1 + (1-lam) * img2
    #   targets = targets1 ∪ targets2
    #
    # 注意：OBB 下混合会产生"幽灵目标"（来自 img2 的框在混合图中
    # 只能看到淡淡痕迹），这是 MixUp 的固有特性。YOLOv8-OBB 默认
    # mixup=0，如果发现效果不好请调小 mixup_prob 或禁用。
    # ==========================================================
    def _apply_mixup(self, img, targets):
        if len(self.img_files) < 2:
            return img, targets

        # ---------- 1. 采样 lam ----------
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # 偏置到 0.5 附近，避免完全覆盖（lam 太小等于丢弃原图）
        lam = max(lam, 1.0 - lam)

        # ---------- 2. 加载另一张图 ----------
        # 只用单图增强，避免递归到 mosaic
        other_idx = random.randint(0, len(self.img_files) - 1)
        img2, targets2 = self._load_single_as_numpy(other_idx)

        # 尺寸必须一致
        if img2.shape[:2] != img.shape[:2]:
            img2 = cv2.resize(img2, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # ---------- 3. 图像混合 ----------
        img_mixed = (img.astype(np.float32) * lam +
                     img2.astype(np.float32) * (1.0 - lam))
        img_mixed = np.clip(img_mixed, 0, 255).astype(np.uint8)

        # ---------- 4. 标签合并 ----------
        targets_mixed = self._merge_targets(targets, targets2)
        targets_mixed['scale'] = 1.0
        targets_mixed['pad_left'] = 0
        targets_mixed['pad_top'] = 0
        targets_mixed['orig_size'] = (self.img_size, self.img_size)
        targets_mixed['flip'] = False
        targets_mixed['flip_dir'] = None

        return img_mixed, targets_mixed

    # ============================================================
    # ★ Copy-Paste
    #
    # 从另一张随机图像中选取若干目标，按旋转框生成 mask 后
    # 把对应像素粘贴到当前图上，并把 GT 添加到当前 targets。
    #
    # OBB 实现要点：
    #   1) 用 cv2.fillPoly 生成旋转框 mask
    #   2) 只在 mask 区域内覆盖像素，边界外保留原图
    #   3) 粘贴的 GT 直接用源图坐标（因为源图与当前图同尺寸）
    # ============================================================
    def _apply_copy_paste(self, img, targets):
        if len(self.img_files) < 2:
            return img, targets

        # ---------- 加载源图 ----------
        src_idx = random.randint(0, len(self.img_files) - 1)
        src_img, src_targets = self._load_single_as_numpy(src_idx)

        if src_img.shape[:2] != img.shape[:2]:
            src_img = cv2.resize(src_img, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        if self.mode == 'obb':
            src_boxes = src_targets.get('obboxes', np.zeros((0, 8), dtype=np.float32))
        else:
            src_boxes = src_targets.get('boxes', np.zeros((0, 4), dtype=np.float32))

        n_src = len(src_boxes)
        if n_src == 0:
            return img, targets

        # ---------- 随机选取要粘贴的目标 ----------
        n_paste = random.randint(1, min(self.copy_paste_max_paste, n_src))
        pick = random.sample(range(n_src), n_paste)

        src_labels = src_targets.get('labels', np.zeros((0,), dtype=np.int64))

        # ---------- 逐个粘贴 ----------
        img_out = img.copy()
        pasted_boxes = []
        pasted_labels = []

        H, W = img_out.shape[:2]

        for i in pick:
            if self.mode == 'obb':
                obbox = src_boxes[i]                          # (8,) 归一化
                pts = (obbox.reshape(4, 2) * np.array([W, H], dtype=np.float32)).astype(np.int32)

                # 快速过滤：粘贴框是否和现有框高度重叠
                # 这里用一个简化判断：如果粘贴框的面积超过原图的 60% 就跳过
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                mask_bool = mask.astype(bool)
                if mask_bool.sum() > 0.6 * H * W:
                    continue

                # 像素粘贴（只在 mask 内覆盖）
                img_out[mask_bool] = src_img[mask_bool]

                pasted_boxes.append(obbox)
                if i < len(src_labels):
                    pasted_labels.append(src_labels[i])
            else:
                box = src_boxes[i]                            # (4,) 归一化 xyxy
                x1 = int(box[0] * W); y1 = int(box[1] * H)
                x2 = int(box[2] * W); y2 = int(box[3] * H)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                img_out[y1:y2, x1:x2] = src_img[y1:y2, x1:x2]

                pasted_boxes.append(box)
                if i < len(src_labels):
                    pasted_labels.append(src_labels[i])

        if len(pasted_boxes) == 0:
            return img, targets

        # ---------- 合并 targets ----------
        pasted_targets = {
            'mode': self.mode,
            'labels': np.array(pasted_labels, dtype=np.int64) if pasted_labels
                      else np.zeros((0,), dtype=np.int64),
        }
        if self.mode == 'obb':
            pasted_targets['obboxes'] = np.array(pasted_boxes, dtype=np.float32).reshape(-1, 8)
            if len(pasted_targets['obboxes']) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    pasted_targets['obboxes'], img_size=self.img_size
                )
            else:
                cxcywh = np.zeros((0, 4), dtype=np.float32)
                angles = np.zeros((0,), dtype=np.float32)
            pasted_targets['cxcywh'] = cxcywh
            pasted_targets['angles'] = angles
        else:
            pasted_targets['boxes'] = np.array(pasted_boxes, dtype=np.float32).reshape(-1, 4)

        targets_merged = self._merge_targets(targets, pasted_targets)
        targets_merged['scale'] = 1.0
        targets_merged['pad_left'] = 0
        targets_merged['pad_top'] = 0
        targets_merged['orig_size'] = (self.img_size, self.img_size)
        targets_merged['flip'] = False
        targets_merged['flip_dir'] = None

        return img_out, targets_merged

    # ==========================================================
    # 合并两个 targets
    # ==========================================================
    def _merge_targets(self, t1, t2):
        merged = {
            'mode': self.mode,
            'labels': np.concatenate([
                t1.get('labels', np.zeros((0,), dtype=np.int64)),
                t2.get('labels', np.zeros((0,), dtype=np.int64)),
            ], axis=0),
            'scale': 1.0,
            'pad_left': 0,
            'pad_top': 0,
            'orig_size': t1.get('orig_size', (self.img_size, self.img_size)),
            'flip': False,
            'flip_dir': None,
        }

        if self.mode == 'detect':
            merged['boxes'] = np.concatenate([
                t1.get('boxes', np.zeros((0, 4), dtype=np.float32)),
                t2.get('boxes', np.zeros((0, 4), dtype=np.float32)),
            ], axis=0)
        else:
            merged['obboxes'] = np.concatenate([
                t1.get('obboxes', np.zeros((0, 8), dtype=np.float32)),
                t2.get('obboxes', np.zeros((0, 8), dtype=np.float32)),
            ], axis=0)
            if len(merged['obboxes']) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    merged['obboxes'], img_size=self.img_size
                )
            else:
                cxcywh = np.zeros((0, 4), dtype=np.float32)
                angles = np.zeros((0,), dtype=np.float32)
            merged['cxcywh'] = cxcywh
            merged['angles'] = angles

        return merged

    # ==========================================================
    # 翻转
    # ==========================================================
    def _apply_hflip(self, img, targets):
        img = cv2.flip(img, 1)
        targets['flip'] = True
        targets['flip_dir'] = 'h'

        if self.mode == 'detect':
            targets['boxes'] = self._flip_detect_boxes_h(targets['boxes'])
        else:
            targets['obboxes'] = self._flip_obb_boxes_h(targets['obboxes'])
            if len(targets['obboxes']) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    targets['obboxes'], img_size=self.img_size
                )
                targets['cxcywh'] = cxcywh
                targets['angles'] = angles
        return img, targets

    def _apply_vflip(self, img, targets):
        img = cv2.flip(img, 0)
        targets['flip'] = True
        targets['flip_dir'] = 'v'

        if self.mode == 'detect':
            targets['boxes'] = self._flip_detect_boxes_v(targets['boxes'])
        else:
            targets['obboxes'] = self._flip_obb_boxes_v(targets['obboxes'])
            if len(targets['obboxes']) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(
                    targets['obboxes'], img_size=self.img_size
                )
                targets['cxcywh'] = cxcywh
                targets['angles'] = angles
        return img, targets

    def _random_flip(self, img, targets):
        if self.fliplr > 0 and random.random() < self.fliplr:
            img, targets = self._apply_hflip(img, targets)
        if self.flipud > 0 and random.random() < self.flipud:
            img, targets = self._apply_vflip(img, targets)
        return img, targets

    # ==========================================================
    # OBB 角度归一化
    # ==========================================================
    def _normalize_obb_angle(self, angle_rad, w, h):
        if self.angle_range == 'pi':
            angle_rad = angle_rad % np.pi
            if w < h:
                w, h = h, w
                angle_rad = (angle_rad + np.pi / 2) % np.pi
        else:
            angle_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
            if angle_rad >= np.pi / 2:
                angle_rad -= np.pi
            elif angle_rad < -np.pi / 2:
                angle_rad += np.pi
            if w < h:
                w, h = h, w
                angle_rad += np.pi / 2
                angle_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
                if angle_rad >= np.pi / 2:
                    angle_rad -= np.pi
                elif angle_rad < -np.pi / 2:
                    angle_rad += np.pi
        return angle_rad, w, h

    def _compute_cxcywh_angle_from_obboxes(self, obboxes, img_size):
        if len(obboxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))

        cxcywh, angles = [], []
        for obbox in obboxes:
            pts = obbox.reshape(4, 2).astype(np.float32) * img_size

            cx = float(0.25 * (pts[0, 0] + pts[1, 0] + pts[2, 0] + pts[3, 0]))
            cy = float(0.25 * (pts[0, 1] + pts[1, 1] + pts[2, 1] + pts[3, 1]))

            v1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
            v2 = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))

            w1 = float(np.linalg.norm(v1))
            h1 = float(np.linalg.norm(v2))

            if w1 >= h1:
                w, h = w1, h1
                angle = float(np.arctan2(v1[1], v1[0]))
            else:
                w, h = h1, w1
                angle = float(np.arctan2(v2[1], v2[0]))

            angle, w, h = self._normalize_obb_angle(angle, w, h)

            cxcywh.append([cx / img_size, cy / img_size,
                           w / img_size, h / img_size])
            angles.append(angle)

        return (np.array(cxcywh, dtype=np.float32),
                np.array(angles, dtype=np.float32))

    # ==========================================================
    # 辅助函数
    # ==========================================================
    def _find_image_path(self, base_name):
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(self.img_dir, f'{base_name}{ext}')
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'Image {base_name} not found')

    def _load_detect_labels(self, label_path):
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                        labels.append(class_id)
                    elif len(parts) == 9:
                        class_id = int(parts[0])
                        pts = np.array(list(map(float, parts[1:])),
                                       dtype=np.float32).reshape(4, 2)
                        boxes.append([float(pts[:, 0].min()), float(pts[:, 1].min()),
                                      float(pts[:, 0].max()), float(pts[:, 1].max())])
                        labels.append(class_id)
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return boxes, labels

    def _load_obb_labels(self, label_path, img_w, img_h):
        obboxes, labels, angles, cxcywh = [], [], [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        class_id = int(parts[0])
                        points = np.array(list(map(float, parts[1:]))).reshape(4, 2)
                    elif len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        points = np.array([
                            [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]
                        ], dtype=np.float32)
                    else:
                        continue

                    pts_pixel = points.copy()
                    pts_pixel[:, 0] *= img_w
                    pts_pixel[:, 1] *= img_h
                    pts_pixel = pts_pixel.astype(np.float32)
                    rect = cv2.minAreaRect(pts_pixel)
                    (cx, cy), (w, h), angle_deg = rect
                    angle = np.deg2rad(angle_deg)

                    angle, w, h = self._normalize_obb_angle(angle, w, h)

                    cxcywh.append([cx / img_w, cy / img_h, w / img_w, h / img_h])
                    angles.append(angle)
                    obboxes.append(points.flatten())
                    labels.append(class_id)

        obboxes = np.array(obboxes, dtype=np.float32) if obboxes else np.zeros((0, 8), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        angles = np.array(angles, dtype=np.float32) if angles else np.zeros((0,), dtype=np.float32)
        cxcywh = np.array(cxcywh, dtype=np.float32) if cxcywh else np.zeros((0, 4), dtype=np.float32)
        return obboxes, labels, angles, cxcywh

    def _transform_detect_boxes(self, boxes, w0, h0, scale, pad_left, pad_top):
        if len(boxes) == 0:
            return boxes
        abs_boxes = boxes * np.array([w0, h0, w0, h0], dtype=np.float32)
        abs_boxes[:, [0, 2]] = abs_boxes[:, [0, 2]] * scale + pad_left
        abs_boxes[:, [1, 3]] = abs_boxes[:, [1, 3]] * scale + pad_top
        boxes = abs_boxes / self.img_size
        return np.clip(boxes, 0, 1)

    def _transform_obb_boxes(self, obboxes, w0, h0, scale, pad_left, pad_top):
        if len(obboxes) == 0:
            return obboxes
        abs_obboxes = obboxes * np.array([w0, h0] * 4, dtype=np.float32)
        for i in range(4):
            abs_obboxes[:, i * 2] = abs_obboxes[:, i * 2] * scale + pad_left
            abs_obboxes[:, i * 2 + 1] = abs_obboxes[:, i * 2 + 1] * scale + pad_top
        obboxes = abs_obboxes / self.img_size
        return obboxes

    def _flip_detect_boxes_h(self, boxes):
        if boxes is None or len(boxes) == 0:
            return boxes
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
        return np.clip(boxes, 0, 1)

    def _flip_obb_boxes_h(self, obboxes):
        if obboxes is None or len(obboxes) == 0:
            return obboxes
        obboxes = np.asarray(obboxes, dtype=np.float32)
        flipped = []
        for obbox in obboxes:
            points = obbox.reshape(4, 2).copy()
            points[:, 0] = 1.0 - points[:, 0]
            flipped.append(np.array([points[0], points[3], points[2], points[1]]).flatten())
        return np.array(flipped, dtype=np.float32)

    def _flip_detect_boxes_v(self, boxes):
        if boxes is None or len(boxes) == 0:
            return boxes
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
        return np.clip(boxes, 0, 1)

    def _flip_obb_boxes_v(self, obboxes):
        if obboxes is None or len(obboxes) == 0:
            return obboxes
        obboxes = np.asarray(obboxes, dtype=np.float32)
        flipped = []
        for obbox in obboxes:
            points = obbox.reshape(4, 2).copy()
            points[:, 1] = 1.0 - points[:, 1]
            flipped.append(np.array([points[0], points[3], points[2], points[1]]).flatten())
        return np.array(flipped, dtype=np.float32)

    # ==========================================================
    # 调试可视化
    # ==========================================================
    def _debug_visualize(self, img, base_name, target):
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        h, w = vis.shape[:2]
        if self.mode == 'obb':
            obboxes = target['obboxes']
            if isinstance(obboxes, torch.Tensor):
                obboxes = obboxes.numpy()
            for obbox in obboxes:
                pts = (obbox * np.array([w, h] * 4)).reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        else:
            boxes = target['boxes']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.numpy()
            for box in boxes:
                x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(np.int32)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.debug_dir, f'{base_name}.jpg'), vis)