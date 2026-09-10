import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import random
from utils import letterbox


class YOLODataset(Dataset):
    def __init__(self, root, img_size=640, split='train',
                 use_flip=True, use_color_jitter=False, color_jitter_params=None,
                 mean=[0, 0, 0], std=[1, 1, 1], mode='detect', angle_range='pi',
                 debug_vis=False,
                 mosaic_prob=0.0, mosaic_disable_epoch=None):
        """
        Args:
            mosaic_prob: 训练时使用 Mosaic 的概率（0~1）
            mosaic_disable_epoch: 从该 epoch 开始禁用（None 表示始终启用）
        """
        self.root = root
        self.img_size = img_size
        self.split = split
        self.use_flip = use_flip and split == 'train'
        self.use_color_jitter = use_color_jitter and split == 'train'
        self.mode = mode
        self.angle_range = angle_range
        self.debug_vis = debug_vis
        self.mosaic_prob = mosaic_prob if split == 'train' else 0.0
        self.mosaic_disable_epoch = mosaic_disable_epoch
        self.epoch = 0

        self.img_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)

        self.img_files = []
        for f in os.listdir(self.img_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                self.img_files.append(os.path.splitext(f)[0])

        if not self.img_files:
            raise RuntimeError(f'在 {self.img_dir} 中未找到任何图片')

        if self.use_color_jitter and color_jitter_params is not None:
            self.color_jitter = T.ColorJitter(*color_jitter_params)
        else:
            self.color_jitter = None

        self.basic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        if self.debug_vis:
            self.debug_dir = os.path.join('./debug_vis', self.split)
            os.makedirs(self.debug_dir, exist_ok=True)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        use_mosaic = (
            self.split == 'train'
            and self.mosaic_prob > 0
            and (self.mosaic_disable_epoch is None or self.epoch < self.mosaic_disable_epoch)
            and random.random() < self.mosaic_prob
        )
        if use_mosaic:
            return self._get_mosaic_item(idx)
        else:
            return self._get_single_item(idx)

    # -------------------- 单图加载 --------------------
    def _get_single_item(self, idx):
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
            angles = None
            cxcywh = None
        else:
            obboxes, labels, angles, cxcywh = self._load_obb_labels(label_path, w0, h0)
            boxes = None

        img_padded, scale, (pad_left, pad_top), _ = letterbox(
            img_rgb, new_shape=(self.img_size, self.img_size), auto=False, stride=32
        )

        if self.mode == 'detect' and boxes is not None and len(boxes) > 0:
            boxes = self._transform_detect_boxes(boxes, w0, h0, scale, pad_left, pad_top)
        elif self.mode == 'obb' and obboxes is not None and len(obboxes) > 0:
            obboxes = self._transform_obb_boxes(obboxes, w0, h0, scale, pad_left, pad_top)
            if len(obboxes) > 0:
                cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(obboxes)

        flip = False
        if self.use_flip and random.random() < 0.5:
            flip = True
            img_padded = cv2.flip(img_padded, 1)
            if self.mode == 'detect' and boxes is not None and len(boxes) > 0:
                boxes = self._flip_detect_boxes(boxes)
            elif self.mode == 'obb' and obboxes is not None and len(obboxes) > 0:
                obboxes = self._flip_obb_boxes(obboxes)
                if len(obboxes) > 0:
                    cxcywh, angles = self._compute_cxcywh_angle_from_obboxes(obboxes)

        img_pil = Image.fromarray(img_padded)
        if self.use_color_jitter and self.color_jitter is not None:
            img_pil = self.color_jitter(img_pil)

        img_tensor = self.basic_transform(img_pil)

        target = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_size': (w0, h0),
            'flip': flip,
            'mode': self.mode,
        }

        if self.mode == 'detect':
            target['boxes'] = torch.from_numpy(boxes) if boxes is not None else torch.zeros((0, 4))
            target['labels'] = torch.from_numpy(labels) if labels is not None else torch.zeros((0,), dtype=torch.int64)
        else:
            target['obboxes'] = torch.from_numpy(obboxes) if obboxes is not None else torch.zeros((0, 8))
            target['labels'] = torch.from_numpy(labels) if labels is not None else torch.zeros((0,), dtype=torch.int64)
            target['angles'] = torch.from_numpy(angles) if angles is not None else torch.zeros((0,))
            target['cxcywh'] = torch.from_numpy(cxcywh) if cxcywh is not None else torch.zeros((0, 4))

        if self.debug_vis and idx < 5:
            self._debug_visualize(img_padded, base_name, boxes, obboxes)

        return img_tensor, target

    # -------------------- Mosaic4 增强 --------------------
    def _get_mosaic_item(self, idx):
        s = self.img_size
        indices = [idx] + random.sample(range(len(self.img_files)), 3)
        random.shuffle(indices)

        xc = random.randint(-s // 2, s * 3 // 2)
        yc = random.randint(-s // 2, s * 3 // 2)

        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        all_boxes = []      # detect
        all_obboxes = []    # obb (顶点)
        all_labels = []

        # 面积保留阈值（保留裁剪后面积 >= 阈值 * 原始面积(画布坐标系)）
        AREA_THRESH = 0.5

        for i, sub_idx in enumerate(indices):
            base_name = self.img_files[sub_idx]
            img_path = self._find_image_path(base_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            scale = random.uniform(0.5, 2.0)
            max_scale = min((s * 2) / w_orig, (s * 2) / h_orig)
            scale = min(scale, max_scale)
            new_w = max(1, int(w_orig * scale))
            new_h = max(1, int(h_orig * scale))
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 放置位置
            if i == 0:      # 左上
                x_place, y_place = xc - new_w, yc - new_h
            elif i == 1:    # 右上
                x_place, y_place = xc, yc - new_h
            elif i == 2:    # 左下
                x_place, y_place = xc - new_w, yc
            else:           # 右下
                x_place, y_place = xc, yc

            # 有效区域（与画布交集）
            x1_canvas = max(0, x_place)
            y1_canvas = max(0, y_place)
            x2_canvas = min(s * 2, x_place + new_w)
            y2_canvas = min(s * 2, y_place + new_h)
            if x2_canvas <= x1_canvas or y2_canvas <= y1_canvas:
                continue

            # 源图对应区域
            x1_src = x1_canvas - x_place
            y1_src = y1_canvas - y_place
            x2_src = x2_canvas - x_place
            y2_src = y2_canvas - y_place
            x1_src = max(0, x1_src)
            y1_src = max(0, y1_src)
            x2_src = min(new_w, x2_src)
            y2_src = min(new_h, y2_src)
            if x2_src <= x1_src or y2_src <= y1_src:
                continue

            # 粘贴
            src_slice = img_resized[y1_src:y2_src, x1_src:x2_src]
            mosaic_img[y1_canvas:y2_canvas, x1_canvas:x2_canvas] = src_slice

            # ---- 加载标签并变换 ----
            label_path = os.path.join(self.label_dir, f'{base_name}.txt')
            if self.mode == 'detect':
                boxes, labels = self._load_detect_labels(label_path)
                if len(boxes) > 0:
                    # 原始框（归一化 -> 像素坐标）
                    boxes_abs_orig = boxes * np.array([w_orig, h_orig, w_orig, h_orig], dtype=np.float32)
                    # 原始面积（像素）
                    orig_areas = (boxes_abs_orig[:, 2] - boxes_abs_orig[:, 0]) * \
                                 (boxes_abs_orig[:, 3] - boxes_abs_orig[:, 1])
                    # 画布坐标系下的原始面积（缩放后）
                    orig_areas_canvas = orig_areas * (scale * scale)

                    # 缩放、平移
                    boxes_abs = boxes_abs_orig.copy()
                    boxes_abs[:, [0, 2]] *= scale
                    boxes_abs[:, [1, 3]] *= scale
                    boxes_abs[:, [0, 2]] += x_place
                    boxes_abs[:, [1, 3]] += y_place

                    # 裁剪到有效区域
                    boxes_abs[:, 0] = np.clip(boxes_abs[:, 0], x1_canvas, x2_canvas)
                    boxes_abs[:, 1] = np.clip(boxes_abs[:, 1], y1_canvas, y2_canvas)
                    boxes_abs[:, 2] = np.clip(boxes_abs[:, 2], x1_canvas, x2_canvas)
                    boxes_abs[:, 3] = np.clip(boxes_abs[:, 3], y1_canvas, y2_canvas)

                    # 裁剪后面积（画布坐标）
                    crop_areas = (boxes_abs[:, 2] - boxes_abs[:, 0]) * \
                                 (boxes_abs[:, 3] - boxes_abs[:, 1])

                    # 有效条件：面积 > 0 且 裁剪后面积 >= 阈值 * 原始面积(画布)
                    valid = (crop_areas > 0) & (crop_areas >= AREA_THRESH * orig_areas_canvas)
                    boxes_abs = boxes_abs[valid]
                    labels = labels[valid]
                    if len(boxes_abs) > 0:
                        boxes_norm = boxes_abs / (s * 2)
                        boxes_norm = np.clip(boxes_norm, 0, 1)
                        all_boxes.extend(boxes_norm.tolist())
                        all_labels.extend(labels.tolist())
            else:  # obb
                obboxes, labels, _, _ = self._load_obb_labels(label_path, w_orig, h_orig)
                if len(obboxes) > 0:
                    # 原始顶点（归一化 -> 像素坐标）
                    obboxes_abs_orig = obboxes * np.array([w_orig, h_orig] * 4, dtype=np.float32)
                    # 计算每个框的原始面积（像素）
                    orig_areas = []
                    for pts in obboxes_abs_orig:
                        pts_reshape = pts.reshape(4, 2).astype(np.float32)
                        area = cv2.contourArea(pts_reshape)
                        orig_areas.append(area)
                    orig_areas = np.array(orig_areas)
                    # 画布坐标系下的原始面积
                    orig_areas_canvas = orig_areas * (scale * scale)

                    # 变换：缩放 + 平移
                    obboxes_abs = obboxes_abs_orig.copy()
                    for j in range(4):
                        obboxes_abs[:, j * 2] *= scale
                        obboxes_abs[:, j * 2 + 1] *= scale
                        obboxes_abs[:, j * 2] += x_place
                        obboxes_abs[:, j * 2 + 1] += y_place

                    # 将顶点 clip 到有效区域
                    for j in range(4):
                        obboxes_abs[:, j * 2] = np.clip(obboxes_abs[:, j * 2], x1_canvas, x2_canvas)
                        obboxes_abs[:, j * 2 + 1] = np.clip(obboxes_abs[:, j * 2 + 1], y1_canvas, y2_canvas)

                    # 计算裁剪后面积（画布坐标）
                    crop_areas = []
                    for b in obboxes_abs:
                        pts = b.reshape(4, 2).astype(np.float32)
                        area = cv2.contourArea(pts)
                        crop_areas.append(area)
                    crop_areas = np.array(crop_areas)

                    valid = (crop_areas > 0) & (crop_areas >= AREA_THRESH * orig_areas_canvas)
                    if np.any(valid):
                        obboxes_abs = obboxes_abs[valid]
                        labels = labels[valid]
                        obboxes_norm = obboxes_abs / (s * 2)
                        obboxes_norm = np.clip(obboxes_norm, 0, 1)
                        all_obboxes.extend(obboxes_norm.tolist())
                        all_labels.extend(labels.tolist())

        # ---- 缩放大图到目标尺寸 ----
        final_img = cv2.resize(mosaic_img, (s, s), interpolation=cv2.INTER_LINEAR)

        # ---- 翻转增强（先做几何变换，再整理标签） ----
        flip = False
        if self.use_flip and random.random() < 0.5:
            flip = True
            final_img = cv2.flip(final_img, 1)
            if self.mode == 'detect' and len(all_boxes) > 0:
                all_boxes = np.array(all_boxes, dtype=np.float32)
                all_boxes = self._flip_detect_boxes(all_boxes)
            elif self.mode == 'obb' and len(all_obboxes) > 0:
                all_obboxes = np.array(all_obboxes, dtype=np.float32)
                all_obboxes = self._flip_obb_boxes(all_obboxes)

        # ---- 整理标签（基于最终几何，重新计算 OBB 的 cxcywh 和 angles） ----
        if self.mode == 'detect':
            if len(all_boxes) > 0:
                all_boxes = np.array(all_boxes, dtype=np.float32)
                all_boxes = np.clip(all_boxes, 0, 1)
            else:
                all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_labels = np.array(all_labels, dtype=np.int64) if all_labels else np.zeros((0,), dtype=np.int64)
            all_obboxes = all_angles = all_cxcywh = None
        else:
            if len(all_obboxes) > 0:
                all_obboxes = np.array(all_obboxes, dtype=np.float32)
                all_obboxes = np.clip(all_obboxes, 0, 1)
                all_labels = np.array(all_labels, dtype=np.int64)
                # 根据最终顶点重新计算 cxcywh 和 angles
                all_cxcywh, all_angles = self._compute_cxcywh_angle_from_obboxes(all_obboxes)
            else:
                all_obboxes = np.zeros((0, 8), dtype=np.float32)
                all_cxcywh = np.zeros((0, 4), dtype=np.float32)
                all_angles = np.zeros((0,), dtype=np.float32)
                all_labels = np.zeros((0,), dtype=np.int64)
            all_boxes = None

        # ---- 颜色抖动 ----
        img_pil = Image.fromarray(final_img)
        if self.use_color_jitter and self.color_jitter is not None:
            img_pil = self.color_jitter(img_pil)

        img_tensor = self.basic_transform(img_pil)

        target = {
            'scale': 1.0,
            'pad_left': 0,
            'pad_top': 0,
            'orig_size': (s, s),
            'flip': flip,
            'mode': self.mode,
        }

        if self.mode == 'detect':
            target['boxes'] = torch.from_numpy(all_boxes)
            target['labels'] = torch.from_numpy(all_labels)
        else:
            target['obboxes'] = torch.from_numpy(all_obboxes)
            target['labels'] = torch.from_numpy(all_labels)
            target['angles'] = torch.from_numpy(all_angles)
            target['cxcywh'] = torch.from_numpy(all_cxcywh)

        # 调试可视化
        if self.debug_vis:
            self._debug_visualize_mosaic(final_img, target, idx)
            self._debug_save_mosaic_big(mosaic_img,
                                        all_boxes if self.mode == 'detect' else all_obboxes,
                                        idx)

        return img_tensor, target

    # -------------------- 辅助函数 --------------------
    def _find_image_path(self, base_name):
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(self.img_dir, f'{base_name}{ext}')
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'Image {base_name} not found')

    def _load_detect_labels(self, label_path):
        """
        加载 detect 标签，兼容两种格式：
          1. 标准 YOLO detect: class cx cy w h              (5 列，归一化)
          2. YOLO OBB:        class x1 y1 x2 y2 x3 y3 x4 y4  (9 列，归一化)
             遇到 9 列时，自动转换为水平外接框 [x1, y1, x2, y2]（归一化）

        返回:
            boxes:  (N, 4) xyxy 归一化
            labels: (N,)
        """
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # ---- 标准 detect 格式 ----
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
                    elif len(parts) == 9:
                        # ---- OBB 格式：转成水平外接框 ----
                        class_id = int(parts[0])
                        pts = np.array(list(map(float, parts[1:])),
                                       dtype=np.float32).reshape(4, 2)
                        x1 = float(pts[:, 0].min())
                        y1 = float(pts[:, 1].min())
                        x2 = float(pts[:, 0].max())
                        y2 = float(pts[:, 1].max())
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return boxes, labels

    def _load_obb_labels(self, label_path, img_w, img_h):
        """
        加载 OBB 标签，兼容两种格式：
          1. YOLO OBB:  class x1 y1 x2 y2 x3 y3 x4 y4  (9 列，归一化)
          2. 标准 YOLO detect: class cx cy w h          (5 列，归一化)
             遇到 5 列时，转成轴对齐 4 顶点，angle=0

        返回:
            obboxes: (N, 8) 归一化顶点
            labels:  (N,)
            angles:  (N,)
            cxcywh:  (N, 4) 归一化
        """
        obboxes = []
        labels = []
        angles = []
        cxcywh = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 9:
                        # ---- OBB 格式 ----
                        class_id = int(parts[0])
                        points = np.array(list(map(float, parts[1:]))).reshape(4, 2)
                    elif len(parts) == 5:
                        # ---- detect 格式：转成轴对齐 4 顶点 ----
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        points = np.array([
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
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
                    if self.angle_range == 'pi':
                        if angle < 0:
                            angle += np.pi
                        if w < h:
                            w, h = h, w
                            angle += np.pi / 2
                            if angle >= np.pi:
                                angle -= np.pi
                    else:
                        angle = np.arctan2(np.sin(angle), np.cos(angle))
                        if angle > np.pi / 2:
                            angle -= np.pi
                        elif angle < -np.pi / 2:
                            angle += np.pi
                        if w < h:
                            w, h = h, w
                            angle += np.pi / 2
                            angle = np.arctan2(np.sin(angle), np.cos(angle))
                            if angle > np.pi / 2:
                                angle -= np.pi
                            elif angle < -np.pi / 2:
                                angle += np.pi
                    cx = cx / img_w
                    cy = cy / img_h
                    w = w / img_w
                    h = h / img_h
                    obbox = points.flatten()
                    obboxes.append(obbox)
                    labels.append(class_id)
                    angles.append(angle)
                    cxcywh.append([cx, cy, w, h])

        obboxes = np.array(obboxes, dtype=np.float32) if obboxes else np.zeros((0, 8), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        angles = np.array(angles, dtype=np.float32) if angles else np.zeros((0,), dtype=np.float32)
        cxcywh = np.array(cxcywh, dtype=np.float32) if cxcywh else np.zeros((0, 4), dtype=np.float32)
        return obboxes, labels, angles, cxcywh

    def _compute_cxcywh_angle_from_obboxes(self, obboxes):
        cxcywh = []
        angles = []
        for obbox in obboxes:
            points = obbox.reshape(4, 2)
            pts_pixel = points * self.img_size
            pts_pixel = pts_pixel.astype(np.float32)
            rect = cv2.minAreaRect(pts_pixel)
            (cx, cy), (w, h), angle_deg = rect
            angle = np.deg2rad(angle_deg)
            if self.angle_range == 'pi':
                if angle < 0:
                    angle += np.pi
                if w < h:
                    w, h = h, w
                    angle += np.pi / 2
                    if angle >= np.pi:
                        angle -= np.pi
            else:
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                if angle > np.pi / 2:
                    angle -= np.pi
                elif angle < -np.pi / 2:
                    angle += np.pi
                if w < h:
                    w, h = h, w
                    angle += np.pi / 2
                    angle = np.arctan2(np.sin(angle), np.cos(angle))
                    if angle > np.pi / 2:
                        angle -= np.pi
                    elif angle < -np.pi / 2:
                        angle += np.pi
            cx = cx / self.img_size
            cy = cy / self.img_size
            w = w / self.img_size
            h = h / self.img_size
            cxcywh.append([cx, cy, w, h])
            angles.append(angle)
        return np.array(cxcywh, dtype=np.float32), np.array(angles, dtype=np.float32)

    def _compute_cxcywh_angle_from_obboxes_given_size(self, obboxes, size):
        cxcywh = []
        angles = []
        for obbox in obboxes:
            points = obbox.reshape(4, 2)
            pts_pixel = points * size
            pts_pixel = pts_pixel.astype(np.float32)
            rect = cv2.minAreaRect(pts_pixel)
            (cx, cy), (w, h), angle_deg = rect
            angle = np.deg2rad(angle_deg)
            if self.angle_range == 'pi':
                if angle < 0:
                    angle += np.pi
                if w < h:
                    w, h = h, w
                    angle += np.pi / 2
                    if angle >= np.pi:
                        angle -= np.pi
            else:
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                if angle > np.pi / 2:
                    angle -= np.pi
                elif angle < -np.pi / 2:
                    angle += np.pi
                if w < h:
                    w, h = h, w
                    angle += np.pi / 2
                    angle = np.arctan2(np.sin(angle), np.cos(angle))
                    if angle > np.pi / 2:
                        angle -= np.pi
                    elif angle < -np.pi / 2:
                        angle += np.pi
            cx = cx / size
            cy = cy / size
            w = w / size
            h = h / size
            cxcywh.append([cx, cy, w, h])
            angles.append(angle)
        return np.array(cxcywh, dtype=np.float32), np.array(angles, dtype=np.float32)

    def _transform_detect_boxes(self, boxes, w0, h0, scale, pad_left, pad_top):
        if len(boxes) == 0:
            return boxes
        abs_boxes = boxes * np.array([w0, h0, w0, h0], dtype=np.float32)
        abs_boxes[:, [0, 2]] = abs_boxes[:, [0, 2]] * scale + pad_left
        abs_boxes[:, [1, 3]] = abs_boxes[:, [1, 3]] * scale + pad_top
        boxes = abs_boxes / self.img_size
        boxes = np.clip(boxes, 0, 1)
        return boxes

    def _transform_obb_boxes(self, obboxes, w0, h0, scale, pad_left, pad_top):
        if len(obboxes) == 0:
            return obboxes
        abs_obboxes = obboxes * np.array([w0, h0] * 4, dtype=np.float32)
        for i in range(4):
            abs_obboxes[:, i * 2] = abs_obboxes[:, i * 2] * scale + pad_left
            abs_obboxes[:, i * 2 + 1] = abs_obboxes[:, i * 2 + 1] * scale + pad_top
        obboxes = abs_obboxes / self.img_size
        obboxes = np.clip(obboxes, 0, 1)
        return obboxes

    def _flip_detect_boxes(self, boxes):
        if len(boxes) == 0:
            return boxes
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
        boxes = np.clip(boxes, 0, 1)
        return boxes

    def _flip_obb_boxes(self, obboxes):
        if len(obboxes) == 0:
            return obboxes
        flipped = []
        for obbox in obboxes:
            points = obbox.reshape(4, 2)
            points[:, 0] = 1.0 - points[:, 0]
            flipped_points = np.array([
                points[0],
                points[3],
                points[2],
                points[1]
            ])
            flipped.append(flipped_points.flatten())
        obboxes = np.array(flipped, dtype=np.float32)
        obboxes = np.clip(obboxes, 0, 1)
        return obboxes

    # -------------------- 调试可视化 --------------------
    def _debug_visualize(self, img, base_name, boxes, obboxes):
        vis_img = img.copy()
        h, w = vis_img.shape[:2]
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        if self.mode == 'obb' and obboxes is not None and len(obboxes) > 0:
            for obbox in obboxes:
                pts = (obbox * np.array([w, h] * 4)).reshape(4, 2).astype(np.int32)
                cv2.polylines(vis_img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        elif self.mode == 'detect' and boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(np.int32)
                cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_path = os.path.join(self.debug_dir, f'{base_name}.jpg')
        cv2.imwrite(save_path, vis_img_bgr)
        print(f"[Debug] 保存单图调试: {save_path}")

    def _debug_visualize_mosaic(self, img, target, idx):
        """保存最终缩放图像并绘制标签"""
        h, w = img.shape[:2]
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.mode == 'detect':
            boxes = target['boxes'].numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
            for box in boxes:
                x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(np.int32)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            obboxes = target['obboxes'].numpy() if isinstance(target['obboxes'], torch.Tensor) else target['obboxes']
            for obbox in obboxes:
                pts = (obbox * np.array([w, h] * 4)).reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        save_path = os.path.join(self.debug_dir, f'mosaic_{idx}.jpg')
        cv2.imwrite(save_path, vis)
        print(f"[Debug] 保存 Mosaic 最终图: {save_path}")

    def _debug_save_mosaic_big(self, mosaic_img, labels, idx):
        """保存拼接大图并绘制标签（画布坐标）"""
        vis = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
        size_big = mosaic_img.shape[0]
        if self.mode == 'detect':
            for box in labels:
                box_canvas = np.array(box) * size_big
                x1, y1, x2, y2 = box_canvas.astype(np.int32)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            for obbox in labels:
                pts = (np.array(obbox).reshape(4, 2) * size_big).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        save_path = os.path.join(self.debug_dir, f'mosaic_big_{idx}.jpg')
        cv2.imwrite(save_path, vis)
        print(f"[Debug] 保存 Mosaic 大图: {save_path}")