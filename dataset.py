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
                 use_flip=True, use_color_jitter=False, color_jitter_params=None, mean=[0,0,0], std=[1,1,1]):
        self.root = root
        self.img_size = img_size
        self.split = split
        self.use_flip = use_flip and split == 'train'
        self.use_color_jitter = use_color_jitter and split == 'train'

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
            T.Normalize(mean=mean,
                        std=std)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        base_name = self.img_files[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, f'{base_name}.jpg')
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(self.img_dir, f'{base_name}{ext}')
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'无法读取图片: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        # 加载标签 (YOLO格式 cx, cy, w, h 归一化)
        label_path = os.path.join(self.label_dir, f'{base_name}.txt')
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # ---------- Letterbox ----------
        img_padded, scale, (pad_left, pad_top), _ = letterbox(
            img_rgb, new_shape=(self.img_size, self.img_size), auto=False, stride=32
        )

        # 调整标签到填充后的图像坐标（归一化）
        if boxes.shape[0] > 0:
            abs_boxes = boxes * np.array([w0, h0, w0, h0], dtype=np.float32)
            abs_boxes[:, [0,2]] = abs_boxes[:, [0,2]] * scale + pad_left
            abs_boxes[:, [1,3]] = abs_boxes[:, [1,3]] * scale + pad_top
            boxes = abs_boxes / self.img_size
            boxes = np.clip(boxes, 0, 1)

        # ---------- 随机水平翻转 ----------
        flip = False
        if self.use_flip and random.random() < 0.5:
            flip = True
            img_padded = cv2.flip(img_padded, 1)
            if boxes.shape[0] > 0:
                boxes[:, [0,2]] = 1.0 - boxes[:, [2,0]]
                boxes = np.clip(boxes, 0, 1)

        img_pil = Image.fromarray(img_padded)
        if self.use_color_jitter and self.color_jitter is not None:
            img_pil = self.color_jitter(img_pil)

        img_tensor = self.basic_transform(img_pil)

        target = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels),
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_size': (w0, h0),
            'flip': flip
        }
        return img_tensor, target