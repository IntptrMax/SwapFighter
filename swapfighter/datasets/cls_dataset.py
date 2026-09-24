import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    from datasets import Dataset as HFDataset  # noqa
    from datasets import DatasetDict as HFDatasetDict  # noqa
    HAS_HF = True
except ImportError:
    HAS_HF = False
    HFDataset = None
    HFDatasetDict = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ============================================================
# ★ 标签行解析器：同时兼容新旧两种格式
# ============================================================
def _parse_label_line(line, num_classes):
    """
    解析一行标签。

    返回 (fname, onehot) 或 None（无法解析）。

    支持：
        - 新格式：filename\\t[0, 14, 26]
        - 新格式：filename [0, 14, 26]      （兼容空格分隔）
        - 旧格式：filename 0 0 1 0 1 ...    （C 个 one-hot 值）
    """
    line = line.strip()
    if not line:
        return None

    # ---------- 分离文件名与标签字符串 ----------
    if '\t' in line:
        fname, label_str = line.split('\t', 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            return None
        fname, label_str = parts[0], parts[1]

    fname = fname.lstrip('\ufeff').strip()
    label_str = label_str.strip()

    # ---------- 新格式：[0, 14, 26] 或 [] ----------
    if label_str.startswith('[') and label_str.endswith(']'):
        inner = label_str[1:-1].strip()
        if inner == '':
            label_ids = []
        else:
            try:
                label_ids = [
                    int(x.strip())
                    for x in inner.split(',')
                    if x.strip() != ''
                ]
            except ValueError:
                return None

        onehot = np.zeros(num_classes, dtype=np.float32)
        for cid in label_ids:
            if 0 <= cid < num_classes:
                onehot[cid] = 1.0
        return fname, onehot

    # ---------- 旧格式：one-hot ----------
    vals = label_str.split()
    if len(vals) != num_classes:
        return None
    try:
        onehot = np.array([float(x) for x in vals], dtype=np.float32)
    except ValueError:
        return None
    onehot = (onehot > 0.5).astype(np.float32)
    return fname, onehot


# ============================================================
# 数据源 1：磁盘 txt 标签文件
# ============================================================
class MultiLabelClassificationDataset(Dataset):
    def __init__(self, root, label_file, img_subdir, num_classes,
                 img_size=224,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 split='train',
                 use_flip=True, fliplr=0.5,
                 use_color_aug=True,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 use_rrc=True, rrc_scale=(0.5, 1.0), rrc_ratio=(3 / 4, 4 / 3),
                 use_mixup=False, mixup_prob=0.0, mixup_alpha=0.2,
                 check_files=True,
                 max_retry=20,
                 class_names=None,
                 debug_vis=False):
        self.root = root
        self.img_dir = os.path.join(root, img_subdir)
        self.label_file = os.path.join(root, label_file)
        self.img_size = int(img_size)
        self.num_classes = int(num_classes)
        self.split = split
        self.use_flip = use_flip and split == 'train'
        self.fliplr = fliplr if self.use_flip else 0.0
        self.use_color_aug = use_color_aug and split == 'train'
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.use_rrc = use_rrc and split == 'train'
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio
        self.use_mixup = use_mixup and split == 'train'
        self.mixup_prob = mixup_prob if self.use_mixup else 0.0
        self.mixup_alpha = mixup_alpha
        self.max_retry = int(max_retry)
        self.debug_vis = debug_vis
        self.epoch = 0
        self._label_id_to_name = {}

        # ---------- 类名 ----------
        if class_names is not None:
            assert len(class_names) == self.num_classes, \
                f'class_names 长度({len(class_names)}) != num_classes({self.num_classes})'
            self.class_names = list(class_names)
        else:
            # 未提供类名时使用占位符，由调用方（trainer）负责解析真实类名
            self.class_names = [f'cls_{i}' for i in range(self.num_classes)]

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f'图像目录不存在: {self.img_dir}')
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f'标签文件不存在: {self.label_file}')

        # ---------- 加载标签 ----------
        self.samples = []
        n_total_lines = 0
        n_mismatch = 0
        with open(self.label_file, 'r', encoding='utf-8-sig') as f:
            for line_no, line in enumerate(f, 1):
                n_total_lines += 1
                result = _parse_label_line(line, self.num_classes)
                if result is None:
                    n_mismatch += 1
                    if n_mismatch <= 3:
                        print(f'[MultiLabelClsDataset] ⚠️ 行 {line_no} 无法解析: '
                              f'{line.strip()[:80]}')
                    continue
                self.samples.append(result)

        if n_mismatch > 3:
            print(f'[MultiLabelClsDataset] 共跳过 {n_mismatch} 行无法解析')
        if len(self.samples) == 0:
            raise RuntimeError(f'标签文件 {self.label_file} 未解析出有效行')

        print(f'[MultiLabelClsDataset] 标签文件共 {n_total_lines} 行，'
              f'解析有效 {len(self.samples)} 行')

        # ---------- 预过滤不存在的图像 ----------
        if check_files:
            print(f'[MultiLabelClsDataset] 检查文件是否存在...')
            valid_samples = []
            missing_samples = []
            for fname, onehot in self.samples:
                path = os.path.join(self.img_dir, fname)
                if os.path.isfile(path):
                    valid_samples.append((fname, onehot))
                else:
                    missing_samples.append(fname)

            n_missing = len(missing_samples)
            if n_missing > 0:
                print(f'[MultiLabelClsDataset] ⚠️ {n_missing} 张图像不存在，已跳过')
                for m in missing_samples[:5]:
                    print(f'    missing: {m}')
                miss_path = os.path.join(self.root, f'missing_{split}.txt')
                try:
                    with open(miss_path, 'w', encoding='utf-8') as mf:
                        for m in missing_samples:
                            mf.write(m + '\n')
                    print(f'  缺失清单已写入: {miss_path}')
                except Exception as e:
                    print(f'  写入缺失清单失败: {e}')
            self.samples = valid_samples

        if len(self.samples) == 0:
            raise RuntimeError('过滤后无有效样本')

        # ---------- 统计 ----------
        all_oh = np.stack([s[1] for s in self.samples], axis=0)
        n_empty = int((all_oh.sum(1) == 0).sum())
        pos_per_img = all_oh.sum(1).mean()
        pos_per_class = all_oh.sum(0).astype(np.int64)
        n_zero_class = int((pos_per_class == 0).sum())
        print(f'[MultiLabelClsDataset] split={split} | '
              f'images={len(self.samples)} | classes={self.num_classes} | '
              f'平均标签数/图={pos_per_img:.2f} | '
              f'空标签图={n_empty} | 零样本类={n_zero_class}')

        # ---------- transform ----------
        norm = T.Normalize(mean=mean, std=std)
        if split == 'train':
            ops = []
            if self.use_rrc:
                ops.append(T.RandomResizedCrop(
                    self.img_size, scale=self.rrc_scale, ratio=self.rrc_ratio,
                    interpolation=T.InterpolationMode.BICUBIC))
            else:
                ops.append(T.Resize(self.img_size,
                                    interpolation=T.InterpolationMode.BICUBIC))
            if self.use_flip:
                ops.append(T.RandomHorizontalFlip(p=self.fliplr))
            if self.use_color_aug:
                ops.append(T.ColorJitter(
                    brightness=self.hsv_v, contrast=self.hsv_v,
                    saturation=self.hsv_s, hue=self.hsv_h))
            ops.extend([T.ToTensor(), norm])
            self.transform = T.Compose(ops)
        else:
            self.transform = T.Compose([
                T.Resize(int(self.img_size * 1.14),
                         interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.img_size),
                T.ToTensor(),
                norm,
            ])

        if self.debug_vis:
            self.debug_dir = os.path.join('./debug_vis', f'cls_{split}')
            os.makedirs(self.debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def _read_image(self, fname):
        path = os.path.join(self.img_dir, fname)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f'无法读取图片: {path}')
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _get_raw(self, idx):
        fname, onehot = self.samples[idx]
        try:
            img = self._read_image(fname)
            return img, onehot, fname
        except Exception:
            return None

    def __getitem__(self, idx):
        img, onehot = None, None
        for attempt in range(self.max_retry):
            cur_idx = idx if attempt == 0 else random.randint(0, len(self.samples) - 1)
            out = self._get_raw(cur_idx)
            if out is not None:
                img, onehot, _ = out
                break

        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            onehot = np.zeros(self.num_classes, dtype=np.float32)

        img_t = self.transform(Image.fromarray(img))
        label_t = torch.from_numpy(onehot)

        if self.use_mixup and random.random() < self.mixup_prob:
            for _ in range(self.max_retry):
                j = random.randint(0, len(self.samples) - 1)
                out2 = self._get_raw(j)
                if out2 is None:
                    continue
                img2, onehot2, _ = out2
                img_t2 = self.transform(Image.fromarray(img2))
                label_t2 = torch.from_numpy(onehot2)
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                img_t = lam * img_t + (1.0 - lam) * img_t2
                label_t = lam * label_t + (1.0 - lam) * label_t2
                break

        return img_t, label_t


# ============================================================
# 数据源 HuggingFace datasets
# ============================================================
class HFMultiLabelClassificationDataset(Dataset):
    def __init__(self, hf_dataset,
                 num_classes=None,
                 class_names=None,
                 img_size=224,
                 split='train',
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 use_flip=True, fliplr=0.5,
                 use_color_aug=True,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 use_rrc=True, rrc_scale=(0.5, 1.0), rrc_ratio=(3 / 4, 4 / 3),
                 use_mixup=False, mixup_prob=0.0, mixup_alpha=0.2,
                 max_retry=5,
                 scan_max_samples=20000,
                 debug_vis=False):
        super().__init__()
        if not HAS_HF:
            raise ImportError("请先安装: pip install datasets")

        self.ds = hf_dataset
        self.split = split
        self.img_size = int(img_size)
        self.max_retry = int(max_retry)
        self.debug_vis = debug_vis
        self.use_flip = use_flip and split == 'train'
        self.fliplr = fliplr if self.use_flip else 0.0
        self.use_color_aug = use_color_aug and split == 'train'
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.use_rrc = use_rrc and split == 'train'
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio
        self.use_mixup = use_mixup and split == 'train'
        self.mixup_prob = mixup_prob if self.use_mixup else 0.0
        self.mixup_alpha = mixup_alpha
        self.scan_max_samples = int(scan_max_samples)
        self.epoch = 0
        self._label_id_to_name = {}

        if len(self.ds) == 0:
            raise RuntimeError("HF dataset 为空")

        sample = self.ds[0]
        if 'image' not in sample:
            raise KeyError(f"HF dataset 缺少 'image' 字段，"
                           f"现有字段: {list(sample.keys())}")
        if 'labels' not in sample:
            raise KeyError(f"HF dataset 缺少 'labels' 字段，"
                           f"现有字段: {list(sample.keys())}")

        raw = sample['labels']
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        raw = list(raw)

        def _all_binary(vs):
            return all(v in (0, 1, 0.0, 1.0, True, False) for v in vs)

        if num_classes is not None and len(raw) == int(num_classes) and _all_binary(raw):
            self.label_mode = 'multihot'
            self.num_classes = int(num_classes)
        elif len(raw) > 0 and _all_binary(raw):
            self.label_mode = 'multihot'
            self.num_classes = len(raw)
        else:
            self.label_mode = 'id_list'
            if num_classes is None:
                # 优先从 label_names 扫描（同时拿到 id→name 映射）
                id_to_name, max_id_names = self._scan_label_names_id_map()
                if max_id_names >= 0:
                    self._label_id_to_name = id_to_name
                    self.num_classes = max_id_names + 1
                    print(f'[HFMultiLabelClsDataset] 从 label_names 推断 '
                          f'num_classes = {self.num_classes}')
                else:
                    max_id = self._scan_max_label_id()
                    self.num_classes = max_id + 1
                    self._label_id_to_name = {}
                    print(f'[HFMultiLabelClsDataset] 从 labels 推断 '
                          f'num_classes = {self.num_classes} (max_id={max_id})')
            else:
                self.num_classes = int(num_classes)
                # 用户指定 num_classes 也扫一下 label_names 供后续用
                id_to_name, _ = self._scan_label_names_id_map()
                self._label_id_to_name = id_to_name

        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = self._infer_class_names()

        if len(self.class_names) != self.num_classes:
            print(f'[HFMultiLabelClsDataset] ⚠️ class_names 长度({len(self.class_names)}) '
                  f'!= num_classes({self.num_classes})，使用通用类名')
            self.class_names = [f'cls_{i}' for i in range(self.num_classes)]

        print(f'[HFMultiLabelClsDataset] split={split} | '
              f'n={len(self.ds)} | label_mode={self.label_mode} | '
              f'num_classes={self.num_classes}')
        sample_names = sample.get('label_names', 'N/A')
        if isinstance(sample_names, (list, tuple)):
            sample_names = list(sample_names)[:8]
        print(f'  示例 label_names: {sample_names}')

        try:
            n_check = min(2000, len(self.ds))
            pos_count = np.zeros(self.num_classes, dtype=np.int64)
            for i in range(n_check):
                y = self._parse_label(self.ds[i]['labels'])
                pos_count += y.astype(np.int64)
            avg_pos = pos_count.sum() / max(n_check, 1)
            n_zero = int((pos_count == 0).sum())
            print(f'  前 {n_check} 张图平均正类数={avg_pos:.2f} | '
                  f'从未出现的类={n_zero}')
        except Exception as e:
            print(f'  统计跳过: {e}')

        norm = T.Normalize(mean=mean, std=std)
        if split == 'train':
            ops = []
            if self.use_rrc:
                ops.append(T.RandomResizedCrop(
                    self.img_size, scale=self.rrc_scale, ratio=self.rrc_ratio,
                    interpolation=T.InterpolationMode.BICUBIC))
            else:
                ops.append(T.Resize(self.img_size,
                                    interpolation=T.InterpolationMode.BICUBIC))
            if self.use_flip:
                ops.append(T.RandomHorizontalFlip(p=self.fliplr))
            if self.use_color_aug:
                ops.append(T.ColorJitter(
                    brightness=self.hsv_v, contrast=self.hsv_v,
                    saturation=self.hsv_s, hue=self.hsv_h))
            ops.extend([T.ToTensor(), norm])
            self.transform = T.Compose(ops)
        else:
            self.transform = T.Compose([
                T.Resize(int(self.img_size * 1.14),
                         interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.img_size),
                T.ToTensor(),
                norm,
            ])

        if self.debug_vis:
            self.debug_dir = os.path.join('./debug_vis', f'hfcls_{split}')
            os.makedirs(self.debug_dir, exist_ok=True)

    def set_epoch(self, epoch):
        self.epoch = epoch

    # ------------------------------------------------------------------
    def _scan_max_label_id(self):
        n = len(self.ds)
        max_samples = min(n, self.scan_max_samples)
        step = n / max_samples
        indices = (int(i * step) for i in range(max_samples))

        max_id = -1
        for i in indices:
            try:
                raw = self.ds[i]['labels']
            except Exception:
                continue
            if isinstance(raw, np.ndarray):
                raw = raw.tolist()
            if not isinstance(raw, (list, tuple)):
                continue
            for v in raw:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv > max_id:
                    max_id = iv
        return max_id

    # ------------------------------------------------------------------
    def _scan_label_names_id_map(self):
        """
        从数据集的 label_names 字段扫描构建 id → name 映射。

        数据集中 labels 和 label_names 是索引对齐的：
            labels       = [12, 45, 78]
            label_names  = ['pizza', 'pasta', 'burger']
        所以可以用 (id, name) 对构建完整映射。

        返回:
            id_to_name: dict[int, str]  id → 类名
            max_id:     int              见过的最大 id（-1 表示没扫到）
        """
        n = len(self.ds)
        max_samples = min(n, self.scan_max_samples)
        if n <= max_samples:
            indices = range(n)
        else:
            step = n / max_samples
            indices = (int(i * step) for i in range(max_samples))

        id_to_name = {}
        for i in indices:
            try:
                sample = self.ds[i]
            except Exception:
                continue

            labels = sample.get('labels', None)
            names = sample.get('label_names', None)
            if labels is None or names is None:
                continue

            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            if isinstance(names, np.ndarray):
                names = names.tolist()
            if not isinstance(labels, (list, tuple)) or \
               not isinstance(names, (list, tuple)):
                continue

            # labels 和 label_names 应该等长；不等长就跳过
            if len(labels) != len(names):
                continue

            for cid, cname in zip(labels, names):
                try:
                    cid = int(cid)
                except Exception:
                    continue
                if cid not in id_to_name:
                    id_to_name[cid] = str(cname)

        max_id = max(id_to_name.keys()) if id_to_name else -1
        return id_to_name, max_id

    # ------------------------------------------------------------------
    def _infer_class_names(self):
        # ---------- 从 features 里直接读 ----------
        try:
            feat = self.ds.features.get('labels', None)
            if hasattr(feat, 'feature') and hasattr(feat.feature, 'names'):
                names = list(feat.feature.names)
                if len(names) == self.num_classes:
                    print(f'[HFMultiLabelClsDataset] 类名来自 features.names '
                          f'({len(names)} 个)')
                    return names
            if hasattr(feat, 'names'):
                names = list(feat.names)
                if len(names) == self.num_classes:
                    print(f'[HFMultiLabelClsDataset] 类名来自 features.names '
                          f'({len(names)} 个)')
                    return names
        except Exception:
            pass

        # ---------- 从 label_names 字段扫描 ----------
        try:
            if getattr(self, '_label_id_to_name', None):
                id_to_name = self._label_id_to_name
            else:
                id_to_name, _ = self._scan_label_names_id_map()

            if id_to_name:
                n_classes = max(self.num_classes,
                                max(id_to_name.keys()) + 1)
                class_names = [
                    id_to_name.get(i, f'cls_{i}')
                    for i in range(n_classes)
                ]
                print(f'[HFMultiLabelClsDataset] 类名来自 label_names 字段扫描 '
                      f'({len(id_to_name)} 个唯一名)')
                return class_names
        except Exception as e:
            print(f'[HFMultiLabelClsDataset] 扫描 label_names 失败: {e}')

        # ---------- 占位兜底 ----------
        print(f'[HFMultiLabelClsDataset] 无法自动推断类名，'
              f'使用占位符 cls_0 ~ cls_{self.num_classes - 1}')
        return [f'cls_{i}' for i in range(self.num_classes)]

    # ------------------------------------------------------------------
    def _parse_label(self, raw):
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        raw = list(raw)

        if self.label_mode == 'multihot':
            vec = np.asarray(raw, dtype=np.float32)
            if vec.shape[0] != self.num_classes:
                out = np.zeros(self.num_classes, dtype=np.float32)
                n = min(vec.shape[0], self.num_classes)
                out[:n] = (vec[:n] > 0.5).astype(np.float32)
                return out
            return (vec > 0.5).astype(np.float32)
        else:
            out = np.zeros(self.num_classes, dtype=np.float32)
            for cid in raw:
                try:
                    cid = int(cid)
                except Exception:
                    continue
                if 0 <= cid < self.num_classes:
                    out[cid] = 1.0
            return out

    def __len__(self):
        return len(self.ds)

    def _get_raw(self, idx):
        try:
            sample = self.ds[idx]
            img = sample['image']
            if not isinstance(img, Image.Image):
                arr = np.array(img)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                img = Image.fromarray(arr)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            y = self._parse_label(sample['labels'])
            return img, y, idx
        except Exception:
            return None

    def __getitem__(self, idx):
        img, y = None, None
        for attempt in range(self.max_retry):
            cur = idx if attempt == 0 else random.randint(0, len(self.ds) - 1)
            out = self._get_raw(cur)
            if out is not None:
                img, y, _ = out
                break

        if img is None:
            img = Image.fromarray(
                np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            y = np.zeros(self.num_classes, dtype=np.float32)

        img_t = self.transform(img)
        label_t = torch.from_numpy(y)

        if self.use_mixup and random.random() < self.mixup_prob:
            for _ in range(self.max_retry):
                j = random.randint(0, len(self.ds) - 1)
                out2 = self._get_raw(j)
                if out2 is None:
                    continue
                img2, y2, _ = out2
                img2_t = self.transform(img2)
                label2_t = torch.from_numpy(y2)
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                img_t = lam * img_t + (1.0 - lam) * img2_t
                label_t = lam * label_t + (1.0 - lam) * label2_t
                break

        return img_t, label_t