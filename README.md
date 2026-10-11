# SWAP-Fighter 🔄⚔️

**"Swap your backbone, fight for better detection."**

SWAP-Fighter is a modular anchor-free framework inspired by **FCOS** and **YOLOv8**. It allows you to plug different `timm` backbones into a unified neck and decoupled multi-task head, supporting:

- **`detect`**: horizontal bounding box detection
- **`obb`**: oriented bounding box detection with angle prediction
- **`segment`**: instance segmentation (YOLOv8-seg style)
- **`keypoint`**: multi-person pose estimation (YOLOv8-pose style)
- **`multilabel`**: multi-label image classification
- **`linear_probe`**: frozen-backbone linear probing for multi-label classification

> Chinese name: **接头霸王** – because you can "connect" almost any backbone you want.
> [中文说明由此进入](README.zh-CN.md)

**GitHub**: https://github.com/IntptrMax/SwapFighter

---

## ✨ Key Features

- **Plug-and-play backbones** via `timm` (`features_only=True`).
  See [BackboneList.md](BackboneList.md) for a curated list.
- **Six task modes**
  - `detect`: horizontal boxes
  - `obb`: oriented boxes with angle regression
  - `segment`: instance segmentation with prototype-based mask head
  - `keypoint`: pose estimation with per-keypoint branch (YOLOv8-pose style)
  - `multilabel`: multi-label classification
  - `linear_probe`: frozen backbone + linear/multi-layer head
- **Unified anchor-free detection head** (for detect/obb/segment/keypoint)
  - **Shared decoupled head** across P3/P4/P5 (via `--head-channels` projection)
  - **Layered neck support** (`--neck-out-channels 64 128 256`)
  - Classification: **Focal Loss** style (BCE with target score normalization).
  - Regression:
    - `detect`/`segment`/`keypoint`: **CIoU Loss** + optional **DFL**.
    - `obb`: **CRIoU + ProbIoU** or **KLD Loss** (selectable) + optional **DFL**.
  - Angle:
    - `sin_cos`: cosine loss + periodic L1 auxiliary loss
    - `direct`: periodic L1 loss
  - Keypoint: normalized absolute-coordinate L1 (YOLOv8-pose style), weight 12.0 by default
- **Instance segmentation head** (YOLOv8-seg style)
  - **Prototype branch** fused from P3+P4+P5
  - **Mask coefficient branch** (coef @ proto → mask)
  - Configurable `proto_dim`, `proto_stride` (2/4/8), `mask_weight`
- **Keypoint head** (YOLOv8-pose style)
  - Dedicated `kpt_branch` outputting normalized absolute coordinates
  - `bias=0.5` initialization → initial prediction at image center
  - L1 loss in normalized coordinate space (weight 12.0)
  - OKS-based pose mAP for evaluation
- **Multi-label classification head**
  - Pooling: `gap`, `gem`, `attention`
  - Head: `linear`, `mlp`, `ml_decoder`
  - Loss: `bce`, `asl` (Asymmetric Loss)
- **Task-Aligned Assigner (TAL)** for positive sample assignment.
- **Full training pipeline**
  - AMP precision: `fp32`, `fp16`, `bf16`
  - `GradScaler` for `fp16`
  - Warmup + cosine / step / multistep / lambda / OneCycle schedulers
  - Gradient clipping, EMA, early stopping, resume from checkpoint
  - **Per-task CSV logs** with task-specific columns
  - **Per-task `results.png`** panels (no empty subplots)
  - **`--verbose-metrics`**: print per-class mAP table
  - **`--grad-checkpointing`**: save GPU memory on large models
- **Flexible dataset support**
  - YOLO-style folder for detect/obb
  - **YOLOv8-seg format** for segment (polygon labels)
  - **YOLOv8-pose format** for keypoint (with or without visibility)
  - txt label files for multilabel (id-list and one-hot formats)
  - HuggingFace datasets for multilabel
- **Built-in evaluation**
  - `detect`: mAP@0.5, mAP@0.5:0.95, AR@100 (YOLOv8-style)
  - `obb`: rotated IoU mAP@0.5, mAP@0.5:0.95 (ProbIoU-based)
  - `segment`: **both bbox mAP and mask mAP** (prototype-scale mask IoU)
  - `keypoint`: **both bbox mAP and OKS pose mAP**
  - `multilabel`: mAP, CF1, OP, OR, OF1, EMR
- **SimMIM self-supervised pre-training**
  - `simmim_pretrain.py` pre-trains a backbone on unlabeled images.
  - Supports Swin / ConvNeXt / ViT (incl. DINOv2 / DINOv3 / Eva) / MobileNet / EfficientNet / generic CNN.
- **Deployment**
  - **ONNX export for all tasks** (raw + end2end)
    - `detect`/`obb`: `num_dets, boxes, scores, classes` (end2end)
    - `segment`: `num_dets, boxes, scores, classes, mask_coefs, proto` (end2end)
    - `keypoint`: `num_dets, boxes, scores, classes, keypoints` (end2end)
    - `multilabel`: `logits (B, C)` or `probs (B, C)` with `--with-sigmoid`
  - **ONNX Runtime inference** via `infer.py` (auto-detects format from file suffix)

---

## 📦 Supported Backbones

The framework uses `timm` to access a wide range of pre-trained models. Any backbone supported by `timm.create_model(..., features_only=True)` can be used in principle.

See **[BackboneList.md](BackboneList.md)** for a curated list of tested backbones, including:

| Family | Example Names |
|---|---|
| ConvNeXtV2 (FCMAE) | `convnextv2_atto.fcmae_ft_in1k`, `convnextv2_tiny.fcmae_ft_in1k` |
| ConvNeXt | `convnext_tiny`, `convnext_base` |
| MobileNet | `mobilenetv3_small_100`, `mobilenetv3_large_100` |
| ResNet / ResNeXt / ResNeSt | `resnet50`, `resnext50_32x4d`, `resnest50` |
| Swin Transformer | `swin_tiny_patch4_window7_224.ms_in1k` |
| Hiera | `hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k` |
| ViT | `vit_base_patch16_224`, `vit_large_patch16_224` |
| DINOv2 / DINOv3 | `vit_small_patch16_dinov3.lvd1689m`, `convnext_tiny.dinov3_lvd1689m` |
| EfficientNet | `efficientnet_b0` ~ `b7`, `efficientnetv2_rw_s` |
| Lightweight | `ghostnet_100`, `fasternet_tiny` |

Backbone feature selection is automatic:

- ViT / DINOv2 / DINOv3-style models: uses `ViTAdapter` to build a multi-scale pyramid from the single-scale tokens.
- Swin / Hiera: selects feature maps with strides `8`, `16`, `32`.
- Other CNNs: uses the last three feature maps.
- Output channels and strides are inferred automatically.

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/IntptrMax/SwapFighter.git
cd SwapFighter
pip install -r requirements.txt
```

**Core requirements:**

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision, timm
- opencv-python, Pillow, numpy
- tqdm, matplotlib
- onnx, onnxruntime
- onnxsim / onnx-simplifier (optional, for `--simplify`)
- datasets (optional, for HuggingFace multi-label datasets)

> No `shapely` or `torchmetrics` required – evaluation and rotated NMS are implemented in pure NumPy / PyTorch.

---

### 2. Dataset Preparation

#### 2.1 Horizontal (`detect`) and Oriented (`obb`) detection

YOLO-style folder structure:

```text
datasets/
└── your_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

**Horizontal detection labels (`detect`):**

```text
class_id cx cy w h
```

**Oriented detection labels (`obb`):**

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All values normalized to `[0, 1]`. The loader also cross-accepts 5-column / 9-column formats.

#### 2.2 Instance segmentation (`segment`)

**YOLOv8-seg polygon format:**

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

All values normalized to `[0, 1]`. Each line is a polygon with **at least 3 vertices**. The dataset automatically rasterizes polygons to prototype-scale masks (`img_size / proto_stride`).

#### 2.3 Keypoint / Pose (`keypoint`)

**YOLOv8-pose format** – one line per instance:

```text
# With visibility (3 values per keypoint)
class_id cx cy w h  x1 y1 v1  x2 y2 v2 ... xK yK vK

# Without visibility (2 values per keypoint)
class_id cx cy w h  x1 y1  x2 y2 ... xK yK
```

- Coordinates are normalized to `[0, 1]`.
- `v ∈ {0, 1, 2}` where `0` = not labeled, `1` = labeled but occluded, `2` = visible.
- `visibility='auto'` auto-detects the format from the first few samples; override with `--keypoint-visibility yes|no`.
- `--flip-idx` should be supplied for left-right symmetric keypoints (e.g. COCO 17).

#### 2.4 Multi-label classification (`multilabel` / `linear_probe`)

Two data sources: **local txt files** or **HuggingFace datasets**.

**Local txt files:**

```text
datasets/
└── your_multilabel_dataset/
    ├── train2017/
    ├── val2017/
    ├── train_labels.txt
    └── val_labels.txt
```

Each line can be:

- **New format**: `filename\t[0, 14, 26]`  (tab-separated, list of class ids)
- **Old format**: `filename 0 0 1 0 1 ...`  (space-separated one-hot vector)

**HuggingFace datasets:** must contain `image` and `labels` fields, optionally `label_names`. Use `--hf-dataset`, `--hf-config`, `--hf-train-split`, `--hf-val-split`.

#### 2.5 SimMIM pre-training (unlabeled images)

```text
datasets/
└── unlabeled_images/
    ├── any_subdir/
    │   ├── a.jpg
    │   └── ...
    └── b.png
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`.

---

### 3. Training

#### Horizontal detection

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_tiny.fcmae_ft_in1k \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --precision fp16 \
    --save-dir ./outputs
```

#### Oriented detection (OBB)

```bash
python train.py \
    --task obb \
    --data-root ./datasets/dota_like \
    --backbone convnextv2_tiny.fcmae_ft_in1k \
    --img-size 1024 \
    --batch-size 8 \
    --angle-format sin_cos \
    --angle-range pi2 \
    --obb-reg-loss probiou \
    --save-dir ./outputs
```

#### Instance segmentation

```bash
python train.py \
    --task segment \
    --data-root ./datasets/my_seg_data \
    --num-classes 2 \
    --backbone convnextv2_tiny.fcmae_ft_in1k \
    --img-size 640 \
    --batch-size 8 \
    --proto-dim 32 \
    --proto-stride 4 \
    --mask-weight 5.0 \
    --epochs 100 \
    --precision fp16 \
    --save-dir ./outputs
```

**Key segment options:**

- `--proto-dim`: prototype / mask coefficient channels (default `32`; YOLOv8-seg uses 32).
- `--proto-stride`: prototype downsample stride, one of `2`/`4`/`8` (default `4`). Smaller = finer masks, more VRAM.
- `--mask-weight`: mask loss weight (default `1.0`; YOLOv8-seg uses `5.0`).

#### Keypoint / Pose estimation

```bash
python train.py \
    --task keypoint \
    --data-root ./datasets/tiger-pose \
    --num-classes 1 \
    --num-keypoints 17 \
    --backbone convnextv2_tiny.fcmae_ft_in1k \
    --img-size 640 \
    --batch-size 8 \
    --keypoint-weight 12.0 \
    --head-kpt-num-convs 3 \
    --epochs 100 \
    --precision fp16 \
    --save-dir ./outputs
```

**Key keypoint options:**

- `--num-keypoints`: number of keypoints per instance (must match your labels).
- `--keypoint-weight`: keypoint loss weight (default `12.0`, following YOLOv8-pose).
- `--head-kpt-num-convs`: number of conv layers in the keypoint branch (default `3`).
- `--keypoint-visibility`: `auto` / `yes` / `no`.
- `--flip-idx`: list of keypoint indices for horizontal flip (e.g. for COCO 17: `0 2 1 4 3 6 5 8 7 10 9 12 11 14 13 16 15`).

#### Multi-label classification

```bash
python train.py \
    --task multilabel \
    --data-root ./datasets/multilabel_data \
    --train-labels train_labels.txt \
    --val-labels val_labels.txt \
    --num-classes 20 \
    --backbone convnextv2_atto.fcmae_ft_in1k \
    --img-size 224 \
    --batch-size 64 \
    --loss asl \
    --epochs 40 \
    --save-dir ./outputs
```

**HuggingFace variant:**

```bash
python train.py \
    --task multilabel \
    --hf-dataset ibrahimdaud/multi-label-food-recognition \
    --num-classes 90 \
    --backbone convnextv2_tiny.fcmae_ft_in1k \
    --img-size 320 \
    --loss bce \
    --epochs 100 \
    --save-dir ./outputs
```

#### Linear probe

```bash
python train.py \
    --task linear_probe \
    --data-root ./datasets/multilabel_data \
    --num-classes 20 \
    --backbone vit_small_patch16_dinov3.lvd1689m \
    --img-size 224 \
    --batch-size 512 \
    --epochs 20 \
    --save-dir ./outputs
```

#### Key training options

- `--task detect|obb|segment|keypoint|multilabel|linear_probe`
- `--data-root`, `--backbone`, `--img-size`, `--batch-size`, `--epochs`, `--lr`
- `--precision fp32|fp16|bf16`
- `--optimizer sgd|adamw`, `--scheduler lambda|step|multistep|cosine|onecycle`
- `--patience`, `--resume`, `--no-ema`, `--save-dir`
- `--head-channels`: shared head channels for detect/obb/segment/keypoint
- `--neck-out-channels`: single value or 3 values (e.g., `64 128 256`)
- `--grad-checkpointing`: enable gradient checkpointing
- `--verbose-metrics`: print per-class mAP table

Any other config item can be overridden by appending `key=value`.

Training automatically creates a `runX` folder:

```text
runX/
├── best.pth
├── last.pth
├── training_log.csv    # per-task columns
├── config.json
└── results.png         # per-task panels
```

---

### 4. Inference

`infer.py` **auto-detects the backend** from the weights suffix (`.pth`/`.pt` → PyTorch, `.onnx` → ONNX). You can still override with `--format pytorch|onnx`.

#### Horizontal detection (PyTorch)

```bash
python infer.py \
    --task detect \
    --weights ./outputs/detect/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.7 --nms 0.3 \
    --output result.jpg \
    --class-names pill capsule tablet
```

#### Oriented detection (PyTorch)

```bash
python infer.py \
    --task obb \
    --weights ./outputs/obb/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.5 \
    --output result_obb.jpg \
    --class-names plane ship vehicle
```

#### Instance segmentation (PyTorch)

```bash
python infer.py \
    --task segment \
    --weights ./outputs/segment/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.5 \
    --output result_seg.jpg \
    --show
```

**Segmentation-specific flags:**

- `--no-mask`: skip mask visualization (bbox only)
- `--mask-alpha`: overlay alpha (0~1, default 0.5)

#### Keypoint / Pose (PyTorch)

```bash
python infer.py \
    --task keypoint \
    --weights ./outputs/keypoint/run1/best.pth \
    --image path/to/person.jpg \
    --conf 0.5 \
    --output result_pose.jpg \
    --show
```

**Keypoint-specific flags:**

- `--no-kpt`: skip keypoint visualization (bbox only)
- `--kpt-radius`: keypoint circle radius (default 6)
- `--kpt-thr`: visibility threshold (default 0.3)

#### Multi-label classification (PyTorch)

```bash
python infer.py \
    --task multilabel \
    --weights ./outputs/multilabel/run1/best.pth \
    --image path/to/image.jpg \
    --threshold 0.5 --top-k 5 \
    --output result.jpg
```

#### ONNX inference (auto-detected)

```bash
# Suffix .onnx → ONNX automatically, no --format needed
python infer.py \
    --weights best_end2end.onnx \
    --image test.jpg \
    --conf 0.7 --nms 0.3 \
    --class-names pill capsule tablet
```

For OBB, also pass `--angle-format` and `--angle-range`. For segment/keypoint, the ONNX predictor handles both raw and end2end outputs automatically.

Additional options:

- `--output-dir`: save multiple images to a directory.
- `--show`: display result window.
- `--show-size`: max window width.
- `--device cpu`: force CPU inference.

---

### 5. ONNX Export

#### Horizontal detection

```bash
python export.py \
    --weights ./outputs/detect/run1/best.pth \
    --dynamic --simplify
```

#### Oriented detection

```bash
python export.py \
    --weights ./outputs/obb/run1/best.pth \
    --dynamic --simplify
```

#### Instance segmentation

```bash
# Raw mode (output0 + proto)
python export.py --weights ./outputs/segment/run1/best.pth

# End2end mode (NMS embedded)
python export.py \
    --weights ./outputs/segment/run1/best.pth \
    --end2end --conf 0.25 --nms 0.5 --max-det 300
```

#### Keypoint / Pose

```bash
# Raw mode (output0)
python export.py --weights ./outputs/keypoint/run1/best.pth

# End2end mode (NMS embedded; keypoints output as pixel coordinates)
python export.py \
    --weights ./outputs/keypoint/run1/best.pth \
    --end2end --conf 0.25 --nms 0.5 --max-det 300
```

#### Multi-label classification

```bash
# Default: outputs logits (B, C)
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --dynamic --simplify

# With embedded sigmoid: outputs probs (B, C) ∈ (0, 1)
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --with-sigmoid --dynamic --simplify
# → best_probs.onnx
```

> **Why `--with-sigmoid`?**
> By default the exported graph outputs raw **logits**, which require a `sigmoid` on the deployment side.
> If you want a **zero-logic deployment** (e.g. on TensorRT / NCNN / TFLite), export with `--with-sigmoid` to get probabilities directly.
> Thresholding and Top-K remain in the application layer.

#### Export modes

- **Raw mode** (default):
  - `detect`: `output0 (B, 4+nc, P)` — `[cx, cy, w, h, cls...]`
  - `obb`: `output0 (B, 5+nc, P)` — `[cx, cy, w, h, angle, cls...]`
  - `segment`: `output0 (B, 4+nc+k, P)` + `proto (B, k, ph, pw)`
  - `keypoint`: `output0 (B, 4+nc+2K, P)` — `[cx, cy, w, h, cls..., kpt_norm...]`
  - `multilabel`: `logits (B, num_classes)` or `probs (B, num_classes)`
- **End2end mode** (`--end2end`):
  - `detect`/`obb`: `num_dets, boxes, scores, classes`
  - `segment`: `num_dets, boxes, scores, classes, mask_coefs, proto`
  - `keypoint`: `num_dets, boxes, scores, classes, keypoints` (all pixel coordinates)
  - NMS is embedded; `--conf`, `--nms`, `--max-det` control its parameters.

Metadata (backbone, img_size, mean, std, strides, proto_dim, proto_stride, num_keypoints, etc.) is embedded in the ONNX file. `ONNXPredictor` reads it back automatically — including whether the multilabel output is `logits` or `probs`.

---

### 6. SimMIM Self-Supervised Pre-training

```bash
python simmim_pretrain.py \
    --data-dir ./datasets/unlabeled_images \
    --backbone convnextv2_atto \
    --img-size 224 \
    --mask-patch-size 32 \
    --mask-ratio 0.6 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --output ./simmim_pretrained/convnextv2_atto_simmim.pth
```

Then use the pre-trained weights:

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_atto \
    --pretrained-source simmim \
    --backbone-pretrained-path ./simmim_pretrained/convnextv2_atto_simmim.pth \
    --img-size 640 --batch-size 16 --epochs 100
```

> Use the **same `--backbone`** in pre-training and downstream. `img-size` does not need to match.

---

### 7. Configuration

All hyperparameters are centralised in `swapfighter/cfg/config.py`. Task-specific defaults are in `swapfighter/tasks/*.py`.

Key parameters:

- `task`: `detect`, `obb`, `segment`, `keypoint`, `multilabel`, `linear_probe`
- `data_root`, `train_split`, `val_split`, `num_classes`
- `backbone_name`, `pretrained`, `img_size`
- `neck_type`, `neck_num_layers`, `neck_out_channels`, `head_channels`
- `head_type`, `head_norm_layer`, `head_act_layer`
- `with_angle`, `angle_range`, `angle_format`
- `reg_max`, `dfl_weight`, `obb_reg_loss`
- `proto_dim`, `proto_stride`, `mask_weight` (segment)
- `num_keypoints`, `keypoint_weight`, `keypoint_visibility`, `flip_idx` (keypoint)
- `batch_size`, `lr`, `backbone_lr_scale`, `weight_decay`, `epochs`
- `warmup_epochs`, `optimizer`, `scheduler`
- `precision`: `fp32`, `fp16`, `bf16`
- `mosaic_prob`, `mosaic_disable_epoch`
- `conf_threshold`, `nms_threshold`, `obb_nms_threshold`
- `eval_map_start_epoch`, `eval_conf_threshold`, `verbose_metrics`
- `patience`, `use_ema`, `ema_decay`
- `tal_topk`, `tal_alpha`, `tal_beta`
- `pool_type`, `head_type`, `loss`, `eval_threshold` (multilabel)

---

## 📁 Project Structure

```text
SWAP-Fighter/
├── swapfighter/
│   ├── api.py                 # SF class
│   ├── cfg/
│   │   └── config.py          # Config class
│   ├── tasks/                 # Task registry and defaults
│   │   ├── __init__.py
│   │   ├── detect.py
│   │   ├── obb.py
│   │   ├── segment.py
│   │   ├── keypoint.py
│   │   ├── multilabel.py
│   │   └── linear_probe.py
│   ├── engine/
│   │   ├── base_trainer.py
│   │   ├── base_predictor.py
│   │   ├── trainers/
│   │   │   ├── detect.py
│   │   │   ├── segment.py
│   │   │   ├── keypoint.py
│   │   │   └── multilabel.py
│   │   ├── predictors/
│   │   │   ├── detect.py
│   │   │   ├── segment.py
│   │   │   ├── keypoint.py
│   │   │   ├── multilabel.py
│   │   │   └── onnx.py
│   │   └── exporters/
│   │       ├── onnx.py
│   │       └── _onnx_impl.py
│   ├── models/
│   │   ├── backbones.py
│   │   ├── backbones_vit_adapter.py
│   │   ├── necks/             # lightfpn, bifpn
│   │   ├── heads/             # decoupled, seg_decoupled, keypoint_decoupled, ml_decoder
│   │   ├── detectors/         # detect_fcos, detect_fcos_seg, detect_fcos_keypoint
│   │   └── classifiers/       # multi_label
│   ├── datasets/
│   │   ├── yolo_det.py
│   │   ├── yolo_seg.py
│   │   ├── yolo_keypoint.py
│   │   └── cls_dataset.py
│   └── utils/
│       ├── box_ops.py
│       ├── mask_ops.py
│       ├── losses.py
│       ├── seg_losses.py
│       ├── keypoint_loss.py
│       ├── metrics.py
│       ├── postprocess.py
│       └── logger.py
├── train.py                    # Training entry
├── infer.py                    # Inference entry (auto-detects format)
├── export.py                   # ONNX export entry
├── simmim_pretrain.py          # SimMIM self-supervised pre-training
├── requirements.txt
├── README.md
├── README.zh-CN.md
└── BackboneList.md
```

---

## 📝 Notes

- The old `--use-amp` flag is replaced by `--precision`.
- `fp16` uses `GradScaler` automatically; `bf16` requires Ampere or newer.
- OBB rotated NMS and OBB mAP are pure NumPy / PyTorch — no `shapely`.
- Segment mask mAP is computed at prototype scale (no `cv2.resize` upsampling) for speed and consistency.
- Keypoint end2end ONNX outputs **pixel coordinates** for both `boxes` and `keypoints`, matching `Segment`'s convention.
- Multilabel ONNX exports `logits` by default; use `--with-sigmoid` to embed `Sigmoid` and get `probs (B, C)`.
- `infer.py` auto-selects `pytorch` / `onnx` by weights suffix (`.pth`/`.pt` → pytorch, `.onnx` → onnx). Override with `--format`.
- CSV logs and `results.png` panels are **per-task** — no empty columns/subplots.
- ONNX post-processing relies on correct metadata (strides, img_size, mean, std, proto_dim, proto_stride, num_keypoints).
- SimMIM pre-training is **optional** — you can use timm ImageNet weights directly.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR at https://github.com/IntptrMax/SwapFighter.

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FCOS](https://github.com/tianzhi0549/FCOS) — anchor-free detection idea.
- [YOLOv8](https://github.com/ultralytics/ultralytics) — TAL assigner, decoupled head, prototype-based segmentation, pose estimation.
- [timm](https://github.com/huggingface/pytorch-image-models) — pre-trained backbones.
- [SimMIM](https://github.com/microsoft/SimMIM) — masked image modeling.

---

**Happy Swapping!** 🔄
