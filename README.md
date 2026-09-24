以下是两个可直接粘贴使用的完整 README 文件内容。

**第一步**：把 `README.md` 整个文件替换为以下内容（全选复制）：

# SWAP-Fighter 🔄⚔️

**"Swap your backbone, fight for better detection."**

SWAP-Fighter is a modular anchor-free object detection framework inspired by **FCOS** and **YOLOv8**. It allows you to plug different `timm` backbones into a unified neck and a decoupled multi-task head, supporting:

- **`detect`**: horizontal bounding box detection
- **`obb`**: oriented bounding box detection with angle prediction
- **`multilabel`**: multi-label image classification

> Chinese name: **接头霸王** – because you can "connect" almost any backbone you want.

**GitHub**: https://github.com/IntptrMax/SwapFighter

---

## ✨ Key Features

- **Plug-and-play backbones** via `timm` (`features_only=True`).
- **Three task modes**
  - `detect`: horizontal boxes
  - `obb`: oriented boxes with angle regression
  - `multilabel`: multi-label classification
- **Unified anchor-free detection head** (for detect/obb)
  - **Decoupled head** with independent convolution towers for classification, regression, and angle.
  - Classification: **Focal Loss** style (BCE with target score normalization).
  - Regression:
    - `detect`: **CIoU Loss** + optional **DFL** (Distribution Focal Loss).
    - `obb`: **CRIoU + ProbIoU** or **KLD Loss** (selectable) + optional **DFL**.
  - Angle:
    - `sin_cos`: cosine loss + periodic L1 auxiliary loss
    - `direct`: periodic L1 loss
- **Multi-label classification head**
  - Pooling: `gap`, `gem`, `attention`
  - Head: `linear` or `mlp`
  - Loss: `bce` or `asl` (Asymmetric Loss)
- **Task-Aligned Assigner (TAL)** for positive sample assignment.
- **Full training pipeline**
  - AMP precision: `fp32`, `fp16`, `bf16`
  - `GradScaler` for `fp16`
  - Warmup + cosine / step / multistep / lambda / OneCycle schedulers
  - Gradient clipping
  - EMA (Exponential Moving Average)
  - Early stopping with patience
  - Resume from checkpoint
- **Flexible dataset support**
  - YOLO-style folder structure for detect/obb
  - txt label files for multilabel (supports both id-list and one-hot formats)
  - HuggingFace datasets for multilabel
- **Built-in evaluation**
  - `detect`: mAP@0.5, mAP@0.5:0.95, AR@100 (YOLOv8-style)
  - `obb`: rotated IoU mAP@0.5 and mAP@0.5:0.95 (ProbIoU-based)
  - `multilabel`: mAP, CF1, OF1, EMR
- **SimMIM self-supervised pre-training**
  - `simmim_pretrain.py` pre-trains a backbone on your own unlabeled images.
  - The resulting weights load directly into SWAP-Fighter's `Backbone` with **no key remapping**.
  - Supports Swin / ConvNeXt / ViT (incl. DINOv2 / DINOv3 / Eva) / MobileNet / EfficientNet / generic CNN.
- **Rich visualisation and logging**
  - CSV training logs
  - Automatic training curves (`results.png`)
  - `runX` output folders
  - Single-image inference with adaptive font and colours
- **Deployment**
  - ONNX export for all tasks (raw or end2end)
  - ONNX Runtime inference via `infer.py --format onnx`

---

## 📦 Supported Backbones

The framework uses `timm` to access a wide range of pre-trained models. Any backbone supported by `timm.create_model(..., features_only=True)` can be used in principle.

The following families are tested and recommended:

| Family | Example Names |
|---|---|
| ConvNeXt | `convnextv2_atto`, `convnextv2_tiny`, `convnext_base` |
| MobileNet | `mobilenetv3_large_100`, `mobilenetv2_140` |
| ResNet | `resnet50`, `resnet101`, `resnext50_32x4d` |
| Swin Transformer | `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224` |
| ViT | `vit_base_patch16_224`, `vit_large_patch16_224` |
| DINOv2 / DINOv3 | `dinov2_vitb14`, `dinov2_vitl14`, and other `timm`-supported variants |
| EfficientNet | `efficientnet_b0` ~ `b7` |

Backbone feature selection is automatic:

- ViT / DINOv2-style models: uses the last three feature maps.
- Swin-style models: selects feature maps with strides `8`, `16`, `32`.
- Other CNNs: uses the last three feature maps.
- Output channels and strides are inferred automatically.

Pass any accepted `timm` model name via `--backbone` or modify `config.py`.

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
- torchvision
- timm
- opencv-python
- Pillow
- numpy
- tqdm
- matplotlib
- onnx
- onnxruntime
- onnxsim / onnx-simplifier (optional, for `--simplify`)
- datasets (optional, for HuggingFace multi-label datasets)

> No `shapely` or `torchmetrics` required – evaluation and rotated NMS are implemented in pure NumPy / PyTorch.

---

### 2. Dataset Preparation

#### 2.1 Horizontal detection (`task = detect`) and Oriented detection (`task = obb`)

Use YOLO-style folder structure:

```text
datasets/
└── your_dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── val/
    │       ├── image2.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   └── ...
        └── val/
            ├── image2.txt
            └── ...
```

**Horizontal detection labels (`detect`):**

```text
class_id cx cy w h
```

All values normalized to `[0, 1]`. The loader also accepts YOLO OBB 9-column labels and converts them to horizontal boxes.

**Oriented detection labels (`obb`):**

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All values normalized to `[0, 1]`. The loader also accepts 5-column YOLO detect labels and converts them to axis-aligned oriented boxes with angle `0`.

Set the dataset root in `config.py` or pass `--data-root`.

#### 2.2 Multi-label classification (`task = multilabel`)

Two data sources are supported: local txt files or HuggingFace datasets.

**Local txt files:**

```text
datasets/
└── your_multilabel_dataset/
    ├── train2017/          # training images
    │   ├── img1.jpg
    │   └── ...
    ├── val2017/            # validation images
    │   ├── img2.jpg
    │   └── ...
    ├── train_labels.txt
    └── val_labels.txt
```

Each line in `train_labels.txt` / `val_labels.txt` can be either:

- **New format**: `filename\t[0, 14, 26]`  (tab-separated, list of class ids)
- **Old format**: `filename 0 0 1 0 1 ...`  (space-separated one-hot vector of length `num_classes`)

Example new format:
```text
img1.jpg	[0, 14, 26]
img2.jpg	[3]
img3.jpg	[]
```

**HuggingFace datasets:**

The dataset must contain `image` and `labels` fields. Optionally `label_names` for class names. Use `--hf-dataset`, `--hf-config`, `--hf-train-split`, `--hf-val-split` to specify.

#### 2.3 SimMIM pre-training (unlabeled images)

SimMIM pre-training only needs a folder of unlabeled images (recursively scanned):

```text
datasets/
└── unlabeled_images/
    ├── any_subdir/
    │   ├── a.jpg
    │   └── ...
    └── b.png
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive).

---

### 3. Training

The training script is `train.py`. It uses the `SF` API internally.

#### Horizontal detection example

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_tiny \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --precision fp16 \
    --optimizer adamw \
    --scheduler cosine \
    --save-dir ./outputs
```

To set additional config options, append `key=value` at the end:

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_tiny \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --precision fp16 \
    warmup_epochs=5 freeze_epochs=5 mosaic_prob=0.8
```

#### Oriented detection example

```bash
python train.py \
    --task obb \
    --data-root ./datasets/dota_like \
    --backbone convnextv2_tiny \
    --img-size 1024 \
    --batch-size 8 \
    --epochs 100 \
    --precision fp16 \
    --angle-format sin_cos \
    --angle-range pi \
    --obb-reg-loss probiou \
    --save-dir ./outputs
```

#### Multi-label classification example (local txt)

```bash
python train.py \
    --task multilabel \
    --data-root ./datasets/multilabel_data \
    --train-labels train_labels.txt \
    --val-labels val_labels.txt \
    --train-img-subdir train2017 \
    --val-img-subdir val2017 \
    --num-classes 20 \
    --backbone convnextv2_atto \
    --img-size 224 \
    --batch-size 64 \
    --epochs 40 \
    --lr 1e-3 \
    --precision fp16 \
    --save-dir ./outputs
```

#### Multi-label classification example (HuggingFace)

```bash
python train.py \
    --task multilabel \
    --hf-dataset my_dataset \
    --num-classes 20 \
    --backbone convnextv2_atto \
    --img-size 224 \
    --batch-size 64 \
    --epochs 40 \
    --lr 1e-3 \
    --save-dir ./outputs
```

#### Key training options

- `--task detect|obb|multilabel`: select task.
- `--data-root`: dataset root.
- `--backbone`: timm model name.
- `--img-size`: input image size.
- `--batch-size`: batch size.
- `--epochs`: number of epochs.
- `--lr`: learning rate.
- `--precision fp32|fp16|bf16`: training precision.
- `--optimizer sgd|adamw`: optimizer.
- `--scheduler lambda|step|multistep|cosine|onecycle`: LR scheduler.
- `--patience`: early stopping patience (0 = disabled).
- `--resume`: resume from checkpoint.
- `--no-ema`: disable EMA (enabled by default).
- `--save-dir`: output directory.

Any other configuration item (e.g., `warmup_epochs`, `freeze_epochs`, `mosaic_prob`, `backbone_lr_scale`, `ema_decay`) can be overridden by appending `key=value` to the command line.

Training automatically creates a `runX` folder under `--save-dir` containing:
```text
runX/
├── best.pth
├── last.pth
├── training_log.csv
├── config.json
└── results.png
```

---

### 4. Inference

#### Horizontal detection (PyTorch)

```bash
python infer.py \
    --task detect \
    --weights ./outputs/detect/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.7 \
    --nms 0.3 \
    --output result.jpg \
    --class-names pill capsule tablet
```

#### Oriented detection (PyTorch)

```bash
python infer.py \
    --task obb \
    --weights ./outputs/obb/run1/best.pth \
    --image path/to/image.jpg \
    --angle-format sin_cos \
    --angle-range pi \
    --conf 0.5 \
    --output result_obb.jpg \
    --class-names plane ship vehicle
```

#### Multi-label classification (PyTorch)

```bash
python infer.py \
    --task multilabel \
    --weights ./outputs/multilabel/run1/best.pth \
    --image path/to/image.jpg \
    --threshold 0.5 \
    --top-k 5 \
    --output result.jpg
```

#### ONNX inference

Use `--format onnx` and point `--weights` to the `.onnx` file.

```bash
python infer.py \
    --format onnx \
    --weights model.onnx \
    --image test.jpg \
    --conf 0.7 \
    --nms 0.3 \
    --class-names pill capsule tablet
```

For OBB, also pass `--angle-format` and `--angle-range`.

> If model strides are not `[8, 16, 32]`, pass `--img-size` and ensure the ONNX metadata contains the correct strides. The ONNX predictor reads strides from metadata if available.

Additional options:
- `--output-dir`: save multiple images to a directory.
- `--show`: display result window.
- `--show-size`: max window width.

---

### 5. ONNX Export

Export a trained model to ONNX:

#### Horizontal detection

```bash
python export.py \
    --weights ./outputs/detect/run1/best.pth \
    --output model.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### Oriented detection

```bash
python export.py \
    --weights ./outputs/obb/run1/best.pth \
    --output model_obb.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### Multi-label classification

```bash
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --output model_cls.onnx \
    --img-size 224 \
    --dynamic \
    --simplify
```

#### Export modes

- **Raw mode** (default): outputs a single tensor `output0`.
  - detect: shape `(B, 4+nc, P)`, values `[cx, cy, w, h, cls...]` normalized to `[0,1]`.
  - obb: shape `(B, 5+nc, P)`, values `[cx, cy, w, h, angle, cls...]` normalized.
  - multilabel: outputs `logits` of shape `(B, num_classes)`.
- **End2end mode** (`--end2end`, only detect/obb): outputs `num_dets`, `boxes`, `scores`, `classes`. NMS is embedded. `--conf`, `--nms`, `--max-det` control the embedded NMS.

Metadata (backbone, img_size, mean, std, strides, etc.) is embedded in the ONNX file.

---

### 6. SimMIM Self-Supervised Pre-training

`simmim_pretrain.py` implements masked image modeling (SimMIM) for pre-training a backbone on your own **unlabeled** images. The saved encoder weights can be loaded directly by SWAP-Fighter's `Backbone` (and by `MultiLabelClassifier` when `pretrained_source='simmim'`), **no key remapping needed**.

#### How it works

1. A random block mask (default `mask_patch_size=32`, `mask_ratio=0.6`) is generated at pixel resolution.
2. The masked image is fed into the timm backbone (`features_only=True`).
3. A mask token is injected right after the stem / patch_embed stage.
4. A lightweight linear decoder reconstructs the pixel values of the masked regions.
5. Loss = L1 on masked pixels only.

The architecture auto-adapts to Swin / ConvNeXt / ViT (incl. DINOv2 / DINOv3 / Eva) / MobileNet / EfficientNet / generic CNN by detecting the correct inner backbone, patch size and stage-1 channel count.

#### Pre-training

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
    --weight-decay 0.05 \
    --warmup-epochs 3 \
    --precision fp16 \
    --output ./simmim_pretrained/convnextv2_atto_simmim.pth
```

#### Key options

- `--data-dir`: folder containing unlabeled images (recursively scanned).
- `--backbone`: timm model name (must match the one used in `train.py`).
- `--img-size`: input size for pre-training (default `224`).
- `--mask-patch-size`: mask block size in pixels (default `32`). Must divide `img_size`.
- `--mask-ratio`: fraction of patches to mask (default `0.6`).
- `--pretrained` / `--no-pretrained`: start from timm ImageNet weights or random init (default: use ImageNet weights).
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--warmup-epochs`.
- `--grad-clip`: gradient clipping norm (default `5.0`).
- `--accumulation-steps`: gradient accumulation for large effective batch size.
- `--precision fp32|fp16|bf16`: AMP precision.
- `--output`: path to save the encoder state dict (`.pth`).

#### Using the pre-trained weights for detection/classification training

**For detect / obb:**

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_atto \
    --pretrained-source simmim \
    --backbone-pretrained-path ./simmim_pretrained/convnextv2_atto_simmim.pth \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --save-dir ./outputs
```

**For multilabel:**

```bash
python train.py \
    --task multilabel \
    --data-root ./datasets/multilabel_data \
    --train-labels train_labels.txt \
    --val-labels val_labels.txt \
    --num-classes 20 \
    --backbone convnextv2_atto \
    --pretrained-source simmim \
    --backbone-pretrained-path ./simmim_pretrained/convnextv2_atto_simmim.pth \
    --img-size 224 \
    --batch-size 64 \
    --epochs 40 \
    --save-dir ./outputs
```

> The saved state dict keys are directly compatible with the `Backbone` class — SWAP-Fighter's loader performs only the standard `module.` / `encoder.` prefix stripping, which matches what SimMIM exports.

#### Tips

- Use the **same `--backbone` name** in pre-training and in `train.py`, otherwise the keys won't match.
- `img-size` in pre-training does **not** have to match the downstream `img_size`. A common setup is `224` for pre-training and `640` (or `1024` for OBB) for detection.
- For ViT / DINOv2 / DINOv3 backbones, `dynamic_img_size=True` is used, so different downstream image sizes work fine.
- If GPU memory is limited, reduce `--batch-size` and increase `--accumulation-steps`.

---

### 7. Configuration

All hyperparameters are centralised in `swapfighter/config.py`. Task-specific defaults are in `swapfighter/tasks/*.py`.

Key parameters:

- `task`: `detect`, `obb`, `multilabel`
- `data_root`, `train_split`, `val_split`, `num_classes`
- `backbone_name`, `pretrained`, `img_size`
- `neck_out_channels`, `head_num_convs`, `use_extra_conv`
- `neck_type`, `neck_num_layers`
- `head_type`, `head_norm_layer`, `head_act_layer`
- `with_angle`, `angle_range`, `angle_format`
- `reg_max`, `dfl_weight`
- `obb_reg_loss`, `kld_weight`, `kld_tau`, `kld_fun`
- `batch_size`, `lr`, `backbone_lr_scale`, `weight_decay`, `epochs`
- `warmup_epochs`, `optimizer`, `scheduler`
- `precision`: `fp32`, `fp16`, `bf16`
- `freeze_epochs`
- `mosaic_prob`, `mosaic_disable_epoch`
- `conf_threshold`, `nms_threshold`, `obb_nms_threshold`
- `mean`, `std`
- `eval_map_start_epoch`, `eval_conf_threshold`
- `patience`, `use_ema`, `ema_decay`
- `tal_topk`, `tal_alpha`, `tal_beta`
- `pool_type`, `head_type`, `loss`, `eval_threshold` (for multilabel)

---

## 📁 Project Structure

```text
SWAP-Fighter/
├── swapfighter/
│   ├── api.py                 # SF class (unified entry)
│   ├── config.py              # Config class
│   ├── tasks/                 # Task registry and defaults
│   │   ├── __init__.py
│   │   ├── detect.py
│   │   ├── obb.py
│   │   └── multilabel.py
│   ├── engine/
│   │   ├── base_trainer.py
│   │   ├── base_predictor.py
│   │   ├── trainers/
│   │   │   ├── detect.py
│   │   │   └── multilabel.py
│   │   ├── predictors/
│   │   │   ├── detect.py
│   │   │   ├── multilabel.py
│   │   │   └── onnx.py
│   │   └── exporters/
│   │       ├── onnx.py
│   │       └── _onnx_impl.py
│   ├── models/
│   │   ├── backbones.py
│   │   ├── necks/
│   │   │   ├── lightfpn.py
│   │   │   └── bifpn.py
│   │   ├── heads/
│   │   │   └── decoupled.py
│   │   ├── detectors/
│   │   │   └── detect_fcos.py
│   │   └── classifiers/
│   │       └── multi_label.py
│   ├── datasets/
│   │   ├── yolo_det.py
│   │   └── cls_dataset.py
│   └── utils/
│       ├── box_ops.py
│       ├── losses.py
│       ├── metrics.py
│       ├── postprocess.py
│       └── logger.py
├── train.py                     # Training entry
├── infer.py                     # Inference entry
├── export.py                    # ONNX export entry
├── simmim_pretrain.py           # SimMIM self-supervised pre-training
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

---

## 📝 Notes

- The old `--use-amp` flag is replaced by `--precision`.
- For `fp16`, `GradScaler` is used automatically.
- `bf16` requires Ampere or newer GPUs; otherwise it falls back to `fp16`.
- Non-CUDA devices automatically use `fp32`.
- OBB rotated NMS and OBB mAP are implemented in pure NumPy / PyTorch – no `shapely` required.
- ONNX post-processing depends on correct metadata (strides, img_size, mean, std).
- For multi-label classification, class names can be provided via `--class-names` or a `class_names.txt` file.
- SimMIM pre-training is **optional** — you can skip it and use timm ImageNet weights directly.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request at https://github.com/IntptrMax/SwapFighter.

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FCOS](https://github.com/tianzhi0549/FCOS) – original anchor-free detection idea.
- [YOLOv8](https://github.com/ultralytics/ultralytics) – TAL assigner, decoupled head, and training pipeline inspiration.
- [timm](https://github.com/huggingface/pytorch-image-models) – pre-trained backbone collection.
- [SimMIM](https://github.com/microsoft/SimMIM) – masked image modeling pre-training idea.

---

**Happy Swapping!** 🔄