# SWAP-Fighter 🔄⚔️

**"Swap your backbone, fight for better detection."**

SWAP-Fighter is a modular anchor-free object detection framework inspired by **FCOS**. It allows you to plug different `timm` backbones into a unified neck and a decoupled multi-task head, supporting both:

- **`detect`**: horizontal bounding box detection
- **`obb`**: oriented bounding box detection with angle prediction

> Chinese name: **接头霸王** – because you can “connect” almost any backbone you want.

---

## ✨ Key Features

- **Plug-and-play backbones** via `timm` (`features_only=True`).
- **Two detection modes**
  - `detect`: horizontal boxes
  - `obb`: oriented boxes with angle regression
- **Unified anchor-free detection head**
  - Classification: **Focal Loss** by default.
  - Regression:
    - `detect`: **GIoU Loss**
    - `obb`: **CRIoU + ProbIoU + Center L1**
  - Angle:
    - `sin_cos`: cosine loss + periodic L1 auxiliary loss
    - `direct`: periodic L1 loss
- **FCOS-style positive sample assignment**
  - center sampling
  - area-based GT assignment
  - minimum positive samples per GT
- **Full training pipeline**
  - AMP precision: `fp32`, `fp16`, `bf16`
  - `GradScaler` for `fp16`
  - warmup + cosine / step / multistep / lambda / OneCycle schedulers
  - gradient clipping
  - staged backbone freezing and unfreezing
  - resume from checkpoint
- **Flexible dataset support**
  - YOLO-style folder structure
  - `detect` labels: `class cx cy w h`
  - `obb` labels: `class x1 y1 x2 y2 x3 y3 x4 y4`
  - Cross-compatible label reading
- **Built-in evaluation**
  - `detect`: mAP@0.5, mAP@0.5:0.95, AR@100 via `torchmetrics`
  - `obb`: rotated IoU mAP@0.5 and mAP@0.5:0.95 via `shapely`
- **Rich visualisation and logging**
  - CSV training logs
  - automatic training curves
  - `runX` output folders
  - single-image inference with adaptive font and colours
- **Deployment**
  - ONNX export for both `detect` and `obb`
  - ONNX Runtime inference demo

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
git clone https://github.com/yourusername/SWAP-Fighter.git
cd SWAP-Fighter
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
- torchmetrics
- shapely
- onnx
- onnxruntime
- onnxsim / onnx-simplifier (optional, for `--simplify`)

> `shapely` is required for rotated NMS and OBB mAP evaluation.  
> `torchmetrics` is used for horizontal-box mAP evaluation.

---

### 2. Dataset Preparation

Your dataset should follow the YOLO-style folder structure:

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

#### Horizontal detection labels (`mode = detect`)

Recommended format:

```text
class_id cx cy w h
```

All values are normalized to `[0, 1]`.

The dataset loader also accepts YOLO OBB 9-column labels and converts them to horizontal bounding boxes.

#### Oriented detection labels (`mode = obb`)

Recommended format:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All values are normalized to `[0, 1]`.

The dataset loader also accepts 5-column YOLO detect labels and converts them to axis-aligned oriented boxes with angle `0`.

Set the dataset root in `config.py`:

```python
data_root = './datasets/your_dataset'
```

or pass it via command line:

```bash
--data-root ./datasets/your_dataset
```

---

### 3. Training

Start training with the default configuration:

```bash
python train.py
```

#### Horizontal detection example

```bash
python train.py \
    --mode detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_tiny \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --precision fp16 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup-epochs 5 \
    --freeze-epochs 5 \
    --eval-map-start-epoch 10
```

#### Oriented detection example

```bash
python train.py \
    --mode obb \
    --angle-format sin_cos \
    --data-root ./datasets/dota_like \
    --backbone convnextv2_tiny \
    --img-size 1024 \
    --batch-size 8 \
    --epochs 100 \
    --precision fp16 \
    --angle-weight 2.0
```

> `angle_range` is configured in `config.py` (`pi` or `pi2`).  
> `angle_format` can also be set in `config.py`, or overridden with `--angle-format`.

#### Key training options

- `--mode detect|obb`: select horizontal or oriented detection.
- `--angle-format sin_cos|direct`: OBB angle encoding.
- `--precision fp32|fp16|bf16`: training precision.
- `--optimizer sgd|adamw`: optimizer type.
- `--scheduler lambda|step|multistep|cosine|onecycle`: LR scheduler.
- `--warmup-epochs N`: warmup epochs for non-OneCycle schedulers.
- `--freeze-epochs N`: freeze backbone for the first N epochs.
- `--unfreeze-lr LR`: learning rate after backbone unfreezing.
- `--eval-fp32`: force FP32 validation loss for more stable evaluation.
- `--eval-map-start-epoch N`: skip mAP computation before epoch N.
- `--mosaic-prob P`: Mosaic probability.
- `--mosaic-disable-epoch N`: disable Mosaic after epoch N.
- `--no-plot`: disable training curve plotting.
- `--resume path/to/checkpoint.pth`: resume training.

The training script automatically creates a new `runX` folder under `--save-dir` (default `./outputs`), containing:

```text
runX/
├── best.pth
├── last.pth
├── final.pth
├── epoch_N.pth
├── training_log.csv
├── config.json
└── training_curves.png
```

---

### 4. Inference on a Single Image

#### Horizontal detection

```bash
python infer.py \
    --image path/to/image.jpg \
    --weights ./outputs/run1/best.pth \
    --mode detect \
    --conf 0.7 \
    --nms 0.3 \
    --output result.jpg \
    --class-names pill capsule tablet
```

#### Oriented detection

```bash
python infer.py \
    --image path/to/image.jpg \
    --weights ./outputs/run1/best.pth \
    --mode obb \
    --angle-format sin_cos \
    --angle-range pi \
    --conf 0.5 \
    --output result_obb.jpg \
    --class-names plane ship vehicle
```

Use `--output-txt` to save detections to a text file.

---

### 5. ONNX Export

Export a trained model to ONNX:

#### Horizontal detection

```bash
python export.py \
    --weights ./outputs/run1/best.pth \
    --output model.onnx \
    --mode detect \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### Oriented detection

```bash
python export.py \
    --weights ./outputs/run1/best.pth \
    --output model_obb.onnx \
    --mode obb \
    --angle-format sin_cos \
    --img-size 640 \
    --dynamic \
    --simplify
```

Exported outputs:

- `detect`
  - `probs`: `(B, N, C)`, sigmoid probabilities
  - `bboxes`: `(B, N, 4)`, normalized `(cx, cy, w, h)`
- `obb`
  - `probs`: `(B, N, C)`, sigmoid probabilities
  - `bboxes`: `(B, N, 4)`, normalized `(cx, cy, w, h)`
  - `angles`: `(B, N, 2)`
    - `sin_cos`: `[sin, cos]`
    - `direct`: `[angle, 0]`, second value is a placeholder

> `angle_range` for OBB export is read from `config.py`.

---

### 6. ONNX Inference Demo

#### Horizontal detection

```bash
python onnx_demo.py \
    --model model.onnx \
    --image test.jpg \
    --mode detect \
    --conf-threshold 0.7 \
    --nms-threshold 0.3 \
    --class-names pill capsule tablet
```

#### Oriented detection

```bash
python onnx_demo.py \
    --model model_obb.onnx \
    --image test.jpg \
    --mode obb \
    --angle-format sin_cos \
    --angle-range pi \
    --conf-threshold 0.5 \
    --nms-threshold 0.3 \
    --class-names plane ship vehicle
```

> Important: if the model strides are not `[8, 16, 32]`, pass `--strides` explicitly so that ONNX post-processing slices the output correctly.

---

## 🧠 How It Works

1. **Backbone** extracts multi-scale feature maps via `timm`.
2. **LightFPN** fuses features from different levels.
3. **Decoupled multi-task head** predicts:
   - classification logits
   - box regression
   - angle prediction for OBB mode
4. **Positive assignment** follows an FCOS-style anchor-free strategy:
   - center sampling
   - area-based assignment
   - minimum positive samples per GT
5. **Losses**
   - classification: Focal Loss
   - `detect` regression: GIoU
   - `obb` regression: CRIoU + ProbIoU + Center L1
   - angle: periodic L1 or cosine + periodic L1
6. **Post-processing**
   - confidence threshold
   - horizontal NMS for `detect`
   - rotated NMS for `obb`
7. **Evaluation**
   - `detect`: torchmetrics mAP
   - `obb`: shapely rotated IoU mAP

The backbone is fully swappable without changing the neck, head, loss, or post-processing code.

---

## 🛠️ Configuration

All global hyperparameters are centralised in `config.py`. Many can be overridden from the command line.

Key parameters:

- `mode`: `detect` or `obb`
- `data_root`, `train_split`, `val_split`, `num_classes`
- `backbone_name`, `pretrained`, `img_size`
- `neck_out_channels`, `head_num_convs`, `use_extra_conv`
- `with_angle`, `angle_range`, `angle_format`
- `batch_size`, `lr`, `unfreeze_lr`, `weight_decay`, `epochs`
- `warmup_epochs`, `optimizer`, `scheduler`
- `precision`: `fp32`, `fp16`, `bf16`
- `freeze_epochs`
- `mosaic_prob`, `mosaic_disable_epoch`
- `conf_threshold`, `nms_threshold`, `obb_nms_threshold`
- `mean`, `std`
- `eval_map_start_epoch`

---

## 📁 Project Structure

```text
SWAP-Fighter/
├── config.py          # Global configuration
├── dataset.py         # YOLO-style dataset with letterbox, augmentation, Mosaic
├── model.py           # Backbone, LightFPN, detection head, DetectFCOS
├── train.py           # Training loop, logging, checkpointing, plotting
├── infer.py           # Single-image PyTorch inference
├── export.py          # ONNX exporter
├── onnx_demo.py       # ONNX Runtime inference demo
├── utils.py           # Losses, post-processing, NMS, mAP, letterbox
├── README.md          # English README
└── README.zh-CN.md    # Chinese README
```

---

## 📝 Notes

- The old `--use-amp` flag is replaced by `--precision`.
- For `fp16`, `GradScaler` is used automatically.
- `bf16` requires Ampere or newer GPUs; otherwise it falls back to `fp16`.
- Non-CUDA devices automatically use `fp32`.
- OBB rotated NMS and OBB mAP require `shapely`.
- ONNX post-processing depends on correct `--strides`.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request for improvements, bug fixes, or new backbone support.

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FCOS](https://github.com/tianzhi0549/FCOS) – original anchor-free detection idea.
- [timm](https://github.com/huggingface/pytorch-image-models) – pre-trained backbone collection.
- [torchmetrics](https://torchmetrics.readthedocs.io/) – mAP computation for horizontal boxes.
- [shapely](https://shapely.readthedocs.io/) – rotated IoU and OBB evaluation.

---

**Happy Swapping!** 🔄