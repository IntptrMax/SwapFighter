# SWAP-Fighter 🔄⚔️

**"Swap your backbone, fight for better detection."**

SWAP-Fighter is a flexible and modular object detection framework built upon **FCOS** (Fully Convolutional One-Stage Object Detection). It allows you to **swap** various state‑of‑the‑art backbones while keeping a unified neck (FPN) and head, making it ideal for rapid experimentation and deployment across different architectures.

> 中文名称：**接头霸王** – 因为你可以任意“接”上不同的骨干网络！

---

## ✨ Key Features

- **Plug‑and‑Play Backbones** – Seamlessly switch between ConvNeXt, MobileNet, ResNet, Swin Transformer, ViT, DINOv3, EfficientNet, and more (via `timm`).
- **Unified FCOS Head** – Classification and regression towers with **Varifocal Loss** and **GIoU Loss**.
- **End‑to‑End Training** – Support for mixed precision (AMP), warmup, cosine annealing, and staged backbone freezing.
- **Easy Deployment** – Export to ONNX with normalized outputs (`probs`, `bboxes` in `cx,cy,w,h` format) for inference on any platform.
- **Built‑in Evaluation** – mAP@0.5, mAP@0.5:0.95, and AR@100 using torchmetrics.
- **Rich Visualisation** – Inference script with adaptive font sizes and coloured boxes; training curves automatically plotted.

---

## 📦 Supported Backbones

The framework leverages `timm` to access a wide range of pre‑trained models. Any backbone that returns feature maps (via `features_only=True`) can be used. The following families are tested and recommended:

| Family          | Example Names                          |
|-----------------|----------------------------------------|
| ConvNeXt        | `convnextv2_tiny`, `convnext_base`     |
| MobileNet       | `mobilenetv3_large_100`, `mobilenetv2_140` |
| ResNet          | `resnet50`, `resnet101`, `resnext50_32x4d` |
| Swin Transformer| `swin_tiny_patch4_window7_224`, `swin_base` |
| ViT             | `vit_base_patch16_224`, `vit_large_patch16_224` |
| DINOv3          | `dinov2_vitb14`, `dinov2_vitl14`       |
| EfficientNet    | `efficientnet_b0` ~ `b7`               |

> You can use any backbone name that `timm.create_model` accepts – just pass it via `--backbone` or modify `config.py`.

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/yourusername/SWAP-Fighter.git
cd SWAP-Fighter
pip install -r requirements.txt
```

**Requirements** (core):
- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- timm
- opencv-python
- Pillow
- numpy
- onnx, onnxruntime (for inference)
- onnx-simplifier (optional)
- torchmetrics (for mAP evaluation)
- matplotlib (for plotting)

### 2. Dataset Preparation

Your dataset should follow the **YOLO‑style** folder structure:

```
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

Each label file contains one line per object in the format:
```
class_id cx cy w h
```
(all values normalized to [0,1]).

Set the path in `config.py` (`data_root = './datasets/your_dataset'`) or pass via command line.

### 3. Training

Start training with the default configuration:

```bash
python train.py
```

**Common command‑line overrides:**

```bash
python train.py \
    --data-root ./datasets/my_data \
    --backbone convnextv2_tiny \
    --img-size 640 \
    --batch-size 32 \
    --lr 1e-3 \
    --epochs 100 \
    --freeze-epochs 10 \
    --use-amp
```

All arguments are optional; they override values in `config.py`. The training script automatically creates a new `runX` folder under `--save-dir` (default `./outputs`) to store checkpoints, logs, config, and plots.

**Key training options:**
- `--freeze-epochs N`: freeze backbone for the first N epochs, then unfreeze with a lower learning rate (`--unfreeze-lr`).
- `--use-amp`: enable automatic mixed precision.
- `--resume`: resume from a checkpoint `.pth` file.

### 4. Inference on a Single Image

```bash
python infer.py \
    --image path/to/image.jpg \
    --weights ./outputs/run1/best.pth \
    --conf 0.8 \
    --output result.jpg \
    --class-names pill capsule tablet    # optional class names
```

The script displays the result and saves it to `--output`. You can also save detections to a text file with `--output-txt`.

### 5. ONNX Export

Export the trained model to ONNX for fast inference on CPU/GPU:

```bash
python export.py \
    --weights ./outputs/run1/best.pth \
    --output model.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

The exported model outputs:
- `probs`: probability scores (sigmoid applied) of shape `(batch, num_points, num_classes)`
- `bboxes`: bounding boxes in `(cx, cy, w, h)` format normalized to [0,1].

### 6. ONNX Inference Demo

Run detection using the exported ONNX model:

```bash
python onnx_demo.py \
    --model model.onnx \
    --image test.jpg \
    --conf-threshold 0.8 \
    --class-names pill capsule tablet
```

It performs letterboxing, normalisation, and post‑processing (NMS) just like the PyTorch inference script.

---

## 🧠 How It Works

1. **Backbone** extracts multi‑scale feature maps.
2. **FPN** fuses features from different levels.
3. **FCOS Head** predicts classification (Varifocal Loss) and bounding‑box regression (GIoU Loss) at each location.
4. **Post‑processing** decodes predictions, applies confidence threshold, and performs NMS.
5. The entire pipeline is modular: you can swap the backbone without changing the rest of the code.

---

## 🛠️ Configuration

All hyperparameters are centralised in `config.py`; you can also override them via command‑line arguments when running `train.py`, `infer.py`, etc.

Key parameters:
- `img_size`: input resolution (default 224, but 640 is recommended for detection).
- `strides`: automatically computed from the backbone.
- `freeze_epochs` / `unfreeze_lr`: staged fine‑tuning.
- `cls_weight` / `reg_weight`: loss balancing.
- `mean` / `std`: normalisation statistics (default to ImageNet values if not changed).

---

## 📁 Project Structure

```
SWAP-Fighter/
├── config.py          # All global settings
├── dataset.py         # YOLO‑style dataset with letterbox and augmentations
├── model.py           # Backbone, FPN, FCOS head, and DetectFCOS wrapper
├── train.py           # Training loop with logging, checkpointing, and plotting
├── infer.py           # Single‑image PyTorch inference
├── export.py          # ONNX exporter with wrapper for normalised outputs
├── onnx_demo.py       # ONNX inference demo
├── utils.py           # Loss functions, post‑processing, mAP evaluation, letterbox
└── README.md          # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request for any improvements, bug fixes, or new backbone support.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FCOS](https://github.com/tianzhi0549/FCOS) – original paper and implementation.
- [timm](https://github.com/huggingface/pytorch-image-models) – for providing a vast collection of pre‑trained backbones.
- [torchmetrics](https://torchmetrics.readthedocs.io/) – for efficient mAP computation.

---

**Happy Swapping!** 🔄