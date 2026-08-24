# SWAP-Fighter 🔄⚔️

**“换掉你的骨干，为更好的检测而战。”**

SWAP-Fighter 是一个灵活、模块化的目标检测框架，基于 **FCOS**（全卷积单阶段目标检测）构建。它允许你自由**替换**各种主流骨干网络，同时保持统一的颈部（FPN）和头部（Head），非常适合在不同架构间快速实验和部署。

> 中文名称：**接头霸王** —— 因为你可以随意“接”上任何你想要的骨干网络！

---

## ✨ 主要特点

- **即插即用的骨干网络** – 无缝切换 ConvNeXt、MobileNet、ResNet、Swin Transformer、ViT、DINOv3、EfficientNet 等（通过 `timm` 支持）。
- **统一的 FCOS 检测头** – 分类和回归分支使用 **Varifocal Loss** 和 **GIoU Loss**。
- **完整的训练流程** – 支持混合精度训练（AMP）、学习率预热、余弦退火以及分阶段冻结骨干网络。
- **轻松部署** – 导出 ONNX 模型，输出归一化的 `probs`（概率）和 `bboxes`（`cx,cy,w,h` 格式），便于在各种平台上推理。
- **内置评估指标** – 使用 torchmetrics 计算 mAP@0.5、mAP@0.5:0.95 和 AR@100。
- **丰富的可视化** – 推理脚本自适应字体大小和边框颜色，训练过程自动绘制曲线图。

---

## 📦 支持的骨干网络

该框架借助 `timm` 库，可以使用大量预训练模型。任何支持 `features_only=True` 并返回特征图的骨干网络都可以使用。以下系列已经过测试并推荐：

| 系列            | 示例名称                                  |
|-----------------|-------------------------------------------|
| ConvNeXt        | `convnextv2_tiny`, `convnext_base`        |
| MobileNet       | `mobilenetv3_large_100`, `mobilenetv2_140`|
| ResNet          | `resnet50`, `resnet101`, `resnext50_32x4d`|
| Swin Transformer| `swin_tiny_patch4_window7_224`, `swin_base`|
| ViT             | `vit_base_patch16_224`, `vit_large_patch16_224`|
| DINOv3          | `dinov2_vitb14`, `dinov2_vitl14`          |
| EfficientNet    | `efficientnet_b0` ~ `b7`                  |

> 你可以使用 `timm.create_model` 接受的任何骨干名称 —— 只需通过 `--backbone` 参数传入或修改 `config.py` 即可。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/yourusername/SWAP-Fighter.git
cd SWAP-Fighter
pip install -r requirements.txt
```

**核心依赖**：
- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- timm
- opencv-python
- Pillow
- numpy
- onnx, onnxruntime（用于 ONNX 推理）
- onnx-simplifier（可选）
- torchmetrics（用于 mAP 评估）
- matplotlib（用于绘图）

### 2. 数据集准备

数据集应遵循 **YOLO 风格** 的目录结构：

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

每个标签文件每行表示一个目标，格式为：
```
class_id cx cy w h
```
（所有值归一化到 [0,1]）。

在 `config.py` 中设置 `data_root = './datasets/your_dataset'`，或通过命令行指定。

### 3. 训练

使用默认配置开始训练：

```bash
python train.py
```

**常用命令行覆盖参数：**

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

所有参数都是可选的，会覆盖 `config.py` 中的对应值。训练脚本会自动在 `--save-dir`（默认为 `./outputs`）下创建 `runX` 文件夹，用于存放检查点、日志、配置和训练曲线图。

**关键训练选项：**
- `--freeze-epochs N`：前 N 轮冻结骨干网络，之后以较低学习率（`--unfreeze-lr`）解冻。
- `--use-amp`：启用自动混合精度。
- `--resume`：从 `.pth` 检查点恢复训练。

### 4. 单张图片推理

```bash
python infer.py \
    --image path/to/image.jpg \
    --weights ./outputs/run1/best.pth \
    --conf 0.8 \
    --output result.jpg \
    --class-names pill capsule tablet    # 可选类别名称
```

脚本会显示检测结果并保存至 `--output`。你也可以通过 `--output-txt` 将检测结果保存为文本文件。

### 5. 导出 ONNX

将训练好的模型导出为 ONNX 格式，以便在 CPU/GPU 上快速推理：

```bash
python export.py \
    --weights ./outputs/run1/best.pth \
    --output model.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

导出的模型输出：
- `probs`：经过 Sigmoid 的概率分数，形状为 `(batch, num_points, num_classes)`
- `bboxes`：归一化到 [0,1] 的边界框，格式为 `(cx, cy, w, h)`，形状为 `(batch, num_points, 4)`

### 6. ONNX 推理演示

使用导出的 ONNX 模型进行检测：

```bash
python onnx_demo.py \
    --model model.onnx \
    --image test.jpg \
    --conf-threshold 0.8 \
    --class-names pill capsule tablet
```

该脚本会执行 Letterbox 填充、归一化以及后处理（NMS），与 PyTorch 推理保持一致。

---

## 🧠 工作原理

1. **骨干网络（Backbone）** 提取多尺度特征图。
2. **FPN** 融合不同层级的特征。
3. **FCOS 检测头** 在每个位置预测分类（Varifocal Loss）和边界框回归（GIoU Loss）。
4. **后处理** 解码预测结果，应用置信度阈值和 NMS。
5. 整个流程高度模块化：替换骨干网络时无需修改其他代码。

---

## 🛠️ 配置说明

所有超参数集中在 `config.py` 中，你也可以通过 `train.py`、`infer.py` 等脚本的命令行参数进行覆盖。

关键参数：
- `img_size`：输入分辨率（默认 224，但检测任务推荐 640）。
- `strides`：由骨干网络自动计算。
- `freeze_epochs` / `unfreeze_lr`：分阶段微调。
- `cls_weight` / `reg_weight`：损失权重。
- `mean` / `std`：归一化统计量（默认 ImageNet 值）。

---

## 📁 项目结构

```
SWAP-Fighter/
├── config.py          # 全局配置
├── dataset.py         # YOLO 风格数据集，含 Letterbox 和数据增强
├── model.py           # 骨干网络、FPN、FCOS 头以及 DetectFCOS 封装
├── train.py           # 训练循环，含日志、检查点和绘图
├── infer.py           # 单张图片 PyTorch 推理
├── export.py          # ONNX 导出器，输出归一化结果
├── onnx_demo.py       # ONNX 推理演示
├── utils.py           # 损失函数、后处理、mAP 评估、Letterbox 工具
└── README.md          # 英文说明（本文件为中文版）
```

---

## 🤝 贡献

欢迎贡献！如有改进、错误修复或新的骨干网络支持，请提交 Issue 或 Pull Request。

---

## 📄 许可证

本项目采用 MIT 许可证 – 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [FCOS](https://github.com/tianzhi0549/FCOS) – 原始论文和实现。
- [timm](https://github.com/huggingface/pytorch-image-models) – 提供了大量预训练骨干网络。
- [torchmetrics](https://torchmetrics.readthedocs.io/) – 高效的 mAP 计算。

---

**快乐“接头”！** 🔄