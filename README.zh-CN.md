# SWAP-Fighter 🔄⚔️

**“换掉你的骨干，为更好的检测而战。”**

SWAP-Fighter 是一个受 **FCOS** 和 **YOLOv8** 启发的模块化、无锚框目标检测框架。它允许你将不同的 `timm` 骨干网络接入统一的 Neck 和解耦多任务 Head，并支持：

- **`detect`**：水平框检测
- **`obb`**：旋转框检测，带角度预测
- **`multilabel`**：多标签图像分类

> 中文名称：**接头霸王** —— 因为你可以随意“接”上任何你想要的骨干网络！

**GitHub 仓库**：https://github.com/IntptrMax/SwapFighter

---

## ✨ 主要特点

- **即插即用的骨干网络**：通过 `timm` 的 `features_only=True` 接入大量预训练模型。
- **三种任务模式**
  - `detect`：水平框
  - `obb`：旋转框 + 角度回归
  - `multilabel`：多标签分类
- **统一的无锚框检测头**（用于 detect/obb）
  - **解耦头**：分类、回归、角度分别使用独立的卷积塔。
  - 分类：**Focal Loss** 风格（BCE + 目标分数归一化）。
  - 回归：
    - `detect`：**CIoU Loss** + 可选 **DFL**（Distribution Focal Loss）。
    - `obb`：**CRIoU + ProbIoU** 或 **KLD Loss**（可选）+ 可选 **DFL**。
  - 角度：
    - `sin_cos`：余弦损失 + 周期 L1 辅助损失
    - `direct`：周期 L1 损失
- **多标签分类头**
  - 池化：`gap`、`gem`、`attention`
  - 分类头：`linear` 或 `mlp`
  - 损失：`bce` 或 `asl`（Asymmetric Loss）
- **Task-Aligned Assigner (TAL)** 正样本分配。
- **完整训练流程**
  - 精度：`fp32`、`fp16`、`bf16`
  - `fp16` 自动使用 `GradScaler`
  - Warmup + cosine / step / multistep / lambda / OneCycle 调度器
  - 梯度裁剪
  - EMA（指数移动平均）
  - 早停（patience）
  - 支持断点恢复
- **灵活的数据集支持**
  - detect/obb 使用 YOLO 风格目录结构
  - multilabel 支持 txt 标签文件（id-list 和 one-hot 两种格式）
  - multilabel 支持 HuggingFace datasets
- **内置评估指标**
  - `detect`：mAP@0.5、mAP@0.5:0.95、AR@100（YOLOv8 风格）
  - `obb`：基于 ProbIoU 的旋转 IoU mAP@0.5、mAP@0.5:0.95
  - `multilabel`：mAP、CF1、OF1、EMR
- **SimMIM 自监督预训练**
  - 通过 `simmim_pretrain.py` 在自有无标注图像上预训练骨干网络。
  - 生成的权重可直接被 SWAP-Fighter 的 `Backbone` 加载，**无需任何键名归一化**。
  - 支持 Swin / ConvNeXt / ViT（含 DINOv2 / DINOv3 / Eva）/ MobileNet / EfficientNet / 通用 CNN。
- **丰富的可视化与日志**
  - CSV 训练日志
  - 自动绘制训练曲线（`results.png`）
  - 自动创建 `runX` 输出目录
  - 单图推理自适应字体和颜色
- **部署支持**
  - 所有任务支持 ONNX 导出（raw 或 end2end）
  - 通过 `infer.py --format onnx` 进行 ONNX Runtime 推理

---

## 📦 支持的骨干网络

框架借助 `timm` 使用大量预训练模型。理论上，任何支持 `timm.create_model(..., features_only=True)` 的骨干都可以使用。

以下系列已经过测试并推荐：

| 系列 | 示例名称 |
|---|---|
| ConvNeXt | `convnextv2_atto`, `convnextv2_tiny`, `convnext_base` |
| MobileNet | `mobilenetv3_large_100`, `mobilenetv2_140` |
| ResNet | `resnet50`, `resnet101`, `resnext50_32x4d` |
| Swin Transformer | `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224` |
| ViT | `vit_base_patch16_224`, `vit_large_patch16_224` |
| DINOv2 / DINOv3 | `dinov2_vitb14`, `dinov2_vitl14` 以及其他 `timm` 支持的变体 |
| EfficientNet | `efficientnet_b0` ~ `b7` |

Backbone 特征选择逻辑是自动的：

- ViT / DINOv2 风格：取最后三个特征图。
- Swin 风格：选择步长为 `8`、`16`、`32` 的特征图。
- 其他 CNN：取最后三个特征图。
- 输出通道数和步长自动推断。

你可以通过 `--backbone` 传入任何 `timm` 接受的模型名，或直接修改 `config.py`。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/IntptrMax/SwapFighter.git
cd SwapFighter
pip install -r requirements.txt
```

**核心依赖：**

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
- onnxsim / onnx-simplifier（可选，用于 `--simplify`）
- datasets（可选，用于 HuggingFace 多标签数据集）

> 不再需要 `shapely` 或 `torchmetrics` —— 评估和旋转 NMS 均使用纯 NumPy / PyTorch 实现。

---

### 2. 数据集准备

#### 2.1 水平检测（`task = detect`）和旋转检测（`task = obb`）

使用 YOLO 风格目录结构：

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

**水平检测标签（`detect`）：**

```text
class_id cx cy w h
```

所有值归一化到 `[0, 1]`。数据加载器也兼容 YOLO OBB 9 列标签，并会将其转换为水平外接框。

**旋转检测标签（`obb`）：**

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

所有值归一化到 `[0, 1]`。数据加载器也兼容 5 列 YOLO 检测标签，并会将其转换为角度为 `0` 的轴对齐旋转框。

在 `config.py` 中设置数据集根目录，或通过 `--data-root` 指定。

#### 2.2 多标签分类（`task = multilabel`）

支持两种数据源：本地 txt 文件或 HuggingFace 数据集。

**本地 txt 文件：**

```text
datasets/
└── your_multilabel_dataset/
    ├── train2017/          # 训练图像
    │   ├── img1.jpg
    │   └── ...
    ├── val2017/            # 验证图像
    │   ├── img2.jpg
    │   └── ...
    ├── train_labels.txt
    └── val_labels.txt
```

`train_labels.txt` / `val_labels.txt` 每行支持两种格式：

- **新格式**：`filename\t[0, 14, 26]`（制表符分隔，类别 id 列表）
- **旧格式**：`filename 0 0 1 0 1 ...`（空格分隔，长度为 `num_classes` 的 one-hot 向量）

新格式示例：
```text
img1.jpg	[0, 14, 26]
img2.jpg	[3]
img3.jpg	[]
```

**HuggingFace 数据集：**

数据集必须包含 `image` 和 `labels` 字段，可选 `label_names` 用于类名。使用 `--hf-dataset`、`--hf-config`、`--hf-train-split`、`--hf-val-split` 指定。

#### 2.3 SimMIM 预训练（无标注图像）

SimMIM 预训练只需要一个无标注图像文件夹（递归扫描）：

```text
datasets/
└── unlabeled_images/
    ├── any_subdir/
    │   ├── a.jpg
    │   └── ...
    └── b.png
```

支持的扩展名：`.jpg`、`.jpeg`、`.png`、`.bmp`（不区分大小写）。

---

### 3. 训练

训练脚本为 `train.py`，内部使用 `SF` API。

#### 水平检测示例

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

如需设置其他配置项，可在命令末尾追加 `key=value`：

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

#### 旋转检测示例

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

#### 多标签分类示例（本地 txt）

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

#### 多标签分类示例（HuggingFace）

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

#### 关键训练选项

- `--task detect|obb|multilabel`：选择任务。
- `--data-root`：数据集根目录。
- `--backbone`：timm 模型名。
- `--img-size`：输入图像尺寸。
- `--batch-size`：批大小。
- `--epochs`：训练轮数。
- `--lr`：学习率。
- `--precision fp32|fp16|bf16`：训练精度。
- `--optimizer sgd|adamw`：优化器。
- `--scheduler lambda|step|multistep|cosine|onecycle`：学习率调度器。
- `--patience`：早停耐心值（0 = 禁用）。
- `--resume`：从检查点恢复。
- `--no-ema`：禁用 EMA（默认启用）。
- `--save-dir`：输出目录。

其他配置项（如 `warmup_epochs`、`freeze_epochs`、`mosaic_prob`、`backbone_lr_scale`、`ema_decay`）可通过在命令末尾追加 `key=value` 覆盖。

训练脚本会自动在 `--save-dir` 下创建新的 `runX` 目录，包含：
```text
runX/
├── best.pth
├── last.pth
├── training_log.csv
├── config.json
└── results.png
```

---

### 4. 推理

#### 水平检测（PyTorch）

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

#### 旋转检测（PyTorch）

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

#### 多标签分类（PyTorch）

```bash
python infer.py \
    --task multilabel \
    --weights ./outputs/multilabel/run1/best.pth \
    --image path/to/image.jpg \
    --threshold 0.5 \
    --top-k 5 \
    --output result.jpg
```

#### ONNX 推理

使用 `--format onnx`，`--weights` 指向 `.onnx` 文件。

```bash
python infer.py \
    --format onnx \
    --weights model.onnx \
    --image test.jpg \
    --conf 0.7 \
    --nms 0.3 \
    --class-names pill capsule tablet
```

对于 OBB，还需传入 `--angle-format` 和 `--angle-range`。

> 如果模型步长不是 `[8, 16, 32]`，请传入 `--img-size`，并确保 ONNX metadata 中包含正确的 strides。ONNX 推理器会优先从 metadata 读取。

其他选项：
- `--output-dir`：将多张图片结果保存到目录。
- `--show`：显示结果窗口。
- `--show-size`：窗口最大宽度。

---

### 5. 导出 ONNX

将训练好的模型导出为 ONNX：

#### 水平检测

```bash
python export.py \
    --weights ./outputs/detect/run1/best.pth \
    --output model.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### 旋转检测

```bash
python export.py \
    --weights ./outputs/obb/run1/best.pth \
    --output model_obb.onnx \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### 多标签分类

```bash
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --output model_cls.onnx \
    --img-size 224 \
    --dynamic \
    --simplify
```

#### 导出模式

- **Raw 模式**（默认）：输出单个张量 `output0`。
  - detect：形状 `(B, 4+nc, P)`，值 `[cx, cy, w, h, cls...]` 归一化到 `[0,1]`。
  - obb：形状 `(B, 5+nc, P)`，值 `[cx, cy, w, h, angle, cls...]` 归一化。
  - multilabel：输出 `logits`，形状 `(B, num_classes)`。
- **End2end 模式**（`--end2end`，仅 detect/obb）：输出 `num_dets`、`boxes`、`scores`、`classes`。NMS 已固化。`--conf`、`--nms`、`--max-det` 控制固化的 NMS 参数。

Metadata（backbone、img_size、mean、std、strides 等）会写入 ONNX 文件。

---

### 6. SimMIM 自监督预训练

`simmim_pretrain.py` 实现了 SimMIM（Masked Image Modeling），可以在你自己的**无标注**图像上对骨干网络做自监督预训练。生成的 encoder 权重可以直接被 SWAP-Fighter 的 `Backbone`（以及 `MultiLabelClassifier` 在 `pretrained_source='simmim'` 时）加载，**无需任何键名重映射**。

#### 工作原理

1. 在像素分辨率生成随机块掩码（默认 `mask_patch_size=32`，`mask_ratio=0.6`）。
2. 掩码后的图像送入 timm 骨干网络（`features_only=True`）。
3. 在 stem / patch_embed 之后立即注入 mask token。
4. 一个轻量级线性解码器重建被掩码区域的像素值。
5. 损失 = 仅在被掩码像素上的 L1 损失。

实现会自动识别内层 backbone、patch size、stage-1 通道数，从而适配 Swin / ConvNeXt / ViT（含 DINOv2 / DINOv3 / Eva）/ MobileNet / EfficientNet / 通用 CNN。

#### 预训练

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

#### 关键选项

- `--data-dir`：无标注图像文件夹（递归扫描）。
- `--backbone`：timm 模型名（必须与 `train.py` 中使用的一致）。
- `--img-size`：预训练输入尺寸（默认 `224`）。
- `--mask-patch-size`：掩码块像素大小（默认 `32`）。必须整除 `img_size`。
- `--mask-ratio`：被掩码 patch 的比例（默认 `0.6`）。
- `--pretrained` / `--no-pretrained`：是否从 timm ImageNet 权重开始（默认：使用 ImageNet 权重）。
- `--epochs`、`--batch-size`、`--lr`、`--weight-decay`、`--warmup-epochs`。
- `--grad-clip`：梯度裁剪范数（默认 `5.0`）。
- `--accumulation-steps`：梯度累积，用于等效增大 batch size。
- `--precision fp32|fp16|bf16`：AMP 精度。
- `--output`：encoder state dict 保存路径（`.pth`）。

#### 使用预训练权重训练检测/分类模型

**用于 detect / obb：**

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

**用于 multilabel：**

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

> 保存的 state dict 键名与 `Backbone` 类完全兼容 —— SWAP-Fighter 的加载器只会做标准的 `module.` / `encoder.` 前缀剥离，与 SimMIM 导出的键名完全匹配。

#### 使用建议

- 预训练和 `train.py` 中使用**相同的 `--backbone` 名**，否则键名不匹配。
- 预训练的 `img-size` **不必**与下游 `img_size` 相同。常见配置是：预训练 `224`，检测 `640`（OBB 常用 `1024`）。
- 对 ViT / DINOv2 / DINOv3 骨干，由于启用了 `dynamic_img_size=True`，下游不同尺寸输入不会出问题。
- 如果显存不足，减小 `--batch-size` 并增大 `--accumulation-steps`。

---

### 7. 配置说明

所有全局超参数集中在 `swapfighter/config.py`。任务默认配置在 `swapfighter/tasks/*.py`。

关键参数：

- `task`：`detect`、`obb`、`multilabel`
- `data_root`、`train_split`、`val_split`、`num_classes`
- `backbone_name`、`pretrained`、`img_size`
- `neck_out_channels`、`head_num_convs`、`use_extra_conv`
- `neck_type`、`neck_num_layers`
- `head_type`、`head_norm_layer`、`head_act_layer`
- `with_angle`、`angle_range`、`angle_format`
- `reg_max`、`dfl_weight`
- `obb_reg_loss`、`kld_weight`、`kld_tau`、`kld_fun`
- `batch_size`、`lr`、`backbone_lr_scale`、`weight_decay`、`epochs`
- `warmup_epochs`、`optimizer`、`scheduler`
- `precision`：`fp32`、`fp16`、`bf16`
- `freeze_epochs`
- `mosaic_prob`、`mosaic_disable_epoch`
- `conf_threshold`、`nms_threshold`、`obb_nms_threshold`
- `mean`、`std`
- `eval_map_start_epoch`、`eval_conf_threshold`
- `patience`、`use_ema`、`ema_decay`
- `tal_topk`、`tal_alpha`、`tal_beta`
- `pool_type`、`head_type`、`loss`、`eval_threshold`（多标签分类）

---

## 📁 项目结构

```text
SWAP-Fighter/
├── swapfighter/
│   ├── api.py                 # SF 类（统一入口）
│   ├── config.py              # Config 类
│   ├── tasks/                 # 任务注册与默认配置
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
├── train.py                     # 训练入口
├── infer.py                     # 推理入口
├── export.py                    # ONNX 导出入口
├── simmim_pretrain.py           # SimMIM 自监督预训练
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

---

## 📝 注意事项

- 旧版 `--use-amp` 已由 `--precision` 取代。
- `fp16` 会自动使用 `GradScaler`。
- `bf16` 需要 Ampere 或更新架构 GPU，否则自动回退到 `fp16`。
- 非 CUDA 设备自动使用 `fp32`。
- OBB 旋转 NMS 和 OBB mAP 均使用纯 NumPy / PyTorch 实现，无需安装 `shapely`。
- ONNX 后处理依赖正确的 metadata（strides、img_size、mean、std）。
- 多标签分类的类名可通过 `--class-names` 或 `class_names.txt` 文件提供。
- SimMIM 预训练是**可选**的 —— 你也可以直接使用 timm ImageNet 权重。

---

## 🤝 贡献

欢迎贡献！如有改进、错误修复或新的骨干网络支持，请到 https://github.com/IntptrMax/SwapFighter 提交 Issue 或 Pull Request。

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 – 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [FCOS](https://github.com/tianzhi0549/FCOS) – 原始无锚框检测思想。
- [YOLOv8](https://github.com/ultralytics/ultralytics) – TAL 分配器、解耦头及训练流程灵感。
- [timm](https://github.com/huggingface/pytorch-image-models) – 大量预训练骨干网络。
- [SimMIM](https://github.com/microsoft/SimMIM) – 掩码图像建模预训练思想。

---

**快乐“接头”！** 🔄