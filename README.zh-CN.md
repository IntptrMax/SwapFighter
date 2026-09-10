# SWAP-Fighter 🔄⚔️

**“换掉你的骨干，为更好的检测而战。”**

SWAP-Fighter 是一个受 **FCOS** 启发的模块化、无锚框目标检测框架。它允许你将不同的 `timm` 骨干网络接入统一的 Neck 和解耦多任务 Head，并同时支持：

- **`detect`**：水平框检测
- **`obb`**：旋转框检测，带角度预测

> 中文名称：**接头霸王** —— 因为你可以随意“接”上任何你想要的骨干网络！

---

## ✨ 主要特点

- **即插即用的骨干网络**：通过 `timm` 的 `features_only=True` 接入大量预训练模型。
- **两种检测模式**
  - `detect`：水平框
  - `obb`：旋转框 + 角度回归
- **统一的无锚框检测头**
  - 分类：默认使用 **Focal Loss**
  - 回归：
    - `detect`：**GIoU Loss**
    - `obb`：**CRIoU + ProbIoU + Center L1**
  - 角度：
    - `sin_cos`：余弦损失 + 周期 L1 辅助损失
    - `direct`：周期 L1 损失
- **FCOS 风格正样本分配**
  - 中心采样
  - 基于面积分配 GT
  - 每个 GT 保证最少正样本数
- **完整训练流程**
  - 精度：`fp32`、`fp16`、`bf16`
  - `fp16` 自动使用 `GradScaler`
  - warmup + cosine / step / multistep / lambda / OneCycle 调度器
  - 梯度裁剪
  - 分阶段冻结 / 解冻 Backbone
  - 支持断点恢复
- **灵活的数据集支持**
  - YOLO 风格目录结构
  - `detect` 标签：`class cx cy w h`
  - `obb` 标签：`class x1 y1 x2 y2 x3 y3 x4 y4`
  - 两种标签格式可互相兼容读取
- **内置评估指标**
  - `detect`：通过 `torchmetrics` 计算 mAP@0.5、mAP@0.5:0.95、AR@100
  - `obb`：通过 `shapely` 计算旋转 IoU 的 mAP@0.5、mAP@0.5:0.95
- **丰富的可视化与日志**
  - CSV 训练日志
  - 自动绘制训练曲线
  - 自动创建 `runX` 输出目录
  - 单图推理自适应字体和颜色
- **部署支持**
  - 支持 `detect` 和 `obb` 的 ONNX 导出
  - 提供 ONNX Runtime 推理示例

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
git clone https://github.com/yourusername/SWAP-Fighter.git
cd SWAP-Fighter
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
- torchmetrics
- shapely
- onnx
- onnxruntime
- onnxsim / onnx-simplifier（可选，用于 `--simplify`）

> `shapely` 用于旋转 NMS 和 OBB mAP 评估。  
> `torchmetrics` 用于水平框 mAP 评估。

---

### 2. 数据集准备

数据集应遵循 YOLO 风格的目录结构：

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

#### 水平检测标签（`mode = detect`）

推荐格式：

```text
class_id cx cy w h
```

所有值归一化到 `[0, 1]`。

数据加载器也兼容 YOLO OBB 9 列标签，并会将其转换为水平外接框。

#### 旋转检测标签（`mode = obb`）

推荐格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

所有值归一化到 `[0, 1]`。

数据加载器也兼容 5 列 YOLO 检测标签，并会将其转换为角度为 `0` 的轴对齐旋转框。

在 `config.py` 中设置数据集根目录：

```python
data_root = './datasets/your_dataset'
```

或通过命令行指定：

```bash
--data-root ./datasets/your_dataset
```

---

### 3. 训练

使用默认配置开始训练：

```bash
python train.py
```

#### 水平检测示例

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

#### 旋转检测示例

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

> `angle_range` 在 `config.py` 中配置，可选 `pi` 或 `pi2`。  
> `angle_format` 也可在 `config.py` 中设置，或用 `--angle-format` 覆盖。

#### 关键训练选项

- `--mode detect|obb`：选择水平框或旋转框检测。
- `--angle-format sin_cos|direct`：OBB 角度编码方式。
- `--precision fp32|fp16|bf16`：训练精度。
- `--optimizer sgd|adamw`：优化器类型。
- `--scheduler lambda|step|multistep|cosine|onecycle`：学习率调度器。
- `--warmup-epochs N`：非 OneCycle 调度器的 warmup 轮数。
- `--freeze-epochs N`：前 N 轮冻结 Backbone。
- `--unfreeze-lr LR`：Backbone 解冻后的学习率。
- `--eval-fp32`：验证损失强制 FP32，更稳定。
- `--eval-map-start-epoch N`：从第 N 轮开始计算 mAP。
- `--mosaic-prob P`：Mosaic 概率。
- `--mosaic-disable-epoch N`：第 N 轮后禁用 Mosaic。
- `--no-plot`：禁用训练曲线绘制。
- `--resume path/to/checkpoint.pth`：从检查点恢复训练。

训练脚本会自动在 `--save-dir`（默认 `./outputs`）下创建新的 `runX` 目录，包含：

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

### 4. 单张图片推理

#### 水平框检测

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

#### 旋转框检测

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

可使用 `--output-txt` 将检测结果保存为文本文件。

---

### 5. 导出 ONNX

将训练好的模型导出为 ONNX：

#### 水平框检测

```bash
python export.py \
    --weights ./outputs/run1/best.pth \
    --output model.onnx \
    --mode detect \
    --img-size 640 \
    --dynamic \
    --simplify
```

#### 旋转框检测

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

导出输出：

- `detect`
  - `probs`：`(B, N, C)`，Sigmoid 概率
  - `bboxes`：`(B, N, 4)`，归一化 `(cx, cy, w, h)`
- `obb`
  - `probs`：`(B, N, C)`，Sigmoid 概率
  - `bboxes`：`(B, N, 4)`，归一化 `(cx, cy, w, h)`
  - `angles`：`(B, N, 2)`
    - `sin_cos`：`[sin, cos]`
    - `direct`：`[angle, 0]`，第二维为占位

> OBB 导出时的 `angle_range` 从 `config.py` 读取。

---

### 6. ONNX 推理演示

#### 水平框检测

```bash
python onnx_demo.py \
    --model model.onnx \
    --image test.jpg \
    --mode detect \
    --conf-threshold 0.7 \
    --nms-threshold 0.3 \
    --class-names pill capsule tablet
```

#### 旋转框检测

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

> 重要：如果模型实际步长不是 `[8, 16, 32]`，请显式传入 `--strides`，确保 ONNX 后处理正确切分输出。

---

## 🧠 工作原理

1. **Backbone** 通过 `timm` 提取多尺度特征图。
2. **LightFPN** 融合不同层级的特征。
3. **解耦多任务 Head** 预测：
   - 分类 logits
   - 边界框回归
   - OBB 模式下的角度
4. **正样本分配** 采用 FCOS 风格的无锚框策略：
   - 中心采样
   - 基于面积分配
   - 每个 GT 保证最少正样本数
5. **损失函数**
   - 分类：Focal Loss
   - `detect` 回归：GIoU
   - `obb` 回归：CRIoU + ProbIoU + Center L1
   - 角度：周期 L1，或余弦 + 周期 L1
6. **后处理**
   - 置信度阈值
   - `detect`：水平 NMS
   - `obb`：旋转 NMS
7. **评估**
   - `detect`：torchmetrics mAP
   - `obb`：shapely 旋转 IoU mAP

整个流程高度模块化：替换 Backbone 时，无需修改 Neck、Head、损失和后处理代码。

---

## 🛠️ 配置说明

所有全局超参数集中在 `config.py` 中，很多参数可通过命令行覆盖。

关键参数：

- `mode`：`detect` 或 `obb`
- `data_root`、`train_split`、`val_split`、`num_classes`
- `backbone_name`、`pretrained`、`img_size`
- `neck_out_channels`、`head_num_convs`、`use_extra_conv`
- `with_angle`、`angle_range`、`angle_format`
- `batch_size`、`lr`、`unfreeze_lr`、`weight_decay`、`epochs`
- `warmup_epochs`、`optimizer`、`scheduler`
- `precision`：`fp32`、`fp16`、`bf16`
- `freeze_epochs`
- `mosaic_prob`、`mosaic_disable_epoch`
- `conf_threshold`、`nms_threshold`、`obb_nms_threshold`
- `mean`、`std`
- `eval_map_start_epoch`

---

## 📁 项目结构

```text
SWAP-Fighter/
├── config.py          # 全局配置
├── dataset.py         # YOLO 风格数据集，含 Letterbox、增强、Mosaic
├── model.py           # Backbone、LightFPN、检测头、DetectFCOS
├── train.py           # 训练循环、日志、检查点、绘图
├── infer.py           # 单张图片 PyTorch 推理
├── export.py          # ONNX 导出器
├── onnx_demo.py       # ONNX Runtime 推理演示
├── utils.py           # 损失函数、后处理、NMS、mAP、Letterbox
├── README.md          # 英文说明
└── README.zh-CN.md    # 中文说明
```

---

## 📝 注意事项

- 旧版 `--use-amp` 已由 `--precision` 取代。
- `fp16` 会自动使用 `GradScaler`。
- `bf16` 需要 Ampere 或更新架构 GPU，否则自动回退到 `fp16`。
- 非 CUDA 设备自动使用 `fp32`。
- OBB 旋转 NMS 和 OBB mAP 需要安装 `shapely`。
- ONNX 后处理依赖正确的 `--strides`。

---

## 🤝 贡献

欢迎贡献！如有改进、错误修复或新的骨干网络支持，请提交 Issue 或 Pull Request。

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 – 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [FCOS](https://github.com/tianzhi0549/FCOS) – 原始无锚框检测思想。
- [timm](https://github.com/huggingface/pytorch-image-models) – 大量预训练骨干网络。
- [torchmetrics](https://torchmetrics.readthedocs.io/) – 水平框 mAP 计算。
- [shapely](https://shapely.readthedocs.io/) – 旋转 IoU 和 OBB 评估。

---

**快乐“接头”！** 🔄