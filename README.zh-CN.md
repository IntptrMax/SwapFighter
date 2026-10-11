# SWAP-Fighter 🔄⚔️

**“换掉你的骨干，为更好的检测而战。”**

SWAP-Fighter 是一个受 **FCOS** 和 **YOLOv8** 启发的模块化、无锚框框架。它允许你将不同的 `timm` 骨干网络接入统一的 Neck 和解耦多任务 Head，并支持：

- **`detect`**：水平框检测
- **`obb`**：旋转框检测（带角度预测）
- **`segment`**：实例分割（YOLOv8-seg 风格）
- **`keypoint`**：多人姿态估计（YOLOv8-pose 风格）
- **`multilabel`**：多标签图像分类
- **`linear_probe`**：冻结骨干 + 线性/多层头的多标签分类

> 中文名称：**接头霸王** —— 因为你可以随意“接”上任何你想要的骨干网络！

**GitHub 仓库**：https://github.com/IntptrMax/SwapFighter

---

## ✨ 主要特点

- **即插即用的骨干网络**：通过 `timm` 的 `features_only=True` 接入大量预训练模型。
  可用的推荐列表见 [BackboneList.md](BackboneList.md)。
- **六种任务模式**
  - `detect`：水平框
  - `obb`：旋转框 + 角度回归
  - `segment`：实例分割（基于 prototype 的 mask 头）
  - `keypoint`：姿态估计（每个关键点独立分支，YOLOv8-pose 风格）
  - `multilabel`：多标签分类
  - `linear_probe`：冻结骨干 + 线性/多层头
- **统一的无锚框检测头**（用于 detect/obb/segment/keypoint）
  - **共享解耦头**：P3/P4/P5 共用同一组 head 权重（通过 `--head-channels` 投影）
  - **支持分层 Neck**（`--neck-out-channels 64 128 256`）
  - 分类：**Focal Loss** 风格（BCE + 目标分数归一化）。
  - 回归：
    - `detect`/`segment`/`keypoint`：**CIoU Loss** + 可选 **DFL**。
    - `obb`：**CRIoU + ProbIoU** 或 **KLD Loss**（可选）+ 可选 **DFL**。
  - 角度：
    - `sin_cos`：余弦损失 + 周期 L1 辅助损失
    - `direct`：周期 L1 损失
  - 关键点：归一化绝对坐标 L1 损失（YOLOv8-pose 风格），默认权重 12.0
- **实例分割头**（YOLOv8-seg 风格）
  - **Prototype 分支**：融合 P3+P4+P5
  - **Mask 系数分支**：`coef @ proto → mask`
  - 可配置 `proto_dim`、`proto_stride`（2/4/8）、`mask_weight`
- **关键点头**（YOLOv8-pose 风格）
  - 独立 `kpt_branch` 输出归一化绝对坐标
  - `bias=0.5` 初始化 → 初始预测图像中心
  - 归一化坐标空间 L1 损失（权重 12.0）
  - 基于 OKS 的 pose mAP 评估
- **多标签分类头**
  - 池化：`gap`、`gem`、`attention`
  - 分类头：`linear`、`mlp`、`ml_decoder`
  - 损失：`bce`、`asl`（Asymmetric Loss）
- **Task-Aligned Assigner (TAL)** 正样本分配。
- **完整训练流程**
  - 精度：`fp32`、`fp16`、`bf16`
  - `fp16` 自动使用 `GradScaler`
  - Warmup + cosine / step / multistep / lambda / OneCycle 调度器
  - 梯度裁剪、EMA、早停、断点恢复
  - **按 task 定制的 CSV 日志**（每个任务各用各的列）
  - **按 task 定制的 `results.png`**（不会出现空子图）
  - **`--verbose-metrics`**：逐类打印 mAP 表格
  - **`--grad-checkpointing`**：大模型时节省显存
- **灵活的数据集支持**
  - detect/obb：YOLO 风格目录
  - **segment：YOLOv8-seg 多边形格式**
  - **keypoint：YOLOv8-pose 格式**（带/不带 visibility）
  - multilabel：txt 标签文件（id-list 和 one-hot）
  - multilabel：HuggingFace datasets
- **内置评估指标**
  - `detect`：mAP@0.5、mAP@0.5:0.95、AR@100（YOLOv8 风格）
  - `obb`：基于 ProbIoU 的旋转 IoU mAP@0.5、mAP@0.5:0.95
  - `segment`：**bbox mAP + mask mAP**（prototype 尺度 mask IoU）
  - `keypoint`：**bbox mAP + OKS pose mAP**
  - `multilabel`：mAP、CF1、OP、OR、OF1、EMR
- **SimMIM 自监督预训练**
  - `simmim_pretrain.py` 在无标注图像上预训练骨干网络。
  - 支持 Swin / ConvNeXt / ViT（含 DINOv2 / DINOv3 / Eva）/ MobileNet / EfficientNet / 通用 CNN。
- **部署支持**
  - **所有任务支持 ONNX 导出**（raw + end2end）
    - `detect`/`obb`：`num_dets, boxes, scores, classes`（end2end）
    - `segment`：`num_dets, boxes, scores, classes, mask_coefs, proto`（end2end）
    - `keypoint`：`num_dets, boxes, scores, classes, keypoints`（end2end）
    - `multilabel`：`logits (B, C)` 或使用 `--with-sigmoid` 输出 `probs (B, C)`
  - **ONNX Runtime 推理**：`infer.py`（按文件后缀自动识别格式）

---

## 📦 支持的骨干网络

框架借助 `timm` 使用大量预训练模型。理论上，任何支持 `timm.create_model(..., features_only=True)` 的骨干都可以使用。

推荐骨干列表见 **[BackboneList.md](BackboneList.md)**，涵盖：

| 系列 | 示例名称 |
|---|---|
| ConvNeXtV2（FCMAE） | `convnextv2_atto.fcmae_ft_in1k`、`convnextv2_tiny.fcmae_ft_in1k` |
| ConvNeXt | `convnext_tiny`、`convnext_base` |
| MobileNet | `mobilenetv3_small_100`、`mobilenetv3_large_100` |
| ResNet / ResNeXt / ResNeSt | `resnet50`、`resnext50_32x4d`、`resnest50` |
| Swin Transformer | `swin_tiny_patch4_window7_224.ms_in1k` |
| Hiera | `hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k` |
| ViT | `vit_base_patch16_224`、`vit_large_patch16_224` |
| DINOv2 / DINOv3 | `vit_small_patch16_dinov3.lvd1689m`、`convnext_tiny.dinov3_lvd1689m` |
| EfficientNet | `efficientnet_b0` ~ `b7`、`efficientnetv2_rw_s` |
| 轻量模型 | `ghostnet_100`、`fasternet_tiny` |

Backbone 特征选择逻辑是自动的：

- ViT / DINOv2 / DINOv3 风格：通过 `ViTAdapter` 从单尺度 token 生成多尺度金字塔。
- Swin / Hiera：选择步长为 `8`、`16`、`32` 的特征图。
- 其他 CNN：取最后三个特征图。
- 输出通道数和步长自动推断。

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
- torchvision, timm
- opencv-python, Pillow, numpy
- tqdm, matplotlib
- onnx, onnxruntime
- onnxsim / onnx-simplifier（可选，用于 `--simplify`）
- datasets（可选，用于 HuggingFace 多标签数据集）

> 不再需要 `shapely` 或 `torchmetrics` —— 评估和旋转 NMS 均使用纯 NumPy / PyTorch 实现。

---

### 2. 数据集准备

#### 2.1 水平检测（`task = detect`）和旋转检测（`task = obb`）

YOLO 风格目录结构：

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

**水平检测标签（`detect`）：**

```text
class_id cx cy w h
```

**旋转检测标签（`obb`）：**

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

所有值归一化到 `[0, 1]`。数据加载器也交叉兼容 5 列 / 9 列格式。

#### 2.2 实例分割（`task = segment`）

**YOLOv8-seg 多边形格式：**

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

所有值归一化到 `[0, 1]`。每行是一个**至少包含 3 个顶点**的多边形。数据集会自动将多边形栅格化为 prototype 尺度的 mask（`img_size / proto_stride`）。

#### 2.3 关键点 / 姿态（`task = keypoint`）

**YOLOv8-pose 格式** —— 每行一个实例：

```text
# 带 visibility（每个关键点 3 个值）
class_id cx cy w h  x1 y1 v1  x2 y2 v2 ... xK yK vK

# 不带 visibility（每个关键点 2 个值）
class_id cx cy w h  x1 y1  x2 y2 ... xK yK
```

- 坐标归一化到 `[0, 1]`。
- `v ∈ {0, 1, 2}`，其中 `0` = 未标注，`1` = 已标注但被遮挡，`2` = 可见。
- `visibility='auto'` 会自动从前几个样本判断格式；也可用 `--keypoint-visibility yes|no` 覆盖。
- 左右对称的关键点（例如 COCO 17 点）建议提供 `--flip-idx`。

#### 2.4 多标签分类（`task = multilabel` / `linear_probe`）

支持两种数据源：**本地 txt 文件**或 **HuggingFace 数据集**。

**本地 txt 文件：**

```text
datasets/
└── your_multilabel_dataset/
    ├── train2017/
    ├── val2017/
    ├── train_labels.txt
    └── val_labels.txt
```

每行支持两种格式：

- **新格式**：`filename\t[0, 14, 26]`（制表符分隔，类别 id 列表）
- **旧格式**：`filename 0 0 1 0 1 ...`（空格分隔，one-hot 向量）

**HuggingFace 数据集：** 必须包含 `image` 和 `labels` 字段，可选 `label_names`。使用 `--hf-dataset`、`--hf-config`、`--hf-train-split`、`--hf-val-split`。

#### 2.5 SimMIM 预训练（无标注图像）

```text
datasets/
└── unlabeled_images/
    ├── any_subdir/
    │   ├── a.jpg
    │   └── ...
    └── b.png
```

支持的扩展名：`.jpg`、`.jpeg`、`.png`、`.bmp`。

---

### 3. 训练

#### 水平检测

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

#### 旋转检测（OBB）

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

#### 实例分割

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

**分割关键参数：**

- `--proto-dim`：prototype / mask 系数通道数（默认 `32`；YOLOv8-seg 用 32）
- `--proto-stride`：prototype 下采样倍数，`2`/`4`/`8`（默认 `4`）。越小 mask 越精细，显存越高
- `--mask-weight`：mask loss 权重（默认 `1.0`；YOLOv8-seg 用 `5.0`）

#### 关键点 / 姿态估计

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

**关键点关键参数：**

- `--num-keypoints`：每个实例的关键点数量（必须与标注一致）
- `--keypoint-weight`：关键点 loss 权重（默认 `12.0`，与 YOLOv8-pose 一致）
- `--head-kpt-num-convs`：关键点分支卷积层数（默认 `3`）
- `--keypoint-visibility`：`auto` / `yes` / `no`
- `--flip-idx`：水平翻转时的关键点索引映射（例如 COCO 17 点：`0 2 1 4 3 6 5 8 7 10 9 12 11 14 13 16 15`）

#### 多标签分类

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

**HuggingFace 变体：**

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

#### 线性探测

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

#### 关键训练选项

- `--task detect|obb|segment|keypoint|multilabel|linear_probe`
- `--data-root`、`--backbone`、`--img-size`、`--batch-size`、`--epochs`、`--lr`
- `--precision fp32|fp16|bf16`
- `--optimizer sgd|adamw`、`--scheduler lambda|step|multistep|cosine|onecycle`
- `--patience`、`--resume`、`--no-ema`、`--save-dir`
- `--head-channels`：detect/obb/segment/keypoint 的共享 head 通道
- `--neck-out-channels`：单值或 3 值（如 `64 128 256`）
- `--grad-checkpointing`：启用梯度检查点
- `--verbose-metrics`：逐类打印 mAP 表格

其他配置项可通过末尾追加 `key=value` 覆盖。

训练脚本会自动在 `--save-dir` 下创建新的 `runX` 目录：

```text
runX/
├── best.pth
├── last.pth
├── training_log.csv    # 按 task 定制列
├── config.json
└── results.png         # 按 task 定制子图
```

---

### 4. 推理

`infer.py` **按权重后缀自动识别后端**（`.pth`/`.pt` → PyTorch，`.onnx` → ONNX）。
仍可用 `--format pytorch|onnx` 显式覆盖。

#### 水平检测（PyTorch）

```bash
python infer.py \
    --task detect \
    --weights ./outputs/detect/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.7 --nms 0.3 \
    --output result.jpg \
    --class-names pill capsule tablet
```

#### 旋转检测（PyTorch）

```bash
python infer.py \
    --task obb \
    --weights ./outputs/obb/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.5 \
    --output result_obb.jpg \
    --class-names plane ship vehicle
```

#### 实例分割（PyTorch）

```bash
python infer.py \
    --task segment \
    --weights ./outputs/segment/run1/best.pth \
    --image path/to/image.jpg \
    --conf 0.5 \
    --output result_seg.jpg \
    --show
```

**分割专属参数：**

- `--no-mask`：不绘制 mask（只画框）
- `--mask-alpha`：mask 叠加透明度（0~1，默认 0.5）

#### 关键点 / 姿态（PyTorch）

```bash
python infer.py \
    --task keypoint \
    --weights ./outputs/keypoint/run1/best.pth \
    --image path/to/person.jpg \
    --conf 0.5 \
    --output result_pose.jpg \
    --show
```

**关键点专属参数：**

- `--no-kpt`：不绘制关键点（只画框）
- `--kpt-radius`：关键点可视化半径（默认 6）
- `--kpt-thr`：关键点显示阈值（默认 0.3）

#### 多标签分类（PyTorch）

```bash
python infer.py \
    --task multilabel \
    --weights ./outputs/multilabel/run1/best.pth \
    --image path/to/image.jpg \
    --threshold 0.5 --top-k 5 \
    --output result.jpg
```

#### ONNX 推理（自动识别）

```bash
# 后缀 .onnx → 自动走 ONNX，无需 --format
python infer.py \
    --weights best_end2end.onnx \
    --image test.jpg \
    --conf 0.7 --nms 0.3 \
    --class-names pill capsule tablet
```

对于 OBB，还需传入 `--angle-format` 和 `--angle-range`。对于 segment / keypoint，ONNX 推理器会自动处理 raw 和 end2end 两种输出。

其他选项：

- `--output-dir`：多图结果保存到目录
- `--show`：显示结果窗口
- `--show-size`：窗口最大宽度
- `--device cpu`：强制 CPU 推理

---

### 5. 导出 ONNX

#### 水平检测

```bash
python export.py \
    --weights ./outputs/detect/run1/best.pth \
    --dynamic --simplify
```

#### 旋转检测

```bash
python export.py \
    --weights ./outputs/obb/run1/best.pth \
    --dynamic --simplify
```

#### 实例分割

```bash
# raw 模式（output0 + proto）
python export.py --weights ./outputs/segment/run1/best.pth

# end2end 模式（NMS 固化）
python export.py \
    --weights ./outputs/segment/run1/best.pth \
    --end2end --conf 0.25 --nms 0.5 --max-det 300
```

#### 关键点 / 姿态

```bash
# raw 模式（output0）
python export.py --weights ./outputs/keypoint/run1/best.pth

# end2end 模式（NMS 固化；keypoints 输出为像素坐标）
python export.py \
    --weights ./outputs/keypoint/run1/best.pth \
    --end2end --conf 0.25 --nms 0.5 --max-det 300
```

#### 多标签分类

```bash
# 默认：输出 logits (B, C)
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --dynamic --simplify

# 图内嵌 sigmoid：输出 probs (B, C) ∈ (0, 1)
python export.py \
    --weights ./outputs/multilabel/run1/best.pth \
    --with-sigmoid --dynamic --simplify
# → best_probs.onnx
```

> **为什么要 `--with-sigmoid`？**
> 默认导出的是原始 **logits**，部署侧需要自己加 `sigmoid`。
> 如果你想**零逻辑部署**（例如 TensorRT / NCNN / TFLite），可用 `--with-sigmoid` 直接得到概率。
> 阈值筛选和 Top-K 仍保留在应用层。

#### 导出模式

- **Raw 模式**（默认）：
  - `detect`：`output0 (B, 4+nc, P)` —— `[cx, cy, w, h, cls...]`
  - `obb`：`output0 (B, 5+nc, P)` —— `[cx, cy, w, h, angle, cls...]`
  - `segment`：`output0 (B, 4+nc+k, P)` + `proto (B, k, ph, pw)`
  - `keypoint`：`output0 (B, 4+nc+2K, P)` —— `[cx, cy, w, h, cls..., kpt_norm...]`
  - `multilabel`：`logits (B, num_classes)` 或 `probs (B, num_classes)`
- **End2end 模式**（`--end2end`）：
  - `detect`/`obb`：`num_dets, boxes, scores, classes`
  - `segment`：`num_dets, boxes, scores, classes, mask_coefs, proto`
  - `keypoint`：`num_dets, boxes, scores, classes, keypoints`（全部像素坐标）
  - NMS 已固化；`--conf`、`--nms`、`--max-det` 控制其参数

Metadata（backbone、img_size、mean、std、strides、proto_dim、proto_stride、num_keypoints 等）会写入 ONNX 文件。`ONNXPredictor` 会自动读取 —— 包括 multilabel 输出的是 `logits` 还是 `probs`。

---

### 6. SimMIM 自监督预训练

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

然后使用预训练权重：

```bash
python train.py \
    --task detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_atto \
    --pretrained-source simmim \
    --backbone-pretrained-path ./simmim_pretrained/convnextv2_atto_simmim.pth \
    --img-size 640 --batch-size 16 --epochs 100
```

> 预训练和下游使用**相同的 `--backbone`**。`img-size` 不必相同。

---

### 7. 配置说明

所有全局超参数集中在 `swapfighter/cfg/config.py`。任务默认配置在 `swapfighter/tasks/*.py`。

关键参数：

- `task`：`detect`、`obb`、`segment`、`keypoint`、`multilabel`、`linear_probe`
- `data_root`、`train_split`、`val_split`、`num_classes`
- `backbone_name`、`pretrained`、`img_size`
- `neck_type`、`neck_num_layers`、`neck_out_channels`、`head_channels`
- `head_type`、`head_norm_layer`、`head_act_layer`
- `with_angle`、`angle_range`、`angle_format`
- `reg_max`、`dfl_weight`、`obb_reg_loss`
- `proto_dim`、`proto_stride`、`mask_weight`（segment）
- `num_keypoints`、`keypoint_weight`、`keypoint_visibility`、`flip_idx`（keypoint）
- `batch_size`、`lr`、`backbone_lr_scale`、`weight_decay`、`epochs`
- `warmup_epochs`、`optimizer`、`scheduler`
- `precision`：`fp32`、`fp16`、`bf16`
- `mosaic_prob`、`mosaic_disable_epoch`
- `conf_threshold`、`nms_threshold`、`obb_nms_threshold`
- `eval_map_start_epoch`、`eval_conf_threshold`、`verbose_metrics`
- `patience`、`use_ema`、`ema_decay`
- `tal_topk`、`tal_alpha`、`tal_beta`
- `pool_type`、`head_type`、`loss`、`eval_threshold`（multilabel）

---

## 📁 项目结构

```text
SWAP-Fighter/
├── swapfighter/
│   ├── api.py                 # SF 类
│   ├── cfg/
│   │   └── config.py          # Config 类
│   ├── tasks/                 # 任务注册与默认配置
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
├── train.py                    # 训练入口
├── infer.py                    # 推理入口（自动识别格式）
├── export.py                   # ONNX 导出入口
├── simmim_pretrain.py          # SimMIM 自监督预训练
├── requirements.txt
├── README.md
├── README.zh-CN.md
└── BackboneList.md
```

---

## 📝 注意事项

- 旧版 `--use-amp` 已由 `--precision` 取代。
- `fp16` 自动使用 `GradScaler`；`bf16` 需要 Ampere 或更新架构 GPU。
- OBB 旋转 NMS 和 OBB mAP 均使用纯 NumPy / PyTorch 实现，无需 `shapely`。
- Segment mask mAP 在 prototype 尺度计算（不上采样到原图），既快又一致。
- Keypoint end2end ONNX 的 `boxes` 和 `keypoints` 均输出**像素坐标**，与 `segment` 的约定一致。
- Multilabel ONNX 默认导出 `logits`；用 `--with-sigmoid` 可在图内嵌 `Sigmoid`，直接得到 `probs (B, C)`。
- `infer.py` 按权重后缀自动选择 `pytorch` / `onnx`（`.pth`/`.pt` → pytorch，`.onnx` → onnx），可用 `--format` 覆盖。
- CSV 日志和 `results.png` 子图**按 task 定制** —— 不会出现空列 / 空子图。
- ONNX 后处理依赖正确的 metadata（strides、img_size、mean、std、proto_dim、proto_stride、num_keypoints）。
- SimMIM 预训练是**可选**的 —— 你也可以直接使用 timm ImageNet 权重。

---

## 🤝 贡献

欢迎贡献！请到 https://github.com/IntptrMax/SwapFighter 提交 Issue 或 PR。

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 – 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [FCOS](https://github.com/tianzhi0549/FCOS) —— 无锚框检测思想。
- [YOLOv8](https://github.com/ultralytics/ultralytics) —— TAL 分配器、解耦头、prototype 分割、姿态估计。
- [timm](https://github.com/huggingface/pytorch-image-models) —— 大量预训练骨干网络。
- [SimMIM](https://github.com/microsoft/SimMIM) —— 掩码图像建模预训练。

---

**快乐“接头”！** 🔄
