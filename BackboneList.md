# SWAP-Fighter 支持的 Backbone 列表

本文档列出经过测试、推荐使用的 `timm` 骨干网络。

> 原理上，任何满足 `timm.create_model(name, features_only=True)` 的模型都可以作为 SWAP-Fighter 的 Backbone 使用，输出通道数和步长会被自动推断。

**标记说明：**

- ⭐ 推荐默认选择
- 🚀 轻量 / 边缘设备友好
- 🎯 高精度（参数量较大）
- 🔬 自监督预训练权重

---

## 1. ConvNeXtV2

ConvNeXtV2 采用 **FCMAE**（全卷积掩码自编码器）自监督预训练，在同等参数量级下精度领先。

> ⚠️ 说明：`convnextv2_atto_100` 这类带 `_100` 后缀的写法并非 timm 官方注册名。timm 中实际可用的名称使用点号分隔，例如 `convnextv2_atto.fcmae_ft_in1k`。

### 1.1 FCMAE 预训练 + ImageNet-1k 微调

| 模型名 | 参数量 | 说明 |
|---|---|---|
| `convnextv2_atto.fcmae_ft_in1k` | 3.7 M | 极小 |
| `convnextv2_femto.fcmae_ft_in1k` | 5.2 M | 很小 |
| `convnextv2_pico.fcmae_ft_in1k` | 9.1 M | 很小 |
| `convnextv2_nano.fcmae_ft_in1k` | 15.6 M | 小 |
| `convnextv2_tiny.fcmae_ft_in1k` | 28.6 M | ⭐ 推荐默认 |
| `convnextv2_base.fcmae_ft_in1k` | 89.0 M | 中大型 |
| `convnextv2_large.fcmae_ft_in1k` | 198 M | 🎯 大 |
| `convnextv2_huge.fcmae_ft_in1k` | 660 M | 🎯 超大 |

### 1.2 FCMAE 预训练 + ImageNet-22k → 1k 微调

| 模型名 | 说明 |
|---|---|
| `convnextv2_base.fcmae_ft_in22k_in1k` | 22k 预训练 |
| `convnextv2_base.fcmae_ft_in22k_in1k_384` | 384×384 输入 |
| `convnextv2_large.fcmae_ft_in22k_in1k` | 22k 预训练 |
| `convnextv2_large.fcmae_ft_in22k_in1k_384` | 384×384 输入 |
| `convnextv2_huge.fcmae_ft_in22k_in1k_384` | 384×384 输入 |
| `convnextv2_huge.fcmae_ft_in22k_in1k_512` | 512×512 输入 |
| `convnextv2_nano.fcmae_ft_in22k_in1k` | 22k 预训练 |
| `convnextv2_nano.fcmae_ft_in22k_in1k_384` | 384×384 输入 |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | 22k 预训练 |
| `convnextv2_tiny.fcmae_ft_in22k_in1k_384` | 384×384 输入 |

### 1.3 仅 FCMAE 预训练（无分类头微调）

适合用 `simmim_pretrain.py` 作为骨干权重来源，或作为进一步微调的起点。

`convnextv2_{atto, femto, pico, nano, tiny, base, large, huge}.fcmae`

---

## 2. ConvNeXt (V1)

| 模型名 | 说明 |
|---|---|
| `convnext_tiny` | 小 |
| `convnext_small` | 中 |
| `convnext_base` | 大 |

> 需要更强的预训练权重时，可直接使用带后缀的变体（如 `convnext_tiny.fb_in22k_ft_in1k`）。

---

## 3. MobileNetV2 / MobileNetV3

适合移动端 / 边缘设备部署。

### 3.1 MobileNetV2

| 模型名 | 说明 |
|---|---|
| `mobilenetv2_100` | 标准宽度 |
| `mobilenetv2_140` | 1.4× 宽度 |

### 3.2 MobileNetV3

| 模型名 | 说明 |
|---|---|
| `mobilenetv3_small_050` | 🚀 0.50× 宽度，最轻量 |
| `mobilenetv3_small_075` | 🚀 0.75× 宽度 |
| `mobilenetv3_small_100` | 🚀 小版本（推荐边缘部署） |
| `mobilenetv3_large_100` | 大版本 |
| `mobilenetv3_rw` | 改进版（Random Width） |

### 3.3 MobileNetV3 预训练变体

**带标签平滑 / 蒸馏 / 21k 预训练：**

- `mobilenetv3_large_100.miil_in21k`
- `mobilenetv3_large_100.miil_in21k_ft_in1k`
- `mobilenetv3_large_100.ra4_e3600_r224_in1k`
- `mobilenetv3_large_100.ra_in1k`
- `mobilenetv3_large_150d.ra4_e3600_r256_in1k`
- `mobilenetv3_rw.rmsp_in1k`
- `mobilenetv3_small_050.lamb_in1k`
- `mobilenetv3_small_075.lamb_in1k`
- `mobilenetv3_small_100.lamb_in1k`

**TensorFlow 移植版：**

- `tf_mobilenetv3_small_075.in1k`
- `tf_mobilenetv3_small_100.in1k`
- `tf_mobilenetv3_small_minimal_100.in1k`
- `tf_mobilenetv3_large_075.in1k`
- `tf_mobilenetv3_large_100.in1k`
- `tf_mobilenetv3_large_minimal_100.in1k`

---

## 4. ResNet / ResNeXt / ResNeSt

### 4.1 经典基线

| 模型名 | 说明 |
|---|---|
| `resnet18` | 小 |
| `resnet34` | 小 |
| `resnet50` | ⭐ 最常用 |
| `resnet101` | 大 |
| `resnet152` | 很大 |
| `resnext50_32x4d` | ResNeXt 50 |
| `resnext101_32x8d` | ResNeXt 101 |
| `resnest50` | ResNeSt 50 |
| `resnest101` | ResNeSt 101 |

### 4.2 常用预训练变体（`timm` 命名：`模型.权重标签`）

**ResNet-18 / 34：**

- `resnet18.a1_in1k`, `resnet18.a2_in1k`, `resnet18.a3_in1k`
- `resnet18d.ra2_in1k`, `resnet18d.ra4_e3600_r224_in1k`
- `resnet18.tv_in1k`, `resnet18.gluon_in1k`
- `resnet18.fb_ssl_yfcc100m_ft_in1k`, `resnet18.fb_swsl_ig1b_ft_in1k`
- `resnet34.a1_in1k`, `resnet34.a2_in1k`, `resnet34.a3_in1k`
- `resnet34d.ra2_in1k`, `resnet34.ra4_e3600_r224_in1k`
- `resnet34.tv_in1k`, `resnet34.gluon_in1k`

**ResNet-50：**

- `resnet50.a1_in1k`, `resnet50.a1h_in1k`, `resnet50.a2_in1k`, `resnet50.a3_in1k`
- `resnet50.ra_in1k`, `resnet50.ram_in1k`
- `resnet50d.ra2_in1k`, `resnet50d.ra4_e3600_r224_in1k`
- `resnet50.tv_in1k`, `resnet50.tv2_in1k`
- `resnet50.fb_ssl_yfcc100m_ft_in1k`, `resnet50.fb_swsl_ig1b_ft_in1k`
- `resnet50_gn.a1h_in1k`

**ResNet-101 / 152：**

- `resnet101.a1_in1k`, `resnet101.a1h_in1k`, `resnet101.a2_in1k`, `resnet101.a3_in1k`
- `resnet101d.ra2_in1k`, `resnet101.tv_in1k`
- `resnet152.a1_in1k`, `resnet152.a1h_in1k`, `resnet152.a2_in1k`, `resnet152.a3_in1k`
- `resnet152d.ra2_in1k`, `resnet152.tv_in1k`

**Wide ResNet：**

- `wide_resnet50_2.tv_in1k`, `wide_resnet50_2.tv2_in1k`, `wide_resnet50_2.racm_in1k`
- `wide_resnet101_2.tv_in1k`, `wide_resnet101_2.tv2_in1k`

**ResNet-RS：**

- `resnetrs50.tf_in1k` … `resnetrs420.tf_in1k`

**ResNetV2：**

- `resnetv2_50.a1h_in1k`, `resnetv2_101.a1h_in1k`
- `resnetv2_50d_evos.ah_in1k`, `resnetv2_50d_gn.ah_in1k`

**SE-ResNet / ECA-ResNet / SK-ResNet / GC-ResNet：**

- `seresnet50.a1_in1k`, `seresnet50.a2_in1k`, `seresnet50.a3_in1k`, `seresnet50.ra2_in1k`
- `seresnet152d.ra2_in1k`
- `legacy_seresnet18.in1k` … `legacy_seresnet152.in1k`
- `ecaresnet26t.ra2_in1k`, `ecaresnet50d.miil_in1k`, `ecaresnet50t.ra2_in1k`
- `ecaresnet101d.miil_in1k`, `ecaresnet269d.ra2_in1k`
- `skresnet18.ra_in1k`, `skresnet34.ra_in1k`
- `gcresnet33ts.ra2_in1k`, `gcresnet50t.ra2_in1k`

**TResNet：**

- `tresnet_m.miil_in1k`, `tresnet_m.miil_in1k_448`, `tresnet_m.miil_in21k_ft_in1k`
- `tresnet_l.miil_in1k`, `tresnet_l.miil_in1k_448`
- `tresnet_xl.miil_in1k`, `tresnet_xl.miil_in1k_448`
- `tresnet_v2_l.miil_in21k_ft_in1k`

**CLIP 预训练 ResNet（多模态）：**

- `resnet50_clip.openai`, `resnet50_clip.cc12m`, `resnet50_clip.yfcc15m`
- `resnet50_clip_gap.openai`, `resnet50_clip_gap.cc12m`, `resnet50_clip_gap.yfcc15m`
- `resnet101_clip.openai`, `resnet101_clip.yfcc15m`
- `resnet50x4_clip.openai`, `resnet50x16_clip.openai`, `resnet50x64_clip.openai`

**其他改进版本：**

- `cspresnet50.ra_in1k`
- `nf_resnet50.ra2_in1k`
- `resnetblur50.bt_in1k`
- `resnet26.bt_in1k`, `resnet26d.bt_in1k`, `resnet26t.ra2_in1k`
- `resnet32ts.ra2_in1k`, `resnet33ts.ra2_in1k`
- `resnet51q.ra2_in1k`, `resnet61q.ra2_in1k`
- `resnet200d.ra2_in1k`
- `resnetaa50.a1h_in1k`, `resnetaa50d.sw_in12k_ft_in1k`, `resnetaa101d.sw_in12k_ft_in1k`
- `inception_resnet_v2.tf_in1k`
- `lambda_resnet26t.c1_in1k`, `lambda_resnet50ts.a1h_in1k`

---

## 5. EfficientNet / EfficientNetV2

### 5.1 EfficientNet (V1)

| 模型名 | 说明 |
|---|---|
| `efficientnet_b0` | 🚀 最小 |
| `efficientnet_b1` | 小 |
| `efficientnet_b2` | 中 |
| `efficientnet_b3` | 中 |
| `efficientnet_b4` | 中 |
| `efficientnet_b5` | 大 |
| `efficientnet_b6` | 大 |
| `efficientnet_b7` | 🎯 最大 |

### 5.2 EfficientNetV2 (RW 系列)

| 模型名 | 说明 |
|---|---|
| `efficientnetv2_rw_t` | V2 tiny |
| `efficientnetv2_rw_s` | V2 small |
| `efficientnetv2_rw_m` | V2 medium |
| `efficientnetv2_rw_l` | V2 large |

### 5.3 常用预训练变体

**EfficientNet V1：**

- `efficientnet_b0.ra_in1k`, `efficientnet_b0.ra4_e3600_r224_in1k`
- `efficientnet_b1.ft_in1k`, `efficientnet_b1.ra4_e3600_r240_in1k`
- `efficientnet_b2.ra_in1k`, `efficientnet_b3.ra2_in1k`, `efficientnet_b4.ra2_in1k`
- `efficientnet_b5.sw_in12k_ft_in1k`
- `efficientnet_es.ra_in1k`, `efficientnet_em.ra2_in1k`, `efficientnet_el.ra_in1k`
- `efficientnet_lite0.ra_in1k`

**EfficientNetV2 (RW)：**

- `efficientnetv2_rw_t.ra2_in1k`
- `efficientnetv2_rw_s.ra2_in1k`
- `efficientnetv2_rw_m.agc_in1k`
- `gc_efficientnetv2_rw_t.agc_in1k`

**TensorFlow 移植版：**

- `tf_efficientnet_b0.in1k` … `tf_efficientnet_b8.ra_in1k`
- `tf_efficientnet_lite0.in1k` … `tf_efficientnet_lite4.in1k`
- `tf_efficientnet_es.in1k`, `tf_efficientnet_em.in1k`, `tf_efficientnet_el.in1k`
- `tf_efficientnetv2_b0.in1k` … `tf_efficientnetv2_b3.in21k_ft_in1k`
- `tf_efficientnetv2_s.in21k_ft_in1k`, `tf_efficientnetv2_m.in21k_ft_in1k`
- `tf_efficientnetv2_l.in21k_ft_in1k`, `tf_efficientnetv2_xl.in21k_ft_in1k`

---

## 6. Swin Transformer (V1 / V2 / S3 / CR)

### 6.1 Swin V1

| 模型名 | 说明 |
|---|---|
| `swin_tiny_patch4_window7_224.ms_in1k` | ⭐ Tiny |
| `swin_small_patch4_window7_224.ms_in1k` | Small |
| `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | Base |
| `swin_large_patch4_window7_224.ms_in22k_ft_in1k` | 🎯 Large |
| `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | 384×384 |
| `swin_large_patch4_window12_384.ms_in22k_ft_in1k` | 384×384 |

**其他后缀：** `.ms_in1k`、`.ms_in22k`、`.ms_in22k_ft_in1k`

### 6.2 Swin S3 / Swin V2-CR（紧凑版）

- `swin_s3_tiny_224.ms_in1k`
- `swin_s3_small_224.ms_in1k`
- `swin_s3_base_224.ms_in1k`
- `swinv2_cr_tiny_ns_224.sw_in1k`
- `swinv2_cr_small_224.sw_in1k`
- `swinv2_cr_small_ns_224.sw_in1k`

### 6.3 Hiera (Hierarchical ViT)

- `hiera_small_abswin_256.sbb2_e200_in12k`
- `hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k`
- `hiera_small_abswin_256.sbb2_pd_e200_in12k`
- `hiera_small_abswin_256.sbb2_pd_e200_in12k_ft_in1k`

---

## 7. Vision Transformer (ViT)

ViT 系列在 SWAP-Fighter 中自动启用 `dynamic_img_size=True`，因此支持非 224 输入尺寸。

| 模型名 | 说明 |
|---|---|
| `vit_tiny_patch16_224` | Tiny |
| `vit_small_patch16_224` | Small |
| `vit_base_patch16_224` | Base |
| `vit_large_patch16_224` | Large |

---

## 8. DINO / DINOv2 / DINOv3（自监督）

> 🔬 所有模型均为自监督预训练，无需分类标签，泛化性通常显著优于监督预训练。

### 8.1 DINOv3

**ViT 系列：**

| 模型名 | 说明 |
|---|---|
| `vit_small_patch16_dinov3.lvd1689m` | Small |
| `vit_small_patch16_dinov3_qkvb.lvd1689m` | Small + QK-VB |
| `vit_small_plus_patch16_dinov3.lvd1689m` | Small+ |
| `vit_small_plus_patch16_dinov3_qkvb.lvd1689m` | Small+ + QK-VB |
| `vit_base_patch16_dinov3.lvd1689m` | Base |
| `vit_base_patch16_dinov3_qkvb.lvd1689m` | Base + QK-VB |
| `vit_large_patch16_dinov3.lvd1689m` | 🎯 Large |
| `vit_large_patch16_dinov3.sat493m` | Large (SAT) |
| `vit_large_patch16_dinov3_qkvb.lvd1689m` | 🎯 Large + QK-VB |
| `vit_huge_plus_patch16_dinov3.lvd1689m` | 🎯 Huge+ |
| `vit_huge_plus_patch16_dinov3_qkvb.lvd1689m` | 🎯 Huge+ + QK-VB |
| `vit_7b_patch16_dinov3.lvd1689m` | 7B 超大模型 |
| `vit_7b_patch16_dinov3.sat493m` | 7B (SAT) |

**ConvNeXt 系列（DINOv3 自监督）：**

- `convnext_tiny.dinov3_lvd1689m`
- `convnext_small.dinov3_lvd1689m`
- `convnext_base.dinov3_lvd1689m`
- `convnext_large.dinov3_lvd1689m`

### 8.2 DINO (V1)

- `vit_small_patch8_224.dino`
- `vit_small_patch16_224.dino`
- `vit_base_patch8_224.dino`
- `vit_base_patch16_224.dino`

---

## 9. 其他轻量级模型

| 模型名 | 说明 |
|---|---|
| `ghostnet_100` | 🚀 GhostNet，推理速度快 |
| `fasternet_tiny` | 🚀 FasterNet，极快 |
| `fasternet_small` | 🚀 FasterNet，极快 |

---

## 10. SimMIM 自监督预训练

SWAP-Fighter 提供 `simmim_pretrain.py`，可在自有未标注图像上对任意 timm 骨干进行 SimMIM 预训练，产出的权重可直接加载：

```bash
python simmim_pretrain.py \
    --data-dir ./datasets/unlabeled_images \
    --backbone convnextv2_atto \
    --img-size 224 \
    --output ./simmim_pretrained/convnextv2_atto_simmim.pth
```

在训练中加载：

```bash
python train.py \
    --mode detect \
    --data-root ./datasets/my_data \
    --backbone convnextv2_atto \
    --pretrained-source simmim \
    --backbone-pretrained-path ./simmim_pretrained/convnextv2_atto_simmim.pth
```

---

## 11. 使用建议

| 场景 | 推荐 Backbone |
|---|---|
| 通用检测（默认） | `convnextv2_tiny.fcmae_ft_in1k` |
| 移动端 / 边缘设备 | `mobilenetv3_small_100`、`ghostnet_100`、`fasternet_tiny` |
| 高精度 | `convnextv2_base.fcmae_ft_in1k`、`swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| 少样本 / 迁移学习 | `vit_base_patch14_dinov2.lvd142m`、`vit_small_patch14_dinov2.lvd142m` |
| 变分辨率输入 | `vit_*_dinov2`、`swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` |
| 极致轻量 | `mobilenetv3_small_050`、`convnextv2_atto.fcmae_ft_in1k` |

---

## 12. 常见错误对照

原始列表中出现的错误写法与 timm 中实际名称的对照：

| ❌ 错误写法 | ✅ 正确写法 | 说明 |
|---|---|---|
| `convnextv2_atto_100` | `convnextv2_atto.fcmae_ft_in1k` | timm 使用点号分隔，不是 `_100` 后缀 |
| `convnextv2_tiny_100` | `convnextv2_tiny.fcmae_ft_in1k` | 同上 |
| `convnextv2_base_100` | `convnextv2_base.fcmae_ft_in1k` | 同上 |
| `swin_tiny_patch4_window7_224` | `swin_tiny_patch4_window7_224.ms_in1k` | 无后缀版本在部分 timm 版本中不可用，需加后缀 |
| `swin_base_patch4_window7_224` | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 同上 |
| `efficientnetv2_rw_l` | （不存在） | timm 中 RW 系列只有 `_t`、`_s`、`_m` 三档 |

> 建议：在 `python` 中执行以下代码，可查看 timm 中所有可用模型的准确名称：
>
> ```python
> import timm
> names = timm.list_models('convnextv2*', pretrained=True)
> for n in names: print(n)
> ```
