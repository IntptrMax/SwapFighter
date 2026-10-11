import os
import glob
import argparse
import math
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T

import timm
from timm.layers import trunc_normal_
from tqdm import tqdm

def _ensure_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img

def _safe_call(owner, attr_name, x):
    module = getattr(owner, attr_name, None)
    if module is not None and callable(module):
        return module(x)
    return x

class MaskGenerator:

    def __init__(self, input_size=224, mask_patch_size=32,
                 model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0, \
            f"input_size ({input_size}) 必须能被 mask_patch_size ({mask_patch_size}) 整除"
        assert self.mask_patch_size % self.model_patch_size == 0, \
            f"mask_patch_size ({mask_patch_size}) 必须能被 model_patch_size ({model_patch_size}) 整除"

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int32)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.mask_patch_size, axis=0) \
                   .repeat(self.mask_patch_size, axis=1)
        return mask

class BackboneForSimMIM(nn.Module):

    def __init__(self, backbone_name, img_size=224, mask_ratio=0.6,
                 mask_patch_size=32, pretrained=True):
        super().__init__()

        name_lower = backbone_name.lower()
        self.is_swin = 'swin' in name_lower
        self.is_convnext = 'convnext' in name_lower
        self.is_vit = ('vit' in name_lower or 'dinov2' in name_lower
                       or 'dinov3' in name_lower or 'eva' in name_lower)
        self.is_mobilenet = 'mobilenet' in name_lower
        self.is_efficientnet = 'efficientnet' in name_lower

        self.is_generic_cnn = not (
            self.is_swin or self.is_convnext or self.is_vit or
            self.is_mobilenet or self.is_efficientnet
        )

        use_kwargs = dict(
            pretrained=pretrained,
            features_only=True,
            num_classes=0,
        )
        if self.is_vit:
            use_kwargs['dynamic_img_size'] = True

        try:
            self.encoder = timm.create_model(backbone_name, img_size=img_size, **use_kwargs)
        except TypeError:
            self.encoder = timm.create_model(backbone_name, **use_kwargs)

        self.inner = self._find_inner_backbone(self.encoder)

        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        channels = self.encoder.feature_info.channels()
        self.embed_dim = channels[-1]

        if self.is_swin:
            pe = self.inner.patch_embed
            ps = getattr(pe, 'patch_size', 4)
            self.model_patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        elif self.is_vit:
            pe = self.inner.patch_embed
            ps = getattr(pe, 'patch_size', 16)
            self.model_patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        elif self.is_convnext:
            self.model_patch_size = 4
        elif self.is_mobilenet:
            self.model_patch_size = 2
        elif self.is_efficientnet:
            self.model_patch_size = 2
        else:
            self.model_patch_size = 4

        all_strides = [info['reduction'] for info in self.encoder.feature_info]
        self.final_downsample = all_strides[-1]
        assert img_size % self.final_downsample == 0, \
            f"img_size ({img_size}) 必须能被 final_downsample ({self.final_downsample}) 整除"
        self.final_res = img_size // self.final_downsample

        self.stage1_dim = self._infer_stage1_dim()

        if self.is_swin:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.stage1_dim))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, self.stage1_dim, 1, 1))
        trunc_normal_(self.mask_token, mean=0., std=0.02)

        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=self.model_patch_size,
            mask_ratio=mask_ratio,
        )

        patch_area = self.final_downsample ** 2 * 3
        self.decoder = nn.Linear(self.embed_dim, patch_area)

        print(f"[SimMIM] backbone = {backbone_name}")
        print(f"[SimMIM] pretrained = {pretrained}")
        print(f"[SimMIM] encoder class = {type(self.encoder).__name__}")
        print(f"[SimMIM] inner class = {type(self.inner).__name__}")
        print(f"[SimMIM] model_patch_size = {self.model_patch_size}")
        print(f"[SimMIM] final_downsample = {self.final_downsample}")
        print(f"[SimMIM] stage1_dim = {self.stage1_dim}")
        print(f"[SimMIM] embed_dim = {self.embed_dim}")

    @staticmethod
    def _find_inner_backbone(encoder):
        print(f"[SimMIM] 检测内层 backbone...")
        print(f"[SimMIM]   encoder 直接子模块: "
              f"{[n for n, _ in encoder.named_children()]}")

        backbone_markers = (
            'patch_embed', 'stem', 'stages',
            'conv_stem', 'blocks', 'features',
            'stem_0', 'stages_0',
        )
        matched_markers = [a for a in backbone_markers if hasattr(encoder, a)]
        if matched_markers:
            wrapper_attrs = ('model', 'body', 'backbone')
            has_wrapper = any(hasattr(encoder, a) for a in wrapper_attrs)
            if not has_wrapper:
                print(f"[SimMIM]   encoder 自身是 backbone（检测到 {matched_markers}）")
                return encoder

        for attr in ('model', 'body', 'backbone', 'encoder', 'net'):
            if hasattr(encoder, attr):
                candidate = getattr(encoder, attr)
                if isinstance(candidate, nn.Module) and candidate is not encoder:
                    print(f"[SimMIM]   找到内层 backbone 通过属性 .{attr} "
                          f"(class={type(candidate).__name__})")
                    return candidate

        children = list(encoder.named_children())
        if len(children) == 1:
            print(f"[SimMIM]   找到内层 backbone 通过唯一子模块 .{children[0][0]}")
            return children[0][1]

        if len(children) > 1:
            total_params = sum(p.numel() for p in encoder.parameters())
            max_child = max(
                children,
                key=lambda x: sum(p.numel() for p in x[1].parameters())
            )
            child_params = sum(p.numel() for p in max_child[1].parameters())
            if child_params > total_params * 0.5:
                print(f"[SimMIM]   找到内层 backbone 通过参数占比 .{max_child[0]} "
                      f"({child_params}/{total_params})")
                return max_child[1]

        print(f"[SimMIM]   使用 encoder 本身作为 backbone")
        return encoder

    def _infer_stage1_dim(self):
        if self.is_swin:
            pe = self.inner.patch_embed
            if hasattr(pe, 'proj') and isinstance(pe.proj, nn.Conv2d):
                return pe.proj.out_channels
            for m in pe.modules():
                if isinstance(m, nn.Conv2d):
                    return m.out_channels
            return self.embed_dim // 8

        elif self.is_convnext:
            if hasattr(self.inner, 'stem'):
                c = None
                for m in self.inner.stem.modules():
                    if isinstance(m, nn.Conv2d):
                        c = m.out_channels
                if c is not None:
                    return c
            if hasattr(self.inner, 'stem_0'):
                stem0 = getattr(self.inner, 'stem_0')
                if isinstance(stem0, nn.Conv2d):
                    return stem0.out_channels
            return self.encoder.feature_info.channels()[0]

        elif self.is_vit:
            pe = self.inner.patch_embed
            if hasattr(pe, 'proj') and isinstance(pe.proj, nn.Conv2d):
                return pe.proj.out_channels
            return getattr(self.inner, 'embed_dim', self.embed_dim)

        elif self.is_mobilenet:
            if hasattr(self.inner, 'conv_stem') and isinstance(self.inner.conv_stem, nn.Conv2d):
                return self.inner.conv_stem.out_channels
            return self.encoder.feature_info.channels()[0]

        elif self.is_efficientnet:
            if hasattr(self.inner, 'conv_stem') and isinstance(self.inner.conv_stem, nn.Conv2d):
                return self.inner.conv_stem.out_channels
            return self.encoder.feature_info.channels()[0]

        else:
            for m in self.inner.modules():
                if isinstance(m, nn.Conv2d):
                    return m.out_channels
            return self.embed_dim

    def _downsample_mask_nhwc(self, mask, H, W):
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask_down = F.adaptive_avg_pool2d(
            mask.unsqueeze(1).float(), (H, W)
        ).squeeze(1)
        return (mask_down > 0.5).float()

    def _collect_swin_layers(self):
        inner = self.inner
        if hasattr(inner, 'layers'):
            layers = inner.layers
            if isinstance(layers, (nn.Sequential, nn.ModuleList)):
                return list(layers)
            elif isinstance(layers, (list, tuple)):
                return list(layers)

        collected = []
        i = 0
        while hasattr(inner, f'layers_{i}'):
            collected.append(getattr(inner, f'layers_{i}'))
            i += 1
        if collected:
            return collected

        raise RuntimeError(
            "无法找到 Swin 的 layers——"
            "既没有 inner.layers，也没有 inner.layers_0/1/2/..."
        )

    def _run_swin(self, x, mask):
        inner = self.inner

        x = inner.patch_embed(x)
        if x.shape[1] == self.stage1_dim and x.shape[-1] != self.stage1_dim:
            x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C1 = x.shape

        mask_down = self._downsample_mask_nhwc(mask, H, W).unsqueeze(-1)
        x = x * (1 - mask_down) + self.mask_token * mask_down

        layers = self._collect_swin_layers()
        for layer in layers:
            x = layer(x)

        x = _safe_call(inner, 'norm', x)
        return x

    def _run_convnext(self, x, mask):
        inner = self.inner

        if hasattr(inner, 'stem'):
            x = inner.stem(x)
            x = _safe_call(inner, 'norm_pre', x)
        elif hasattr(inner, 'stem_0'):
            i = 0
            while hasattr(inner, f'stem_{i}'):
                x = getattr(inner, f'stem_{i}')(x)
                i += 1
        else:
            raise RuntimeError(
                "ConvNeXt 无法找到 stem——既没有 inner.stem 也没有 inner.stem_0"
            )

        B, C1, H, W = x.shape

        mask_down = self._downsample_mask_nhwc(mask, H, W).unsqueeze(1)
        x = x * (1 - mask_down) + self.mask_token * mask_down

        if hasattr(inner, 'stages'):
            x = inner.stages(x)
        elif hasattr(inner, 'stages_0'):
            i = 0
            while hasattr(inner, f'stages_{i}'):
                x = getattr(inner, f'stages_{i}')(x)
                i += 1
        else:
            raise RuntimeError(
                "ConvNeXt 无法找到 stages——既没有 inner.stages 也没有 inner.stages_0"
            )

        return x

    def _run_mobilenet(self, x, mask):
        inner = self.inner

        x = inner.conv_stem(x)
        x = _safe_call(inner, 'bn1', x)
        x = _safe_call(inner, 'act1', x)

        B, C1, H, W = x.shape

        mask_down = self._downsample_mask_nhwc(mask, H, W).unsqueeze(1)
        x = x * (1 - mask_down) + self.mask_token * mask_down

        x = inner.blocks(x)

        x = _safe_call(inner, 'conv_head', x)
        x = _safe_call(inner, 'bn2', x)
        x = _safe_call(inner, 'act2', x)

        return x

    def _run_efficientnet(self, x, mask):
        return self._run_mobilenet(x, mask)

    def _run_generic_cnn(self, x, mask):
        inner = self.inner
        stem_attr = None
        for name in ('stem', 'conv_stem', 'features', 'conv1', 'layer0'):
            if hasattr(inner, name):
                stem_attr = name
                break

        if stem_attr is None:
            print("[SimMIM] 警告: 无法拆分 backbone，直接 forward（不注入 mask）")
            return self.encoder(x)

        x = getattr(inner, stem_attr)(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        B, C1, H, W = x.shape

        mask_down = self._downsample_mask_nhwc(mask, H, W).unsqueeze(1)
        x = x * (1 - mask_down) + self.mask_token * mask_down

        for name, module in inner.named_children():
            if name == stem_attr:
                continue
            try:
                x = module(x)
                if not isinstance(x, torch.Tensor):
                    x = x[0]
            except Exception:
                break

        return x

    def _run_vit(self, x, mask):
        inner = self.inner

        pe_out = inner.patch_embed(x)

        if pe_out.dim() == 3:

            tokens = pe_out
            pe_layout = 'NLC'
            B, N, C1 = tokens.shape
            h = w = int(math.sqrt(N))
        elif pe_out.dim() == 4:
            if pe_out.shape[1] == self.stage1_dim:

                B, C1, h, w = pe_out.shape
                N = h * w
                tokens = pe_out.flatten(2).transpose(1, 2)
                pe_layout = 'NCHW'
            elif pe_out.shape[-1] == self.stage1_dim:

                B, h, w, C1 = pe_out.shape
                N = h * w
                tokens = pe_out.reshape(B, N, C1)
                pe_layout = 'NHWC'
            else:
                raise ValueError(
                    f"无法判断 patch_embed 输出布局，shape={tuple(pe_out.shape)}，"
                    f"stage1_dim={self.stage1_dim}"
                )
        else:
            raise ValueError(f"patch_embed 输出维度异常: {pe_out.dim()}")

        assert N == h * w, f"token 数不匹配: N={N}, h*w={h*w}"

        mask_down = self._downsample_mask_nhwc(mask, h, w).view(B, N, 1)
        mask_token_flat = self.mask_token.view(1, 1, C1)
        tokens = tokens * (1 - mask_down) + mask_token_flat * mask_down

        pos_embed_fn = getattr(inner, '_pos_embed', None)
        if pos_embed_fn is not None and callable(pos_embed_fn):

            tokens = pos_embed_fn(tokens)
        else:

            cls_token = getattr(inner, 'cls_token', None)
            pos_embed = getattr(inner, 'pos_embed', None)

            if cls_token is not None:
                cls_tok = cls_token.expand(B, -1, -1)
                tokens = torch.cat([cls_tok, tokens], dim=1)

            if pos_embed is not None:
                pe = pos_embed
                if pe.dim() == 3:

                    tokens = tokens + pe
                elif pe.dim() == 4:

                    if pe.shape[1] == C1:
                        pe = pe.flatten(2).transpose(1, 2)
                    else:
                        pe = pe.flatten(2)
                    if tokens.shape[1] == pe.shape[1] + 1:
                        cls_pos = pe[:, :1, :]
                        pe = torch.cat([cls_pos, pe], dim=1)
                    tokens = tokens + pe

        tokens = _safe_call(inner, 'patch_drop', tokens)
        tokens = _safe_call(inner, 'norm_pre', tokens)

        blocks = getattr(inner, 'blocks', None)
        if blocks is None:
            raise RuntimeError("ViT 无法找到 inner.blocks")
        for blk in blocks:
            tokens = blk(tokens)

        tokens = _safe_call(inner, 'norm', tokens)

        cls_token = getattr(inner, 'cls_token', None)
        if cls_token is not None:

            if tokens.shape[1] == N + 1:
                tokens = tokens[:, 1:, :]

        B, N_out, C_out = tokens.shape
        h = w = int(math.sqrt(N_out))
        tokens = tokens.transpose(1, 2).view(B, C_out, h, w)
        return tokens

    def _run_encoder(self, x, mask):
        if self.is_swin:
            return self._run_swin(x, mask)
        elif self.is_convnext:
            return self._run_convnext(x, mask)
        elif self.is_vit:
            return self._run_vit(x, mask)
        elif self.is_mobilenet:
            return self._run_mobilenet(x, mask)
        elif self.is_efficientnet:
            return self._run_efficientnet(x, mask)
        else:
            return self._run_generic_cnn(x, mask)

    def forward(self, x, mask):
        features = self._run_encoder(x, mask)

        if features.dim() == 4:
            if features.shape[-1] == self.embed_dim and features.shape[1] != self.embed_dim:
                features = features.permute(0, 3, 1, 2).contiguous()
        elif features.dim() == 3:
            B, N, C = features.shape
            h = w = int(math.sqrt(N))
            features = features.transpose(1, 2).view(B, C, h, w)

        B, C, H, W = features.shape
        assert C == self.embed_dim, f"期望 embed_dim={self.embed_dim}，实际 {C}"

        flat = features.flatten(2).transpose(1, 2)
        pred = self.decoder(flat)

        ds = self.final_downsample
        pred = pred.permute(0, 2, 1).view(B, 3, ds, ds, H, W)
        pred = pred.permute(0, 1, 4, 2, 5, 3).contiguous()
        pred = pred.view(B, 3, H * ds, W * ds)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()

        if pred.shape[-2:] != mask.shape[-2:]:
            pred = F.interpolate(
                pred, size=mask.shape[-2:], mode='bilinear', align_corners=False
            )

        loss = F.l1_loss(pred * mask, x * mask, reduction='sum')
        num_masked = mask.sum() * 3 + 1e-6
        loss = loss / num_masked
        return loss

class SimMIMDataset(Dataset):

    def __init__(self, img_dir, img_size=224, mask_patch_size=32,
                 model_patch_size=4, mask_ratio=0.6):
        self.img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))
            self.img_paths.extend(glob.glob(os.path.join(img_dir, '**', ext.upper()), recursive=True))

        if len(self.img_paths) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到任何图像文件")

        self.img_size = img_size
        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

        self.transform = T.Compose([
            T.Lambda(_ensure_rgb),
            T.RandomResizedCrop(
                img_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        mask = self.mask_generator()
        mask = torch.from_numpy(mask).float()
        return img, mask

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0.0
    for p in parameters:
        total_norm += p.grad.data.norm(norm_type).item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def parse_args():
    parser = argparse.ArgumentParser('SimMIM 预训练脚本 (features_only=True 与 SWAP-Fighter 对齐)')

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--mask-patch-size', type=int, default=32)
    parser.add_argument('--mask-ratio', type=float, default=0.6)

    parser.add_argument('--backbone', type=str, default='convnextv2_atto')

    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='加载 timm 官方 ImageNet 预训练权重（默认开启）')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='从头随机初始化，不加载任何预训练权重')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=1)

    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'])

    parser.add_argument('--output', type=str, default='simmim_pretrained.pth',
                        help='最终编码器权重保存路径')

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_amp = False
    amp_dtype = torch.float32
    scaler = None

    if device.type == 'cuda':
        if args.precision == 'fp16':
            use_amp = True
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler('cuda')
            print("使用 FP16 混合精度 + GradScaler")
        elif args.precision == 'bf16':
            if torch.cuda.is_bf16_supported():
                use_amp = True
                amp_dtype = torch.bfloat16
                print("使用 BF16 混合精度")
            else:
                print("警告: 当前 GPU 不支持 BF16，回退到 FP16")
                use_amp = True
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler('cuda')
        else:
            print("使用 FP32 全精度")
    else:
        print("使用 FP32 全精度 (CPU)")

    print(f"加载数据集: {args.data_dir}")
    dataset = SimMIMDataset(
        img_dir=args.data_dir,
        img_size=args.img_size,
        mask_patch_size=args.mask_patch_size,
        model_patch_size=4,
        mask_ratio=args.mask_ratio,
    )
    print(f"数据集大小: {len(dataset)} 张图像")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"创建模型: {args.backbone} (pretrained={args.pretrained})")
    model = BackboneForSimMIM(
        backbone_name=args.backbone,
        img_size=args.img_size,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        pretrained=args.pretrained,
    ).to(device)

    actual_ps = model.model_patch_size
    dataset.mask_generator = MaskGenerator(
        input_size=args.img_size,
        mask_patch_size=args.mask_patch_size,
        model_patch_size=actual_ps,
        mask_ratio=args.mask_ratio,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    no_decay = ['bias', 'norm', 'mask_token']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    steps_per_epoch = max(len(dataloader) // args.accumulation_steps, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("=" * 60)
    print("开始 SimMIM 预训练 (features_only=True，与 SWAP-Fighter 对齐)")
    print(f"  骨干网络: {args.backbone}")
    print(f"  ImageNet 预训练: {args.pretrained}")
    print(f"  输入尺寸: {args.img_size}")
    print(f"  掩码块大小: {args.mask_patch_size}, 掩码比例: {args.mask_ratio}")
    print(f"  批次大小: {args.batch_size}, 梯度累积: {args.accumulation_steps}")
    print(f"  学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"  Warmup: {args.warmup_epochs} epochs, 总 epochs: {args.epochs}")
    print(f"  精度: {args.precision}")
    print(f"  设备: {device}")
    print(f"  最终输出: {args.output}")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        start_time = time.time()

        optimizer.zero_grad()
        accumulation_counter = 0
        accumulated_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f'Epoch [{epoch+1}/{args.epochs}]',
            leave=True,
            mininterval=0.0,
        )

        for step, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = model(images, masks)

            loss = loss / args.accumulation_steps
            accumulated_loss += loss.item()

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            if accumulation_counter == args.accumulation_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = get_grad_norm(model.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = get_grad_norm(model.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                accumulation_counter = 0

                loss_meter.update(accumulated_loss)
                norm_meter.update(grad_norm)
                accumulated_loss = 0.0

                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'lr': f'{lr:.2e}',
                    'gn': f'{grad_norm:.3f}',
                })

        pbar.close()

        epoch_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] 完成 | "
              f"平均损失: {loss_meter.avg:.6f} | "
              f"LR: {lr:.2e} | 耗时: {epoch_time:.1f}s")

    print("=" * 60)
    print("预训练完成，保存编码器权重...")

    encoder_state = model.encoder.state_dict()
    print(f"  encoder 键数: {len(encoder_state)}")
    print(f"  键名前 5 个示例: {list(encoder_state.keys())[:5]}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(encoder_state, args.output)
    print(f"编码器权重已保存至: {args.output}")
    print(f"该权重可直接被 SWAP-Fighter 的 Backbone 加载，无需任何归一化")
    print("=" * 60)

if __name__ == '__main__':
    main()