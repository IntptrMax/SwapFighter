import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T

from ..base_predictor import BasePredictor
from ...utils.logger import LINE, LINE_EQ, _title, _kv, suppress_stdout


_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_RESIZE_RATIO = 1.14


_INTERP_MAP = {
    'bicubic': T.InterpolationMode.BICUBIC,
    'bilinear': T.InterpolationMode.BILINEAR,
    'nearest': T.InterpolationMode.NEAREST,
    'lanczos': T.InterpolationMode.LANCZOS,
}


def _build_transform(img_size, mean, std, resize_ratio=1.14,
                     interpolation='bicubic'):
    interp = _INTERP_MAP.get(interpolation, T.InterpolationMode.BICUBIC)
    return T.Compose([
        T.Resize(int(img_size * resize_ratio), interpolation=interp),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class MultiLabelPredictor(BasePredictor):

    def __init__(self, cfg_dict, weights=None):
        super().__init__(cfg_dict, weights)
        self._weights_meta = {}
        self._load_status = 'N/A'
        self._has_meta = False
        self._info = {}

    # ------------------------------------------------------------------
    def build_model(self):
        from ...tasks import get_task
        task_def = get_task(self.cfg.task)

        # 屏蔽 MultiLabelClassifier / timm 初始化时的散乱打印
        with suppress_stdout():
            model = task_def['model'](self.cfg)

        ckpt = torch.load(self.weights, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            meta = ckpt.get('model_meta', {}) or {}
            self._has_meta = bool(meta)
            self._weights_meta = meta
        else:
            sd = ckpt
            meta = {}

        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)

        real_missing = [k for k in missing if 'num_batches_tracked' not in k]
        real_unexpected = [k for k in unexpected if 'num_batches_tracked' not in k]
        if not real_missing and not real_unexpected:
            self._load_status = 'OK (full match)'
        else:
            parts = []
            if real_missing:
                parts.append(f'missing {len(real_missing)}')
            if real_unexpected:
                parts.append(f'unexpected {len(real_unexpected)}')
            self._load_status = 'WARN: ' + ', '.join(parts)

        # ---------- 从 meta 里恢复信息 ----------
        arch = meta.get('architecture', {}) or {}
        preprocess = meta.get('preprocess', {}) or {}
        inference = meta.get('inference', {}) or {}
        train_info = meta.get('train_info', {}) or {}

        def _pick(*vals, default=None):
            for v in vals:
                if v is not None:
                    return v
            return default

        self._info = {
            'backbone': _pick(arch.get('backbone_name'),
                              meta.get('backbone_name'),
                              self.cfg.backbone_name),
            'num_classes': _pick(arch.get('num_classes'),
                                 meta.get('num_classes'),
                                 self.cfg.num_classes),
            'img_size': _pick(arch.get('img_size'),
                              meta.get('img_size'),
                              self.cfg.img_size),
            'pool_type': _pick(arch.get('pool_type'),
                               meta.get('pool_type'),
                               self.cfg.pool_type),
            'head_type': _pick(arch.get('head_type'),
                               meta.get('head_type'),
                               self.cfg.head_type),
            'head_norm': _pick(arch.get('head_norm'),
                               meta.get('head_norm'), True),
            'multi_scale': _pick(arch.get('multi_scale'),
                                 meta.get('multi_scale'),
                                 self.cfg.multi_scale, default=False),
            'hidden_dim': _pick(arch.get('hidden_dim'),
                                meta.get('hidden_dim'),
                                self.cfg.hidden_dim, default=0),
            'dropout': _pick(arch.get('dropout'),
                             meta.get('dropout'),
                             self.cfg.dropout, default=0.0),
            'mean': list(_pick(preprocess.get('mean'),
                               meta.get('mean'), DEFAULT_MEAN)),
            'std': list(_pick(preprocess.get('std'),
                              meta.get('std'), DEFAULT_STD)),
            'resize_ratio': _pick(preprocess.get('resize_ratio'),
                                  DEFAULT_RESIZE_RATIO),
            'interpolation': _pick(preprocess.get('interpolation'), 'bicubic'),
            'crop': _pick(preprocess.get('crop'), 'center'),
            'best_threshold': inference.get('best_threshold'),
            'default_threshold': _pick(inference.get('default_threshold'),
                                       0.5),
            'epoch': _pick(train_info.get('epoch'),
                           meta.get('epoch'), default='N/A'),
            'metrics': _pick(train_info.get('metrics'),
                             meta.get('metrics'), default=None),
            'class_names': meta.get('class_names'),
        }

        return model.eval().to(self.device)

    # ------------------------------------------------------------------
    def run(self, source):
        if self.model is None:
            self.model = self.build_model()

        # ---------- 收集图片 ----------
        if os.path.isdir(source):
            images = sorted(
                os.path.join(source, f) for f in os.listdir(source)
                if f.lower().endswith(_IMG_EXTS)
            )
            if not images:
                raise FileNotFoundError(f"目录下没有图片: {source}")
        else:
            images = [source]

        # ---------- 类名 + 阈值 ----------
        class_names, class_names_src = self._resolve_class_names()
        threshold, threshold_src = self._resolve_threshold()

        # ---------- Header ----------
        if len(images) == 1:
            self._print_header_single(images[0], threshold, threshold_src,
                                       class_names, class_names_src)
        else:
            self._print_header_multi(source, len(images), threshold,
                                      threshold_src, class_names,
                                      class_names_src)

        # ---------- 预处理 ----------
        transform = _build_transform(
            self._info['img_size'],
            mean=self._info['mean'],
            std=self._info['std'],
            resize_ratio=self._info['resize_ratio'],
            interpolation=self._info['interpolation'],
        )

        # ---------- 逐图推理 ----------
        results = []
        for img_path in images:
            r = self._predict_one(img_path, transform, threshold, class_names)
            results.append(r)

            self._print_result(r, threshold, class_names)

            if self.output_image:
                try:
                    vis = self._draw(r['img'], r['probs'], class_names,
                                      threshold, getattr(self.cfg, 'top_k', 5))
                    cv2.imwrite(self.output_image, vis)
                except Exception as e:
                    print(f"  可视化保存失败: {e}")

            if self.output_json:
                try:
                    with open(self.output_json, 'w', encoding='utf-8') as f:
                        json.dump(self._to_json(r), f, indent=2,
                                  ensure_ascii=False)
                except Exception as e:
                    print(f"  JSON 保存失败: {e}")

        # ---------- 多图汇总 ----------
        if len(images) > 1:
            print()
            print(LINE_EQ)
            print(f"  Summary  ({len(images)} images)")
            print(LINE_EQ)
            for r in results:
                name = os.path.basename(r['image'])
                n_pos = int((r['probs'] >= threshold).sum())
                top1 = int(np.argmax(r['probs']))
                top1_name = (class_names[top1] if top1 < len(class_names)
                             else f'cls_{top1}')
                top1_p = float(r['probs'][top1])
                print(f"    {name:<44s}  "
                      f"top1={top1_name}({top1_p:.3f})  pos={n_pos}")
            print()
            print(LINE_EQ)

        return results

    # ------------------------------------------------------------------
    def _resolve_class_names(self):
        # 1. CLI 传入（cfg.class_names）
        if self.class_names:
            return list(self.class_names), 'CLI'
        # 2. ckpt.meta
        meta_names = self._info.get('class_names')
        if meta_names and len(meta_names) == self._info['num_classes']:
            return list(meta_names), 'checkpoint.meta'
        # 3. 占位
        n = self._info['num_classes']
        return [f'cls_{i}' for i in range(n)], 'placeholder'

    def _resolve_threshold(self):
        # CLI 传了非默认值 → CLI
        eval_threshold = getattr(self.cfg, 'eval_threshold', None)
        if eval_threshold is not None and float(eval_threshold) != 0.5:
            return float(eval_threshold), 'CLI'
        # meta.best
        if self._info['best_threshold'] is not None:
            return float(self._info['best_threshold']), 'meta.best'
        # meta.default
        return float(self._info['default_threshold']), 'meta.default'

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_one(self, image_path, transform, threshold, class_names):
        img = Image.open(image_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()

        top_k = getattr(self.cfg, 'top_k', 5)
        order = np.argsort(-probs)

        topk = [{
            'rank': int(r + 1),
            'class_id': int(idx),
            'class_name': (class_names[idx] if idx < len(class_names)
                           else f'cls_{idx}'),
            'prob': float(probs[idx]),
        } for r, idx in enumerate(order[:top_k])]

        positives = [{
            'class_id': int(idx),
            'class_name': (class_names[idx] if idx < len(class_names)
                           else f'cls_{idx}'),
            'prob': float(probs[idx]),
        } for idx in np.where(probs >= threshold)[0]]
        positives.sort(key=lambda x: -x['prob'])

        return {
            'image': image_path,
            'img': img,
            'probs': probs,
            'threshold': threshold,
            'topk': topk,
            'positives': positives,
        }

    def _to_json(self, r):
        return {
            'image': os.path.basename(r['image']),
            'image_path': os.path.abspath(r['image']),
            'threshold': r['threshold'],
            'topk': [{'rank': t['rank'],
                      'class_id': t['class_id'],
                      'class_name': t['class_name'],
                      'prob': round(t['prob'], 6)}
                     for t in r['topk']],
            'positives': [{'class_id': p['class_id'],
                           'class_name': p['class_name'],
                           'prob': round(p['prob'], 6)}
                          for p in r['positives']],
        }

    # ------------------------------------------------------------------
    # Header 打印
    # ------------------------------------------------------------------
    def _print_header_single(self, image_path, threshold, threshold_src,
                              class_names, class_names_src):
        try:
            w, h = Image.open(image_path).size
            size_str = f"{w} x {h}"
        except Exception:
            size_str = "N/A"

        print()
        print(LINE_EQ)
        print("  SWAP-Fighter Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Image", image_path)
        _kv("Image Size", size_str)
        _kv("Device", str(self.device))

        self._print_weights_block()
        self._print_model_block()
        self._print_preprocess_block()
        self._print_inference_block(
            threshold, threshold_src, class_names, class_names_src, n_images=1)

        print()
        print(LINE_EQ)

    def _print_header_multi(self, source, n_images, threshold, threshold_src,
                             class_names, class_names_src):
        print()
        print(LINE_EQ)
        print("  SWAP-Fighter Inference")
        print(LINE_EQ)

        _title("Input")
        _kv("Source", source)
        _kv("Images", n_images)
        _kv("Device", str(self.device))

        self._print_weights_block()
        self._print_model_block()
        self._print_preprocess_block()
        self._print_inference_block(
            threshold, threshold_src, class_names, class_names_src,
            n_images=n_images)

        print()
        print(LINE_EQ)

    def _print_weights_block(self):
        _title("Weights")
        _kv("Path", self.weights)
        if self._has_meta:
            _kv("Model Meta", f"{len(self._weights_meta)} fields")
        else:
            _kv("Model Meta", "not found, using config defaults")
        _kv("Load Status", self._load_status)

        epoch = self._info.get('epoch')
        if epoch is not None and epoch != 'N/A':
            _kv("Epoch", str(epoch))

        metrics = self._info.get('metrics')
        if metrics:
            try:
                mAP = float(metrics.get('mAP', 0))
                CF1 = float(metrics.get('CF1', 0))
                OF1 = float(metrics.get('OF1', 0))
                _kv("Metrics", f"mAP={mAP:.4f}, CF1={CF1:.4f}, OF1={OF1:.4f}")
            except Exception:
                pass

    def _print_model_block(self):
        info = self._info
        _title("Model")
        _kv("Backbone", info['backbone'])
        _kv("Num Classes", info['num_classes'])
        _kv("Image Size", info['img_size'])
        _kv("Pool / Head", f"{info['pool_type']} / {info['head_type']}")
        _kv("Multi-Scale", "yes" if info['multi_scale'] else "no")
        hidden = info['hidden_dim']
        _kv("Hidden Dim", hidden if hidden and hidden > 0 else "-")
        _kv("Dropout", info['dropout'])

    def _print_preprocess_block(self):
        info = self._info
        _title("Preprocess")
        _kv("Mean", str(info['mean']))
        _kv("Std", str(info['std']))
        _kv("Resize Ratio", info['resize_ratio'])
        _kv("Interpolation", info['interpolation'])
        _kv("Crop", info['crop'])

    def _print_inference_block(self, threshold, threshold_src, class_names,
                                class_names_src, n_images):
        _title("Inference")
        _kv("Images", n_images)
        _kv("Threshold", f"{threshold}  [from {threshold_src}]")
        _kv("Top-K", getattr(self.cfg, 'top_k', 5))
        _kv("Class Names Source", class_names_src)
        if class_names:
            preview = class_names[:5]
            _kv("Class Names",
                f"{preview}{' ...' if len(class_names) > 5 else ''}")

    # ------------------------------------------------------------------
    # Result 打印
    # ------------------------------------------------------------------
    def _print_result(self, r, threshold, class_names):
        probs = r['probs']
        name = os.path.basename(r['image'])
        n_pos = int((probs >= threshold).sum())
        top_k = getattr(self.cfg, 'top_k', 5)
        print_all = getattr(self.cfg, 'print_all', False)

        print()
        print(LINE_EQ)
        print(f"  {name}")
        print(LINE_EQ)

        print(f"\n  预测正类数 (score >= {threshold}): {n_pos}")

        print(f"\n  Top-{top_k}:")
        print(f"  {'#':>4}  {'Class':<24}  {'Score':>9}  {'Hit':^4}")
        print("  " + "─" * 48)
        for item in r['topk']:
            mark = "✓" if item['prob'] >= threshold else " "
            print(f"  {item['rank']:>4}  {item['class_name']:<24}  "
                  f"{item['prob']:>9.4f}  {mark:^4}")

        if print_all and r['positives']:
            print(f"\n  全部正类 (score >= {threshold}):")
            for item in r['positives']:
                print(f"    {item['class_name']:<24}  {item['prob']:.4f}")

        if self.output_image:
            print()
            print("  Output Files:")
            _kv("Image", self.output_image, indent=4)

    # ------------------------------------------------------------------
    def _draw(self, img, probs, class_names, threshold, top_k):
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        order = np.argsort(-probs)
        items = [(idx, probs[idx], "✓" if probs[idx] >= threshold else " ")
                 for idx in order[:top_k]]

        pad, line_h = 8, 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.55, 1

        overlay = img_bgr.copy()
        panel_w = 340
        panel_h = pad * 2 + line_h * len(items)
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        img_bgr = cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0)

        for i, (idx, sc, mark) in enumerate(items):
            cls_name = (class_names[idx] if idx < len(class_names)
                        else f"cls_{idx}")
            text = f"{mark} {cls_name:<18s}  {sc:.3f}"
            y0 = pad + i * line_h
            color = (0, 255, 0) if mark == "✓" else (200, 200, 200)
            cv2.putText(img_bgr, text, (pad + 3, y0 + line_h - 6),
                        font, fs, color, th, cv2.LINE_AA)

        return img_bgr