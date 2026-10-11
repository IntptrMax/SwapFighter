import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from .detect import DetectPredictor, _get_class_color
from ...utils.box_ops import letterbox
from ...utils.logger import LINE, LINE_EQ, _title, _kv

class SegPredictor(DetectPredictor):

    def __init__(self, cfg_dict, weights=None):
        self.mask_alpha = cfg_dict.pop('mask_alpha', 0.5)
        self.draw_mask = cfg_dict.pop('draw_mask', True)
        self.draw_bbox = cfg_dict.pop('draw_bbox', True)

        super().__init__(cfg_dict, weights)

    @torch.no_grad()
    def _predict_one(self, image_path):
        cfg = self.cfg
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_padded, scale, (pl, pt), _ = letterbox(
            img_rgb, new_shape=(cfg.img_size, cfg.img_size),
            auto=False, stride=32)

        x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])(Image.fromarray(img_padded)).unsqueeze(0).to(self.device)

        out = self.model(x)
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise RuntimeError(
                f"SegPredictor 期望 seg 模型输出 4 项，实际 {type(out)}")
        cls_l, reg_l, coef_l, proto = out[0], out[1], out[2], out[3]

        from ...utils.postprocess import post_process

        result = post_process(
            cls_l, reg_l,
            orig_w, orig_h, cfg.img_size,
            conf_threshold=cfg.conf_threshold,
            nms_threshold=cfg.nms_threshold,
            strides=cfg.strides,
            scale=scale,
            pad_left=pl, pad_top=pt,
            reg_max=cfg.reg_max,
            return_indices=True,
        )
        boxes, scores, labels, origin_idx = result

        masks = []
        if len(scores) > 0:

            all_coef = []
            for coef in coef_l:
                _, c, h, w = coef.shape
                all_coef.append(coef[0].permute(1, 2, 0).reshape(-1, c))
            all_coef = torch.cat(all_coef, dim=0)
            proto_img = proto[0]
            C = proto_img.shape[0]
            ph, pw = proto_img.shape[1], proto_img.shape[2]

            for bi, oi in zip(boxes, origin_idx):
                coef_vec = all_coef[oi]
                mask_logits = torch.einsum('c,chw->hw', coef_vec, proto_img)
                mask_np = torch.sigmoid(mask_logits).cpu().numpy()

                mask_up = cv2.resize(mask_np, (orig_w, orig_h),
                                     interpolation=cv2.INTER_LINEAR)
                mask_bin = (mask_up > 0.5).astype(np.uint8)

                x1i = max(0, int(bi[0])); y1i = max(0, int(bi[1]))
                x2i = min(orig_w, int(bi[2])); y2i = min(orig_h, int(bi[3]))
                cropped = np.zeros_like(mask_bin)
                if x2i > x1i and y2i > y1i:
                    cropped[y1i:y2i, x1i:x2i] = mask_bin[y1i:y2i, x1i:x2i]

                masks.append(cropped)

        img_vis = self._draw_seg(img_rgb, boxes, scores, labels, masks)

        if self.output_image:
            cv2.imwrite(self.output_image, img_vis)

        return {
            'image': image_path,
            'orig_size': (orig_w, orig_h),
            'mode': 'segment',
            'boxes': boxes,
            'obboxes': None,
            'scores': scores,
            'labels': labels,
            'masks': masks,
            'img_vis': img_vis,
        }

    def _print_model_block(self):
        cfg = self.cfg
        _title("Model")
        _kv("Mode", "Segment (instance segmentation)")

        if hasattr(self.model, 'backbone'):
            bb = self.model.backbone
            src = getattr(bb, 'resolved_source', 'N/A')
            _kv("Backbone", f"{getattr(cfg, 'backbone_name', 'N/A')} ({src})")
            _kv("Feature Indices",
                str(getattr(bb, 'feature_indices', 'N/A')))
            _kv("Feature Channels",
                str(getattr(bb, 'out_channels', 'N/A')))
            _kv("Strides", str(getattr(bb, 'strides', cfg.strides)))
            if getattr(bb, 'need_permute', False):
                _kv("NHWC → NCHW", "yes")
        else:
            _kv("Backbone", getattr(cfg, 'backbone_name', 'N/A'))
            _kv("Strides", str(getattr(cfg, 'strides', 'N/A')))

        neck_extra = "yes" if getattr(cfg, 'use_extra_conv', True) else "no"
        neck_type = getattr(cfg, 'neck_type', 'lightfpn')
        _kv("Neck", f"{neck_type} "
                    f"(out={getattr(cfg, 'neck_out_channels', 128)}, "
                    f"extra_conv={neck_extra})")

        _kv("Head", f"{getattr(cfg, 'head_type', 'decoupled')} "
                    f"(convs={getattr(cfg, 'head_num_convs', 3)}, "
                    f"norm={getattr(cfg, 'head_norm_layer', 'group')}, "
                    f"act={getattr(cfg, 'head_act_layer', 'silu')})")

        proto_dim = getattr(self.model, 'proto_dim',
                            getattr(cfg, 'proto_dim', 32))
        proto_stride = getattr(self.model, 'proto_stride',
                               getattr(cfg, 'proto_stride', 4))
        _kv("Task", f"cls={cfg.num_classes}, reg=4, "
                    f"mask_coef={proto_dim}")
        _kv("Proto Dim / Stride", f"{proto_dim} / {proto_stride}")

        reg_max = getattr(cfg, 'reg_max', 0)
        _kv("DFL", "disabled" if reg_max == 0
                  else f"reg_max={reg_max}")

    def _print_result(self, r):
        cfg = self.cfg
        boxes = r.get('boxes')
        scores = r.get('scores')
        labels = r.get('labels')
        masks = r.get('masks')

        n_total = 0 if scores is None else len(scores)

        print()
        print(LINE_EQ)
        print(f"  Result  ({n_total} detections)")
        print(LINE_EQ)

        if n_total == 0:
            print("  (no objects detected)")
            self._print_output_files()
            print()
            print(LINE_EQ)
            return

        from collections import Counter
        class_names = self.class_names
        if class_names:
            print("\n  Class Breakdown:")
            counts = Counter(int(l) for l in labels)
            max_cnt = max(counts.values())
            for cls_id in sorted(counts.keys()):
                name = (class_names[cls_id] if cls_id < len(class_names)
                        else f"cls_{cls_id}")
                cnt = counts[cls_id]
                bar = "█" * min(int(cnt / max_cnt * 20), 20)
                print(f"    {name:<20}  {cnt:>4d}  {bar}")

        top_k = getattr(cfg, 'top_k', 20)
        order = np.argsort(-np.array(scores))[:top_k]
        print()
        print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  "
              f"{'Geometry (x1,y1,x2,y2)':<40}  {'Mask':>8}")
        print("  " + "─" * 88)

        for i, idx in enumerate(order, 1):
            name = self._get_label_name(labels[idx])
            box = boxes[idx]
            geom = (f"({box[0]:.0f}, {box[1]:.0f}, "
                    f"{box[2]:.0f}, {box[3]:.0f})")

            mask_str = "N/A"
            if masks is not None and idx < len(masks):
                m = masks[idx]
                if m is not None:
                    mask_pixels = int(m.sum())
                    mask_str = f"{mask_pixels:>5d}px"

            print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  "
                  f"{geom:<40}  {mask_str:>8}")

        if n_total > top_k:
            print(f"  {'':>4}  ... ({n_total - top_k} more detections omitted)")

        self._print_output_files()
        print()
        print(LINE_EQ)

    def _draw_seg(self, img_rgb, boxes, scores, labels, masks):
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font_scale = max(0.4, min(1.8, min(vis.shape[:2]) / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        if self.draw_mask and len(masks) > 0:
            overlay = vis.copy()
            for mask, l in zip(masks, labels):
                color = _get_class_color(l)
                overlay[mask > 0] = color
            vis = cv2.addWeighted(overlay, self.mask_alpha,
                                  vis, 1.0 - self.mask_alpha, 0)

        if self.draw_bbox:
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, b)
                color = _get_class_color(l)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                name = self._get_label_name(l)
                cv2.putText(vis, f'{name}: {s:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, thickness)

        return vis