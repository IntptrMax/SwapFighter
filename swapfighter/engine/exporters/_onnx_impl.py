import os
import json
import math
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.box_ops import LOG_SCALE_MIN, LOG_SCALE_MAX
from ...utils.logger import LINE, LINE_EQ, _title, _kv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")

try:
    from torchvision.ops import batched_nms as tv_batched_nms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

def _cxcywh_angle_to_8pts(cx, cy, w, h, angle):
    N = cx.shape[0]
    half_w = w / 2
    half_h = h / 2

    corners = torch.zeros((N, 4, 2), device=cx.device, dtype=cx.dtype)
    corners[:, 0, 0] = half_w; corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w; corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w; corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w; corners[:, 3, 1] = -half_h

    cos_a = torch.cos(angle); sin_a = torch.sin(angle)
    rot = torch.stack([
        torch.stack([cos_a, sin_a], dim=1),
        torch.stack([-sin_a, cos_a], dim=1),
    ], dim=1)

    corners_rot = torch.matmul(corners, rot)
    center = torch.stack([cx, cy], dim=1).unsqueeze(1)
    return (corners_rot + center).reshape(N, 8)

def _pad_indices_with_dummy(keep, max_det, device):
    keep_padded = F.pad(keep, (0, max_det), value=-1)[:max_det]
    valid_mask = keep_padded >= 0
    padded = torch.where(valid_mask, keep_padded,
                         torch.zeros_like(keep_padded))
    return padded, valid_mask

def _print_export_header(cfg, output, opset, dynamic, end2end,
                         conf_val=None, nms_val=None, max_det=None):
    print()
    print(LINE_EQ)
    print("  SWAP-Fighter ONNX Export")
    print(LINE_EQ)

    _title("Config")
    _kv("Output", output)
    _kv("Task", cfg.task)
    _kv("Backbone", getattr(cfg, 'backbone_name', 'N/A'))
    _kv("Image Size", cfg.img_size)
    _kv("Num Classes", cfg.num_classes)

    if cfg.task in ('detect', 'obb', 'segment'):
        _kv("Reg Max", getattr(cfg, 'reg_max', 0))

    if cfg.task == 'obb':
        _kv("Angle Format",
            f"{getattr(cfg, 'angle_format', 'sin_cos')} "
            f"({getattr(cfg, 'angle_range', 'pi2')})")

    if cfg.task == 'segment':
        _kv("Proto Dim", getattr(cfg, 'proto_dim', 32))
        _kv("Proto Stride", getattr(cfg, 'proto_stride', 4))

    if cfg.task == 'keypoint':
        _kv("Num Keypoints", getattr(cfg, 'num_keypoints', 17))

    _kv("Opset", opset)
    _kv("Dynamic Batch", "yes" if dynamic else "no")
    _kv("End2End", "yes" if end2end else "no")

    if end2end:
        if conf_val is not None:
            _kv("Conf Threshold", f"{conf_val:.3f}")
        if nms_val is not None:
            _kv("NMS Threshold", f"{nms_val:.3f}")
        if max_det is not None:
            _kv("Max Det", max_det)

    print()
    print(LINE_EQ)

def _print_export_footer(output, end2end, output_names,
                         output_kind_str=None):
    size_mb = os.path.getsize(output) / (1024 * 1024)

    if output_kind_str is None:
        output_kind_str = "End2End (with NMS)" if end2end else "Raw"

    print()
    print(LINE_EQ)
    print("  Export Complete")
    print(LINE_EQ)
    _kv("Output", output)
    _kv("File Size", f"{size_mb:.2f} MB")
    _kv("Output Kind", output_kind_str)
    _kv("Outputs", " / ".join(output_names))
    print(LINE_EQ)

class DetectFCOSRawWrapper(nn.Module):
    def __init__(self, model, img_size, task='detect',
                 angle_format='sin_cos', angle_range='pi', reg_max=0):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = list(model.strides)
        self.task = task
        self.angle_format = angle_format
        self.angle_range = angle_range
        self.reg_max = reg_max
        self.use_dfl = (reg_max > 0)

    def _decode_dfl(self, reg_l, B, H, W):
        reg_4d = reg_l.view(B, 4, self.reg_max + 1, H * W).permute(0, 3, 1, 2)
        probs = reg_4d.softmax(dim=-1)
        idx = torch.arange(self.reg_max + 1, device=reg_l.device,
                           dtype=torch.float32)
        distances = (probs * idx).sum(dim=-1)
        return distances.unbind(-1)

    def forward(self, x):
        if self.task == 'obb':
            cls_list, reg_list, angle_list = self.model(x)
        else:
            cls_list, reg_list = self.model(x)
            angle_list = None

        all_levels = []

        for lvl in range(len(cls_list)):
            cls_l = cls_list[lvl]
            reg_l = reg_list[lvl]
            B = cls_l.shape[0]
            nc = cls_l.shape[1]
            H = cls_l.shape[2]
            W = cls_l.shape[3]
            stride = self.strides[lvl]

            y_coords = torch.arange(H, device=x.device, dtype=torch.float32) * stride + stride / 2
            x_coords = torch.arange(W, device=x.device, dtype=torch.float32) * stride + stride / 2
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            gx_flat = gx.reshape(-1)
            gy_flat = gy.reshape(-1)

            if self.use_dfl:
                l_d, t_d, r_d, b_d = self._decode_dfl(reg_l, B, H, W)
                cx_local = (r_d - l_d) * 0.5 * stride
                cy_local = (b_d - t_d) * 0.5 * stride
                w_pix = (l_d + r_d) * stride
                h_pix = (t_d + b_d) * stride
            else:
                reg_flat = reg_l.view(B, 4, -1)
                cx_local = reg_flat[:, 0, :] * stride
                cy_local = reg_flat[:, 1, :] * stride
                w_pix = torch.exp(reg_flat[:, 2, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                h_pix = torch.exp(reg_flat[:, 3, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

            if self.task == 'obb':
                angle_pred = angle_list[lvl]
                if self.angle_format == 'sin_cos':
                    angle_flat = angle_pred.view(B, 2, -1).permute(0, 2, 1)
                    angle = torch.atan2(angle_flat[..., 0], angle_flat[..., 1])
                else:
                    angle_flat = angle_pred.view(B, 1, -1).permute(0, 2, 1)
                    angle = angle_flat[..., 0]

                cos_a = torch.cos(angle); sin_a = torch.sin(angle)
                cx_pix = gx_flat.unsqueeze(0) + cx_local * cos_a - cy_local * sin_a
                cy_pix = gy_flat.unsqueeze(0) + cx_local * sin_a + cy_local * cos_a
            else:
                cx_pix = gx_flat.unsqueeze(0) + cx_local
                cy_pix = gy_flat.unsqueeze(0) + cy_local

            cx_n = cx_pix / self.img_size
            cy_n = cy_pix / self.img_size
            w_n = w_pix / self.img_size
            h_n = h_pix / self.img_size

            cls_sig = torch.sigmoid(cls_l).reshape(B, nc, H * W)

            if self.task == 'obb':
                boxes = torch.stack([cx_n, cy_n, w_n, h_n, angle], dim=2)
            else:
                boxes = torch.stack([cx_n, cy_n, w_n, h_n], dim=2)
            boxes = boxes.permute(0, 2, 1)
            out_l = torch.cat([boxes, cls_sig], dim=1)
            all_levels.append(out_l)

        return torch.cat(all_levels, dim=2)

class DetectFCOSEnd2EndWrapper(nn.Module):
    def __init__(self, model, img_size, task='detect',
                 angle_format='sin_cos', angle_range='pi', reg_max=0,
                 conf_threshold=0.5, nms_threshold=0.5, max_det=300):
        super().__init__()
        self.raw = DetectFCOSRawWrapper(
            model, img_size, task=task,
            angle_format=angle_format, angle_range=angle_range, reg_max=reg_max)
        self.task = task
        self.img_size = img_size
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_det = int(max_det)

    def forward(self, images):
        output0 = self.raw(images)
        out = output0[0]
        device = images.device
        if self.task == 'detect':
            return self._forward_detect(out, device)
        return self._forward_obb(out, device)

    def _forward_detect(self, out, device):
        boxes_n = out[:4, :]
        cls_scores = out[4:, :]

        cx = boxes_n[0] * self.img_size
        cy = boxes_n[1] * self.img_size
        w = boxes_n[2] * self.img_size
        h = boxes_n[3] * self.img_size
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        boxes_xyxy = torch.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        max_scores, max_labels = cls_scores.max(dim=0)
        max_labels = max_labels.to(torch.int64)
        max_scores = torch.nan_to_num(max_scores, nan=0.0, posinf=0.0, neginf=0.0)

        keep = tv_batched_nms(boxes_xyxy, max_scores, max_labels, self.nms_threshold)
        keep_padded, valid_mask = _pad_indices_with_dummy(keep, self.max_det, device)

        sel_boxes = boxes_xyxy[keep_padded]
        sel_scores = max_scores[keep_padded]
        sel_labels = max_labels[keep_padded]

        conf_valid = sel_scores > self.conf_threshold
        valid = valid_mask & conf_valid

        sel_scores = torch.where(valid, sel_scores, torch.zeros_like(sel_scores))
        sel_labels = torch.where(valid, sel_labels, torch.zeros_like(sel_labels))
        sel_boxes = sel_boxes * valid.to(sel_boxes.dtype).unsqueeze(1)

        num_dets = valid.sum().to(torch.int32).reshape(1, 1)
        return (num_dets, sel_boxes.unsqueeze(0),
                sel_scores.unsqueeze(0), sel_labels.to(torch.int32).unsqueeze(0))

    def _forward_obb(self, out, device):
        boxes_n = out[:4, :]
        angle = out[4, :]
        cls_scores = out[5:, :]

        cx = boxes_n[0] * self.img_size
        cy = boxes_n[1] * self.img_size
        w = boxes_n[2] * self.img_size
        h = boxes_n[3] * self.img_size

        cx = torch.nan_to_num(cx, nan=0.0)
        cy = torch.nan_to_num(cy, nan=0.0)
        w = torch.nan_to_num(w, nan=0.0)
        h = torch.nan_to_num(h, nan=0.0)
        angle = torch.nan_to_num(angle, nan=0.0)

        cos_a = torch.cos(angle); sin_a = torch.sin(angle)
        aabb_w = torch.abs(w * cos_a) + torch.abs(h * sin_a)
        aabb_h = torch.abs(w * sin_a) + torch.abs(h * cos_a)

        x1 = cx - aabb_w / 2; y1 = cy - aabb_h / 2
        x2 = cx + aabb_w / 2; y2 = cy + aabb_h / 2
        aabb_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        aabb_xyxy = torch.nan_to_num(aabb_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        max_scores, max_labels = cls_scores.max(dim=0)
        max_labels = max_labels.to(torch.int64)
        max_scores = torch.nan_to_num(max_scores, nan=0.0, posinf=0.0, neginf=0.0)

        keep = tv_batched_nms(aabb_xyxy, max_scores, max_labels, self.nms_threshold)
        keep_padded, valid_mask = _pad_indices_with_dummy(keep, self.max_det, device)

        sel_cx = cx[keep_padded]
        sel_cy = cy[keep_padded]
        sel_w = w[keep_padded]
        sel_h = h[keep_padded]
        sel_angle = angle[keep_padded]
        sel_scores = max_scores[keep_padded]
        sel_labels = max_labels[keep_padded]

        boxes_8 = _cxcywh_angle_to_8pts(sel_cx, sel_cy, sel_w, sel_h, sel_angle)

        conf_valid = sel_scores > self.conf_threshold
        valid = valid_mask & conf_valid

        sel_scores = torch.where(valid, sel_scores, torch.zeros_like(sel_scores))
        sel_labels = torch.where(valid, sel_labels, torch.zeros_like(sel_labels))
        boxes_8 = boxes_8 * valid.to(boxes_8.dtype).unsqueeze(1)

        num_dets = valid.sum().to(torch.int32).reshape(1, 1)
        return (num_dets, boxes_8.unsqueeze(0),
                sel_scores.unsqueeze(0), sel_labels.to(torch.int32).unsqueeze(0))

class SegRawWrapper(nn.Module):

    def __init__(self, model, img_size, num_classes,
                 num_mask_coef=32, reg_max=0):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = list(model.strides)
        self.num_classes = num_classes
        self.num_mask_coef = num_mask_coef
        self.reg_max = reg_max
        self.use_dfl = (reg_max > 0)

    def _decode_dfl(self, reg_l, B, H, W):
        reg_4d = reg_l.view(B, 4, self.reg_max + 1, H * W).permute(0, 3, 1, 2)
        probs = reg_4d.softmax(dim=-1)
        idx = torch.arange(self.reg_max + 1, device=reg_l.device,
                           dtype=torch.float32)
        distances = (probs * idx).sum(dim=-1)
        return distances.unbind(-1)

    def forward(self, x):
        cls_list, reg_list, coef_list, proto = self.model(x)

        all_levels = []
        for lvl in range(len(cls_list)):
            cls_l = cls_list[lvl]
            reg_l = reg_list[lvl]
            coef_l = coef_list[lvl]
            B = cls_l.shape[0]
            nc = cls_l.shape[1]
            H = cls_l.shape[2]
            W = cls_l.shape[3]
            stride = self.strides[lvl]

            y_coords = torch.arange(H, device=x.device,
                                    dtype=torch.float32) * stride + stride / 2
            x_coords = torch.arange(W, device=x.device,
                                    dtype=torch.float32) * stride + stride / 2
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            gx_flat = gx.reshape(-1)
            gy_flat = gy.reshape(-1)

            if self.use_dfl:
                l_d, t_d, r_d, b_d = self._decode_dfl(reg_l, B, H, W)
                cx_local = (r_d - l_d) * 0.5 * stride
                cy_local = (b_d - t_d) * 0.5 * stride
                w_pix = (l_d + r_d) * stride
                h_pix = (t_d + b_d) * stride
            else:
                reg_flat = reg_l.view(B, 4, -1)
                cx_local = reg_flat[:, 0, :] * stride
                cy_local = reg_flat[:, 1, :] * stride
                w_pix = torch.exp(reg_flat[:, 2, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                h_pix = torch.exp(reg_flat[:, 3, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

            cx_pix = gx_flat.unsqueeze(0) + cx_local
            cy_pix = gy_flat.unsqueeze(0) + cy_local

            cx_n = cx_pix / self.img_size
            cy_n = cy_pix / self.img_size
            w_n = w_pix / self.img_size
            h_n = h_pix / self.img_size

            cls_sig = torch.sigmoid(cls_l).reshape(B, nc, H * W)
            coef_flat = coef_l.reshape(B, self.num_mask_coef, H * W)

            out_l = torch.cat([
                torch.stack([cx_n, cy_n, w_n, h_n], dim=1),
                cls_sig,
                coef_flat,
            ], dim=1)
            all_levels.append(out_l)

        output0 = torch.cat(all_levels, dim=2)
        return output0, proto

class SegEnd2EndWrapper(nn.Module):

    def __init__(self, model, img_size, num_classes,
                 num_mask_coef=32, reg_max=0,
                 conf_threshold=0.5, nms_threshold=0.5, max_det=300):
        super().__init__()
        self.raw = SegRawWrapper(
            model=model, img_size=img_size,
            num_classes=num_classes,
            num_mask_coef=num_mask_coef,
            reg_max=reg_max)
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_mask_coef = num_mask_coef
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_det = int(max_det)

    def forward(self, images):
        output0, proto = self.raw(images)
        out = output0[0]
        device = images.device

        boxes_n = out[:4, :]
        cls_scores = out[4:4 + self.num_classes, :]
        coefs = out[4 + self.num_classes:, :]

        cx = boxes_n[0] * self.img_size
        cy = boxes_n[1] * self.img_size
        w = boxes_n[2] * self.img_size
        h = boxes_n[3] * self.img_size
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        boxes_xyxy = torch.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        max_scores, max_labels = cls_scores.max(dim=0)
        max_labels = max_labels.to(torch.int64)
        max_scores = torch.nan_to_num(max_scores, nan=0.0, posinf=0.0, neginf=0.0)

        keep = tv_batched_nms(boxes_xyxy, max_scores, max_labels,
                              self.nms_threshold)
        keep_padded, valid_mask = _pad_indices_with_dummy(
            keep, self.max_det, device)

        sel_boxes = boxes_xyxy[keep_padded]
        sel_scores = max_scores[keep_padded]
        sel_labels = max_labels[keep_padded]
        sel_coefs = coefs[:, keep_padded].transpose(0, 1)

        conf_valid = sel_scores > self.conf_threshold
        valid = valid_mask & conf_valid

        sel_scores = torch.where(valid, sel_scores,
                                  torch.zeros_like(sel_scores))
        sel_labels = torch.where(valid, sel_labels,
                                  torch.zeros_like(sel_labels))
        sel_boxes = sel_boxes * valid.to(sel_boxes.dtype).unsqueeze(1)
        sel_coefs = sel_coefs * valid.to(sel_coefs.dtype).unsqueeze(1)

        num_dets = valid.sum().to(torch.int32).reshape(1, 1)

        return (num_dets,
                sel_boxes.unsqueeze(0),
                sel_scores.unsqueeze(0),
                sel_labels.to(torch.int32).unsqueeze(0),
                sel_coefs.unsqueeze(0),
                proto)

def _inject_metadata(onnx_path, metadata):
    try:
        import onnx
        model = onnx.load(onnx_path)
        del model.metadata_props[:]
        written = 0
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict)):
                v_str = json.dumps(v)
            elif isinstance(v, bool):
                v_str = 'true' if v else 'false'
            else:
                v_str = str(v)
            prop = model.metadata_props.add()
            prop.key = k
            prop.value = v_str
            written += 1
        onnx.save(model, onnx_path)
        print(f"\n  [Metadata] 写入 {written} 个字段")
        return written
    except Exception as e:
        print(f"\n  [Metadata] 写入失败: {e}")
        return 0

def export_detect_onnx(model, cfg, weights, output=None, batch_size=1,
                       dynamic=False, simplify=False, opset=11,
                       end2end=False, conf=None, nms=None, max_det=300,
                       **kwargs):
    if output is None:
        output = os.path.splitext(weights)[0] + '.onnx'
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    conf_val = None
    nms_val = None

    if end2end:
        if not HAS_TORCHVISION:
            raise ImportError("end2end 需要 torchvision")
        if batch_size != 1:
            print("  [提示] end2end 强制 batch_size=1")
            batch_size = 1
        if dynamic:
            print("  [提示] end2end 不支持动态 batch，自动关闭")
            dynamic = False

        conf_val = conf if conf is not None else cfg.conf_threshold
        nms_val = nms if nms is not None else (
            cfg.obb_nms_threshold if cfg.task == 'obb'
            else cfg.nms_threshold)

        wrapper = DetectFCOSEnd2EndWrapper(
            model=model, img_size=cfg.img_size, task=cfg.task,
            angle_format=cfg.angle_format, angle_range=cfg.angle_range,
            reg_max=cfg.reg_max,
            conf_threshold=conf_val, nms_threshold=nms_val, max_det=max_det)
        output_names = ['num_dets', 'boxes', 'scores', 'classes']
    else:
        wrapper = DetectFCOSRawWrapper(
            model=model, img_size=cfg.img_size, task=cfg.task,
            angle_format=cfg.angle_format, angle_range=cfg.angle_range,
            reg_max=cfg.reg_max)
        output_names = ['output0']

    _print_export_header(cfg, output, opset, dynamic, end2end,
                          conf_val=conf_val, nms_val=nms_val,
                          max_det=max_det)

    wrapper = wrapper.cpu().eval()
    dummy = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output,
                input_names=['images'],
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False,
            )

    if simplify and not end2end:
        try:
            import onnx, onnxsim
            onnx_model = onnx.load(output)
            model_simp, check = onnxsim.simplify(
                onnx_model, input_shapes={'images': tuple(dummy.shape)})
            if check:
                onnx.save(model_simp, output)
                print("  [Simplify] 简化成功")
        except ImportError:
            print("  [Simplify] 警告: onnx-simplifier 未安装")
        except Exception as e:
            print(f"  [Simplify] 出错: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        'project_name': 'SwapFighter',
        'project_url': 'https://github.com/IntptrMax/SwapFighter',
        'export_time': now,
        'export_mode': 'end2end' if end2end else 'raw',
        'export_opset': opset,
        'train_backbone': cfg.backbone_name,
        'train_num_classes': cfg.num_classes,
        'train_img_size': cfg.img_size,
        'train_task': cfg.task,
        'train_reg_max': cfg.reg_max,
        'train_with_angle': cfg.with_angle,
        'train_angle_format': cfg.angle_format,
        'train_angle_range': cfg.angle_range,
        'train_mean': list(cfg.mean),
        'train_std': list(cfg.std),
        'model_strides': list(cfg.strides) if cfg.strides else [8, 16, 32],
        'conf_threshold': cfg.conf_threshold,
        'nms_threshold': cfg.nms_threshold,
        'class_names': getattr(cfg, 'class_names', None),
    }
    if end2end:
        meta['end2end_conf'] = conf_val
        meta['end2end_nms'] = nms_val
        meta['max_det'] = max_det

    _inject_metadata(output, meta)
    _print_export_footer(output, end2end, output_names)
    return output

def export_classify_onnx(model, cfg, weights, output=None,
                         batch_size=1, dynamic=False,
                         simplify=False, opset=11, **kwargs):
    if output is None:
        output = os.path.splitext(weights)[0] + '.onnx'
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _print_export_header(cfg, output, opset, dynamic, end2end=False)

    wrapper = model.cpu().eval()
    dummy = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch'}, 'logits': {0: 'batch'}}

    output_names = ['logits']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output,
                input_names=['images'],
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False,
            )

    if simplify:
        try:
            import onnx, onnxsim
            onnx_model = onnx.load(output)
            model_simp, check = onnxsim.simplify(
                onnx_model, input_shapes={'images': tuple(dummy.shape)})
            if check:
                onnx.save(model_simp, output)
                print("  [Simplify] 简化成功")
        except ImportError:
            print("  [Simplify] 警告: onnx-simplifier 未安装")
        except Exception as e:
            print(f"  [Simplify] 出错: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        'project_name': 'SwapFighter',
        'project_url': 'https://github.com/IntptrMax/SwapFighter',
        'export_time': now,
        'train_task': 'classify',
        'train_backbone': cfg.backbone_name,
        'train_num_classes': cfg.num_classes,
        'train_img_size': cfg.img_size,
        'train_pool_type': cfg.pool_type,
        'train_head_type': cfg.head_type,
        'train_mean': list(cfg.mean),
        'train_std': list(cfg.std),
        'class_names': getattr(model, 'class_names', None),
    }
    _inject_metadata(output, meta)
    _print_export_footer(output, end2end=False, output_names=output_names)
    return output

def export_seg_onnx(model, cfg, weights, output=None, batch_size=1,
                    dynamic=False, simplify=False, opset=11,
                    end2end=False, conf=None, nms=None, max_det=300,
                    **kwargs):
    if output is None:
        output = os.path.splitext(weights)[0] + '.onnx'
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_mask_coef = getattr(model, 'proto_dim', None)
    if num_mask_coef is None:
        num_mask_coef = getattr(model, 'num_mask_coef', None)
    if num_mask_coef is None:
        num_mask_coef = getattr(cfg, 'proto_dim', 32)

    conf_val = None
    nms_val = None

    if end2end:
        if not HAS_TORCHVISION:
            raise ImportError("end2end 需要 torchvision")
        if batch_size != 1:
            print("  [提示] end2end 强制 batch_size=1")
            batch_size = 1
        if dynamic:
            print("  [提示] end2end 不支持动态 batch，自动关闭")
            dynamic = False

        conf_val = conf if conf is not None else cfg.conf_threshold
        nms_val = nms if nms is not None else cfg.nms_threshold

        wrapper = SegEnd2EndWrapper(
            model=model, img_size=cfg.img_size,
            num_classes=cfg.num_classes,
            num_mask_coef=num_mask_coef,
            reg_max=getattr(cfg, 'reg_max', 0),
            conf_threshold=conf_val, nms_threshold=nms_val,
            max_det=max_det)
        output_names = ['num_dets', 'boxes', 'scores',
                        'classes', 'mask_coefs', 'proto']
    else:
        wrapper = SegRawWrapper(
            model=model, img_size=cfg.img_size,
            num_classes=cfg.num_classes,
            num_mask_coef=num_mask_coef,
            reg_max=getattr(cfg, 'reg_max', 0))
        output_names = ['output0', 'proto']

    _print_export_header(cfg, output, opset, dynamic, end2end,
                          conf_val=conf_val, nms_val=nms_val,
                          max_det=max_det)

    wrapper = wrapper.cpu().eval()
    dummy = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output,
                input_names=['images'],
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False,
            )

    if simplify and not end2end:
        try:
            import onnx, onnxsim
            onnx_model = onnx.load(output)
            model_simp, check = onnxsim.simplify(
                onnx_model, input_shapes={'images': tuple(dummy.shape)})
            if check:
                onnx.save(model_simp, output)
                print("  [Simplify] 简化成功")
        except ImportError:
            print("  [Simplify] 警告: onnx-simplifier 未安装")
        except Exception as e:
            print(f"  [Simplify] 出错: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        'project_name': 'SwapFighter',
        'project_url': 'https://github.com/IntptrMax/SwapFighter',
        'export_time': now,
        'export_mode': 'end2end' if end2end else 'raw',
        'export_opset': opset,
        'train_task': 'segment',
        'train_backbone': cfg.backbone_name,
        'train_num_classes': cfg.num_classes,
        'train_img_size': cfg.img_size,
        'train_reg_max': getattr(cfg, 'reg_max', 0),
        'train_proto_dim': num_mask_coef,
        'train_proto_stride': getattr(model, 'proto_stride',
                                       getattr(cfg, 'proto_stride', 4)),
        'train_mean': list(cfg.mean),
        'train_std': list(cfg.std),
        'model_strides': list(cfg.strides) if cfg.strides else [8, 16, 32],
        'conf_threshold': cfg.conf_threshold,
        'nms_threshold': cfg.nms_threshold,
        'class_names': getattr(cfg, 'class_names', None),
    }
    if end2end:
        meta['end2end_conf'] = conf_val
        meta['end2end_nms'] = nms_val
        meta['max_det'] = max_det

    _inject_metadata(output, meta)
    _print_export_footer(output, end2end, output_names)
    return output

class KeypointRawWrapper(nn.Module):

    def __init__(self, model, img_size, num_classes,
                 num_keypoints=17, reg_max=0):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = list(model.strides)
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.reg_max = reg_max
        self.use_dfl = (reg_max > 0)

    def _decode_dfl(self, reg_l, B, H, W):
        reg_4d = reg_l.view(B, 4, self.reg_max + 1, H * W).permute(0, 3, 1, 2)
        probs = reg_4d.softmax(dim=-1)
        idx = torch.arange(self.reg_max + 1, device=reg_l.device,
                            dtype=torch.float32)
        distances = (probs * idx).sum(dim=-1)
        return distances.unbind(-1)

    def forward(self, x):
        cls_list, reg_list, kpt_list = self.model(x)

        all_levels = []
        for lvl in range(len(cls_list)):
            cls_l = cls_list[lvl]
            reg_l = reg_list[lvl]
            kpt_l = kpt_list[lvl]
            B = cls_l.shape[0]
            nc = cls_l.shape[1]
            H = cls_l.shape[2]
            W = cls_l.shape[3]
            stride = self.strides[lvl]

            y = torch.arange(H, device=x.device,
                            dtype=torch.float32) * stride + stride / 2
            xx = torch.arange(W, device=x.device,
                             dtype=torch.float32) * stride + stride / 2
            gy, gx = torch.meshgrid(y, xx, indexing='ij')
            gx_flat = gx.reshape(-1)
            gy_flat = gy.reshape(-1)

            if self.use_dfl:
                l_d, t_d, r_d, b_d = self._decode_dfl(reg_l, B, H, W)
                cx_local = (r_d - l_d) * 0.5 * stride
                cy_local = (b_d - t_d) * 0.5 * stride
                w_pix = (l_d + r_d) * stride
                h_pix = (t_d + b_d) * stride
            else:
                reg_flat = reg_l.view(B, 4, -1)
                cx_local = reg_flat[:, 0, :] * stride
                cy_local = reg_flat[:, 1, :] * stride
                w_pix = torch.exp(reg_flat[:, 2, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                h_pix = torch.exp(reg_flat[:, 3, :].clamp(
                    LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

            cx_pix = gx_flat.unsqueeze(0) + cx_local
            cy_pix = gy_flat.unsqueeze(0) + cy_local

            cx_n = cx_pix / self.img_size
            cy_n = cy_pix / self.img_size
            w_n = w_pix / self.img_size
            h_n = h_pix / self.img_size

            cls_sig = torch.sigmoid(cls_l).reshape(B, nc, H * W)
            kpt_flat = kpt_l.reshape(B, self.num_keypoints * 2, H * W)

            out_l = torch.cat([
                torch.stack([cx_n, cy_n, w_n, h_n], dim=1),
                cls_sig,
                kpt_flat,
            ], dim=1)
            all_levels.append(out_l)

        return torch.cat(all_levels, dim=2)

class KeypointEnd2EndWrapper(nn.Module):

    def __init__(self, model, img_size, num_classes,
                 num_keypoints=17, reg_max=0,
                 conf_threshold=0.5, nms_threshold=0.5, max_det=300):
        super().__init__()
        self.raw = KeypointRawWrapper(
            model=model, img_size=img_size,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            reg_max=reg_max)
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_det = int(max_det)

    def forward(self, images):
        output0 = self.raw(images)
        out = output0[0]
        device = images.device
        K = self.num_keypoints

        boxes_n = out[:4, :]
        cls_scores = out[4:4 + self.num_classes, :]
        kpt_norm = out[4 + self.num_classes:, :]

        cx = boxes_n[0] * self.img_size
        cy = boxes_n[1] * self.img_size
        w = boxes_n[2] * self.img_size
        h = boxes_n[3] * self.img_size
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        boxes_xyxy = torch.nan_to_num(boxes_xyxy, nan=0.0,
                                       posinf=0.0, neginf=0.0)

        max_scores, max_labels = cls_scores.max(dim=0)
        max_labels = max_labels.to(torch.int64)
        max_scores = torch.nan_to_num(max_scores, nan=0.0,
                                       posinf=0.0, neginf=0.0)

        keep = tv_batched_nms(boxes_xyxy, max_scores, max_labels,
                              self.nms_threshold)
        keep_padded, valid_mask = _pad_indices_with_dummy(
            keep, self.max_det, device)

        sel_boxes = boxes_xyxy[keep_padded]
        sel_scores = max_scores[keep_padded]
        sel_labels = max_labels[keep_padded]

        sel_kpt = kpt_norm[:, keep_padded].transpose(0, 1)
        sel_kpt = sel_kpt.reshape(self.max_det, K, 2) * self.img_size

        conf_valid = sel_scores > self.conf_threshold
        valid = valid_mask & conf_valid
        v_f = valid.to(sel_boxes.dtype)

        sel_scores = torch.where(valid, sel_scores,
                                  torch.zeros_like(sel_scores))
        sel_labels = torch.where(valid, sel_labels,
                                  torch.zeros_like(sel_labels))
        sel_boxes = sel_boxes * v_f.unsqueeze(1)
        sel_kpt = sel_kpt * v_f.view(-1, 1, 1)

        num_dets = valid.sum().to(torch.int32).reshape(1, 1)
        return (num_dets,
                sel_boxes.unsqueeze(0),
                sel_scores.unsqueeze(0),
                sel_labels.to(torch.int32).unsqueeze(0),
                sel_kpt.unsqueeze(0))

def export_keypoint_onnx(model, cfg, weights, output=None, batch_size=1,
                         dynamic=False, simplify=False, opset=11,
                         end2end=False, conf=None, nms=None, max_det=300,
                         **kwargs):
    if output is None:
        output = os.path.splitext(weights)[0] + '.onnx'
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    K = getattr(model, 'num_keypoints', 17)

    conf_val = None
    nms_val = None

    if end2end:
        if not HAS_TORCHVISION:
            raise ImportError("end2end 需要 torchvision")
        if batch_size != 1:
            print("  [提示] end2end 强制 batch_size=1")
            batch_size = 1
        if dynamic:
            print("  [提示] end2end 不支持动态 batch，自动关闭")
            dynamic = False

        conf_val = conf if conf is not None else cfg.conf_threshold
        nms_val = nms if nms is not None else cfg.nms_threshold

        wrapper = KeypointEnd2EndWrapper(
            model=model, img_size=cfg.img_size,
            num_classes=cfg.num_classes, num_keypoints=K,
            reg_max=getattr(cfg, 'reg_max', 0),
            conf_threshold=conf_val, nms_threshold=nms_val,
            max_det=max_det)
        output_names = ['num_dets', 'boxes', 'scores', 'classes',
                        'keypoints']
    else:
        wrapper = KeypointRawWrapper(
            model=model, img_size=cfg.img_size,
            num_classes=cfg.num_classes, num_keypoints=K,
            reg_max=getattr(cfg, 'reg_max', 0))
        output_names = ['output0']

    _print_export_header(cfg, output, opset, dynamic, end2end,
                          conf_val=conf_val, nms_val=nms_val,
                          max_det=max_det)

    wrapper = wrapper.cpu().eval()
    dummy = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'images': {0: 'batch'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output,
                input_names=['images'],
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False)

    if simplify and not end2end:
        try:
            import onnx, onnxsim
            m = onnx.load(output)
            m_simp, ok = onnxsim.simplify(
                m, input_shapes={'images': tuple(dummy.shape)})
            if ok:
                onnx.save(m_simp, output)
                print("  [Simplify] 简化成功")
        except Exception as e:
            print(f"  [Simplify] 跳过: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        'project_name': 'SwapFighter',
        'project_url': 'https://github.com/IntptrMax/SwapFighter',
        'export_time': now,
        'export_mode': 'end2end' if end2end else 'raw',
        'export_opset': opset,
        'train_task': 'keypoint',
        'train_backbone': cfg.backbone_name,
        'train_num_classes': cfg.num_classes,
        'train_img_size': cfg.img_size,
        'train_reg_max': getattr(cfg, 'reg_max', 0),
        'train_num_keypoints': K,
        'train_keypoint_visibility':
            getattr(cfg, 'keypoint_visibility', 'auto'),
        'train_mean': list(cfg.mean),
        'train_std': list(cfg.std),
        'model_strides': list(cfg.strides) if cfg.strides else [8, 16, 32],
        'conf_threshold': cfg.conf_threshold,
        'nms_threshold': cfg.nms_threshold,
        'class_names': getattr(cfg, 'class_names', None),
    }
    if end2end:
        meta['end2end_conf'] = conf_val
        meta['end2end_nms'] = nms_val
        meta['max_det'] = max_det

    _inject_metadata(output, meta)
    _print_export_footer(output, end2end, output_names)
    return output

class MultiLabelSigmoidWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

def export_multilabel_onnx(model, cfg, weights, output=None,
                           batch_size=1, dynamic=False,
                           simplify=False, opset=11,
                           with_sigmoid=False,
                           **kwargs):
    if output is None:
        suffix = '_probs.onnx' if with_sigmoid else '.onnx'
        output = os.path.splitext(weights)[0] + suffix
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_name = 'probs' if with_sigmoid else 'logits'
    output_kind_str = (
        'Probs (sigmoid embedded)' if with_sigmoid else 'Raw (logits)'
    )

    if kwargs.get('end2end'):
        print("  [提示] multilabel 无 end2end 模式，"
              "已忽略 --end2end。如需图内 sigmoid 请用 --with-sigmoid。")

    _print_export_header(cfg, output, opset, dynamic, end2end=False)

    if with_sigmoid:
        wrapper = MultiLabelSigmoidWrapper(model).cpu().eval()
    else:
        wrapper = model.cpu().eval()

    dummy = torch.randn(batch_size, 3, cfg.img_size, cfg.img_size)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch'},
            output_name: {0: 'batch'},
        }

    output_names = [output_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output,
                input_names=['images'],
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                keep_initializers_as_inputs=False)

    if simplify:
        try:
            import onnx, onnxsim
            m = onnx.load(output)
            m_simp, ok = onnxsim.simplify(
                m, input_shapes={'images': tuple(dummy.shape)})
            if ok:
                onnx.save(m_simp, output)
                print("  [Simplify] 简化成功")
        except Exception as e:
            print(f"  [Simplify] 跳过: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        'project_name': 'SwapFighter',
        'project_url': 'https://github.com/IntptrMax/SwapFighter',
        'export_time': now,
        'export_mode': 'raw',
        'output_kind': output_name,
        'export_opset': opset,
        'train_task': cfg.task,
        'train_backbone': cfg.backbone_name,
        'train_num_classes': cfg.num_classes,
        'train_img_size': cfg.img_size,
        'train_pool_type': getattr(cfg, 'pool_type', 'gem'),
        'train_head_type': getattr(cfg, 'head_type', 'mlp'),
        'train_mean': list(cfg.mean),
        'train_std': list(cfg.std),
        'train_eval_threshold': getattr(cfg, 'eval_threshold', 0.5),
        'train_resize_ratio': 1.14,
        'class_names': getattr(cfg, 'class_names', None) or
                       getattr(model, 'class_names', None),
    }
    _inject_metadata(output, meta)

    _print_export_footer(output, end2end=False,
                         output_names=output_names,
                         output_kind_str=output_kind_str)
    return output