import os
import sys
import io
import json
import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from contextlib import contextmanager

from config import Config
from model import DetectFCOS
from utils import post_process, post_process_obb, letterbox


# ============================================================
# 日志打印辅助
# ============================================================
@contextmanager
def suppress_stdout():
    """临时抑制 stdout，用于折叠 model.py 内部的散乱 print"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


LINE = "─" * 64
LINE_EQ = "═" * 64


def _title(text):
    print(f"\n  [{text}]")


def _kv(key, value, indent=4, total_width=40):
    """点号填充对齐：`    Key .......... Value`"""
    prefix = " " * indent
    label = f"{key} "
    pad_len = max(2, total_width - len(label))
    dots = " " + "." * pad_len + " "
    print(f"{prefix}{label}{dots} {value}")


def _bool_en(b):
    return "yes" if b else "no"


# ============================================================
# 类别颜色调色板（BGR），循环使用
# ============================================================
CLASS_COLORS = [
    (0, 200, 0),       # 绿
    (255, 100, 0),     # 蓝
    (0, 100, 255),     # 红
    (200, 200, 0),     # 青
    (200, 0, 200),     # 紫
    (0, 200, 200),     # 黄
    (100, 100, 255),   # 橙红
    (255, 0, 100),     # 紫红
    (100, 255, 100),   # 浅绿
    (255, 200, 100),   # 浅蓝
]


def _get_class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]


def draw_obb(img, obbox, color, thickness=2):
    """在图像上绘制旋转框（8 个顶点）"""
    if isinstance(obbox, np.ndarray):
        if obbox.shape == (8,):
            pts = obbox.reshape(4, 2).astype(np.int32)
        else:
            pts = obbox.astype(np.int32)
    else:
        pts = np.array(obbox).reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


# ============================================================
# 推理主函数（不显示图片，只返回结果和绘制好的图像）
# ============================================================
def infer(model, image_path, device, cfg, conf_threshold=None, nms_threshold=None,
          class_names=None, output_image='output.jpg', output_txt=None,
          output_json=None):
    """
    对单张图片进行目标检测，支持 detect 和 obb 模式。

    返回：
        detect: (boxes, scores, labels, img_vis)
        obb:    (obboxes, scores, labels, img_vis)
    """
    model.eval()

    mode = cfg.mode
    img_size = cfg.img_size
    strides = cfg.strides
    mean = cfg.mean
    std = cfg.std
    reg_max = getattr(cfg, 'reg_max', 0)
    angle_format = cfg.angle_format if mode == 'obb' else None
    angle_range = cfg.angle_range if mode == 'obb' else None

    if conf_threshold is None:
        conf_threshold = cfg.conf_threshold
    if nms_threshold is None:
        nms_threshold = cfg.obb_nms_threshold if mode == 'obb' else cfg.nms_threshold

    # ---------- 读取图片 ----------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # ---------- Letterbox ----------
    img_padded, scale, (pad_left, pad_top), _ = letterbox(
        img_rgb,
        new_shape=(img_size, img_size),
        auto=False,
        stride=32
    )

    # ---------- ToTensor + Normalize ----------
    img_pil = Image.fromarray(img_padded)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ---------- 推理 ----------
    with torch.no_grad():
        if mode == 'detect':
            cls_logits_list, reg_preds_list = model(input_tensor)
            boxes, scores, labels = post_process(
                cls_logits_list, reg_preds_list,
                orig_w, orig_h, img_size,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                strides=strides,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                reg_max=reg_max,
            )
            obboxes = None
        else:  # obb
            cls_logits_list, reg_preds_list, angle_preds_list = model(input_tensor)
            obboxes, scores, labels = post_process_obb(
                cls_logits_list, reg_preds_list, angle_preds_list,
                orig_w, orig_h, img_size,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                strides=strides,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                angle_format=angle_format,
                angle_range=angle_range,
                reg_max=reg_max,
            )
            boxes = None

    # ---------- 字体大小自适应 ----------
    short_side = min(orig_w, orig_h)
    font_scale = max(0.4, min(1.8, short_side / 800.0))
    thickness = max(1, int(round(font_scale * 2)))
    text_thickness = max(1, int(round(font_scale * 1.5)))

    # ---------- 可视化 ----------
    img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if mode == 'detect':
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = _get_class_color(label)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
            label_text = _format_label(label, score, class_names)
            _draw_text(img_vis, label_text, (x1, y1),
                       font_scale, text_thickness, color, (255, 255, 255))
    else:  # obb
        for obbox, score, label in zip(obboxes, scores, labels):
            color = _get_class_color(label)
            draw_obb(img_vis, obbox, color, thickness)
            pts = obbox.reshape(4, 2).astype(np.int32)
            x1, y1 = pts[0]
            label_text = _format_label(label, score, class_names)
            _draw_text(img_vis, label_text, (x1, y1),
                       font_scale, text_thickness, color, (255, 255, 255))

    # ---------- 保存图像 ----------
    if output_image:
        cv2.imwrite(output_image, img_vis)

    # ---------- 保存 TXT ----------
    if output_txt:
        with open(output_txt, 'w') as f:
            if mode == 'detect':
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box)
                    label_name = _get_label_name(label, class_names)
                    f.write(f"{label_name} {x1} {y1} {x2} {y2} {score:.4f}\n")
            else:
                for obbox, score, label in zip(obboxes, scores, labels):
                    pts = obbox.reshape(4, 2).flatten()
                    pts_str = ' '.join([f"{p:.2f}" for p in pts])
                    label_name = _get_label_name(label, class_names)
                    f.write(f"{label_name} {pts_str} {score:.4f}\n")

    # ---------- 保存 JSON ----------
    if output_json:
        result = {
            'image': image_path,
            'orig_size': [orig_w, orig_h],
            'mode': mode,
            'detections': []
        }
        if mode == 'detect':
            for box, score, label in zip(boxes, scores, labels):
                result['detections'].append({
                    'label': int(label),
                    'label_name': _get_label_name(label, class_names),
                    'score': float(score),
                    'bbox_xyxy': [float(v) for v in box],
                })
        else:
            for obbox, score, label in zip(obboxes, scores, labels):
                result['detections'].append({
                    'label': int(label),
                    'label_name': _get_label_name(label, class_names),
                    'score': float(score),
                    'obbox_8': [float(v) for v in obbox],
                })
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ---------- 返回 ----------
    if mode == 'detect':
        return boxes, scores, labels, img_vis
    else:
        return obboxes, scores, labels, img_vis


# ============================================================
# 辅助函数
# ============================================================
def _format_label(label, score, class_names):
    if class_names and int(label) < len(class_names):
        return f'{class_names[int(label)]}: {score:.2f}'
    return f'{int(label)}: {score:.2f}'


def _get_label_name(label, class_names):
    if class_names and int(label) < len(class_names):
        return class_names[int(label)]
    return str(int(label))


def _draw_text(img, text, pos, font_scale, thickness, bg_color, text_color):
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(img, (x, y - text_h - 8), (x + text_w + 4, y + 4),
                  bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter 单图推理 (Detect / OBB)')
    parser.add_argument('--image', type=str, default='test.jpg')
    parser.add_argument('--weights', type=str, default='./outputs/run1/best.pth')
    parser.add_argument('--output', type=str, default='output.jpg')
    parser.add_argument('--output-txt', type=str, default=None)
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # ---------- 命令行覆盖项 ----------
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--nms', type=float, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['detect', 'obb'])
    parser.add_argument('--angle-format', type=str, default=None,
                        choices=['sin_cos', 'direct'])
    parser.add_argument('--angle-range', type=str, default=None,
                        choices=['pi', 'pi2'])
    parser.add_argument('--class-names', type=str, nargs='+', default=None)
    parser.add_argument('--show-size', type=int, default=640)
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='不打开显示窗口（无 GUI 环境用）')

    parser.add_argument('--neck-out-channels', type=int, default=None)
    parser.add_argument('--header-num-convs', type=int, default=None)
    parser.add_argument('--use-extra-conv', action='store_true', default=None)
    parser.add_argument('--no-extra-conv', dest='use_extra_conv',
                        action='store_false')
    parser.add_argument('--neck-type', type=str, default=None,
                        choices=['lightfpn', 'bifpn'])
    parser.add_argument('--neck-num-layers', type=int, default=None)
    parser.add_argument('--reg-max', type=int, default=None)

    # 解耦头相关
    parser.add_argument('--head-type', type=str, default=None,
                        choices=['decoupled'])
    parser.add_argument('--head-norm-layer', type=str, default=None,
                        choices=['group', 'batch', 'none'])
    parser.add_argument('--head-act-layer', type=str, default=None,
                        choices=['silu', 'relu', 'gelu'])
    parser.add_argument('--head-cls-num-convs', type=int, default=None)
    parser.add_argument('--head-reg-num-convs', type=int, default=None)
    parser.add_argument('--head-angle-num-convs', type=int, default=None)

    # 输出控制
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='只输出最终结果，不显示配置区块')
    parser.add_argument('--top-k', type=int, default=20,
                        help='结果表格显示前 K 个检测框（按置信度排序）')

    return parser.parse_args()


# ============================================================
# 加载模型
# ============================================================
def load_model(weights, device, cli_overrides=None, quiet=False):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    checkpoint = torch.load(weights, map_location=device, weights_only=False)

    # ---------- 解析 .pt ----------
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        meta = checkpoint.get('model_meta', {})
        has_meta = bool(meta)
    else:
        state_dict = checkpoint
        meta = {}
        has_meta = False

    # ---------- 应用命令行覆盖 ----------
    applied_overrides = []
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                meta[k] = v
                applied_overrides.append(k)

    # ---------- 从 meta 中提取模型结构参数 ----------
    cfg = Config()

    backbone_name = meta.get('backbone_name', cfg.backbone_name)
    num_classes = meta.get('num_classes', cfg.num_classes)
    img_size = meta.get('img_size', cfg.img_size)
    reg_max = meta.get('reg_max', getattr(cfg, 'reg_max', 0))
    neck_out_channels = meta.get('neck_out_channels', getattr(cfg, 'neck_out_channels', 128))
    head_num_convs = meta.get('head_num_convs', getattr(cfg, 'head_num_convs', 2))
    use_extra_conv = meta.get('use_extra_conv', getattr(cfg, 'use_extra_conv', True))
    neck_type = meta.get('neck_type', getattr(cfg, 'neck_type', 'lightfpn'))
    neck_num_layers = meta.get('neck_num_layers', getattr(cfg, 'neck_num_layers', 2))
    with_angle = meta.get('with_angle', False)
    angle_format = meta.get('angle_format', 'sin_cos')
    angle_range = meta.get('angle_range', 'pi')
    mode = meta.get('mode', 'detect')

    head_type = meta.get('head_type', getattr(cfg, 'head_type', 'decoupled'))
    head_norm_layer = meta.get('head_norm_layer', getattr(cfg, 'head_norm_layer', 'group'))
    head_act_layer = meta.get('head_act_layer', getattr(cfg, 'head_act_layer', 'silu'))
    head_cls_num_convs = meta.get('head_cls_num_convs', getattr(cfg, 'head_cls_num_convs', None))
    head_reg_num_convs = meta.get('head_reg_num_convs', getattr(cfg, 'head_reg_num_convs', None))
    head_angle_num_convs = meta.get('head_angle_num_convs', getattr(cfg, 'head_angle_num_convs', None))

    # ---------- 构建模型 ----------
    with suppress_stdout():
        model = DetectFCOS(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=False,
            pretrained_path=None,
            pretrained_source='none',
            with_angle=with_angle,
            angle_format=angle_format,
            neck_out_channels=neck_out_channels,
            head_num_convs=head_num_convs,
            use_extra_conv=use_extra_conv,
            img_size=img_size,
            neck_type=neck_type,
            neck_num_layers=neck_num_layers,
            reg_max=reg_max,
            head_type=head_type,
            head_norm_layer=head_norm_layer,
            head_act_layer=head_act_layer,
            head_cls_num_convs=head_cls_num_convs,
            head_reg_num_convs=head_reg_num_convs,
            head_angle_num_convs=head_angle_num_convs,
        ).to(device)

    # ---------- 加载权重 ----------
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    try:
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        real_missing = [k for k in missing if 'num_batches_tracked' not in k]
        real_unexpected = [k for k in unexpected if 'num_batches_tracked' not in k]

        if not real_missing and not real_unexpected:
            load_status = "OK (full match)"
        else:
            parts = []
            if real_missing:
                parts.append(f"missing {len(real_missing)}")
            if real_unexpected:
                parts.append(f"unexpected {len(real_unexpected)}")
            load_status = "WARN: " + ", ".join(parts)
    except RuntimeError as e:
        print()
        print(LINE_EQ)
        print("  [ERROR] 权重加载失败")
        print(LINE_EQ)
        _kv("head_type", head_type)
        _kv("head_norm_layer", head_norm_layer)
        _kv("head_act_layer", head_act_layer)
        _kv("neck_type", neck_type)
        _kv("neck_out_channels", neck_out_channels)
        _kv("reg_max", reg_max)
        print()
        print("  原始报错:")
        print(f"    {e}")
        print(LINE_EQ)
        raise

    model.eval()

    # ---------- 组装返回 ----------
    final_meta = {
        'mode': mode,
        'backbone_name': backbone_name,
        'num_classes': num_classes,
        'img_size': img_size,
        'reg_max': reg_max,
        'neck_out_channels': neck_out_channels,
        'head_type': head_type,
        'head_num_convs': head_num_convs,
        'head_norm_layer': head_norm_layer,
        'head_act_layer': head_act_layer,
        'head_cls_num_convs': head_cls_num_convs,
        'head_reg_num_convs': head_reg_num_convs,
        'head_angle_num_convs': head_angle_num_convs,
        'use_extra_conv': use_extra_conv,
        'neck_type': neck_type,
        'neck_num_layers': neck_num_layers,
        'with_angle': with_angle,
        'angle_format': angle_format,
        'angle_range': angle_range,
        'angle_loss_type': meta.get('angle_loss_type', 'l1'),
        'mean': meta.get('mean', list(cfg.mean)),
        'std': meta.get('std', list(cfg.std)),
        'strides': meta.get('strides', getattr(cfg, 'strides', None)),
        'conf_threshold': meta.get('conf_threshold', cfg.conf_threshold),
        'nms_threshold': meta.get('nms_threshold', cfg.nms_threshold),
        'obb_nms_threshold': meta.get('obb_nms_threshold',
                                      getattr(cfg, 'obb_nms_threshold', None)),
    }

    load_info = {
        'has_meta': has_meta,
        'meta_fields': len(meta),
        'applied_overrides': applied_overrides,
        'load_status': load_status,
    }

    return model, final_meta, load_info


# ============================================================
# 打印推理配置
# ============================================================
def _print_config(weights, image_path, orig_w, orig_h, device, meta, load_info,
                  conf_threshold, nms_threshold):
    print()
    print(LINE_EQ)
    print("  SWAP-Fighter Inference")
    print(LINE_EQ)

    # ---- [Input] ----
    _title("Input")
    _kv("Image", f"{image_path}")
    _kv("Image Size", f"{orig_w} x {orig_h}")
    _kv("Device", str(device))

    # ---- [Weights] ----
    _title("Weights")
    _kv("Path", weights)
    if load_info['has_meta']:
        _kv("Model Meta", f"{load_info['meta_fields']} fields")
    else:
        _kv("Model Meta", "not found, using config defaults")
    if load_info['applied_overrides']:
        ov = ", ".join(load_info['applied_overrides'][:5])
        if len(load_info['applied_overrides']) > 5:
            ov += f" (+{len(load_info['applied_overrides']) - 5} more)"
        _kv("CLI Overrides", ov)
    else:
        _kv("CLI Overrides", "none")
    _kv("Load Status", load_info['load_status'])

    # ---- [Model] ----
    _title("Model")
    mode_str = ("OBB (rotated bbox)" if meta['mode'] == 'obb'
                else "Detect (axis-aligned bbox)")
    _kv("Mode", mode_str)
    _kv("Backbone", meta['backbone_name'])
    _kv("Strides", str(meta['strides']))
    neck_extra = _bool_en(meta['use_extra_conv'])
    neck_line = f"{meta['neck_type']} (out={meta['neck_out_channels']}, extra_conv={neck_extra})"
    if meta['neck_type'] == 'bifpn':
        neck_line += f", layers={meta['neck_num_layers']}"
    _kv("Neck", neck_line)
    _kv("Head", f"{meta['head_type']} (convs={meta['head_num_convs']}, "
                f"norm={meta['head_norm_layer']}, act={meta['head_act_layer']})")

    if meta['mode'] == 'obb':
        _kv("Task", f"cls={meta['num_classes']}, reg=4, angle=2 ({meta['angle_format']})")
    else:
        _kv("Task", f"cls={meta['num_classes']}, reg=4")

    _kv("DFL", "disabled" if meta['reg_max'] == 0
              else f"reg_max={meta['reg_max']} (bin={meta['reg_max'] + 1})")

    # ---- [Inference] ----
    _title("Inference")
    _kv("Conf Threshold", f"{conf_threshold:.3f}")
    _kv("NMS Threshold", f"{nms_threshold:.3f}")
    if meta['mode'] == 'obb':
        _kv("Angle Range", meta['angle_range'])

    print()
    print(LINE_EQ)


# ============================================================
# 打印推理结果
# ============================================================
def _print_result(mode, boxes, obboxes, scores, labels, class_names,
                  output_image, output_txt, output_json, top_k=20):
    n_total = len(scores)

    print()
    print(LINE_EQ)
    print(f"  Result  ({n_total} detections)")
    print(LINE_EQ)

    if n_total == 0:
        print("  (no objects detected)")
    else:
        # ---- 类别分布 ----
        if class_names:
            print("\n  Class Breakdown:")
            from collections import Counter
            counts = Counter(int(l) for l in labels)
            for cls_id in sorted(counts.keys()):
                name = _get_label_name(cls_id, class_names)
                cnt = counts[cls_id]
                bar = "█" * min(int(cnt / max(counts.values()) * 20), 20)
                print(f"    {name:<20}  {cnt:>4d}  {bar}")

        # ---- 检测详情（按分数排序）----
        order = np.argsort(-np.array(scores))[:top_k]
        print()
        if mode == 'detect':
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  {'Geometry (x1,y1,x2,y2)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = _get_label_name(labels[idx], class_names)
                box = boxes[idx]
                geom = f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")
        else:
            print(f"  {'#':>4}  {'Label':<16}  {'Score':>7}  {'Geometry (cx,cy,w,h,angle)':<40}")
            print("  " + "─" * 76)
            for i, idx in enumerate(order, 1):
                name = _get_label_name(labels[idx], class_names)
                pts = obboxes[idx].reshape(4, 2)
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                v1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
                v2 = 0.5 * ((pts[2] - pts[1]) + (pts[3] - pts[0]))
                w = float(np.linalg.norm(v1))
                h = float(np.linalg.norm(v2))
                angle = float(np.degrees(np.arctan2(v1[1], v1[0])))
                geom = f"({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f}, {angle:+.1f}°)"
                print(f"  {i:>4}  {name:<16}  {scores[idx]:>7.4f}  {geom:<40}")

        if n_total > top_k:
            print(f"  {'':>4}  ... ({n_total - top_k} more detections omitted)")

    # ---- 输出文件 ----
    print()
    print("  Output Files:")
    if output_image:
        _kv("Image", output_image, indent=4)
    if output_txt:
        _kv("TXT", output_txt, indent=4)
    if output_json:
        _kv("JSON", output_json, indent=4)

    print()
    print(LINE_EQ)
    print()


# ============================================================
# 显示图片
# ============================================================
def _show_image(img_vis, show_size=640):
    """在窗口里显示检测结果，按任意键关闭"""
    try:
        if show_size and show_size > 0:
            h, w = img_vis.shape[:2]
            display_width = show_size
            display_height = int(display_width * h / w)
            display_height = max(1, display_height)
            display_img = cv2.resize(img_vis, (display_width, display_height))
        else:
            display_img = img_vis

        print("  [Display] 按任意键关闭窗口...")
        cv2.imshow('Detection', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"  [Display] 无法打开显示窗口（无 GUI 环境？）: {e}")


# ============================================================
# 主入口
# ============================================================
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device = torch.device('cpu')

    cli_overrides = {
        'mode': args.mode,
        'backbone_name': args.backbone,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'reg_max': args.reg_max,
        'neck_out_channels': args.neck_out_channels,
        'head_num_convs': args.header_num_convs,
        'use_extra_conv': args.use_extra_conv,
        'angle_format': args.angle_format,
        'angle_range': args.angle_range,
        'neck_type': args.neck_type,
        'neck_num_layers': args.neck_num_layers,
        'head_type': args.head_type,
        'head_norm_layer': args.head_norm_layer,
        'head_act_layer': args.head_act_layer,
        'head_cls_num_convs': args.head_cls_num_convs,
        'head_reg_num_convs': args.head_reg_num_convs,
        'head_angle_num_convs': args.head_angle_num_convs,
    }

    # ---------- 加载权重 ----------
    model, meta, load_info = load_model(
        weights=args.weights,
        device=device,
        cli_overrides=cli_overrides,
        quiet=args.quiet,
    )

    # ---------- 用 meta 更新 cfg ----------
    cfg = Config()
    cfg.mode = meta['mode']
    cfg.with_angle = meta['with_angle']
    cfg.backbone_name = meta['backbone_name']
    cfg.num_classes = meta['num_classes']
    cfg.img_size = meta['img_size']
    cfg.reg_max = meta['reg_max']
    cfg.neck_out_channels = meta['neck_out_channels']
    cfg.head_num_convs = meta['head_num_convs']
    cfg.use_extra_conv = meta['use_extra_conv']
    cfg.neck_type = meta['neck_type']
    cfg.neck_num_layers = meta['neck_num_layers']
    cfg.angle_format = meta['angle_format']
    cfg.angle_range = meta['angle_range']
    cfg.angle_loss_type = meta['angle_loss_type']
    cfg.mean = meta['mean']
    cfg.std = meta['std']
    cfg.strides = meta['strides']
    cfg.conf_threshold = meta['conf_threshold']
    cfg.nms_threshold = meta['nms_threshold']
    cfg.obb_nms_threshold = meta['obb_nms_threshold'] or cfg.nms_threshold

    # ---------- 计算最终生效阈值 ----------
    conf_threshold = args.conf if args.conf is not None else cfg.conf_threshold
    if args.nms is not None:
        nms_threshold = args.nms
    else:
        nms_threshold = cfg.obb_nms_threshold if cfg.mode == 'obb' else cfg.nms_threshold

    # ---------- 读取图像尺寸（用于配置打印）----------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")
    orig_h, orig_w = img_bgr.shape[:2]

    # ---------- 打印配置区块 ----------
    if not args.quiet:
        _print_config(
            weights=args.weights,
            image_path=args.image,
            orig_w=orig_w,
            orig_h=orig_h,
            device=device,
            meta=meta,
            load_info=load_info,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

    # ---------- 推理 ----------
    result = infer(
        model=model,
        image_path=args.image,
        device=device,
        cfg=cfg,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        class_names=args.class_names,
        output_image=args.output,
        output_txt=args.output_txt,
        output_json=args.output_json,
    )

    # ---------- 解包结果 ----------
    if cfg.mode == 'detect':
        boxes, scores, labels, img_vis = result
        obboxes = None
    else:
        obboxes, scores, labels, img_vis = result
        boxes = None

    # ---------- ★ 第一步：先打印结果 ----------
    if not args.quiet:
        _print_result(
            mode=cfg.mode,
            boxes=boxes,
            obboxes=obboxes,
            scores=scores,
            labels=labels,
            class_names=args.class_names,
            output_image=args.output,
            output_txt=args.output_txt,
            output_json=args.output_json,
            top_k=args.top_k,
        )
    else:
        print(f"Detected {len(scores)} objects. Saved to {args.output}")

    # ---------- ★ 第二步：后显示图片 ----------
    if not args.no_show:
        _show_image(img_vis, show_size=args.show_size)


if __name__ == '__main__':
    main()