import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

from config import Config
from model import DetectFCOS
from utils import post_process, post_process_obb, letterbox


def draw_obb(img, obbox, color, thickness=2):
    """
    在图像上绘制旋转框（8个顶点）
    obbox: (8,) 或 (2,4) 顶点坐标 (x,y)
    """
    if isinstance(obbox, np.ndarray):
        if obbox.shape == (8,):
            pts = obbox.reshape(4, 2).astype(np.int32)
        else:
            pts = obbox.astype(np.int32)
    else:
        pts = np.array(obbox).reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def infer(model, image_path, device, cfg, conf_threshold=None, nms_threshold=None,
          class_names=None, output_image='output.jpg', output_txt=None, show_size=640):
    """
    对单张图片进行目标检测，支持 detect 和 obb 模式
    """
    model.eval()

    mode = cfg.mode
    img_size = cfg.img_size
    strides = cfg.strides
    mean = cfg.mean
    std = cfg.std
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

    # ---------- Letterbox 处理 ----------
    img_padded, scale, (pad_left, pad_top), _ = letterbox(
        img_rgb,
        new_shape=(img_size, img_size),
        auto=False,
        stride=32
    )

    # ---------- 转换为 Tensor 并归一化 ----------
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
                pad_top=pad_top
            )
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
            )

    # ---------- 颜色配置 ----------
    box_color = (0, 150, 0)          # 深绿色边框
    bg_color = (50, 120, 50)         # 暗灰绿色背景
    text_color = (255, 255, 255)     # 白色文字

    short_side = min(orig_w, orig_h)
    font_scale = max(0.4, min(1.8, short_side / 800.0))
    thickness = max(1, int(round(font_scale * 2)))
    text_thickness = max(1, int(round(font_scale * 1.5)))

    # ---------- 可视化并保存 ----------
    img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if mode == 'detect':
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, thickness)
            label_text = _format_label(label, score, class_names)
            _draw_text(img_vis, label_text, (x1, y1), font_scale, text_thickness, bg_color, text_color)
    else:  # obb
        for obbox, score, label in zip(obboxes, scores, labels):
            draw_obb(img_vis, obbox, box_color, thickness)
            pts = obbox.reshape(4, 2).astype(np.int32)
            x1, y1 = pts[0]
            label_text = _format_label(label, score, class_names)
            _draw_text(img_vis, label_text, (x1, y1), font_scale, text_thickness, bg_color, text_color)

    cv2.imwrite(output_image, img_vis)
    print(f"检测结果图像已保存至: {output_image}")

    # ---------- 保存文本结果 ----------
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
        print(f"检测结果已保存至: {output_txt}")

    # ---------- 显示 ----------
    if show_size and show_size > 0:
        display_width = show_size
        display_height = int(display_width * orig_h / orig_w)
        if display_height < 1:
            display_height = 1
        display_img = cv2.resize(img_vis, (display_width, display_height))
    else:
        display_img = img_vis

    cv2.imshow('Detection', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"检测到 {len(scores)} 个目标")
    if mode == 'detect':
        return boxes, scores, labels
    else:
        return obboxes, scores, labels


def _format_label(label, score, class_names):
    if class_names and int(label) < len(class_names):
        return f'{class_names[int(label)]}: {score:.2f}'
    else:
        return f'{int(label)}: {score:.2f}'


def _get_label_name(label, class_names):
    if class_names and int(label) < len(class_names):
        return class_names[int(label)]
    else:
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


def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter 单图推理 (支持 Detect / OBB)')
    parser.add_argument('--image', type=str, default='test.jpg',
                        help='待检测图片路径')
    parser.add_argument('--weights', type=str, default='./outputs/run1/best.pth',
                        help='模型权重文件路径')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='输出图像路径')
    parser.add_argument('--output-txt', type=str, default=None,
                        help='输出检测结果文本路径（可选）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='推理设备')
    parser.add_argument('--conf', type=float, default=None,
                        help='置信度阈值（覆盖 config）')
    parser.add_argument('--nms', type=float, default=None,
                        help='NMS IoU 阈值（覆盖 config）')
    parser.add_argument('--img-size', type=int, default=None,
                        help='模型输入尺寸（覆盖 config）')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数（覆盖 config）')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone 名称（覆盖 config）')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['detect', 'obb'],
                        help='检测模式（覆盖 config）')
    parser.add_argument('--angle-format', type=str, default=None,
                        choices=['sin_cos', 'direct'],
                        help='OBB 角度编码方式（覆盖 config）')
    parser.add_argument('--angle-range', type=str, default=None,
                        choices=['pi', 'pi2'],
                        help='OBB 角度范围（覆盖 config）')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='类别名称列表（按顺序，例如 "pill capsule"）')
    parser.add_argument('--show-size', type=int, default=640,
                        help='显示窗口宽度（高度自动适配），设为 0 则显示原始大小')
    return parser.parse_args()


def load_model(weights, backbone, num_classes, device, with_angle, angle_format):
    model = DetectFCOS(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        with_angle=with_angle,
        angle_format=angle_format
    ).to(device)

    checkpoint = torch.load(weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 去除多卡前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print(f"模型加载成功: {weights}")
    return model


def main():
    args = parse_args()

    # ---------- 加载配置 ----------
    cfg = Config()
    if args.mode is not None:
        cfg.mode = args.mode
    if args.angle_format is not None:
        cfg.angle_format = args.angle_format
    if args.angle_range is not None:
        cfg.angle_range = args.angle_range
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    if args.backbone is not None:
        cfg.backbone_name = args.backbone

    # 验证并设置配置
    cfg.validate_and_setup()

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device = torch.device('cpu')

    # ---------- 构建模型 ----------
    model = load_model(
        weights=args.weights,
        backbone=cfg.backbone_name,
        num_classes=cfg.num_classes,
        device=device,
        with_angle=cfg.with_angle,
        angle_format=cfg.angle_format
    )

    # ---------- 运行推理 ----------
    infer(
        model=model,
        image_path=args.image,
        device=device,
        cfg=cfg,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        class_names=args.class_names,
        output_image=args.output,
        output_txt=args.output_txt,
        show_size=args.show_size,
    )


if __name__ == '__main__':
    main()