import torch
import cv2
import argparse
from PIL import Image
from torchvision import transforms

from config import Config
from model import DetectFCOS
from utils import post_process, letterbox


def infer(model, image_path, device, img_size, strides, conf_threshold=0.5, nms_threshold=0.5,
          mean=[0, 0, 0], std=[1, 1, 1], class_names=None, output_image='output.jpg',
          output_txt=None, show_size=640):
    """
    对单张图片进行目标检测，保存图像和可选文本结果。
    """
    model.eval()

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
        cls_logits_list, reg_preds_list = model(input_tensor)

    # ---------- 后处理 ----------
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

    # ---------- 颜色配置 ----------
    # 边框颜色 (BGR) 柔和的暗绿色
    box_color = (0, 150, 0)          # 深绿色边框
    bg_color = (50, 120, 50)         # 暗灰绿色背景
    text_color = (255, 255, 255)     # 白色文字

    # 根据图像尺寸动态调整字体和线宽
    short_side = min(orig_w, orig_h)
    font_scale = max(0.4, min(1.8, short_side / 800.0))
    thickness = max(1, int(round(font_scale * 2)))
    text_thickness = max(1, int(round(font_scale * 1.5)))
    print(f"自适应参数: font_scale={font_scale:.2f}, thickness={thickness}, text_thickness={text_thickness}")

    # ---------- 可视化并保存 ----------
    img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, thickness)

        if class_names and int(label) < len(class_names):
            label_text = f'{class_names[int(label)]}: {score:.2f}'
        else:
            label_text = f'{int(label)}: {score:.2f}'

        # 计算文本尺寸，绘制背景矩形
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        # 背景矩形（略微比文字大）
        cv2.rectangle(img_vis, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1 + 4),
                      bg_color, -1)
        # 文字（白色）
        cv2.putText(img_vis, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

    # 保存图像（原始尺寸）
    cv2.imwrite(output_image, img_vis)
    print(f"检测结果图像已保存至: {output_image}")

    # 如果指定了输出文本文件，保存检测结果
    if output_txt:
        with open(output_txt, 'w') as f:
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                if class_names and int(label) < len(class_names):
                    label_name = class_names[int(label)]
                else:
                    label_name = str(int(label))
                f.write(f"{label_name} {x1} {y1} {x2} {y2} {score:.4f}\n")
        print(f"检测结果已保存至: {output_txt}")

    # 显示图像（可缩放）
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

    print(f"检测到 {len(boxes)} 个目标")
    return boxes, scores, labels


def parse_args():
    parser = argparse.ArgumentParser(description='ConvNeXtV2-FCOS 单图推理')
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
    parser.add_argument('--conf', type=float, default=0.8,
                        help='置信度阈值')
    parser.add_argument('--nms', type=float, default=0.5,
                        help='NMS IoU 阈值')
    parser.add_argument('--img-size', type=int, default=None,
                        help='模型输入尺寸（若不指定则使用 config 中的值）')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数（若不指定则使用 config 中的值）')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone 名称（若不指定则使用 config 中的值）')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='类别名称列表（按顺序，例如 "person car dog"）')
    parser.add_argument('--show-size', type=int, default=640,
                        help='显示窗口宽度（高度自动适配），设为 0 则显示原始大小')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ---------- 加载配置 ----------
    cfg = Config()
    img_size = args.img_size if args.img_size is not None else cfg.img_size
    num_classes = args.num_classes if args.num_classes is not None else cfg.num_classes
    backbone_name = args.backbone if args.backbone is not None else cfg.backbone_name

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        device = torch.device('cpu')

    # ---------- 构建模型 ----------
    model = DetectFCOS(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=False
    ).to(device)

    # ---------- 加载权重 ----------
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"模型加载成功: {args.weights}")

    # ---------- 运行推理 ----------
    infer(
        model=model,
        image_path=args.image,
        device=device,
        img_size=img_size,
        strides=cfg.strides,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        mean=getattr(cfg, 'mean', [0, 0, 0]),
        std=getattr(cfg, 'std', [1, 1, 1]),
        class_names=args.class_names,
        output_image=args.output,
        output_txt=args.output_txt,
        show_size=args.show_size
    )