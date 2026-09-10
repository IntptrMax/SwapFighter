import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import DetectFCOS
from utils import LOG_SCALE_MIN, LOG_SCALE_MAX


class DetectFCOSWrapper(nn.Module):
    """
    统一的 ONNX 导出包装。

    detect 模式输出:
        probs:   (B, N, C)  已经过 sigmoid 的概率
        bboxes:  (B, N, 4)  归一化 (cx, cy, w, h)

    obb 模式输出:
        probs:   (B, N, C)  已经过 sigmoid 的概率
        bboxes:  (B, N, 4)  归一化 (cx, cy, w, h)
        angles:  (B, N, 2)  sin_cos 模式 -> [sin, cos]
                            direct 模式 -> [angle, 0]（第二维恒为 0 占位）
    """
    def __init__(self, model: DetectFCOS, img_size: int,
                 mode: str = 'detect', angle_format: str = 'sin_cos'):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = model.strides
        self.mode = mode
        self.angle_format = angle_format

    def forward(self, x):
        if self.mode == 'obb':
            cls_logits_list, reg_preds_list, angle_preds_list = self.model(x)
        else:
            cls_logits_list, reg_preds_list = self.model(x)
            angle_preds_list = None

        all_probs = []
        all_bboxes = []
        all_angles = []

        for level_idx, (cls_l, reg_l) in enumerate(zip(cls_logits_list, reg_preds_list)):
            b, c, h, w = cls_l.shape
            stride = self.strides[level_idx]

            # 网格点坐标（绝对坐标）
            y_coords = torch.arange(h, device=x.device).float() * stride + stride / 2
            x_coords = torch.arange(w, device=x.device).float() * stride + stride / 2
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_x_flat = grid_x.reshape(-1)
            grid_y_flat = grid_y.reshape(-1)

            # reg_l: (B,4,H,W) -> (B,4,H*W)
            reg_flat = reg_l.view(b, 4, -1)

            if self.mode == 'obb':
                # ---- OBB 解码：[dx, dy, dw, dh] ----
                dx = reg_flat[:, 0, :]
                dy = reg_flat[:, 1, :]
                dw = reg_flat[:, 2, :]
                dh = reg_flat[:, 3, :]

                cx = grid_x_flat.unsqueeze(0) + dx * stride
                cy = grid_y_flat.unsqueeze(0) + dy * stride
                w_ = torch.exp(dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                h_ = torch.exp(dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
            else:
                # ---- detect 解码：[l, t, r, b]（与 utils.post_process 严格对齐） ----
                l = F.relu(reg_flat[:, 0, :])
                t = F.relu(reg_flat[:, 1, :])
                r = F.relu(reg_flat[:, 2, :])
                b_ = F.relu(reg_flat[:, 3, :])

                x1 = grid_x_flat.unsqueeze(0) - l
                y1 = grid_y_flat.unsqueeze(0) - t
                x2 = grid_x_flat.unsqueeze(0) + r
                y2 = grid_y_flat.unsqueeze(0) + b_

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w_ = x2 - x1
                h_ = y2 - y1

            # 归一化到 [0,1]
            cx = cx / self.img_size
            cy = cy / self.img_size
            w_ = w_ / self.img_size
            h_ = h_ / self.img_size

            cx = torch.clamp(cx, 0, 1)
            cy = torch.clamp(cy, 0, 1)
            w_ = torch.clamp(w_, 0, 1)
            h_ = torch.clamp(h_, 0, 1)

            bboxes_decoded = torch.stack([cx, cy, w_, h_], dim=2)  # (B, H*W, 4)

            # 分类：sigmoid 得到概率
            cls_prob = torch.sigmoid(cls_l)                        # (B, C, H, W)
            prob_flat = cls_prob.view(b, c, -1).permute(0, 2, 1)   # (B, H*W, C)

            all_probs.append(prob_flat)
            all_bboxes.append(bboxes_decoded)

            # OBB 角度（原始输出，不做归一化，客户端自行处理）
            if self.mode == 'obb':
                angle_pred = angle_preds_list[level_idx]  # (B, 1 or 2, H, W)
                if self.angle_format == 'sin_cos':
                    # (B, 2, H*W) -> (B, H*W, 2)
                    angle_flat = angle_pred.view(b, 2, -1).permute(0, 2, 1)
                else:
                    # direct: (B, 1, H*W) -> (B, H*W, 1)
                    angle_flat = angle_pred.view(b, 1, -1).permute(0, 2, 1)
                    # 占位第二维，保持输出 shape 统一为 (B, N, 2)
                    angle_flat = torch.cat(
                        [angle_flat, torch.zeros_like(angle_flat)], dim=2
                    )
                all_angles.append(angle_flat)

        probs_out = torch.cat(all_probs, dim=1)     # (B, N, C)
        bboxes_out = torch.cat(all_bboxes, dim=1)   # (B, N, 4)

        if self.mode == 'obb':
            angles_out = torch.cat(all_angles, dim=1)  # (B, N, 2)
            return probs_out, bboxes_out, angles_out
        return probs_out, bboxes_out


def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter ONNX Exporter')
    parser.add_argument('--weights', required=True, help='模型权重文件')
    parser.add_argument('--output', default='model.onnx', help='输出 ONNX 路径')
    parser.add_argument('--mode', default=None, choices=['detect', 'obb'],
                        help='检测模式（覆盖 config）')
    parser.add_argument('--angle-format', default=None, choices=['sin_cos', 'direct'],
                        help='OBB 角度编码（覆盖 config）')
    parser.add_argument('--backbone', default=None, help='Backbone 名称')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数')
    parser.add_argument('--img-size', type=int, default=None, help='输入尺寸')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--dynamic', action='store_true', help='动态批次')
    parser.add_argument('--simplify', action='store_true', help='简化模型')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')
    return parser.parse_args()


def load_model(weights, backbone, num_classes, mode, angle_format, device):
    model = DetectFCOS(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        with_angle=(mode == 'obb'),
        angle_format=angle_format,
    )
    ckpt = torch.load(weights, map_location=device)
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    # 去除多卡前缀
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            k = k[7:]
        new_sd[k] = v
    model.load_state_dict(new_sd)
    model.to(device).eval()
    print(f"模型加载成功: {weights}")
    print(f"  Backbone: {backbone}, 类别数: {num_classes}, 模式: {mode}")
    return model


def remove_initializer_from_input(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        init_names = {init.name for init in model.graph.initializer}
        new_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
        model.graph.ClearField('input')
        model.graph.input.extend(new_inputs)
        onnx.save(model, onnx_path)
        print(f"  已移除 {len(init_names)} 个初始值设定项")
    except Exception as e:
        print(f"  移除初始值设定项失败: {e}")


def export_onnx(wrapper, dummy_input, output_path, dynamic, simplify, opset, mode):
    wrapper = wrapper.cpu()
    dummy_input = dummy_input.cpu()

    input_names = ['input']
    if mode == 'obb':
        output_names = ['probs', 'bboxes', 'angles']
    else:
        output_names = ['probs', 'bboxes']

    dynamic_axes = {}
    if dynamic:
        dynamic_axes['input'] = {0: 'batch_size'}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}

    print(f"导出 ONNX 到 {output_path}")
    print(f"  Opset: {opset}, 动态批次: {dynamic}")
    print(f"  输入形状: {dummy_input.shape}")
    if mode == 'obb':
        print(f"  输出: probs (概率 0~1), bboxes (cx,cy,w,h 归一化), angles (sin_cos 或 angle+占位)")
    else:
        print(f"  输出: probs (概率 0~1), bboxes (cx,cy,w,h 归一化)")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
            dynamic_axes=dynamic_axes if dynamic else None,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=True,
        )
    print("  导出成功")
    remove_initializer_from_input(output_path)

    if simplify:
        try:
            import onnx, onnxsim
            onnx_model = onnx.load(output_path)
            model_simp, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=dynamic,
                input_shapes={'input': dummy_input.shape}
            )
            if check:
                onnx.save(model_simp, output_path)
                remove_initializer_from_input(output_path)
                print("  简化成功")
            else:
                print("  简化失败")
        except ImportError:
            print("  警告: onnx-simplifier 未安装")
        except Exception as e:
            print(f"  简化出错: {e}")

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX 模型验证通过")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")
    except Exception as e:
        print(f"  验证警告: {e}")


def main():
    args = parse_args()
    cfg = Config()

    device = torch.device('cpu')
    img_size = args.img_size or cfg.img_size
    num_classes = args.num_classes or cfg.num_classes
    backbone = args.backbone or cfg.backbone_name
    mode = args.mode or cfg.mode
    angle_format = args.angle_format or cfg.angle_format

    print("=" * 60)
    print("SWAP-Fighter ONNX Exporter")
    print("=" * 60)
    print(f"模式: {mode}")
    if mode == 'obb':
        print(f"角度编码: {angle_format}")
    print(f"输入尺寸: {img_size}")

    model = load_model(args.weights, backbone, num_classes, mode, angle_format, device)
    wrapper = DetectFCOSWrapper(model, img_size, mode=mode, angle_format=angle_format)
    wrapper.to(device).eval()

    dummy = torch.randn(args.batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        outputs = wrapper(dummy)
        if mode == 'obb':
            test_probs, test_bboxes, test_angles = outputs
            print(f"示例输出形状: probs {test_probs.shape}, bboxes {test_bboxes.shape}, angles {test_angles.shape}")
        else:
            test_probs, test_bboxes = outputs
            print(f"示例输出形状: probs {test_probs.shape}, bboxes {test_bboxes.shape}")
        print(f"总点数: {test_probs.shape[1]}")

    export_onnx(wrapper, dummy, args.output, args.dynamic, args.simplify, args.opset, mode)
    print(f"\n导出完成: {args.output}")


if __name__ == '__main__':
    main()