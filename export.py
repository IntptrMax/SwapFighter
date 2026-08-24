import os
import argparse
import torch
import torch.nn as nn

from config import Config
from model import DetectFCOS


class DetectFCOSWrapper(nn.Module):
    """将多层输出拼接，并解码为 (cx,cy,w,h) 归一化，probs 输出概率"""
    def __init__(self, model: DetectFCOS, img_size: int):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = model.strides

    def forward(self, x):
        cls_logits_list, reg_preds_list = self.model(x)
        all_probs = []
        all_bboxes = []

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
            l = reg_flat[:, 0, :]
            t = reg_flat[:, 1, :]
            r = reg_flat[:, 2, :]
            b_ = reg_flat[:, 3, :]

            # 解码为 x1,y1,x2,y2
            x1 = grid_x_flat.unsqueeze(0) - l
            y1 = grid_y_flat.unsqueeze(0) - t
            x2 = grid_x_flat.unsqueeze(0) + r
            y2 = grid_y_flat.unsqueeze(0) + b_

            # 计算 cx,cy,w,h
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w_ = x2 - x1
            h_ = y2 - y1

            # 归一化到 [0,1]
            cx = cx / self.img_size
            cy = cy / self.img_size
            w_ = w_ / self.img_size
            h_ = h_ / self.img_size

            # 裁剪确保在 [0,1] 内
            cx = torch.clamp(cx, 0, 1)
            cy = torch.clamp(cy, 0, 1)
            w_ = torch.clamp(w_, 0, 1)
            h_ = torch.clamp(h_, 0, 1)

            bboxes_decoded = torch.stack([cx, cy, w_, h_], dim=2)  # (B, H*W, 4)

            # 分类：先应用 sigmoid 得到概率，再展平
            cls_prob = torch.sigmoid(cls_l)                     # (B, C, H, W)
            prob_flat = cls_prob.view(b, c, -1).permute(0, 2, 1) # (B, H*W, C)

            all_probs.append(prob_flat)
            all_bboxes.append(bboxes_decoded)

        probs_out = torch.cat(all_probs, dim=1)   # (B, total_points, C)
        bboxes_out = torch.cat(all_bboxes, dim=1) # (B, total_points, 4)
        return probs_out, bboxes_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='模型权重文件')
    parser.add_argument('--output', default='model.onnx', help='输出 ONNX 路径')
    parser.add_argument('--backbone', default=None, help='Backbone 名称')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数')
    parser.add_argument('--img-size', type=int, default=None, help='输入尺寸')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--dynamic', action='store_true', help='动态批次')
    parser.add_argument('--simplify', action='store_true', help='简化模型')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')
    return parser.parse_args()


def load_model(weights, backbone, num_classes, device):
    cfg = Config()
    backbone = backbone or cfg.backbone_name
    num_classes = num_classes or cfg.num_classes
    model = DetectFCOS(num_classes=num_classes, backbone_name=backbone, pretrained=False)
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
    print(f"  Backbone: {backbone}, 类别数: {num_classes}")
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


def export_onnx(model, dummy_input, output_path, dynamic, simplify, opset):
    model = model.cpu()
    dummy_input = dummy_input.cpu()
    input_names = ['input']
    output_names = ['probs', 'bboxes']
    dynamic_axes = {}
    if dynamic:
        dynamic_axes['input'] = {0: 'batch_size'}
        dynamic_axes['probs'] = {0: 'batch_size'}
        dynamic_axes['bboxes'] = {0: 'batch_size'}

    print(f"导出 ONNX 到 {output_path}")
    print(f"  Opset: {opset}, 动态批次: {dynamic}")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出: probs (概率值 0~1), bboxes (cx,cy,w,h 归一化)")

    with torch.no_grad():
        torch.onnx.export(
            model,
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
            model_simp, check = onnxsim.simplify(onnx_model,
                                                  dynamic_input_shape=dynamic,
                                                  input_shapes={'input': dummy_input.shape})
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
        size_mb = os.path.getsize(output_path) / (1024*1024)
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

    print("="*60)
    print("SWAP-Fighter ONNX Exporter")
    print("="*60)
    print(f"输入尺寸: {img_size}")

    model = load_model(args.weights, backbone, num_classes, device)
    wrapper = DetectFCOSWrapper(model, img_size).to(device).eval()

    dummy = torch.randn(args.batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        test_probs, test_bboxes = wrapper(dummy)
        print(f"示例输出形状: probs (概率) {test_probs.shape}, bboxes {test_bboxes.shape}")
        print(f"总点数: {test_probs.shape[1]}")

    export_onnx(wrapper, dummy, args.output, args.dynamic, args.simplify, args.opset)
    print(f"\n导出完成: {args.output}")


if __name__ == '__main__':
    main()