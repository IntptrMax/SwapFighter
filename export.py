

import os
import sys
import io
import json
import argparse
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import DetectFCOS
from utils import LOG_SCALE_MIN, LOG_SCALE_MAX


# ============================================================
# 抑制已知警告（不影响功能）
# ============================================================
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
warnings.filterwarnings("ignore", message=".*input_shapes.*deprecated.*")


# ============================================================
# 日志打印辅助
# ============================================================
@contextmanager
def suppress_stdout():
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
    prefix = " " * indent
    label = f"{key} "
    pad_len = max(2, total_width - len(label))
    dots = " " + "." * pad_len + " "
    print(f"{prefix}{label}{dots} {value}")


def _bool_en(b):
    return "yes" if b else "no"


# ============================================================
# ONNX 导出包装
# ============================================================
class DetectFCOSWrapper(nn.Module):
    """
    detect 模式输出:
        probs:   (B, N, C)  sigmoid 后概率
        bboxes:  (B, N, 4)  归一化 (cx, cy, w, h)，不做 clamp（下游处理）

    obb 模式输出:
        probs:   (B, N, C)
        bboxes:  (B, N, 4)  归一化 (cx, cy, w, h)
        angles:  (B, N, 2)  sin_cos → [sin, cos]
                            direct  → [angle, 0]（占位）
    """

    def __init__(self, model, img_size, mode='detect', angle_format='sin_cos',
                 angle_range='pi', reg_max=0):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.strides = model.strides
        self.mode = mode
        self.angle_format = angle_format
        self.angle_range = angle_range
        self.reg_max = reg_max
        self.use_dfl = (reg_max > 0)

    def _decode_dfl(self, reg_l, b, h, w, stride):
        """
        DFL 解码：
        - detect: 图像坐标系 → l, t, r, b
        - obb:    局部坐标系 → l, t, r, b
        """
        # reg_l: (B, 4*(R+1), H, W)
        reg_4d = reg_l.view(b, 4, self.reg_max + 1, h * w)  # (B, 4, R+1, N)
        reg_4d = reg_4d.permute(0, 3, 1, 2)                 # (B, N, 4, R+1)
        probs = reg_4d.softmax(dim=-1)
        idx = torch.arange(self.reg_max + 1, device=reg_l.device, dtype=torch.float32)
        distances = (probs * idx).sum(dim=-1)               # (B, N, 4)
        l_dist, t_dist, r_dist, b_dist = distances.unbind(-1)
        return l_dist, t_dist, r_dist, b_dist

    def forward(self, x):
        if self.mode == 'obb':
            cls_logits_list, reg_preds_list, angle_preds_list = self.model(x)
        else:
            cls_logits_list, reg_preds_list = self.model(x)
            angle_preds_list = None

        all_probs, all_bboxes, all_angles = [], [], []

        for level_idx, (cls_l, reg_l) in enumerate(zip(cls_logits_list, reg_preds_list)):
            b, c, h, w = cls_l.shape
            stride = self.strides[level_idx]

            # 网格中心点坐标（像素，非归一化）
            y_coords = torch.arange(h, device=x.device, dtype=torch.float32) * stride + stride / 2
            x_coords = torch.arange(w, device=x.device, dtype=torch.float32) * stride + stride / 2
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_x_flat = grid_x.reshape(-1)
            grid_y_flat = grid_y.reshape(-1)

            # ==================== OBB ====================
            if self.mode == 'obb':
                # ---------- 角度 ----------
                angle_pred = angle_preds_list[level_idx]  # (B, 1 or 2, H, W)
                if self.angle_format == 'sin_cos':
                    angle_flat = angle_pred.view(b, 2, -1).permute(0, 2, 1)  # (B, N, 2)
                    sin_val = angle_flat[..., 0]
                    cos_val = angle_flat[..., 1]
                    angle = torch.atan2(sin_val, cos_val)
                else:
                    angle_flat = angle_pred.view(b, 1, -1).permute(0, 2, 1)  # (B, N, 1)
                    angle = angle_flat[..., 0]

                # ---------- 回归 ----------
                if self.use_dfl:
                    l_dist, t_dist, r_dist, b_dist = self._decode_dfl(reg_l, b, h, w, stride)
                    cx_local = (r_dist - l_dist) * 0.5 * stride
                    cy_local = (b_dist - t_dist) * 0.5 * stride
                    w_pix = (l_dist + r_dist) * stride
                    h_pix = (t_dist + b_dist) * stride
                else:
                    reg_flat = reg_l.view(b, 4, -1)  # (B, 4, N)
                    dx = reg_flat[:, 0, :]
                    dy = reg_flat[:, 1, :]
                    dw = reg_flat[:, 2, :]
                    dh = reg_flat[:, 3, :]
                    cx_local = dx * stride
                    cy_local = dy * stride
                    w_pix = torch.exp(dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                    h_pix = torch.exp(dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride

                # 局部 → 全局（用预测角度旋转）
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                cx = grid_x_flat.unsqueeze(0) + cx_local * cos_a - cy_local * sin_a
                cy = grid_y_flat.unsqueeze(0) + cx_local * sin_a + cy_local * cos_a

                # 归一化（不 clamp，下游处理）
                cx_n = cx / self.img_size
                cy_n = cy / self.img_size
                w_n = w_pix / self.img_size
                h_n = h_pix / self.img_size
                bboxes_decoded = torch.stack([cx_n, cy_n, w_n, h_n], dim=2)

                # 角度输出：sin_cos → (B, N, 2)；direct → (B, N, 2) with [angle, 0]
                if self.angle_format == 'sin_cos':
                    all_angles.append(angle_flat)
                else:
                    angle_flat_2d = torch.cat(
                        [angle.unsqueeze(-1), torch.zeros_like(angle).unsqueeze(-1)],
                        dim=-1
                    )
                    all_angles.append(angle_flat_2d)

            # ==================== detect ====================
            else:
                if self.use_dfl:
                    l_dist, t_dist, r_dist, b_dist = self._decode_dfl(reg_l, b, h, w, stride)
                    x1 = grid_x_flat.unsqueeze(0) - l_dist * stride
                    y1 = grid_y_flat.unsqueeze(0) - t_dist * stride
                    x2 = grid_x_flat.unsqueeze(0) + r_dist * stride
                    y2 = grid_y_flat.unsqueeze(0) + b_dist * stride
                else:
                    reg_flat = reg_l.view(b, 4, -1)
                    dx = reg_flat[:, 0, :]
                    dy = reg_flat[:, 1, :]
                    dw = reg_flat[:, 2, :]
                    dh = reg_flat[:, 3, :]
                    cx_pix = grid_x_flat.unsqueeze(0) + dx * stride
                    cy_pix = grid_y_flat.unsqueeze(0) + dy * stride
                    w_ = torch.exp(dw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                    h_ = torch.exp(dh.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)) * stride
                    x1 = cx_pix - w_ / 2
                    y1 = cy_pix - h_ / 2
                    x2 = cx_pix + w_ / 2
                    y2 = cy_pix + h_ / 2

                cx_n = (x1 + x2) / 2 / self.img_size
                cy_n = (y1 + y2) / 2 / self.img_size
                w_n = (x2 - x1) / self.img_size
                h_n = (y2 - y1) / self.img_size
                bboxes_decoded = torch.stack([cx_n, cy_n, w_n, h_n], dim=2)

            # ==================== 分类 ====================
            cls_prob = torch.sigmoid(cls_l)                          # (B, C, H, W)
            prob_flat = cls_prob.view(b, c, -1).permute(0, 2, 1)     # (B, N, C)

            all_probs.append(prob_flat)
            all_bboxes.append(bboxes_decoded)

        probs_out = torch.cat(all_probs, dim=1)
        bboxes_out = torch.cat(all_bboxes, dim=1)

        if self.mode == 'obb':
            angles_out = torch.cat(all_angles, dim=1)
            return probs_out, bboxes_out, angles_out
        return probs_out, bboxes_out


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SWAP-Fighter ONNX Exporter')
    parser.add_argument('--weights', required=True, help='模型权重文件 (.pth)')
    parser.add_argument('--output', default='model.onnx', help='输出 ONNX 路径')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--dynamic', action='store_true', help='动态批次')
    parser.add_argument('--simplify', action='store_true', help='简化模型')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')

    # ---------- 命令行覆盖项（优先级高于 .pth 中的 meta） ----------
    parser.add_argument('--mode', default=None, choices=['detect', 'obb'])
    parser.add_argument('--angle-format', default=None, choices=['sin_cos', 'direct'])
    parser.add_argument('--angle-range', default=None, choices=['pi', 'pi2'])
    parser.add_argument('--backbone', default=None, help='Backbone 名称')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数')
    parser.add_argument('--img-size', type=int, default=None, help='输入尺寸')
    parser.add_argument('--reg-max', type=int, default=None, help='DFL bin 数')
    parser.add_argument('--neck-out-channels', type=int, default=None)
    parser.add_argument('--head-num-convs', type=int, default=None)
    parser.add_argument('--no-extra-conv', dest='use_extra_conv',
                        action='store_false', default=None)
    parser.add_argument('--neck-type', default=None, choices=['lightfpn', 'bifpn'])
    parser.add_argument('--neck-num-layers', type=int, default=None)
    parser.add_argument('--head-type', default=None, choices=['decoupled'])
    parser.add_argument('--head-norm-layer', default=None,
                        choices=['group', 'batch', 'none'])
    parser.add_argument('--head-act-layer', default=None,
                        choices=['silu', 'relu', 'gelu'])
    parser.add_argument('--head-cls-num-convs', type=int, default=None)
    parser.add_argument('--head-reg-num-convs', type=int, default=None)
    parser.add_argument('--head-angle-num-convs', type=int, default=None)

    return parser.parse_args()


# ============================================================
# 加载权重 + 从 meta 提取模型结构（与 infer.py 一致）
# ============================================================
def load_model_from_ckpt(weights, device, cli_overrides=None):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    checkpoint = torch.load(weights, map_location=device, weights_only=False)

    # ---------- 解析 .pth ----------
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

    # ---------- 从 meta 提取模型结构参数 ----------
    cfg = Config()

    backbone_name = meta.get('backbone_name', cfg.backbone_name)
    num_classes = meta.get('num_classes', cfg.num_classes)
    img_size = meta.get('img_size', cfg.img_size)
    reg_max = meta.get('reg_max', getattr(cfg, 'reg_max', 0))
    neck_out_channels = meta.get('neck_out_channels',
                                 getattr(cfg, 'neck_out_channels', 128))
    head_num_convs = meta.get('head_num_convs',
                              getattr(cfg, 'head_num_convs', 2))
    use_extra_conv = meta.get('use_extra_conv',
                              getattr(cfg, 'use_extra_conv', True))
    neck_type = meta.get('neck_type', getattr(cfg, 'neck_type', 'lightfpn'))
    neck_num_layers = meta.get('neck_num_layers',
                               getattr(cfg, 'neck_num_layers', 2))
    with_angle = meta.get('with_angle', False)
    angle_format = meta.get('angle_format', 'sin_cos')
    angle_range = meta.get('angle_range', 'pi')
    mode = meta.get('mode', 'detect')

    head_type = meta.get('head_type', getattr(cfg, 'head_type', 'decoupled'))
    head_norm_layer = meta.get('head_norm_layer',
                               getattr(cfg, 'head_norm_layer', 'group'))
    head_act_layer = meta.get('head_act_layer',
                              getattr(cfg, 'head_act_layer', 'silu'))
    head_cls_num_convs = meta.get('head_cls_num_convs',
                                  getattr(cfg, 'head_cls_num_convs', None))
    head_reg_num_convs = meta.get('head_reg_num_convs',
                                  getattr(cfg, 'head_reg_num_convs', None))
    head_angle_num_convs = meta.get('head_angle_num_convs',
                                    getattr(cfg, 'head_angle_num_convs', None))

    # ---------- 构建模型（折叠内部打印） ----------
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
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_sd[k] = v

    try:
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        real_missing = [k for k in missing if 'num_batches_tracked' not in k]
        real_unexpected = [k for k in unexpected
                           if 'num_batches_tracked' not in k]
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
        _kv("neck_type", neck_type)
        _kv("neck_out_channels", neck_out_channels)
        _kv("reg_max", reg_max)
        print(f"\n  原始报错:\n    {e}\n")
        print(LINE_EQ)
        raise

    model.eval()

    # ---------- 组装最终 meta ----------
    final_meta = {
        'mode': mode,
        'backbone_name': backbone_name,
        'num_classes': num_classes,
        'img_size': img_size,
        'reg_max': reg_max,
        'neck_out_channels': neck_out_channels,
        'head_num_convs': head_num_convs,
        'head_norm_layer': head_norm_layer,
        'head_act_layer': head_act_layer,
        'head_cls_num_convs': head_cls_num_convs,
        'head_reg_num_convs': head_reg_num_convs,
        'head_angle_num_convs': head_angle_num_convs,
        'head_type': head_type,
        'use_extra_conv': use_extra_conv,
        'neck_type': neck_type,
        'neck_num_layers': neck_num_layers,
        'with_angle': with_angle,
        'angle_format': angle_format,
        'angle_range': angle_range,
        'mean': meta.get('mean', list(cfg.mean)),
        'std': meta.get('std', list(cfg.std)),
        'strides': meta.get('strides', list(model.strides)),
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
# 打印导出配置
# ============================================================
def _print_export_config(weights, output, batch_size, dynamic, simplify, opset,
                         meta, load_info):
    print()
    print(LINE_EQ)
    print("  SWAP-Fighter ONNX Exporter")
    print(LINE_EQ)

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

    _title("Model")
    mode_str = ("OBB (rotated bbox)" if meta['mode'] == 'obb'
                else "Detect (axis-aligned bbox)")
    _kv("Mode", mode_str)
    _kv("Backbone", meta['backbone_name'])
    _kv("Image Size", f"{meta['img_size']} x {meta['img_size']}")
    _kv("Strides", str(meta['strides']))
    _kv("Neck", f"{meta['neck_type']} (out={meta['neck_out_channels']}, "
                f"extra_conv={_bool_en(meta['use_extra_conv'])})")
    _kv("Head", f"{meta['head_type']} (convs={meta['head_num_convs']}, "
                f"norm={meta['head_norm_layer']}, act={meta['head_act_layer']})")
    if meta['mode'] == 'obb':
        _kv("Task", f"cls={meta['num_classes']}, reg=4, angle=2 "
                    f"({meta['angle_format']}, {meta['angle_range']})")
    else:
        _kv("Task", f"cls={meta['num_classes']}, reg=4")
    _kv("DFL", "disabled" if meta['reg_max'] == 0
              else f"reg_max={meta['reg_max']} (bin={meta['reg_max'] + 1})")

    _title("Export")
    _kv("Output", output)
    _kv("Batch Size", batch_size)
    _kv("Dynamic Batch", _bool_en(dynamic))
    _kv("Opset", opset)
    _kv("Simplify", _bool_en(simplify))

    print()
    print(LINE_EQ)


# ============================================================
# ONNX 工具函数
# ============================================================
def remove_initializer_from_input(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        init_names = {init.name for init in model.graph.initializer}
        new_inputs = [inp for inp in model.graph.input if inp.name not in init_names]
        model.graph.ClearField('input')
        model.graph.input.extend(new_inputs)
        onnx.save(model, onnx_path)
        return True
    except Exception as e:
        print(f"  移除初始值设定项失败: {e}")
        return False


def inject_metadata_into_onnx(onnx_path, meta):
    """把 meta 写入 ONNX metadata_props"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        del model.metadata_props[:]

        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict)):
                v_str = json.dumps(list(v) if isinstance(v, (list, tuple)) else v)
            elif isinstance(v, bool):
                v_str = 'true' if v else 'false'
            else:
                v_str = str(v)
            prop = model.metadata_props.add()
            prop.key = k
            prop.value = v_str

        onnx.save(model, onnx_path)
        print(f"  已写入 {len(model.metadata_props)} 个 metadata 字段")
        return True
    except Exception as e:
        print(f"  写入 ONNX metadata 失败: {e}")
        return False


def _simplify_onnx_compat(onnx_model, dynamic, input_shape):
    """
    兼容新旧版本的 onnx-simplifier API
    返回: (simplified_model, success_flag)
    """
    import onnxsim
    import inspect

    # 检查 simplify 的签名
    sig = inspect.signature(onnxsim.simplify)
    params = set(sig.parameters.keys())

    common_kwargs = {
        'dynamic_input_shape': dynamic,
    }

    if 'overwrite_input_shapes' in params:
        # 新 API
        if dynamic:
            common_kwargs['overwrite_input_shapes'] = {'input': tuple(input_shape)}
        common_kwargs['test_input_shapes'] = {'input': tuple(input_shape)}
    else:
        # 旧 API
        common_kwargs['input_shapes'] = {'input': input_shape}

    try:
        return onnxsim.simplify(onnx_model, **common_kwargs)
    except TypeError:
        # 回退到纯旧 API
        return onnxsim.simplify(
            onnx_model,
            dynamic_input_shape=dynamic,
            input_shapes={'input': input_shape}
        )


# ============================================================
# 执行导出
# ============================================================
def export_onnx(wrapper, dummy_input, output_path, dynamic, simplify, opset,
                mode, meta):
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

    print(f"\n  导出 ONNX 到 {output_path}")
    print(f"    Opset: {opset}, 动态批次: {dynamic}")
    print(f"    输入形状: {tuple(dummy_input.shape)}")
    if mode == 'obb':
        print(f"    输出: probs(概率) / bboxes(cx,cy,w,h归一化) / angles(sin_cos)")
    else:
        print(f"    输出: probs(概率) / bboxes(cx,cy,w,h归一化)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    print("    导出成功")

    remove_initializer_from_input(output_path)
    inject_metadata_into_onnx(output_path, meta)

    # ---------- 简化 ----------
    if simplify:
        try:
            import onnx
            import onnxsim  # noqa: F401
            onnx_model = onnx.load(output_path)
            meta_props = [(p.key, p.value) for p in onnx_model.metadata_props]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_simp, check = _simplify_onnx_compat(
                    onnx_model, dynamic, tuple(dummy_input.shape)
                )

            if check:
                del model_simp.metadata_props[:]
                for k, v in meta_props:
                    prop = model_simp.metadata_props.add()
                    prop.key = k
                    prop.value = v
                onnx.save(model_simp, output_path)
                remove_initializer_from_input(output_path)
                print("    简化成功")
            else:
                print("    简化失败")
        except ImportError:
            print("    警告: onnx-simplifier 未安装，跳过简化")
        except Exception as e:
            print(f"    简化出错: {e}")

    # ---------- 验证 ----------
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    ONNX 模型验证通过，文件大小: {size_mb:.2f} MB")

        if len(onnx_model.metadata_props) > 0:
            print(f"    嵌入 metadata: {len(onnx_model.metadata_props)} 个字段")
    except Exception as e:
        print(f"    验证警告: {e}")


# ============================================================
# 主入口
# ============================================================
def main():
    args = parse_args()
    device = torch.device('cpu')

    # ---------- 收集命令行覆盖项 ----------
    cli_overrides = {
        'mode': args.mode,
        'backbone_name': args.backbone,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'reg_max': args.reg_max,
        'neck_out_channels': args.neck_out_channels,
        'head_num_convs': args.head_num_convs,
        'use_extra_conv': args.use_extra_conv,
        'neck_type': args.neck_type,
        'neck_num_layers': args.neck_num_layers,
        'head_type': args.head_type,
        'head_norm_layer': args.head_norm_layer,
        'head_act_layer': args.head_act_layer,
        'head_cls_num_convs': args.head_cls_num_convs,
        'head_reg_num_convs': args.head_reg_num_convs,
        'head_angle_num_convs': args.head_angle_num_convs,
        'angle_format': args.angle_format,
        'angle_range': args.angle_range,
    }

    # ---------- 加载模型 ----------
    model, meta, load_info = load_model_from_ckpt(
        weights=args.weights,
        device=device,
        cli_overrides=cli_overrides,
    )

    # ---------- 打印配置 ----------
    _print_export_config(
        weights=args.weights,
        output=args.output,
        batch_size=args.batch_size,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        meta=meta,
        load_info=load_info,
    )

    # ---------- 构建 wrapper ----------
    wrapper = DetectFCOSWrapper(
        model=model,
        img_size=meta['img_size'],
        mode=meta['mode'],
        angle_format=meta['angle_format'],
        angle_range=meta['angle_range'],
        reg_max=meta['reg_max'],
    )
    wrapper.to(device).eval()

    dummy = torch.randn(args.batch_size, 3, meta['img_size'], meta['img_size'],
                        device=device)

    # ---------- 冒烟测试 ----------
    print("\n  [Smoke Test]")
    with torch.no_grad():
        outputs = wrapper(dummy)
        if meta['mode'] == 'obb':
            test_probs, test_bboxes, test_angles = outputs
            print(f"    probs  shape: {tuple(test_probs.shape)}")
            print(f"    bboxes shape: {tuple(test_bboxes.shape)}")
            print(f"    angles shape: {tuple(test_angles.shape)}")
        else:
            test_probs, test_bboxes = outputs
            print(f"    probs  shape: {tuple(test_probs.shape)}")
            print(f"    bboxes shape: {tuple(test_bboxes.shape)}")
        print(f"    总点数 (N): {test_probs.shape[1]}")
        print(f"    probs 范围: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
        print(f"    bboxes 范围: [{test_bboxes.min():.4f}, {test_bboxes.max():.4f}]")

    # ---------- 导出 ----------
    export_onnx(
        wrapper=wrapper,
        dummy_input=dummy,
        output_path=args.output,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        mode=meta['mode'],
        meta=meta,
    )

    print(f"\n  导出完成: {args.output}")
    print(f"  提示: 可直接使用 onnx_demo.py 推理（会自动读取 ONNX 内嵌的 meta）\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        # ★ 强制退出，绕过 onnxsim/nanobind 的引用计数泄漏检查
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)