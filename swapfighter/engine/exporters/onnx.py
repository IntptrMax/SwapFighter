from ..base_predictor import BasePredictor


class ONNXExporter(BasePredictor):

    def build_model(self):
        from ...tasks import get_task
        model = get_task(self.cfg.task)['model'](self.cfg)
        import torch
        ckpt = torch.load(self.weights, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and \
            'model_state_dict' in ckpt else ckpt
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        return model.eval().to(self.device)

    def run(self, format='onnx', **kwargs):
        if format != 'onnx':
            raise ValueError(f"当前仅支持 onnx，收到: {format}")

        from ._onnx_impl import (
            export_detect_onnx, export_classify_onnx,
        )

        model = self.build_model()
        if self.cfg.task in ('detect', 'obb'):
            return export_detect_onnx(model, self.cfg, self.weights, **kwargs)
        elif self.cfg.task == 'classify':
            return export_classify_onnx(model, self.cfg, self.weights, **kwargs)
        else:
            raise ValueError(f"不支持的 task: {self.cfg.task}")