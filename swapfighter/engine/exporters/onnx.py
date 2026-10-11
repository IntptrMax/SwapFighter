from ..base_predictor import BasePredictor

class ONNXExporter(BasePredictor):

    def build_model(self):
        from ...tasks import get_task
        from ...utils.logger import suppress_stdout
        import torch

        with suppress_stdout():
            model = get_task(self.cfg.task)['model'](self.cfg)

            ckpt = torch.load(self.weights, map_location='cpu',
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            else:
                sd = ckpt
            sd = {k[7:] if k.startswith('module.') else k: v
                  for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)

        return model.eval().to(self.device)

    def run(self, format='onnx', **kwargs):
        if format != 'onnx':
            raise ValueError(f"当前仅支持 onnx，收到: {format}")

        from ._onnx_impl import (
            export_detect_onnx, export_classify_onnx,
            export_seg_onnx, export_keypoint_onnx,
            export_multilabel_onnx,
        )

        model = self.build_model()

        if self.cfg.task in ('detect', 'obb'):
            return export_detect_onnx(model, self.cfg, self.weights, **kwargs)
        elif self.cfg.task == 'segment':
            return export_seg_onnx(model, self.cfg, self.weights, **kwargs)
        elif self.cfg.task == 'keypoint':
            return export_keypoint_onnx(model, self.cfg, self.weights,
                                         **kwargs)
        elif self.cfg.task in ('multilabel', 'linear_probe'):
            return export_multilabel_onnx(model, self.cfg, self.weights,
                                           **kwargs)
        elif self.cfg.task == 'classify':
            return export_classify_onnx(model, self.cfg, self.weights,
                                         **kwargs)
        else:
            raise ValueError(f"不支持的 task: {self.cfg.task}")