import os
from typing import Any, Dict, Optional, Union

import torch

class SF:
    def __init__(self, weights: Optional[str] = None,
                 task: Optional[str] = None,
                 cfg: Optional[Dict[str, Any]] = None):
        from .tasks import get_task

        self.weights = weights
        self.cfg: Dict[str, Any] = dict(cfg or {})
        meta: Dict[str, Any] = {}

        if weights is not None:
            if not os.path.exists(weights):
                raise FileNotFoundError(f"权重文件不存在: {weights}")
            ckpt = torch.load(weights, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model_meta' in ckpt:
                meta = ckpt['model_meta'] or {}
            inferred = meta.get('task') or 'detect'
            self.task = task or inferred

            _skip_keys = {'save_dir', 'resume'}
            for k, v in meta.items():
                if k in _skip_keys:
                    continue
                if k not in self.cfg:
                    self.cfg[k] = v
        else:
            self.task = task or 'detect'

        self.cfg['task'] = self.task
        self.cfg.setdefault('mode', self.task)

        self._task = get_task(self.task)

    @classmethod
    def from_config(cls, **kwargs) -> 'SF':
        task = kwargs.pop('task', 'detect')
        return cls(weights=None, task=task, cfg=kwargs)

    def train(self, **kwargs) -> Any:
        merged = {**self.cfg, **kwargs, 'task': self.task}
        return self._task['trainer'](merged).run()

    def val(self, **kwargs) -> Any:
        merged = {**self.cfg, **kwargs, 'task': self.task}
        return self._task['trainer'](merged).validate_only()

    def predict(self, source: Union[str, list], **kwargs) -> Any:
        merged = {**self.cfg, **kwargs, 'task': self.task}
        return self._task['predictor'](merged, weights=self.weights).run(source)

    def export(self, format: str = 'onnx', **kwargs) -> Any:
        merged = {**self.cfg, **kwargs, 'task': self.task}
        return self._task['exporter'](merged, weights=self.weights).run(format, **kwargs)

    def __repr__(self):
        return (f"SF(task={self.task!r}, weights={self.weights!r}, "
                f"backbone={self.cfg.get('backbone_name', 'N/A')!r})")