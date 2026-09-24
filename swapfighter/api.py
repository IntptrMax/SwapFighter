import os
from typing import Any, Dict, Optional, Union

import torch


class SF:
    """
    统一入口。既可从权重加载，也可从配置构建。

    >>> from swapfighter import SF
    >>> model = SF('best.pth')
    >>> model.predict('image.jpg')

    >>> model = SF.from_config(task='detect',
    ...                        backbone_name='convnextv2_atto.fcmae_ft_in1k',
    ...                        num_classes=80, img_size=640)
    >>> model.train(data_root='./data', epochs=100)
    """
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
            inferred = meta.get('task') or meta.get('mode') or 'detect'
            self.task = task or inferred

            #   从 meta 恢复默认值时，跳过这些字段：
            #   save_dir → 否则 resume 会在老 run 目录下再嵌套一层 runN
            #   resume   → 与本次运行的 checkpoint 无关
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

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, **kwargs) -> 'SF':
        """从配置构建未训练的模型。"""
        task = kwargs.pop('task', 'detect')
        return cls(weights=None, task=task, cfg=kwargs)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"SF(task={self.task!r}, weights={self.weights!r}, "
                f"backbone={self.cfg.get('backbone_name', 'N/A')!r})")