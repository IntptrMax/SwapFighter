"""
任务注册表。

每个任务的条目包含：
    model          (cfg) -> nn.Module
    dataset_cls    数据集类
    trainer        训练器类
    predictor      推理器类
    exporter       导出器类
    default_cfg    任务默认配置
"""
from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {}


def register_task(name: str, **components):
    if name in TASKS:
        raise ValueError(f"任务 {name} 已注册")
    TASKS[name] = dict(components)


def get_task(name: str) -> Dict[str, Any]:
    if name not in TASKS:
        raise KeyError(f"未知任务 {name!r}。已注册: {list(TASKS.keys())}")
    return TASKS[name]


def list_tasks():
    return list(TASKS.keys())


# 触发注册
from . import detect        # noqa: E402, F401
from . import obb           # noqa: E402, F401
from . import multilabel    # noqa: E402, F401