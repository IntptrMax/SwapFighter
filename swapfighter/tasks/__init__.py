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

from . import detect
from . import obb
from . import multilabel
from . import linear_probe
from . import segment
from . import keypoint