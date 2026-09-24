from . import register_task
from ..engine.trainers.detect import DetectTrainer
from ..engine.predictors.detect import DetectPredictor
from ..engine.exporters.onnx import ONNXExporter
from ..models.detectors.detect_fcos import build_detect_model
from ..datasets.yolo_det import YOLODataset


register_task(
    'detect',
    model=build_detect_model,
    dataset_cls=YOLODataset,
    trainer=DetectTrainer,
    predictor=DetectPredictor,
    exporter=ONNXExporter,
    default_cfg=dict(
        mode='detect',
        with_angle=False,
        angle_format=None,
        angle_range=None,
        conf_threshold=0.7,
        nms_threshold=0.5,
        img_size=640,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
)