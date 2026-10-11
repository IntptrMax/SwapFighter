from . import register_task
from ..engine.trainers.keypoint import KeypointTrainer
from ..engine.predictors.keypoint import KeypointPredictor
from ..engine.exporters.onnx import ONNXExporter
from ..models.detectors.detect_fcos_keypoint import (
    build_detect_keypoint_model,
)
from ..datasets.yolo_keypoint import YOLOKeypointDataset

register_task(
    'keypoint',
    model=build_detect_keypoint_model,
    dataset_cls=YOLOKeypointDataset,
    trainer=KeypointTrainer,
    predictor=KeypointPredictor,
    exporter=ONNXExporter,
    default_cfg=dict(
        with_angle=False,
        angle_format=None,
        angle_range=None,
        conf_threshold=0.7,
        nms_threshold=0.5,
        img_size=640,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        neck_out_channels=[64, 128, 256],
        head_channels=128,
        head_num_convs=2,
        head_kpt_num_convs=3, 
        num_keypoints=17,
        keypoint_weight=12.0,
        keypoint_visibility='auto',
        batch_size=8,
        lr=1e-4,
        epochs=100,
        patience=20,
        tal_topk=10,
        tal_alpha=1.0,
        tal_beta=6.0,
        reg_max=0,
        dfl_weight=0.0,
    ),
)