from . import register_task
from ..engine.trainers.segment import SegTrainer
from ..engine.predictors.segment import SegPredictor
from ..engine.exporters.onnx import ONNXExporter
from ..models.detectors.detect_fcos_seg import build_detect_seg_model
from ..datasets.yolo_seg import YOLOSegDataset

register_task(
    'segment',                       
    model=build_detect_seg_model,
    dataset_cls=YOLOSegDataset,
    trainer=SegTrainer,
    predictor=SegPredictor,
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
        proto_dim=32,
        proto_stride=4,
        mask_weight=1.0,
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