from . import register_task
from ..engine.trainers.multilabel import MultiLabelTrainer
from ..engine.predictors.multilabel import MultiLabelPredictor
from ..engine.exporters.onnx import ONNXExporter
from ..models.classifiers.multi_label import build_multilabel_classifier
from ..datasets.cls_dataset import MultiLabelClassificationDataset

register_task(
    'multilabel',
    model=build_multilabel_classifier,
    dataset_cls=MultiLabelClassificationDataset,
    trainer=MultiLabelTrainer,
    predictor=MultiLabelPredictor,
    exporter=ONNXExporter,
    default_cfg=dict(
        img_size=224,
        pool_type='gem',
        head_type='mlp',
        multi_scale=False,
        eval_threshold=0.5,
        loss='asl',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        epochs=40,
        batch_size=64,
        lr=1e-3,
        patience=15,
    ),
)