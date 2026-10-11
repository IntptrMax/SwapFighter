from . import register_task
from ..engine.trainers.multilabel import MultiLabelTrainer
from ..engine.predictors.multilabel import MultiLabelPredictor
from ..engine.exporters.onnx import ONNXExporter
from ..models.classifiers.multi_label import build_classifier
from ..datasets.cls_dataset import MultiLabelClassificationDataset

register_task(
    'linear_probe',
    model=build_classifier,
    dataset_cls=MultiLabelClassificationDataset,
    trainer=MultiLabelTrainer,
    predictor=MultiLabelPredictor,
    exporter=ONNXExporter,
    default_cfg=dict(
        img_size=224,
        batch_size=128,
        accum_steps=1,
        epochs=20,
        lr=1e-3,
        backbone_lr_scale=0.0,
        weight_decay=1e-4,
        warmup_epochs=1,
        patience=5,
        pool_type='gem',
        head_type='mlp',
        head_norm=True,
        hidden_dim=0,
        dropout=0.0,
        pool_from='auto',
        frozen_backbone=True,
        loss='bce',
        eval_threshold=0.5,
        use_flip=True, fliplr=0.5,
        use_color_aug=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        use_rrc=True, rrc_min_scale=0.7,
        use_ema=False, ema_decay=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
)