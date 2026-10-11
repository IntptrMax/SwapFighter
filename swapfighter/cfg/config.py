import os
from typing import Any, Dict

import torch

class Config:

    task = 'detect'

    data_root = './datasets/medical-pills'
    train_split = 'train'
    val_split = 'val'
    num_classes = 1

    train_labels = 'train_labels.txt'
    val_labels = 'val_labels.txt'
    train_img_subdir = 'train2017'
    val_img_subdir = 'val2017'

    hf_dataset = None
    hf_config = None
    hf_train_split = 'train'
    hf_val_split = 'validation'
    hf_cache_dir = None

    class_names_file = None
    hf_class_names_file = None

    backbone_name = 'convnextv2_atto.fcmae_ft_in1k'
    pretrained = True
    backbone_pretrained_source = 'auto'
    backbone_pretrained_path = None
    img_size = 640

    neck_type = 'lightfpn'
    neck_num_layers = 2
    neck_out_channels = [64, 128, 256]
    use_extra_conv = True

    head_channels = 128

    head_type = 'decoupled'
    head_num_convs = 3
    head_norm_layer = 'group'
    head_act_layer = 'silu'
    head_cls_num_convs = None
    head_reg_num_convs = None
    head_angle_num_convs = None
    head_kpt_num_convs = None

    reg_max = 16
    dfl_weight = 1.5

    with_angle = False
    angle_format = 'sin_cos'
    angle_range = 'pi2'
    obb_reg_loss = 'probiou'
    kld_weight = 1.0
    kld_tau = 2.0
    kld_fun = 'log1p'

    proto_dim = 32
    proto_stride = 4
    mask_weight = 1.0

    num_keypoints = 17
    keypoint_weight = 12.0
    keypoint_visibility = 'auto'
    flip_idx = None

    batch_size = 16
    accum_steps = 1
    lr = 5e-4
    backbone_lr_scale = 0.2
    weight_decay = 1e-2
    epochs = 100
    warmup_epochs = 3
    grad_clip_norm = 3.0

    optimizer = 'adamw'
    scheduler = 'cosine'
    scheduler_gamma = 0.95
    scheduler_step_size = 1
    scheduler_milestones = [30, 60]

    patience = 50

    precision = 'fp16'
    use_amp = True
    device = 'cuda'
    seed = 42
    num_workers = 4

    freeze_epochs = 0

    cls_weight = 0.5
    reg_weight = 7.5
    angle_weight = 3.0

    tal_topk = 10
    tal_alpha = 1.0
    tal_beta = 6.0

    mosaic_prob = 0.5
    mosaic_disable_epoch = None
    degrees = 0.0
    translate = 0.1
    scale = 0.5
    shear = 0.0
    perspective = 0.0

    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    color_blur_prob = 0.01
    color_median_blur_prob = 0.01
    color_to_gray_prob = 0.01
    color_clahe_prob = 0.01
    use_color_aug = True

    use_flip = True
    fliplr = 0.5
    flipud = 0.0

    mixup_prob = 0.0
    mixup_alpha = 32.0
    copy_paste_prob = 0.0
    copy_paste_max_paste = 3

    pool_type = 'gem'
    hidden_dim = 0
    dropout = 0.0
    drop_path_rate = 0.0
    multi_scale = False
    grad_checkpointing = False
    loss = 'asl'
    asl_gamma_neg = 4.0
    asl_gamma_pos = 0.0
    asl_clip = 0.05
    eval_threshold = 0.5
    eval_tta = False
    use_rrc = True
    rrc_min_scale = 0.5

    ml_decoder_layers = 1
    ml_decoder_heads = 8
    ml_decoder_group_size = 1

    pool_from = 'auto'
    frozen_backbone = False
    freeze_stage = 'none'
    freeze_last_n = 2

    conf_threshold = 0.7
    nms_threshold = 0.5
    obb_nms_threshold = 0.5
    eval_conf_threshold = None
    eval_diag = False

    use_ema = True
    ema_decay = None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class_names = None

    save_dir = None
    save_interval = 0
    eval_interval = 1
    eval_map_start_epoch = 0
    eval_start_epoch = 0
    scan_threshold = False
    run_name = None
    verbose_metrics = False

    strides = None

    @classmethod
    def from_kwargs(cls, task: str, **kwargs) -> 'Config':
        from ..tasks import get_task
        task_def = get_task(task)
        cfg = cls()

        for k, v in task_def.get('default_cfg', {}).items():
            setattr(cfg, k, v)

        for k, v in kwargs.items():
            if v is None:
                continue
            if k == 'mode':
                continue
            setattr(cfg, k, v)

        cfg.task = task
        cfg.validate_and_setup()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in dir(self):
            if k.startswith('_'):
                continue
            if k == 'mode':
                continue
            v = getattr(self, k)
            if callable(v):
                continue
            out[k] = v

        out['proto_dim'] = getattr(self, 'proto_dim', 32)
        out['proto_stride'] = getattr(self, 'proto_stride', 4)
        out['mask_weight'] = getattr(self, 'mask_weight', 1.0)
        out['num_keypoints'] = getattr(self, 'num_keypoints', 17)
        out['keypoint_weight'] = getattr(self, 'keypoint_weight', 1.0)
        out['keypoint_visibility'] = getattr(self, 'keypoint_visibility',
                                              'auto')
        out['flip_idx'] = getattr(self, 'flip_idx', None)
        return out

    def update(self, **kwargs) -> 'Config':
        for k, v in kwargs.items():
            if v is None:
                continue
            if k == 'mode':
                continue
            setattr(self, k, v)
        return self

    def validate_and_setup(self) -> 'Config':

        if self.task not in ('detect', 'obb', 'segment', 'keypoint',
                             'multilabel', 'linear_probe'):
            raise ValueError(
                f"未知 task: {self.task!r}，必须为 detect / obb / segment / "
                f"keypoint / multilabel / linear_probe")

        if self.task == 'obb':
            self.with_angle = True
        elif self.task in ('detect', 'segment', 'keypoint'):
            self.with_angle = False

        if self.eval_conf_threshold is None:
            if self.task == 'obb':
                self.eval_conf_threshold = 0.01
            elif self.task in ('detect', 'segment', 'keypoint'):
                self.eval_conf_threshold = 0.001
            else:
                self.eval_conf_threshold = 0.5

        if self.task == 'obb':
            if self.angle_range not in ('pi', 'pi2'):
                raise ValueError(
                    f"angle_range 必须为 'pi' 或 'pi2'，"
                    f"当前为 {self.angle_range!r}")
            if self.angle_format not in ('sin_cos', 'direct'):
                raise ValueError(
                    f"angle_format 必须为 'sin_cos' 或 'direct'，"
                    f"当前为 {self.angle_format!r}")
            if self.obb_nms_threshold is None:
                self.obb_nms_threshold = self.nms_threshold
            if self.obb_reg_loss not in ('probiou', 'kld'):
                raise ValueError(
                    f"obb_reg_loss 必须为 'probiou' 或 'kld'，"
                    f"当前为 {self.obb_reg_loss!r}")
            if self.obb_reg_loss == 'kld':
                if self.kld_fun not in ('log1p', 'sqrt', None):
                    raise ValueError(
                        f"kld_fun 必须为 'log1p' / 'sqrt' / None，"
                        f"当前为 {self.kld_fun!r}")
                if self.kld_weight <= 0:
                    raise ValueError(
                        f"kld_weight 必须 > 0，当前为 {self.kld_weight}")
                if self.kld_tau <= 0:
                    raise ValueError(
                        f"kld_tau 必须 > 0，当前为 {self.kld_tau}")

        if self.task == 'segment':
            if self.proto_dim < 4:
                raise ValueError(
                    f'proto_dim 必须 >= 4，当前 {self.proto_dim}')
            if self.proto_stride not in (2, 4, 8):
                raise ValueError(
                    f'proto_stride 必须是 2/4/8，当前 {self.proto_stride}')
            if self.mask_weight < 0:
                raise ValueError(
                    f'mask_weight 必须 >= 0，当前 {self.mask_weight}')

        if self.task == 'keypoint':
            if self.num_keypoints < 1:
                raise ValueError(
                    f'num_keypoints 必须 >= 1，当前 {self.num_keypoints}')
            if self.keypoint_weight < 0:
                raise ValueError(
                    f'keypoint_weight 必须 >= 0，'
                    f'当前 {self.keypoint_weight}')
            if self.keypoint_visibility not in ('auto', 'yes', 'no'):
                raise ValueError(
                    f"keypoint_visibility 必须是 auto/yes/no，"
                    f"当前 {self.keypoint_visibility!r}")
            if self.flip_idx is not None:
                if len(self.flip_idx) != self.num_keypoints:
                    raise ValueError(
                        f'flip_idx 长度 ({len(self.flip_idx)}) 必须等于 '
                        f'num_keypoints ({self.num_keypoints})')

        if self.task in ('detect', 'segment', 'keypoint') and \
           self.degrees > 5.0:
            print(f"[Config] ⚠️ {self.task} 模式下 degrees={self.degrees} "
                  f"会让 GT 外接框膨胀，自动调整为 0.0")
            self.degrees = 0.0

        if self.precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError(
                f"precision 必须为 fp32/fp16/bf16，"
                f"当前为 {self.precision!r}")
        self.use_amp = (self.precision != 'fp32')

        if self.warmup_epochs >= self.epochs:
            print(f"[Config] warmup_epochs ({self.warmup_epochs}) >= "
                  f"epochs ({self.epochs})，自动调整为 epochs 的 10%")
            self.warmup_epochs = max(1, int(self.epochs * 0.1))

        if self.mosaic_disable_epoch is None:
            self.mosaic_disable_epoch = int(self.epochs * 0.9)
        elif self.mosaic_disable_epoch >= self.epochs:
            print(f"[Config] ⚠️ mosaic_disable_epoch "
                  f"({self.mosaic_disable_epoch}) >= epochs "
                  f"({self.epochs})，Mosaic 将始终启用")

        if self.optimizer not in ('sgd', 'adamw'):
            raise ValueError(
                f"optimizer 必须为 sgd/adamw，"
                f"当前为 {self.optimizer!r}")

        valid_schedulers = ('lambda', 'step', 'multistep',
                            'cosine', 'onecycle')
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"scheduler 必须为 {valid_schedulers} 之一，"
                f"当前为 {self.scheduler!r}")

        if self.task in ('detect', 'obb', 'segment', 'keypoint'):
            if self.head_type not in ('decoupled',):
                raise ValueError(
                    f"{self.task} 仅支持 head_type='decoupled'，"
                    f"当前为 {self.head_type!r}")
            if self.head_norm_layer not in ('group', 'batch', 'none'):
                raise ValueError(
                    f"head_norm_layer 必须为 group/batch/none，"
                    f"当前为 {self.head_norm_layer!r}")
            if self.head_act_layer not in ('silu', 'relu', 'gelu'):
                raise ValueError(
                    f"head_act_layer 必须为 silu/relu/gelu，"
                    f"当前为 {self.head_act_layer!r}")
            if self.neck_type not in ('lightfpn', 'bifpn'):
                raise ValueError(
                    f"neck_type 必须为 lightfpn/bifpn，"
                    f"当前为 {self.neck_type!r}")
            if self.reg_max < 0:
                raise ValueError(
                    f"reg_max 必须 >= 0，当前为 {self.reg_max}")

        elif self.task in ('multilabel', 'linear_probe'):
            if self.head_type not in ('linear', 'mlp', 'ml_decoder'):
                raise ValueError(
                    f"multilabel/linear_probe 仅支持 head_type="
                    f"'linear' / 'mlp' / 'ml_decoder'，"
                    f"当前为 {self.head_type!r}")

            if self.head_type != 'ml_decoder' and \
               self.pool_type not in ('gap', 'gem', 'attention'):
                raise ValueError(
                    f"pool_type 必须为 gap/gem/attention，"
                    f"当前为 {self.pool_type!r}")

            if self.head_type == 'ml_decoder' and self.multi_scale:
                raise ValueError(
                    "ML-Decoder 暂不支持 multi_scale，"
                    "请设置 multi_scale=False")

            if self.loss not in ('bce', 'asl'):
                raise ValueError(
                    f"loss 必须为 bce 或 asl，"
                    f"当前为 {self.loss!r}")

            if self.loss == 'asl':
                if self.asl_gamma_neg < 0:
                    raise ValueError(
                        f"asl_gamma_neg 必须 >= 0，"
                        f"当前为 {self.asl_gamma_neg}")
                if self.asl_clip < 0:
                    raise ValueError(
                        f"asl_clip 必须 >= 0，"
                        f"当前为 {self.asl_clip}")

            if self.head_type == 'ml_decoder':
                if self.ml_decoder_layers < 1:
                    raise ValueError(
                        f"ml_decoder_layers 必须 >= 1，"
                        f"当前为 {self.ml_decoder_layers}")
                if self.ml_decoder_heads < 1:
                    raise ValueError(
                        f"ml_decoder_heads 必须 >= 1，"
                        f"当前为 {self.ml_decoder_heads}")
                if self.ml_decoder_group_size < 1:
                    raise ValueError(
                        f"ml_decoder_group_size 必须 >= 1，"
                        f"当前为 {self.ml_decoder_group_size}")

        if self.task == 'linear_probe':
            self.frozen_backbone = True

        if self.pool_from not in ('auto', 'cls_token',
                                   'avg_pool', 'spatial'):
            raise ValueError(
                f"pool_from 必须为 auto/cls_token/avg_pool/spatial，"
                f"当前为 {self.pool_from!r}")

        if self.task in ('multilabel', 'linear_probe') and \
           self.head_type == 'ml_decoder':
            if self.pool_from not in ('auto', 'spatial'):
                print(f"[Config] ⚠️ head_type='ml_decoder' 需要 spatial "
                      f"特征，pool_from 从 {self.pool_from!r} 改为 'spatial'")
                self.pool_from = 'spatial'

        if self.freeze_stage not in ('none', 'head_only',
                                      'neck_head', 'last_n'):
            raise ValueError(
                f"freeze_stage 必须为 none/head_only/neck_head/last_n，"
                f"当前为 {self.freeze_stage!r}")

        if self.freeze_last_n < 1:
            raise ValueError(
                f"freeze_last_n 必须 >= 1，当前为 {self.freeze_last_n}")

        if self.backbone_pretrained_source == 'simmim':
            if not self.backbone_pretrained_path or \
               not os.path.exists(self.backbone_pretrained_path):
                raise FileNotFoundError(
                    f"backbone_pretrained_path 无效: "
                    f"{self.backbone_pretrained_path}")

        if self.save_dir is None:
            self.save_dir = os.path.join('./outputs', self.task)

        return self