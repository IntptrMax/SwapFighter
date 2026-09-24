import os
from typing import Any, Dict

import torch


class Config:
    # ---------- 基础 ----------
    task = 'detect'              # detect / obb / mutilabel
    mode = 'detect'

    # ---------- 数据 ----------
    data_root = './datasets/medical-pills'
    train_split = 'train'
    val_split = 'val'
    num_classes = 1

    # ---------- 分类数据集专用 ----------
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

    # ---------- 模型 ----------
    backbone_name = 'convnextv2_atto.fcmae_ft_in1k'
    pretrained = True
    backbone_pretrained_source = 'auto'
    backbone_pretrained_path = None
    img_size = 640

    neck_type = 'lightfpn'
    neck_num_layers = 3
    neck_out_channels = 128
    use_extra_conv = True

    head_type = 'decoupled'
    head_num_convs = 3
    head_norm_layer = 'group'
    head_act_layer = 'silu'
    head_cls_num_convs = None
    head_reg_num_convs = None
    head_angle_num_convs = None

    # ---------- DFL ----------
    reg_max = 0
    dfl_weight = 0.0

    # ---------- OBB ----------
    with_angle = False
    angle_format = 'sin_cos'
    angle_range = 'pi2'
    angle_loss_type = 'l1'
    obb_reg_loss = 'probiou'
    kld_weight = 1.0
    kld_tau = 2.0
    kld_fun = 'log1p'

    # ---------- 训练 ----------
    batch_size = 16
    lr = 5e-4
    weight_decay = 1e-2
    epochs = 100
    warmup_epochs = 3
    optimizer = 'adamw'
    scheduler = 'lambda'
    scheduler_gamma = 0.95
    scheduler_step_size = 1
    scheduler_milestones = [30, 60]
    grad_clip_norm = 3.0
    backbone_lr_scale = 0.1
    patience = 50
    precision = 'fp16'
    use_amp = True
    freeze_epochs = 0
    seed = 42
    num_workers = 4
    device = 'cuda'

    # ---------- 损失权重 ----------
    cls_weight = 0.5
    reg_weight = 7.5
    angle_weight = 1.0

    # ---------- TAL ----------
    tal_topk = 13
    tal_alpha = 1.0
    tal_beta = 6.0

    # ---------- 数据增强 ----------
    mosaic_prob = 0.8
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

    # ---------- 分类专用 ----------
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
    accum_steps = 1
    ema = True

    # ---------- SWA ----------
    swa = False
    swa_start_ratio = 0.5
    swa_lr = 1e-4

    # ---------- 后处理 ----------
    conf_threshold = 0.7
    nms_threshold = 0.5
    obb_nms_threshold = 0.5
    eval_conf_threshold = None
    eval_diag = False

    # ---------- EMA ----------
    use_ema = True
    ema_decay = 0.99

    # ---------- 归一化 ----------
    # mean = [0, 0, 0]
    # std = [1, 1, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ---------- 类别名称 ----------
    class_names = None
    
    # ---------- 输出 ----------
    save_dir = None
    save_dir_explicit = False          # ★ 用户是否显式指定了 save_dir
    save_interval = 0
    eval_interval = 1
    eval_map_start_epoch = 0
    eval_start_epoch = 0
    scan_threshold = False
    run_name = None

    # ---------- 自动计算 ----------
    strides = None

    # ============================================================
    @classmethod
    def from_kwargs(cls, task: str, **kwargs) -> 'Config':
        """从 task 默认值 + 用户 kwargs 构建。"""
        from ..tasks import get_task
        task_def = get_task(task)
        cfg = cls()

        # task 默认
        for k, v in task_def.get('default_cfg', {}).items():
            setattr(cfg, k, v)

        # 在应用用户 kwargs 之前检测 save_dir 是否被显式指定
        if kwargs.get('save_dir') is not None:
            cfg.save_dir_explicit = True

        # 用户 kwargs
        for k, v in kwargs.items():
            if v is None:
                continue
            setattr(cfg, k, v)

        cfg.task = task
        cfg.validate_and_setup()
        return cfg


    def to_dict(self) -> Dict[str, Any]:
        """导出所有配置项（含类属性默认值）。"""
        out: Dict[str, Any] = {}
        for k in dir(self):
            if k.startswith('_'):
                continue
            v = getattr(self, k)
            if callable(v):          # 跳过 from_kwargs / to_dict / validate_and_setup 等方法
                continue
            out[k] = v
        return out

    def update(self, **kwargs) -> 'Config':
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        return self

    # ============================================================
    def validate_and_setup(self) -> 'Config':
        # ---------- task / mode 对齐 ----------
        if self.task in ('detect', 'obb'):
            self.mode = self.task
        elif self.task == 'classify':
            self.mode = 'classify'

        if self.mode == 'obb':
            self.with_angle = True
        elif self.mode == 'detect':
            self.with_angle = False

        # ---------- eval_conf_threshold ----------
        if self.eval_conf_threshold is None:
            self.eval_conf_threshold = 0.01 if self.mode == 'obb' else 0.001

        # ---------- OBB ----------
        if self.mode == 'obb':
            if self.angle_range not in ('pi', 'pi2'):
                raise ValueError(f"angle_range 必须为 'pi' 或 'pi2'")
            if self.angle_format not in ('sin_cos', 'direct'):
                raise ValueError(f"angle_format 必须为 'sin_cos' 或 'direct'")
            if self.obb_nms_threshold is None:
                self.obb_nms_threshold = self.nms_threshold

        # ---------- detect 下禁大角度旋转 ----------
        if self.mode == 'detect' and self.degrees > 5.0:
            self.degrees = 0.0

        # ---------- 精度 ----------
        if self.precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError(f"precision 必须为 fp32/fp16/bf16")
        self.use_amp = (self.precision != 'fp32')

        # ---------- warmup / mosaic ----------
        if self.warmup_epochs >= self.epochs:
            self.warmup_epochs = max(1, int(self.epochs * 0.1))
        if self.mosaic_disable_epoch is None:
            self.mosaic_disable_epoch = int(self.epochs * 0.9)

        # ---------- 优化器 / 调度器 ----------
        if self.optimizer not in ('sgd', 'adamw'):
            raise ValueError(f"optimizer 必须为 sgd/adamw")
        if self.scheduler not in ('lambda', 'step', 'multistep', 'cosine', 'onecycle'):
            raise ValueError(f"不支持的 scheduler: {self.scheduler}")

        # ---------- 头结构（按 task 分别校验） ----------
        if self.task in ('detect', 'obb'):
            if self.head_type not in ('decoupled',):
                raise ValueError(
                    f"detect/obb 仅支持 head_type='decoupled'，"
                    f"当前为 {self.head_type!r}")
            if self.head_norm_layer not in ('group', 'batch', 'none'):
                raise ValueError(
                    f"head_norm_layer 必须为 group/batch/none，"
                    f"当前为 {self.head_norm_layer!r}")
            if self.head_act_layer not in ('silu', 'relu', 'gelu'):
                raise ValueError(
                    f"head_act_layer 必须为 silu/relu/gelu，"
                    f"当前为 {self.head_act_layer!r}")
        elif self.task == 'multilabel':                     # ← 改这里
            if self.head_type not in ('linear', 'mlp'):
                raise ValueError(
                    f"multilabel 仅支持 head_type='linear' 或 'mlp'，"
                    f"当前为 {self.head_type!r}")
            if self.pool_type not in ('gap', 'gem', 'attention'):
                raise ValueError(
                    f"pool_type 必须为 gap/gem/attention，"
                    f"当前为 {self.pool_type!r}")
            if self.loss not in ('bce', 'asl'):
                raise ValueError(
                    f"loss 必须为 bce 或 asl，当前为 {self.loss!r}")
            
        # return self
        # ---------- simmim 路径校验 ----------
        if self.backbone_pretrained_source == 'simmim':
            if not self.backbone_pretrained_path or \
               not os.path.exists(self.backbone_pretrained_path):
                raise FileNotFoundError(
                    f"backbone_pretrained_path 无效: {self.backbone_pretrained_path}")

        # ---------- ★ save_dir 默认值 ----------
        # 用户未指定 → ./outputs/{task}（如 ./outputs/detect）
        # 用户已指定 → 保持用户值
        if self.save_dir is None:
            self.save_dir = os.path.join('./outputs', self.task)

        return self