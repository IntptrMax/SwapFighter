import torch
import os


class Config:
    # ==================== 基础配置 ====================
    mode = 'detect'          # 'detect' 或 'obb'

    # ---------- 数据相关 ----------
    data_root = './datasets/medical-pills'
    train_split = 'train'
    val_split = 'val'
    num_classes = 1

    # ---------- 模型相关 ----------
    backbone_name = 'convnextv2_atto.fcmae_ft_in1k'
    pretrained = True
    img_size = 640
    neck_out_channels = 128
    head_num_convs = 3
    use_extra_conv = True

    # ★ DFL 相关：默认关闭
    #    0 => 关闭 DFL，回归头输出 4 通道（dx,dy,dw,dh）
    #    >0 => 启用 DFL，回归头输出 4*(reg_max+1) 通道
    reg_max = 0
    dfl_weight = 0.0

    # ---------- Neck 相关 ----------
    neck_type = 'lightfpn'          # 'lightfpn' 或 'bifpn'
    neck_num_layers = 2

    # ---------- ★ 解耦头（DecoupledHead） ----------
    head_type = 'decoupled'         # 目前仅支持 'decoupled'
    head_norm_layer = 'group'       # 'group' / 'batch' / 'none'
    head_act_layer = 'silu'         # 'silu' / 'relu' / 'gelu'
    head_cls_num_convs = None       # None 表示使用 head_num_convs
    head_reg_num_convs = None
    head_angle_num_convs = None

    # ---------- 预训练权重来源 ----------
    backbone_pretrained_source = 'auto'    # 'auto' / 'timm' / 'simmim' / 'none'
    backbone_pretrained_path = None

    # ---------- OBB 相关 ----------
    with_angle = False              # 由 mode='obb' 自动设置
    angle_range = 'pi2'             # 'pi' 或 'pi2'（YOLOv8-OBB 推荐 pi2）
    angle_format = 'sin_cos'        # 'sin_cos' 或 'direct'
    angle_loss_type = 'l1'

    # ---------- ★ OBB 回归损失 ----------
    #   'probiou' = CRIoU + 0.5*ProbIoU（稳定，对角度敏感，少数据推荐）
    #   'kld'     = KLD 损失（大长宽比目标强，但需要数据量支撑）
    obb_reg_loss = 'probiou'
    kld_weight = 1.0                # KLD 内部权重（有效权重 = reg_weight × kld_weight）
    kld_tau = 2.0                   # KLD 温度，越大梯度越平滑
    kld_fun = 'log1p'               # 'log1p' / 'sqrt' / None

    # ---------- 训练相关 ----------
    batch_size = 16
    lr = 5e-4                       
    weight_decay = 1e-2
    epochs = 100
    warmup_epochs = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = min(4, os.cpu_count() // 2)
    save_dir = './outputs'
    log_interval = 0
    save_interval = 0
    eval_interval = 1
    grad_clip_norm = 3.0            

    # ---------- 早停 ----------
    # 连续 N 个 epoch 未刷新 best fitness 则停止训练
    # 0 = 禁用早停，跑满所有 epoch
    patience = 50

    # ---------- 分参数组学习率 ----------
    backbone_lr_scale = 0.1         

    # ---------- 优化器与调度器 ----------
    optimizer = 'adamw'             # 'sgd' 或 'adamw'
    scheduler = 'lambda'            # 'lambda' / 'step' / 'multistep' / 'cosine' / 'onecycle'
    scheduler_gamma = 0.95
    scheduler_step_size = 1
    scheduler_milestones = [30, 60]

    # ---------- 损失权重 ----------
    #   有效 KLD 权重 = reg_weight × kld_weight
    cls_weight = 0.5
    reg_weight = 7.5
    angle_weight = 1.0              

    # ---------- 混合精度 ----------
    precision = 'fp16'              # 'fp32' / 'fp16' / 'bf16'
    use_amp = True

    # ---------- 冻结 Backbone ----------
    freeze_epochs = 0               # 0 = 从不冻结

    # ---------- Mosaic ----------
    mosaic_prob = 0.8
    mosaic_disable_epoch = None       

    # ---------- 后处理（推理时） ----------
    conf_threshold = 0.7
    nms_threshold = 0.5
    obb_nms_threshold = 0.5        

    # ---------- 训练评估配置 ----------
    eval_conf_threshold = None      
    eval_diag = False

    # ---------- 数据增强（几何） ----------
    degrees = 0.0                  # OBB 必须开启；detect 建议设为 0
    translate = 0.1
    scale = 0.5
    shear = 0.0
    perspective = 0.0

    # ---------- ★ 颜色增强（YOLOv8 风格 HSV） ----------
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    color_blur_prob = 0.01
    color_median_blur_prob = 0.01
    color_to_gray_prob = 0.01
    color_clahe_prob = 0.01

    # 颜色增强总开关（True 时启用 YOLOv8 风格 HSV 增强）
    use_color_aug = True

    # ---------- ★ 翻转增强（YOLOv8 风格） ----------
    use_flip = True
    fliplr = 0.5
    flipud = 0.0

    # ---------- ★ MixUp / Copy-Paste（YOLOv8 风格） ----------
    mixup_prob = 0.0
    mixup_alpha = 32.0
    copy_paste_prob = 0.0
    copy_paste_max_paste = 3

    # ---------- EMA ----------
    use_ema = True
    ema_decay = 0.99

    # ---------- 图像归一化 ----------
    mean = [0, 0, 0]
    std = [1, 1, 1]

    # ---------- 随机种子 ----------
    seed = 42

    # ---------- mAP 计算起始轮次 ----------
    eval_map_start_epoch = 0

    # ---------- 自动计算（不要手动设置） ----------
    strides = None

    # ---------- TAL 正样本分配 ----------
    tal_topk = 13
    tal_alpha = 1.0
    tal_beta = 6.0

    # ============================================================
    # 校验与自动设置
    #   注：所有详细配置打印由 train.py 统一负责，这里只输出警告/错误
    # ============================================================
    def validate_and_setup(self):
        # ---------- mode 一致性 ----------
        if self.mode == 'obb':
            self.with_angle = True
        elif self.mode == 'detect':
            self.with_angle = False
        else:
            raise ValueError(f"不支持的 mode: {self.mode}")
        
        # ---------- eval_conf_threshold 按 mode 自动设置 ----------
        if self.eval_conf_threshold is None:
            if self.mode == 'obb':
                self.eval_conf_threshold = 0.01
            else:
                self.eval_conf_threshold = 0.001

        # ---------- OBB 参数校验 ----------
        if self.angle_range not in ['pi', 'pi2']:
            raise ValueError(f"angle_range 必须为 'pi' 或 'pi2'，当前为 {self.angle_range}")
        if self.angle_format not in ['sin_cos', 'direct']:
            raise ValueError(f"angle_format 必须为 'sin_cos' 或 'direct'，当前为 {self.angle_format}")

        if self.mode == 'obb' and self.obb_nms_threshold is None:
            self.obb_nms_threshold = self.nms_threshold
            
        # detect 模式下强制关闭旋转增强（旋转会让 GT 外接框膨胀 2~3 倍）
        if self.mode == 'detect' and self.degrees > 5.0:
            print(f"[Config] ⚠️  detect 模式下 degrees={self.degrees} 会让 GT 外接框膨胀 2~3 倍，"
                  f"自动调整为 0.0")
            self.degrees = 0.0
        elif self.mode == 'detect' and self.degrees > 0.0:
            print(f"[Config] detect 模式下 degrees={self.degrees} ≤ 5，允许")

        # ---------- 优化器 / 调度器 ----------
        if self.optimizer not in ['sgd', 'adamw']:
            raise ValueError(f"optimizer 必须为 'sgd' 或 'adamw'，当前为 {self.optimizer}")

        valid_schedulers = ['lambda', 'step', 'multistep', 'cosine', 'onecycle']
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler 必须为 {valid_schedulers} 之一，当前为 {self.scheduler}")

        # ---------- 精度 ----------
        if self.precision is None:
            self.precision = 'fp16' if getattr(self, 'use_amp', True) else 'fp32'

        valid_precisions = ['fp32', 'fp16', 'bf16']
        if self.precision not in valid_precisions:
            raise ValueError(f"precision 必须为 {valid_precisions} 之一，当前为 {self.precision}")
        self.use_amp = (self.precision != 'fp32')

        # ---------- 预训练来源 ----------
        valid_sources = ['auto', 'timm', 'simmim', 'none']
        if self.backbone_pretrained_source not in valid_sources:
            raise ValueError(f"backbone_pretrained_source 必须为 {valid_sources} 之一")

        if self.backbone_pretrained_source == 'simmim':
            if not self.backbone_pretrained_path:
                raise ValueError("backbone_pretrained_source='simmim' 时必须提供 backbone_pretrained_path")
            if not os.path.exists(self.backbone_pretrained_path):
                raise FileNotFoundError(f"backbone_pretrained_path 不存在: {self.backbone_pretrained_path}")

        # ---------- warmup / mosaic ----------
        if getattr(self, 'warmup_epochs', 0) >= self.epochs:
            print(f"[Config] 警告: warmup_epochs >= epochs，自动调整为 epochs 的 10%")
            self.warmup_epochs = max(1, int(self.epochs * 0.1))

        # mosaic_disable_epoch 自动设置
        if self.mosaic_disable_epoch is None:
            self.mosaic_disable_epoch = int(self.epochs * 0.9)
        elif self.mosaic_disable_epoch >= self.epochs:
            print(f"[Config] 警告: mosaic_disable_epoch ({self.mosaic_disable_epoch}) >= epochs "
                  f"({self.epochs})，Mosaic 将始终启用")

        if self.eval_map_start_epoch < 0:
            self.eval_map_start_epoch = 0
        if self.eval_map_start_epoch >= self.epochs:
            print(f"[Config] 警告: eval_map_start_epoch >= epochs，将只会在最后一轮计算 mAP")

        # ---------- 数据增强校验 ----------
        if self.degrees < 0:
            raise ValueError(f"degrees 必须 >= 0")
        if self.translate < 0:
            raise ValueError(f"translate 必须 >= 0")
        if self.scale < 0:
            raise ValueError(f"scale 必须 >= 0")
        if self.shear < 0:
            raise ValueError(f"shear 必须 >= 0")
        if self.perspective < 0:
            raise ValueError(f"perspective 必须 >= 0")

        # ---------- 颜色增强校验 ----------
        for name in ['hsv_h', 'hsv_s', 'hsv_v']:
            v = getattr(self, name, 0.0)
            if v < 0:
                raise ValueError(f"{name} 必须 >= 0，当前为 {v}")
        for name in ['color_blur_prob', 'color_median_blur_prob',
                     'color_to_gray_prob', 'color_clahe_prob']:
            v = getattr(self, name, 0.0)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前为 {v}")

        # ---------- 翻转增强校验 ----------
        if not (0.0 <= self.fliplr <= 1.0):
            raise ValueError(f"fliplr 必须在 [0, 1] 范围内，当前为 {self.fliplr}")
        if not (0.0 <= self.flipud <= 1.0):
            raise ValueError(f"flipud 必须在 [0, 1] 范围内，当前为 {self.flipud}")

        # ---------- MixUp / Copy-Paste 校验 ----------
        if not (0.0 <= self.mixup_prob <= 1.0):
            raise ValueError(f"mixup_prob 必须在 [0, 1] 范围内，当前为 {self.mixup_prob}")
        if self.mixup_alpha <= 0:
            raise ValueError(f"mixup_alpha 必须 > 0，当前为 {self.mixup_alpha}")
        if not (0.0 <= self.copy_paste_prob <= 1.0):
            raise ValueError(f"copy_paste_prob 必须在 [0, 1] 范围内，当前为 {self.copy_paste_prob}")
        if self.copy_paste_max_paste < 1:
            raise ValueError(f"copy_paste_max_paste 必须 >= 1，当前为 {self.copy_paste_max_paste}")

        # ---------- lr 相关 ----------
        if not (0.0 < self.backbone_lr_scale <= 1.0):
            raise ValueError(f"backbone_lr_scale 必须在 (0, 1] 范围内")


        # 因为上面已经自动填好，这里的校验只针对显式传值
        if not (0.0 < self.eval_conf_threshold < 1.0):
            raise ValueError(f"eval_conf_threshold 必须在 (0, 1) 范围内")

        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"ema_decay 必须在 (0, 1) 范围内")

        # ---------- DFL 校验 ----------
        if not isinstance(self.reg_max, int) or self.reg_max < 0:
            raise ValueError(f"reg_max 必须为 >=0 的整数（0 表示关闭 DFL），当前为 {self.reg_max}")

        # ---------- KLD 校验 ----------
        if self.obb_reg_loss not in ['kld', 'probiou']:
            raise ValueError(f"obb_reg_loss 必须为 'kld' 或 'probiou'，当前为 {self.obb_reg_loss}")
        if self.obb_reg_loss == 'kld':
            if self.kld_fun not in ['log1p', 'sqrt', None]:
                raise ValueError(f"kld_fun 必须为 'log1p' / 'sqrt' / None，当前为 {self.kld_fun}")
            if self.kld_weight <= 0:
                raise ValueError(f"kld_weight 必须 > 0，当前为 {self.kld_weight}")
            if self.kld_tau <= 0:
                raise ValueError(f"kld_tau 必须 > 0，当前为 {self.kld_tau}")

        # ---------- 解耦头校验 ----------
        if self.head_type not in ['decoupled']:
            raise ValueError(f"当前仅支持 head_type='decoupled'，当前为 {self.head_type}")
        if self.head_norm_layer not in ['group', 'batch', 'none']:
            raise ValueError(f"head_norm_layer 必须为 'group' / 'batch' / 'none'")
        if self.head_act_layer not in ['silu', 'relu', 'gelu']:
            raise ValueError(f"head_act_layer 必须为 'silu' / 'relu' / 'gelu'")

        return self

    # ============================================================
    # 导出为字典（保存到 run 目录）
    # ============================================================
    def to_dict(self):
        return {
            # 基础
            'mode': self.mode,
            'with_angle': self.with_angle,
            'backbone_name': self.backbone_name,
            'pretrained': self.pretrained,
            'backbone_pretrained_source': self.backbone_pretrained_source,
            'backbone_pretrained_path': self.backbone_pretrained_path,
            'num_classes': self.num_classes,
            'img_size': self.img_size,

            # DFL
            'reg_max': self.reg_max,
            'dfl_weight': self.dfl_weight,

            # KLD
            'obb_reg_loss': self.obb_reg_loss,
            'kld_weight': self.kld_weight,
            'kld_tau': self.kld_tau,
            'kld_fun': self.kld_fun,

            # 解耦头
            'head_type': self.head_type,
            'head_norm_layer': self.head_norm_layer,
            'head_act_layer': self.head_act_layer,
            'head_cls_num_convs': self.head_cls_num_convs,
            'head_reg_num_convs': self.head_reg_num_convs,
            'head_angle_num_convs': self.head_angle_num_convs,

            # 后处理
            'strides': self.strides,
            'mean': self.mean,
            'std': self.std,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'obb_nms_threshold': self.obb_nms_threshold if self.mode == 'obb' else None,
            'eval_conf_threshold': self.eval_conf_threshold,
            'eval_diag': self.eval_diag,

            # OBB 角度
            'angle_range': self.angle_range if self.mode == 'obb' else None,
            'angle_format': self.angle_format if self.mode == 'obb' else None,
            'angle_weight': self.angle_weight if self.mode == 'obb' else None,

            # 训练
            'freeze_epochs': self.freeze_epochs,
            'lr': self.lr,
            'backbone_lr_scale': self.backbone_lr_scale,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'precision': self.precision,
            'use_amp': self.use_amp,
            'seed': self.seed,
            'data_root': self.data_root,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'cls_weight': self.cls_weight,
            'reg_weight': self.reg_weight,
            'grad_clip_norm': self.grad_clip_norm,

            # Mosaic
            'mosaic_prob': self.mosaic_prob,
            'mosaic_disable_epoch': self.mosaic_disable_epoch,

            # 优化器/调度器
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'scheduler_gamma': self.scheduler_gamma,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_milestones': self.scheduler_milestones,

            # Neck / Head
            'neck_out_channels': self.neck_out_channels,
            'head_num_convs': self.head_num_convs,
            'use_extra_conv': self.use_extra_conv,
            'neck_type': self.neck_type,
            'neck_num_layers': self.neck_num_layers,

            # mAP 起始
            'eval_map_start_epoch': self.eval_map_start_epoch,

            # 几何增强
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,

            # 颜色增强
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'color_blur_prob': self.color_blur_prob,
            'color_median_blur_prob': self.color_median_blur_prob,
            'color_to_gray_prob': self.color_to_gray_prob,
            'color_clahe_prob': self.color_clahe_prob,
            'use_color_aug': self.use_color_aug,

            # 翻转增强
            'use_flip': self.use_flip,
            'fliplr': self.fliplr,
            'flipud': self.flipud,

            # MixUp / Copy-Paste
            'mixup_prob': self.mixup_prob,
            'mixup_alpha': self.mixup_alpha,
            'copy_paste_prob': self.copy_paste_prob,
            'copy_paste_max_paste': self.copy_paste_max_paste,

            # EMA
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,

            # TAL
            'tal_topk': self.tal_topk,
            'tal_alpha': self.tal_alpha,
            'tal_beta': self.tal_beta,
        }