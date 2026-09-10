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
    backbone_name = 'convnextv2_atto'
    pretrained = True
    img_size = 640
    neck_out_channels = 128
    head_num_convs = 2
    use_extra_conv = True

    # OBB 相关
    with_angle = False
    angle_range = 'pi'
    angle_format = 'sin_cos'
    angle_loss_type = 'l1'

    # ---------- 训练相关 ----------
    batch_size = 16
    lr = 5e-4                 # AdamW 推荐 5e-4
    unfreeze_lr = 1e-4
    weight_decay = 1e-2       # AdamW 推荐 1e-2
    epochs = 50
    warmup_epochs = 5         # 前 5 轮线性 warmup，然后进入主衰减
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = min(4, os.cpu_count() // 2)
    save_dir = './outputs'
    log_interval = 10
    save_interval = 10
    eval_interval = 1
    grad_clip_norm = 5.0      # 梯度裁剪阈值

    # ---------- 优化器与调度器 ----------
    optimizer = 'adamw'          # 'sgd' 或 'adamw'
    scheduler = 'cosine'         # 'lambda', 'step', 'multistep', 'cosine', 'onecycle'
    scheduler_gamma = 0.95       # 用于 lambda, step, multistep
    scheduler_step_size = 1      # 用于 step
    scheduler_milestones = [30, 60]  # 用于 multistep（列表）
    scheduler_warmup_epochs = 5  # 保留字段，实际 warmup 由 warmup_epochs 控制

    # ---------- 损失权重 ----------
    cls_weight = 1.0
    reg_weight = 1.0
    angle_weight = 2.0        

    # ---------- 混合精度 ----------
    # 'fp32' | 'fp16' | 'bf16'
    #   fp32: 全精度训练（无 autocast）
    #   fp16: 混合精度 + GradScaler（动态范围小，需要梯度缩放）
    #   bf16: 混合精度，无需 GradScaler（需 Ampere 及更新架构）
    precision = 'fp16'
    # 仅当 precision 未显式设置时作为回退依据（见 validate_and_setup）
    use_amp = True

    # ---------- 冻结 Backbone ----------
    freeze_epochs = 5

    # ---------- Mosaic ----------
    mosaic_prob = 0.3
    mosaic_disable_epoch = 20

    # ---------- 后处理 ----------
    conf_threshold = 0.7
    nms_threshold = 0.3
    obb_nms_threshold = 0.3

    # ---------- 数据增强 ----------
    use_flip = True
    use_color_jitter = True
    color_jitter_params = (0.25, 0.25, 0.5, 0.015)

    # ---------- 图像归一化 ----------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ---------- 随机种子 ----------
    seed = 42

    # ---------- mAP 计算起始轮次 ----------
    # 0 表示从第一轮开始计算；设为较大值可跳过前期不稳定阶段，加快训练
    # 推荐设为 warmup_epochs + 5 左右，避免前期浪费时间在无意义的 mAP 上
    eval_map_start_epoch = 10

    # ---------- 自动计算 ----------
    strides = None

    def validate_and_setup(self):
        """验证并自动设置参数"""
        # ---------- 模式校验 ----------
        if self.mode == 'obb':
            self.with_angle = True
            print(f"[Config] OBB 模式已启用，角度编码: {self.angle_format}")
        elif self.mode == 'detect':
            self.with_angle = False
            print("[Config] 水平检测模式已启用")
        else:
            raise ValueError(f"不支持的 mode: {self.mode}")

        # ---------- 角度参数校验 ----------
        if self.angle_range not in ['pi', 'pi2']:
            raise ValueError(f"angle_range 必须为 'pi' 或 'pi2'，当前为 {self.angle_range}")
        if self.angle_format not in ['sin_cos', 'direct']:
            raise ValueError(f"angle_format 必须为 'sin_cos' 或 'direct'，当前为 {self.angle_format}")

        if self.mode == 'obb' and self.obb_nms_threshold is None:
            self.obb_nms_threshold = self.nms_threshold

        # ---------- 优化器校验 ----------
        if self.optimizer not in ['sgd', 'adamw']:
            raise ValueError(f"optimizer 必须为 'sgd' 或 'adamw'，当前为 {self.optimizer}")

        # ---------- 调度器校验 ----------
        valid_schedulers = ['lambda', 'step', 'multistep', 'cosine', 'onecycle']
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler 必须为 {valid_schedulers} 之一，当前为 {self.scheduler}")

        # ---------- 精度校验与兼容处理 ----------
        # 若 precision 未显式赋值，根据 use_amp 回退
        if self.precision is None:
            self.precision = 'fp16' if getattr(self, 'use_amp', True) else 'fp32'
            print(f"[Config] precision 未设置，根据 use_amp={self.use_amp} 推断为 {self.precision}")

        valid_precisions = ['fp32', 'fp16', 'bf16']
        if self.precision not in valid_precisions:
            raise ValueError(
                f"precision 必须为 {valid_precisions} 之一，当前为 {self.precision}"
            )

        # 保持 use_amp 与实际精度一致，避免下游代码误用
        self.use_amp = (self.precision != 'fp32')

        # ---------- warmup 与 epochs 一致性校验 ----------
        if getattr(self, 'warmup_epochs', 0) >= self.epochs:
            print(f"[Config] 警告: warmup_epochs({self.warmup_epochs}) >= epochs({self.epochs})，"
                  f"自动调整为 epochs 的 10%")
            self.warmup_epochs = max(1, int(self.epochs * 0.1))

        # ---------- Mosaic 禁用轮次校验 ----------
        if self.mosaic_disable_epoch is not None and self.mosaic_disable_epoch >= self.epochs:
            print(f"[Config] 警告: mosaic_disable_epoch({self.mosaic_disable_epoch}) >= epochs({self.epochs})，"
                  f"Mosaic 将始终启用")

        # ---------- eval_map_start_epoch 校验 ----------
        if self.eval_map_start_epoch < 0:
            self.eval_map_start_epoch = 0
        if self.eval_map_start_epoch >= self.epochs:
            print(f"[Config] 警告: eval_map_start_epoch({self.eval_map_start_epoch}) >= epochs({self.epochs})，"
                  f"将只会在最后一轮计算 mAP")

        # ---------- 打印汇总 ----------
        print(f"[Config] 优化器: {self.optimizer}, 调度器: {self.scheduler}, warmup: {self.warmup_epochs} epochs")
        print(f"[Config] 训练精度: {self.precision.upper()} (use_amp={self.use_amp})")
        print(f"[Config] 梯度裁剪: {self.grad_clip_norm}, 冻结 Backbone: {self.freeze_epochs} epochs")
        print(f"[Config] Mosaic 概率: {self.mosaic_prob}, 禁用轮次: {self.mosaic_disable_epoch}")
        print(f"[Config] mAP 计算起始轮次: {self.eval_map_start_epoch}")

        return self

    # ---------- 辅助方法 ----------
    def get_head_params(self):
        return {
            'num_classes': self.num_classes,
            'with_angle': self.with_angle,
            'angle_format': self.angle_format,
        }

    def get_loss_params(self):
        return {
            'cls_weight': self.cls_weight,
            'reg_weight': self.reg_weight,
            'angle_weight': self.angle_weight if self.mode == 'obb' else 0.0,
            'mode': self.mode,
        }

    def get_postprocess_params(self):
        params = {
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'img_size': self.img_size,
            'strides': self.strides,
        }
        if self.mode == 'obb':
            params.update({
                'obb_nms_threshold': self.obb_nms_threshold,
                'angle_range': self.angle_range,
                'angle_format': self.angle_format,
            })
        return params

    def get_dataset_params(self):
        return {
            'root': self.data_root,
            'img_size': self.img_size,
            'split': None,
            'use_flip': self.use_flip,
            'use_color_jitter': self.use_color_jitter,
            'color_jitter_params': self.color_jitter_params,
            'mean': self.mean,
            'std': self.std,
            'mode': self.mode,
            'angle_range': self.angle_range if self.mode == 'obb' else None,
            'mosaic_prob': self.mosaic_prob,
            'mosaic_disable_epoch': self.mosaic_disable_epoch,
        }

    def get_training_params(self):
        return {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'unfreeze_lr': self.unfreeze_lr,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'warmup_epochs': self.warmup_epochs,
            'device': self.device,
            'num_workers': self.num_workers,
            'save_dir': self.save_dir,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
            'precision': self.precision,
            'use_amp': self.use_amp,
            'freeze_epochs': self.freeze_epochs,
            'seed': self.seed,
            'mode': self.mode,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'grad_clip_norm': self.grad_clip_norm,
        }

    def get_export_params(self):
        return {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'backbone_name': self.backbone_name,
            'mode': self.mode,
            'angle_range': self.angle_range if self.mode == 'obb' else None,
            'angle_format': self.angle_format if self.mode == 'obb' else None,
        }

    def to_dict(self):
        config_dict = {
            'mode': self.mode,
            'with_angle': self.with_angle,
            'backbone_name': self.backbone_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'strides': self.strides,
            'mean': self.mean,
            'std': self.std,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'obb_nms_threshold': self.obb_nms_threshold if self.mode == 'obb' else None,
            'angle_range': self.angle_range if self.mode == 'obb' else None,
            'angle_format': self.angle_format if self.mode == 'obb' else None,
            'angle_weight': self.angle_weight if self.mode == 'obb' else None,
            'freeze_epochs': self.freeze_epochs,
            'unfreeze_lr': self.unfreeze_lr,
            'lr': self.lr,
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
            'mosaic_prob': self.mosaic_prob,
            'mosaic_disable_epoch': self.mosaic_disable_epoch,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'scheduler_gamma': self.scheduler_gamma,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_milestones': self.scheduler_milestones,
            # 模型结构配置参数
            'neck_out_channels': self.neck_out_channels,
            'head_num_convs': self.head_num_convs,
            'use_extra_conv': self.use_extra_conv,
            # mAP 起始轮次
            'eval_map_start_epoch': self.eval_map_start_epoch,
        }
        return config_dict