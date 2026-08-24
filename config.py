import torch

class Config:
    # ---------- 数据相关 ----------
    data_root = './datasets/medical-pills'   # 请修改为实际路径
    train_split = 'train'
    val_split = 'val'
    num_classes = 1

    # ---------- 模型相关 ----------
    backbone_name = 'convnext_tiny.dinov3_lvd1689m'
    pretrained = True
    img_size = 640
    # strides = [8, 16, 32]               # FPN 各层下采样步长
    strides = None   # 将由模型自动计算

    # ---------- 训练相关 ----------
    batch_size = 8
    lr = 1e-3
    unfreeze_lr = lr/10                 # 解冻 Backbone 后的学习率（通常为 lr 的 0.1 倍）
    weight_decay = 1e-4
    epochs = 20
    warmup_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    save_dir = './outputs'
    log_interval = 10
    save_interval = 10
    eval_interval = 1                   # 每 n 个 epoch 验证一次

    # ---------- loss相关 ----------
    cls_weight = 1.0 
    reg_weight = 2.0 

    # ---------- 混合精度训练 ----------
    use_amp = True                      # 是否启用自动混合精度

    # ---------- 冻结 Backbone ----------
    freeze_epochs = 10                   # 前 N 轮冻结 Backbone，0 表示不冻结

    # ---------- 推理相关 ----------
    conf_threshold = 0.5
    nms_threshold = 0.5

    # ---------- 数据增强 ----------
    use_flip = True
    use_color_jitter = True
    color_jitter_params = (0.25, 0.25, 0.5, 0.015)

    # ---------- 图像处理 ----------
    mean = [0, 0, 0]
    std = [1, 1, 1]

    # ---------- 随机种子 ----------
    seed = 42