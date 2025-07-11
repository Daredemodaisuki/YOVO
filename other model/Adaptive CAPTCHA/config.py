import torch


class Config:
    # 数据参数
    img_width = 200
    img_height = 100
    target_width = 192  # 论文目标尺寸
    target_height = 64
    # num_classes = 62  # 大小写字母+数字 (26+26+10)
    num_classes = 29  # 真实赶集网
    num_chars = 4

    # 模型参数
    affn_layers = 2  # AFFN层数
    lstm_hidden = 512
    dropout = 0.3

    # 训练参数
    batch_size = 32
    epochs = 100
    lr = 0.00035
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径设置
    train_dir = "dataset/annotated_Ganji_remixed/train/images/"
    val_dir =   "dataset/annotated_Ganji_remixed/val/images/"
    save_dir = "other model/Adaptive CAPTCHA/runs/remote/3-real_Ganji_mixed/"


config = Config()
