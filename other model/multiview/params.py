import torch
from multiview import MultiViewCAPTCHANet


def calculate_model_parameters(num_views=3, num_chars=63, img_size=(100, 200)):
    """
    计算MultiviewCAPTCHANet模型的参数量
    :param num_views: 视图数量
    :param num_chars: 字符类别数（包括空白符）
    :param img_size: 图像尺寸 (高度, 宽度)
    """
    # 创建模型实例
    model = MultiViewCAPTCHANet(
        num_views=num_views,
        num_chars=num_chars,
        img_size=img_size
    )

    # 计算总参数数量和可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型大小（假设32位浮点数）
    model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32

    # 打印结果
    print("=" * 60)
    print(f"模型配置: {num_views}视图, {num_chars}字符类别, 图像尺寸{img_size}")
    print("=" * 60)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小 (32位浮点): {model_size_mb:.2f} MB")

    # 按层细分参数
    print("\n" + "=" * 60)
    print("按层参数明细:")
    print("=" * 60)

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {module_params:,} 参数")

        # 如果是ModuleList，进一步细分
        if isinstance(module, torch.nn.ModuleList):
            for i, sub_module in enumerate(module):
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"  - View {i + 1}: {sub_params:,} 参数")

    # 计算卷积层参数占比
    conv_params = 0
    for name, param in model.named_parameters():
        if "cnn" in name or "conv" in name:
            conv_params += param.numel()

    conv_percent = conv_params / total_params * 100

    print("\n" + "=" * 60)
    print(f"卷积层参数占比: {conv_percent:.1f}% ({conv_params:,}/{total_params:,})")
    print("=" * 60)


if __name__ == "__main__":
    CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    NUM_CHARS = len(CHAR_SET) + 1  # 字符集 + 空白符

    calculate_model_parameters(
        num_views=3,
        num_chars=NUM_CHARS,
        img_size=(100, 200)  # hw
    )