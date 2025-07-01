import torch
import torch.nn as nn
from model import CRNN  # 假设您的模型定义在model.py中


def count_parameters(model):
    """计算模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_layer_params(model):
    """打印每层的名称、参数量和占比"""
    print("\n{:<30} {:<15} {:<15} {:<10}".format(
        "Layer", "Parameters", "Trainable", "Percentage"
    ))
    print("-" * 70)

    total_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable = "Yes"
        else:
            trainable = "No"
        params = param.numel()
        percentage = f"{params / total_params * 100:.2f}%"
        print("{:<30} {:<15,} {:<15} {:<10}".format(
            name, params, trainable, percentage
        ))


if __name__ == "__main__":
    # 初始化模型（与训练脚本相同的配置）
    num_chars = len("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") + 1  # 62字符+1空白符
    model = CRNN(num_chars)

    # 计算参数量
    total_params, trainable_params = count_parameters(model)

    # 打印结果
    print("\n" + "=" * 50)
    print(f"{'CRNN Parameter Count':^50}")
    print("=" * 50)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")

    # 打印各层详细参数
    print_layer_params(model)

    # 内存占用估算（FP32）
    print("\nMemory Usage Estimation (FP32):")
    print(f"- Model Size: {total_params * 4 / (1024 ** 2):.2f} MB")
    print(f"- GPU Memory (Training): ~{total_params * 16 / (1024 ** 2):.2f} MB (with gradients)")