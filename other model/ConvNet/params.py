import torch
import torch.nn as nn
from model import CAPTCHAModel  # 替换为您的模型类
from collections import OrderedDict


def count_parameters(model, verbose=True):
    """统计模型参数量

    Args:
        model: PyTorch模型
        verbose: 是否打印详细分层信息

    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    # 初始化统计
    total_params = 0
    trainable_params = 0
    param_dist = OrderedDict()

    # 逐层统计
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params

        # 按层类型分类统计
        layer_type = name.split('.')[0]  # 取第一级模块名
        if layer_type not in param_dist:
            param_dist[layer_type] = 0
        param_dist[layer_type] += params

    # 打印结果
    if verbose:
        print(f"\n{'=' * 50}\n模型参数量统计\n{'=' * 50}")
        print(f"{'总参数量':<20}: {total_params:,}")
        print(f"{'可训练参数量':<20}: {trainable_params:,}")
        print(f"{'不可训练参数量':<20}: {total_params - trainable_params:,}")

        print(f"\n{'=' * 50}\n各层参数分布\n{'=' * 50}")
        for layer, params in param_dist.items():
            print(f"{layer:<20}: {params:,} ({params / total_params:.2%})")

        print(f"\n{'=' * 50}\n参数类型统计\n{'=' * 50}")
        param_types = {}
        for name, param in model.named_parameters():
            ptype = str(param.dtype).replace("torch.", "")
            size = "x".join(map(str, param.shape))
            key = f"{ptype} ({size})"
            param_types[key] = param_types.get(key, 0) + param.numel()

        for ptype, count in sorted(param_types.items()):
            print(f"{ptype:<30}: {count:,}")

    return total_params, trainable_params


if __name__ == "__main__":
    # 初始化模型
    model = CAPTCHAModel(num_chars=62, num_positions=4)

    # 统计参数量
    total, trainable = count_parameters(model)