import torch
from model import AdaptiveCAPTCHA
from config import Config

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_parameter_details(model):
    """打印各层参数详情"""
    print("{:<30} {:<20} {:<15}".format('Layer Name', 'Parameters', 'Trainable'))
    print("-" * 65)
    for name, param in model.named_parameters():
        print("{:<30} {:<20,} {:<15}".format(
            name, 
            param.numel(), 
            str(param.requires_grad)
        ))

def main():
    # 初始化模型
    model = AdaptiveCAPTCHA(Config)
    
    # 统计参数量
    total_params, trainable_params = count_parameters(model)
    
    # 打印结果
    print("\n" + "="*50)
    print(f"{'Total Parameters:':<20} {total_params:,}")
    print(f"{'Trainable Parameters:':<20} {trainable_params:,}")
    print(f"{'Non-trainable Parameters:':<20} {total_params - trainable_params:,}")
    print("="*50 + "\n")
    
    # 打印详细参数分布（可选）
    print_parameter_details(model)
    
    # 计算模型大小（近似）
    param_size = total_params * 4 / (1024 ** 2)  # 假设float32占4字节，转换为MB
    print(f"\nApproximate Model Size: {param_size:.2f} MB (float32)")

if __name__ == "__main__":
    main()