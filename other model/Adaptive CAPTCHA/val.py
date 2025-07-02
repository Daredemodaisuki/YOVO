import torch
import difflib
from model import AdaptiveCAPTCHA
from data_loader import CaptchaDataset
from config import Config
from torch.utils.data import DataLoader
# import argparse


def lcs_length(a, b):
    """计算两个字符串的最长公共子序列长度"""
    s = difflib.SequenceMatcher(None, a, b)
    return sum(block.size for block in s.get_matching_blocks())


def validate_model(model_path):
    # 加载模型
    checkpoint = torch.load(model_path, map_location=Config.device)
    model = AdaptiveCAPTCHA(Config).to(Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载验证集和字符映射
    val_dataset = CaptchaDataset(Config.val_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4
    )
    idx2char = {v: k for k, v in val_dataset.char2idx.items()}

    # 验证统计
    total_lcs = 0
    total_chars = 0
    correct_full = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)

            # 获取预测字符
            predictions = []
            for i in range(Config.num_chars):
                _, pred = outputs[i].max(1)
                predictions.append(pred)
            predictions = torch.stack(predictions, dim=1)  # [B, 4]

            # 转换为字符串比较
            for i in range(labels.size(0)):
                # 真实标签字符串
                true_str = ''.join([idx2char[idx.item()] for idx in labels[i]])

                # 预测字符串
                pred_str = ''.join([idx2char[idx.item()] for idx in predictions[i]])

                # 计算LCS
                lcs_len = lcs_length(true_str, pred_str)
                total_lcs += lcs_len
                total_chars += len(true_str)

                # 完整匹配计数
                if true_str == pred_str:
                    correct_full += 1

    # 计算指标
    lcs_acc = total_lcs / total_chars
    full_acc = correct_full / len(val_dataset)

    print(f"测试结果 (模型: {model_path})")
    print(f"LCS字符准确率: {lcs_acc:.4f}")
    print(f"完整验证码准确率: {full_acc:.4f}")
    print(f"错误率: {(1 - full_acc):.4f}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default=Config.save_path,
    #                     help="模型路径")
    # args = parser.parse_args()
    MODEL_PATH = ''

    validate_model(MODEL_PATH)
