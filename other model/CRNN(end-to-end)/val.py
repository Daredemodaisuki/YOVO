import torch
from model import CRNN
from dataset import create_data_loader
from utils import CHARSET, NUM_CHARS, ctc_decode
import os
import numpy as np
from tqdm import tqdm
import difflib


def lcs_length(a, b):
    """计算两个字符串的最长公共子序列长度"""
    s = difflib.SequenceMatcher(None, a, b)
    return sum(block.size for block in s.get_matching_blocks())


def calculate_metrics(pred_texts, true_texts):
    """计算字符准确率、全图准确率和字符数准确率"""
    total_images = len(true_texts)
    full_correct = 0
    length_correct = 0
    total_chars = 0
    correct_chars = 0

    for pred, true in zip(pred_texts, true_texts):
        # 全图准确率
        if pred == true:
            full_correct += 1

        # 字符数准确率
        if len(pred) == len(true):
            length_correct += 1

        # 字符准确率
        total_chars += len(true)
        correct_chars += lcs_length(pred, true)

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    full_image_accuracy = full_correct / total_images
    length_accuracy = length_correct / total_images

    return char_accuracy, full_image_accuracy, length_accuracy


def save_results(result_file, image_names, pred_texts, true_texts):
    """保存识别结果到文件"""
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入每个样本的结果
        for name, pred, true in zip(image_names, pred_texts, true_texts):
            is_correct = pred == true
            f.write(f'图片「{name}」：识别为「{pred}」，{is_correct}\n')

        # 计算并写入统计结果
        char_acc, full_acc, len_acc = calculate_metrics(pred_texts, true_texts)

        f.write('\n' + '=' * 10 + ' 统计结果 ' + '=' * 10 + '\n')
        f.write(f'总图片数: {len(true_texts)}\n')
        f.write(f'字符准确率: {char_acc * 100:.2f}%\n')
        f.write(f'全图准确率: {full_acc * 100:.2f}%\n')
        f.write(f'字符数准确率: {len_acc * 100:.2f}%\n')


def validate(model, data_loader, device, result_dir='results'):
    """验证模型并保存结果"""
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'result.txt')

    model.eval()
    all_preds = []
    all_targets = []
    all_names = []

    val_loop = tqdm(data_loader, desc='Validating', leave=True)
    with torch.no_grad():
        for batch in val_loop:
            images, targets, target_lengths, image_names = batch
            images = images.to(device)

            # 前向传播
            outputs = model(images)

            # 解码预测
            pred_texts = ctc_decode(outputs, CHARSET)
            true_texts = [''.join([CHARSET[i] for i in t[:l]])
                          for t, l in zip(targets.cpu(), target_lengths.cpu())]

            all_preds.extend(pred_texts)
            all_targets.extend(true_texts)
            all_names.extend(image_names)

            # 更新进度条
            correct = sum(1 for p, t in zip(pred_texts, true_texts) if p == t)
            val_loop.set_postfix(acc=f"{correct / len(true_texts):.2%}" if true_texts else "N/A")

    # 保存结果到文件
    save_results(result_file, all_names, all_preds, all_targets)

    # 计算并返回准确率
    char_acc, full_acc, len_acc = calculate_metrics(all_preds, all_targets)
    return char_acc, full_acc, len_acc


if __name__ == '__main__':
    # 配置参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    VAL_DIR = '../../dataset/3-6char/val/images'
    MODEL_PATH = './runs/remote/3-6训/best_model_epoch26_val-acc0.9680.pth'  # 使用最终训练的模型
    RESULT_DIR = './runs/local/test/3-6训6测'
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 加载模型
    model = CRNN(NUM_CHARS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # 数据加载
    val_loader = create_data_loader(VAL_DIR, CHARSET, batch_size=32, shuffle=False)

    # 执行验证
    char_acc, full_acc, len_acc = validate(model, val_loader, DEVICE, RESULT_DIR)

    # 打印最终结果
    print('\n' + '=' * 10 + ' 最终统计结果 ' + '=' * 10)
    print(f'字符准确率: {char_acc * 100:.2f}%')
    print(f'全图准确率: {full_acc * 100:.2f}%')
    print(f'字符数准确率: {len_acc * 100:.2f}%')
    print(f'结果已保存至: {os.path.join(RESULT_DIR, "结果.txt")}')