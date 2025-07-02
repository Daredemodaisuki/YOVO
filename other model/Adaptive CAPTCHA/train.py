import torch
import torch.nn as nn
from model import AdaptiveCAPTCHA
from data_loader import get_dataloaders
from config import Config
import time
import os
import difflib
from tqdm import tqdm


def lcs_length(a, b):
    """计算两个字符串的最长公共子序列长度"""
    s = difflib.SequenceMatcher(None, a, b)
    return sum(block.size for block in s.get_matching_blocks())

def train():
    # 初始化
    device = Config.device
    os.makedirs(os.path.dirname(Config.save_dir), exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(Config.save_dir, 'recording.txt')
    with open(log_file, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")

    # 数据加载
    train_loader, val_loader, char2idx = get_dataloaders()

    # 模型设置
    model = AdaptiveCAPTCHA(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    # 训练循环
    best_acc = 0.0
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        correct_full = 0
        correct_chars = 0
        total_full = 0
        total_chars = 0

        # 获取字符映射
        idx2char = {v: k for k, v in val_loader.dataset.char2idx.items()}

        start_time = time.time()
        # 使用tqdm包装训练数据加载器
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs} [Train]", leave=True)
        running_loop = 0
        for images, labels in train_loop:
            running_loop += 1
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失 (4个字符的交叉熵)
            predictions = []
            loss = 0
            for i in range(Config.num_chars):
                _, pred = outputs[i].max(1)
                predictions.append(pred)
                loss += nn.NLLLoss()(outputs[i], labels[:, i])
            predictions = torch.stack(predictions, dim=1)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(labels.size(0)):
                true_str = ''.join([idx2char[idx.item()] for idx in labels[i]])
                pred_str = ''.join([idx2char[predictions[i, j].item()] for j in range(Config.num_chars)])

                # 计算LCS
                correct_chars += lcs_length(true_str, pred_str)
                total_chars += len(true_str)

                # 完整匹配计数
                total_full += 1
                if true_str == pred_str:
                    correct_full += 1

            # 统计
            total_loss += loss.item()

            # 更新训练进度条描述
            running_loss = total_loss / total_full
            train_loop.set_postfix(loss=f"{running_loss:.4f}")

        # 训练统计
        avg_loss = total_loss / total_full
        char_acc = correct_chars / total_chars

        # 验证
        val_acc, val_char_acc, val_loss = validate(model, val_loader)

        # 记录日志
        log_line = f"Epoch {epoch + 1}/{Config.epochs}: Train loss: {avg_loss:.4f} | Train char acc: {char_acc:.4f}" \
                   f" | Train char acc: {char_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}" \
                   f" | Val char acc: {val_char_acc:.4f} \n"
        with open(log_file, 'a') as f:
            f.write(log_line)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_name = f'best_model_epoch{epoch + 1}_val-acc{val_acc:.4f}.pth'
            save_path = os.path.join(Config.save_dir, save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'char2idx': char2idx
            }, save_path)

            # 记录模型保存信息
            with open(log_file, 'a') as f:
                f.write(f"Saved best model: {save_path}\n")

        # 打印进度
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{Config.epochs}] | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

    print(f"训练完成! 最佳验证准确率: {best_acc:.4f}")


def validate(model, val_loader):
    model.eval()
    device = Config.device
    total_lcs = 0
    total_chars = 0
    correct_full = 0
    total_loss = 0.0

    # 获取字符映射
    idx2char = {v: k for k, v in val_loader.dataset.char2idx.items()}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 获取预测字符、计算损失 (4个字符的交叉熵)
            predictions = []
            loss = 0
            for i in range(Config.num_chars):
                _, pred = outputs[i].max(1)
                predictions.append(pred)
                loss += nn.NLLLoss()(outputs[i], labels[:, i])
            predictions = torch.stack(predictions, dim=1)  # [B, 4]
            total_loss += loss.item()

            # 转换为字符串比较
            for i in range(labels.size(0)):
                true_str = ''.join([idx2char[idx.item()] for idx in labels[i]])
                pred_str = ''.join([idx2char[idx.item()] for idx in predictions[i]])

                # 计算LCS
                lcs_len = lcs_length(true_str, pred_str)
                total_lcs += lcs_len
                total_chars += len(true_str)

                # 完整匹配计数
                if true_str == pred_str:
                    correct_full += 1

    lcs_acc = total_lcs / total_chars
    full_acc = correct_full / len(val_loader.dataset)
    val_loss = total_loss / len(val_loader.dataset)

    print(f"验证结果: LCS准确率 {lcs_acc:.4f} | 完整验证码准确率 {full_acc:.4f}")
    return full_acc, lcs_acc, val_loss  # 仍返回完整准确率用于模型选择


if __name__ == "__main__":
    train()
