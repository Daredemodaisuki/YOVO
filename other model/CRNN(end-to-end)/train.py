import torch
import torch.nn as nn
from torch.optim import Adam
from model import CRNN
from dataset import create_data_loader
from utils import CHARSET, NUM_CHARS, ctc_decode, calculate_accuracy
import os
import numpy as np
from tqdm import tqdm

# 配置参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 0.0007
EPOCHS = 160
TRAIN_DIR = 'dataset/4char/train/images'
VAL_DIR =   'dataset/4char/val/images'
SAVE_DIR = 'other model/CRNN(end-to-end)/runs/remote/6-是改了什么导致Ganji不行吗？重测4训验证模型有效性'
LOG_FILE = os.path.join(SAVE_DIR, 'recording.txt')
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化模型
model = CRNN(NUM_CHARS).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CTCLoss(blank=NUM_CHARS)

# 微调
# checkpoint_path = 'other model/RCNN(end-to-end)/runs/remote/1/best_model_epoch13_val-acc0.9403.pth'
# model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))


# 数据加载
train_loader = create_data_loader(TRAIN_DIR, CHARSET, BATCH_SIZE)
val_loader = create_data_loader(VAL_DIR, CHARSET, BATCH_SIZE, shuffle=False)


def append_text_to_file(text, file_path):
    """将文本追加到文件末尾"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def train_epoch(model, data_loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    train_loop = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True)
    for images, targets, target_lengths, _ in train_loop:  # _=image_name
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # 前向传播
        outputs = model(images)

        # 计算CTC损失
        input_lengths = torch.full(
            size=(outputs.size(0),),
            fill_value=outputs.size(1),
            dtype=torch.long,
            device=device
        )

        loss = criterion(
            outputs.permute(1, 0, 2),
            targets,
            input_lengths,
            target_lengths
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        pred_texts = ctc_decode(outputs, CHARSET)
        true_texts = [''.join([CHARSET[i] for i in t[:l]])
                      for t, l in zip(targets.cpu(), target_lengths.cpu())]

        for pred, true in zip(pred_texts, true_texts):
            if pred == true:
                correct += 1
            total += 1

        # 更新进度条
        avg_loss = total_loss / len(train_loop)
        acc = correct / total if total > 0 else 0.0
        train_loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2%}")

    return total_loss / len(data_loader), correct / total if total > 0 else 0.0


def validate(model, data_loader, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    val_loop = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", leave=True)
    with torch.no_grad():
        for images, targets, target_lengths, _ in val_loop:  # _=img_name
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # 前向传播
            outputs = model(images)

            # 计算CTC损失
            input_lengths = torch.full(
                size=(outputs.size(0),),
                fill_value=outputs.size(1),
                dtype=torch.long,
                device=device
            )

            loss = criterion(
                outputs.permute(1, 0, 2),
                targets,
                input_lengths,
                target_lengths
            )
            total_loss += loss.item()

            # 计算准确率
            pred_texts = ctc_decode(outputs, CHARSET)
            true_texts = [''.join([CHARSET[i] for i in t[:l]])
                          for t, l in zip(targets.cpu(), target_lengths.cpu())]

            for pred, true in zip(pred_texts, true_texts):
                if pred == true:
                    correct += 1
                total += 1

            # 更新进度条
            val_loop.set_postfix(acc=f"{correct / total:.2%}" if total else "N/A")

    return total_loss / len(data_loader), correct / total if total > 0 else 0.0


# 训练循环
best_val_acc = 0.0
for epoch in range(EPOCHS):
    # 训练阶段
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)

    # 验证阶段
    val_loss, val_acc = validate(model, val_loader, DEVICE)

    # 打印epoch总结
    print(f'\nEpoch {epoch + 1}/{EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # 记录到日志文件
    log_text = (f'Epoch {epoch + 1}/{EPOCHS}: '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    append_text_to_file(log_text, LOG_FILE)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(SAVE_DIR, f'best_model_epoch{epoch + 1}_val-acc{val_acc:.4f}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with val acc {val_acc:.4f} at epoch {epoch + 1}!')
        append_text_to_file(f'Saved best model: {best_model_path}', LOG_FILE)

    print('-' * 80)

print(f'\nTraining complete. Best validation accuracy: {best_val_acc:.4f}')