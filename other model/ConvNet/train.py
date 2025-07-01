import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import CAPTCHAModel
import difflib


# 配置参数
class Config:
    # 数据路径
    TRAIN_DIR = "dataset/4char/train/images"
    VAL_DIR = "dataset/4char/val/images"
    SAVE_DIR = "other model/ConvNet/runs/remote/2"
    LOG_FILE = "other model/ConvNet/runs/remote/2/recording.txt"

    # 训练参数
    NUM_EPOCHS = 250
    BATCH_SIZE = 32
    LR = 0.0007
    MOMENTUM = 0.99
    NUM_WORKERS = 4

    # 模型参数
    NUM_CHARS = 62  # A-Z, a-z, 0-9
    NUM_POSITIONS = 4  # 4个字符
    CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集
class CAPTCHADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 从文件名提取标签 (第一个"_"前的部分)
        filename = self.image_files[idx]
        label_str = filename.split('_')[0]

        # 将字符标签转换为数字索引
        char_to_idx = {char: i for i, char in enumerate(Config.CHAR_SET)}

        label = []
        for char in label_str:
            label.append(char_to_idx[char])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label), filename, label_str


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 160)),  # 调整到模型输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 辅助函数
def append_text_to_file(text, filename):
    with open(filename, 'a') as f:
        f.write(text + '\n')


def lcs_length(a, b):
    """计算两个字符串的最长公共子序列长度"""
    matcher = difflib.SequenceMatcher(None, a, b)
    return matcher.find_longest_match(0, len(a), 0, len(b)).size


def compare_prediction(outputs, labels, idx_to_char):
    """计算三种准确率指标"""
    # outputs: [B, 4, 62]
    # labels: [B, 4]
    batch_size = outputs.size(0)

    # 获取预测结果
    _, preds = torch.max(outputs, dim=2)  # [B, 4]

    # 初始化统计指标
    correct_char_count = 0
    total_char_count = 0
    correct_sequence_count = 0
    correct_length_count = 0

    # 处理每个样本
    for i in range(batch_size):
        # 转换预测结果为字符串
        pred_str = ''.join([idx_to_char[idx.item()] for idx in preds[i]])

        # 真实标签字符串
        true_str = ''.join([idx_to_char[idx.item()] for idx in labels[i]])

        # 计算LCS长度
        lcs_len = lcs_length(pred_str, true_str)
        correct_char_count += lcs_len
        total_char_count += len(true_str)

        # 检查序列是否完全正确
        if pred_str == true_str:
            correct_sequence_count += 1

        # 检查长度是否正确
        if len(pred_str) == len(true_str):
            correct_length_count += 1

    # 计算准确率
    # char_acc = correct_char_count / total_char_count if total_char_count > 0 else 0
    # seq_acc = correct_sequence_count / batch_size if batch_size > 0 else 0
    # len_acc = correct_length_count / batch_size if batch_size > 0 else 0

    return correct_char_count, correct_sequence_count, correct_length_count


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, idx_to_char):
    model.train()
    running_loss = 0.0
    running_char = 0
    running_seq = 0
    # running_len_acc = 0.0
    running_total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]")
    loop = 0
    for images, labels, _, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)  # [B, 4, 62]

        # 计算损失 (每个字符单独计算)
        loss = 0
        for i in range(Config.NUM_POSITIONS):
            loss += criterion(outputs[:, i, :], labels[:, i])

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        correct_chars, correct_seqs, _ = compare_prediction(outputs, labels, idx_to_char)

        # 统计信息
        running_loss += loss.item()
        running_char += correct_chars
        running_seq += correct_seqs
        # running_len_acc += correct_lens
        running_total += len(images)
        loop += 1

        # 更新进度条
        progress_bar.set_postfix({
            'acc': f"{running_seq / running_total * 100:.2f}%",
            'loss': f"{running_loss / loop:.4f}"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_char_acc = running_char / (running_total * 4)
    epoch_seq_acc = running_seq / running_total
    epoch_len_acc = 1

    return epoch_loss, epoch_char_acc, epoch_seq_acc, epoch_len_acc


def validate(model, dataloader, criterion, device, epoch, total_epochs, idx_to_char):
    model.eval()
    running_loss = 0.0
    running_char = 0
    running_seq = 0
    running_len = 0
    running_total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs} [Val]")
    with torch.no_grad():
        for images, labels, _, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = 0
            for i in range(Config.NUM_POSITIONS):
                loss += criterion(outputs[:, i, :], labels[:, i])

            # 计算准确率
            chars, seqs, lens = compare_prediction(outputs, labels, idx_to_char)

            # 统计信息
            running_loss += loss.item()
            running_char += chars
            running_seq += seqs
            running_len += lens
            running_total += len(images)

            # 更新进度条
            progress_bar.set_postfix({
                'acc': f"{running_seq / running_total * 100:.2f}%"
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_char_acc = running_char / (running_total * 4)
    epoch_seq_acc = running_seq / running_total
    epoch_len_acc = running_len / running_total

    return epoch_loss, epoch_char_acc, epoch_seq_acc, epoch_len_acc


def main():
    # 创建目录
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    # 初始化记录文件
    with open(Config.LOG_FILE, 'w') as f:
        f.write("CAPTCHA Recognition Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Device: {Config.DEVICE}\n")
        f.write(f"Number of epochs: {Config.NUM_EPOCHS}\n")
        f.write(f"Batch size: {Config.BATCH_SIZE}\n")
        f.write(f"Learning rate: {Config.LR}\n\n")

    # 字符映射
    idx_to_char = {i: char for i, char in enumerate(Config.CHAR_SET)}

    # 数据加载器
    train_dataset = CAPTCHADataset(Config.TRAIN_DIR, transform=transform)
    val_dataset = CAPTCHADataset(Config.VAL_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # 初始化模型
    model = CAPTCHAModel(num_chars=Config.NUM_CHARS, num_positions=Config.NUM_POSITIONS)
    model.to(Config.DEVICE)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)

    # 自适应学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5  # , verbose=True
    )

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # 训练阶段
        train_loss, train_char_acc, train_seq_acc, train_len_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, epoch, Config.NUM_EPOCHS, idx_to_char
        )

        # 验证阶段
        val_loss, val_char_acc, val_seq_acc, val_len_acc = validate(
            model, val_loader, criterion, Config.DEVICE, epoch, Config.NUM_EPOCHS, idx_to_char
        )

        # 记录结果
        epoch_time = time.time() - start_time
        log_line = (f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}: \n"
                    f"Train loss: {train_loss:.4f} | Train acc: {train_seq_acc:.4f} | Train ChAR: {train_char_acc:.4f}\n"
                    f"Val loss: {val_loss:.4f} | Val acc: {val_seq_acc:.4f} | Val ChAR: {val_char_acc:.4f}\n"
                    f"Time: {epoch_time:.2f}s")

        print("\n" + log_line)
        print("-" * 50)
        append_text_to_file(log_line, Config.LOG_FILE)

        # 保存最佳模型 (基于序列准确率)
        if val_seq_acc > best_val_acc:
            best_val_acc = val_seq_acc
            model_name = f"best_model_epoch{epoch + 1}_val-acc{val_seq_acc:.4f}.pth"
            model_path = os.path.join(Config.SAVE_DIR, model_name)
            torch.save(model.state_dict(), model_path)

            save_msg = f"Saved best model with val CAAR {val_seq_acc:.4f} at epoch {epoch + 1}!"
            print(save_msg)
            append_text_to_file(f"Saved best model: {model_path}", Config.LOG_FILE)

        # 学习率调整
        scheduler.step(val_seq_acc)

    print("Training completed!")


if __name__ == "__main__":
    main()