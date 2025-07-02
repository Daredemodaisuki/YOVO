import os
import torch
import difflib
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import CAPTCHAModel


class Config:
    VAL_DIR = "../../dataset/4char/val/images"
    MODEL_PATH = "runs/remote/1/best_model_epoch35_val-acc0.8267.pth"  # 替换为最佳模型路径
    RESULT_FILE = "runs/local/test/2-4训4测（定长）/result.txt"  # 结果输出文件
    NUM_CHARS = 62
    NUM_POSITIONS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_WORKERS = 4


class CAPTCHADataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 从文件名提取标签
        filename = self.image_files[idx]
        true_label = filename.split('_')[0]

        # 字符到索引映射
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        char_to_idx = {char: i for i, char in enumerate(chars)}

        # 只返回标签索引用于模型输入
        label_indices = [char_to_idx[char] for char in true_label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_indices), filename, true_label


def calculate_accuracy(outputs, labels):
    """计算字符准确率和序列准确率（传统方式）"""
    _, preds = torch.max(outputs, dim=2)  # [B, 4]

    # 字符准确率（位置对应）
    char_acc = (preds == labels).float().mean()

    # 序列准确率（所有字符都正确）
    seq_acc = (preds == labels).all(dim=1).float().mean()

    return char_acc.item(), seq_acc.item()


def lcs_length(a, b):
    """计算两个字符串的最长公共子序列长度"""
    s = difflib.SequenceMatcher(None, a, b)
    return sum(block.size for block in s.get_matching_blocks())


def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = CAPTCHADataset(Config.VAL_DIR, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # 加载模型
    model = CAPTCHAModel(num_chars=Config.NUM_CHARS, num_positions=Config.NUM_POSITIONS)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # 字符映射
    idx_to_char = {}
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for i, char in enumerate(chars):
        idx_to_char[i] = char

    # 初始化统计指标
    total_images = 0
    correct_char_count = 0
    total_char_count = 0
    correct_sequence_count = 0
    correct_length_count = 0

    # 打开结果文件
    with open(Config.RESULT_FILE, 'w', encoding='utf-8') as result_file:
        progress_bar = tqdm(dataloader, desc="Validating")
        with torch.no_grad():
            for images, labels, filenames, true_labels_str in progress_bar:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                outputs = model(images)  # [B, 4, 62]

                # 获取预测结果
                _, preds = torch.max(outputs, dim=2)  # [B, 4]

                # 处理每个样本
                for i in range(len(images)):
                    total_images += 1

                    # 转换预测结果为字符串
                    pred_str = ''.join([idx_to_char[idx.item()] for idx in preds[i]])

                    # 真实标签字符串
                    true_str = true_labels_str[i]

                    # 计算LCS长度
                    lcs_len = lcs_length(pred_str, true_str)
                    correct_char_count += lcs_len
                    total_char_count += len(true_str)

                    # 检查序列是否完全正确
                    is_correct = (pred_str == true_str)
                    if is_correct:
                        correct_sequence_count += 1

                    # 检查长度是否正确
                    if len(pred_str) == len(true_str):
                        correct_length_count += 1

                    # 写入结果行
                    result_line = f"图片「{filenames[i]}」：识别为「{pred_str}」，{str(is_correct)}"
                    result_file.write(result_line + '\n')

        # 计算最终指标
        char_acc = correct_char_count / total_char_count if total_char_count > 0 else 0
        seq_acc = correct_sequence_count / total_images if total_images > 0 else 0
        len_acc = correct_length_count / total_images if total_images > 0 else 0

        # 写入统计结果
        stats = f"\n字符准确率：{char_acc:.6f} | 字符数识别准确率：{len_acc:.6f} | 全图准确率：{seq_acc:.6f}"
        result_file.write(stats)

    # 打印最终结果
    print("\nValidation Results:")
    print(stats)
    print(f"Total images processed: {total_images}")
    print(f"Results saved to: {Config.RESULT_FILE}")


if __name__ == "__main__":
    main()