import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from multiview import MultiViewCAPTCHANet, ctc_decode, decode_predictions
from tqdm import tqdm
import difflib


class PredictionDataset(Dataset):
    def __init__(self, img_dir, img_size=(100, 200)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        # 创建多视图
        views = create_views(image)  # (3, H, W)
        views = views[:, np.newaxis, :, :]  # (3, 1, H, W)

        # 从文件名获取真实标签
        filename = self.img_files[idx]
        true_label = filename.split('_')[0]

        return torch.tensor(views, dtype=torch.float32) / 255.0, true_label, filename


def create_views(image):
    """为单张图像创建三种处理视图"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    view1 = cv2.bilateralFilter(gray, 9, 75, 75)
    _, view2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    view3 = cv2.medianBlur(gray, 5)

    return np.stack([view1, view2, view3], axis=0)


def predict_images(model, img_dir, result_file, batch_size=64, device='cuda'):
    # 创建数据集
    dataset = PredictionDataset(img_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 统计指标
    total_images = 0
    correct_images = 0
    correct_chars = 0
    total_chars = 0
    length_mismatch = 0

    with open(result_file, 'w', encoding='utf-8') as f_out:
        model.eval()

        def lcs_length(a, b):
            """计算两个字符串的最长公共子序列长度"""
            matcher = difflib.SequenceMatcher(None, a, b)
            return matcher.find_longest_match(0, len(a), 0, len(b)).size

        with torch.no_grad():
            for views, true_labels, filenames in tqdm(loader, desc='预测进度'):
                views = views.to(device)

                # 预测
                logits = model(views)
                log_probs = torch.log_softmax(logits, dim=2)
                predictions = ctc_decode(log_probs)
                pred_strings = decode_predictions(predictions.cpu(), model.idx2char)

                # 写入结果并统计
                for filename, true_label, pred in zip(filenames, true_labels, pred_strings):
                    # 写入文件
                    f_out.write(f"图片「{filename}」：识别为「{pred}」，{str(pred == true_label)}\n")

                    # 统计准确率
                    total_images += 1
                    if pred == true_label:
                        correct_images += 1

                    # 字符级统计（LCS）
                    correct_chars += lcs_length(pred, true_label)
                    total_chars += len(true_label)

                    # 统计长度错误
                    if len(pred) != len(true_label):
                        length_mismatch += 1

    # 计算指标
    char_acc = correct_chars / total_chars * 100
    img_acc = correct_images / total_images * 100
    len_error_rate = length_mismatch / total_images * 100

    # 追加统计信息
    with open(result_file, 'a', encoding='utf-8') as f_out:
        f_out.write(f"\n===== 统计结果 =====\n")
        f_out.write(f"总图片数: {total_images}\n")
        f_out.write(f"字符准确率: {char_acc:.2f}%\n")
        f_out.write(f"全图准确率: {img_acc:.2f}%\n")
        f_out.write(f"字符数错误率: {len_error_rate:.2f}%\n")

    print(f"\n预测完成！结果已保存到 {result_file}")
    print(f"字符准确率: {char_acc:.2f}%")
    print(f"全图准确率: {img_acc:.2f}%")
    print(f"字符数错误率: {len_error_rate:.2f}%")


def main():
    # ===== 参数配置 =====
    MODEL_PATH = './runs/remote/2/best_model_132_val-acc0.87725.pth'  # 模型权重路径
    IMAGE_DIR = '../../dataset/3-6char/val/images'  # 验证码图片目录
    RESULT_FILE = './runs/local/test/3-4训测6/结果.txt'  # 结果输出文件
    CHAR_SET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'  # 字符集
    BATCH_SIZE = 64  # 批处理大小
    IMG_SIZE = (100, 200)  # 图像尺寸 (高, 宽)
    # ===================

    # 检查路径
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")

    # 创建字符映射
    char2idx = {char: idx + 1 for idx, char in enumerate(CHAR_SET)}
    char2idx[''] = 0
    idx2char = {idx: char for char, idx in char2idx.items()}

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    model = MultiViewCAPTCHANet(
        num_views=3,
        num_chars=len(CHAR_SET) + 1,
        img_size=IMG_SIZE
    )
    model.to(device)

    # 加载模型权重 - 修复KeyError问题
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 检查模型文件类型
    if isinstance(checkpoint, dict):
        # 如果是字典格式，尝试可能的键名
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # 如果找不到标准键名，尝试直接加载整个状态
            model.load_state_dict(checkpoint)
    else:
        # 如果文件是直接保存的模型状态
        model.load_state_dict(checkpoint)

    # 添加字符映射到模型
    model.char2idx = char2idx
    model.idx2char = idx2char

    # 执行预测
    print(f"开始预测 {len(os.listdir(IMAGE_DIR))} 张图片...")
    predict_images(
        model=model,
        img_dir=IMAGE_DIR,
        result_file=RESULT_FILE,
        batch_size=BATCH_SIZE,
        device=device
    )


if __name__ == '__main__':
    main()