import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def create_views(image):
    """
    为单张图像创建三种处理视图
    返回: (3, H, W) numpy数组
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 视图1: 双边滤波
    view1 = cv2.bilateralFilter(gray, 9, 75, 75)

    # 视图2: 阈值处理
    _, view2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 视图3: 中值滤波
    view3 = cv2.medianBlur(gray, 5)

    # 堆叠视图
    views = np.stack([view1, view2, view3], axis=0)

    return views


class CAPTCHADataset(Dataset):
    def __init__(self, root_dir, char_set, img_size=(100, 200)):
        """
        root_dir: 图片目录 (e.g., 'dataset/4char/train/images')
        char_set: 所有可能字符的字符串 (e.g., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        img_size: 目标图像尺寸 (高度, 宽度)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(root_dir) if f.endswith('.png')]

        # 创建字符到索引的映射
        self.char2idx = {char: idx + 1 for idx, char in enumerate(char_set)}  # 索引从1开始
        self.char2idx[''] = 0  # 空白符索引为0
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.num_chars = len(char_set) + 1  # 包括空白符

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        # 创建多视图
        views = create_views(image)  # (3, H, W)

        # 添加通道维度 (3, 1, H, W)
        views = views[:, np.newaxis, :, :]

        # 从文件名获取标签 (前4字符)
        label_str = os.path.splitext(self.filenames[idx])[0][:4]
        label = [self.char2idx[c] for c in label_str]

        # 转换为Tensor
        views_tensor = torch.tensor(views, dtype=torch.float32) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return views_tensor, label_tensor

    def get_char_mappings(self):
        """返回字符映射字典"""
        return self.char2idx, self.idx2char