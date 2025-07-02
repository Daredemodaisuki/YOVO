import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CaptchaDataset(Dataset):
    def __init__(self, root_dir, charset, max_label_len=6):
        self.root_dir = root_dir
        self.image_dir = root_dir  # os.path.join(root_dir, 'images')
        self.image_files = os.listdir(self.image_dir)
        self.charset = charset
        self.max_label_len = max_label_len
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # 文件名格式: "label_otherinfo.jpg" - 第一个"_"前是标签
        label = img_name.split('_')[0]

        # 加载图像
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # 应用变换
        img = self.transform(img)

        # 标签转换为索引序列
        target = [self.charset.index(c) for c in label]
        target_len = len(target)

        # 填充标签到固定长度
        if target_len < self.max_label_len:
            target += [len(self.charset)] * (self.max_label_len - target_len)

        return img, torch.tensor(target), torch.tensor(target_len), img_name


def create_data_loader(root_dir, charset, batch_size=32, shuffle=True):
    dataset = CaptchaDataset(root_dir, charset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)