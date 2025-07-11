import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import Config


class CaptchaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度
            transforms.Resize((Config.target_height, Config.target_width)),
            transforms.ToTensor(),
        ])

        # 构建字符到索引的映射
        self.char2idx = {}
        # idx = 0
        # # 数字0-9
        # for i in range(10):
        #     self.char2idx[str(i)] = idx
        #     idx += 1
        # # 大写字母A-Z
        # for i in range(65, 91):
        #     self.char2idx[chr(i)] = idx
        #     idx += 1
        # # 小写字母a-z
        # for i in range(97, 123):
        #     self.char2idx[chr(i)] = idx
        #     idx += 1
        
        # Ganji
        ganji = ['2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', ]
        for i in range(len(ganji)):
            self.char2idx[ganji[i]] = i

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = self.transform(image)

        # 从文件名提取标签 (第一个'_'前的内容)
        filename = self.image_files[idx]
        label_str, _ = os.path.splitext(filename)
        label_str = label_str.split('_')[0]

        # 确保标签长度正确
        if len(label_str) != Config.num_chars:
            raise ValueError(f"标签长度错误: {label_str} (文件: {filename})")

        # 将标签转为张量
        label = torch.zeros(Config.num_chars, dtype=torch.long)
        for i, char in enumerate(label_str):
            if char not in self.char2idx:
                raise ValueError(f"未知字符: {char} (文件: {filename})")
            label[i] = self.char2idx[char]

        return image, label


def get_dataloaders():
    train_dataset = CaptchaDataset(Config.train_dir)
    val_dataset = CaptchaDataset(Config.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, train_dataset.char2idx