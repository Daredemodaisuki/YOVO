import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        # 卷积层 (输入: [b, 3, 100, 200])
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),  # [64, 50, 100]
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
            # [128, 25, 50]
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(256),  # [256, 25, 50]
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(256), nn.MaxPool2d((2, 1), (2, 1)),
            # [256, 12, 50]
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(512),  # [512, 12, 50]
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(512), nn.MaxPool2d((2, 1), (2, 1)),
            # [512, 6, 50]
            nn.Conv2d(512, 512, 2), nn.LeakyReLU(0.2), nn.BatchNorm2d(512)  # [512, 5, 49]
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=512 * 5,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 输出层 (时间步分类)
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        # CNN特征提取
        x = self.cnn(x)
        b, c, h, w = x.size()

        # 重塑为序列: [b, w, c*h]
        x = x.permute(0, 3, 2, 1)  # [b, w, h, c]
        x = x.reshape(b, w, c * h)  # [b, w, c*h]

        # LSTM处理序列
        x, _ = self.lstm(x)  # [b, w, hidden_size*2]

        # 时间步分类
        x = self.fc(x)  # [b, w, num_chars+1]
        return x