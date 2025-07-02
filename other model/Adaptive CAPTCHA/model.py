import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class AFFN(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.alphas = nn.ParameterList()

        # 创建编码器/解码器对
        in_ch = 1
        for i in range(layers):
            out_ch = 4 ** (i + 1)  # 4, 16, ...
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(out_ch, in_ch, kernel_size=5, padding=2),
                nn.BatchNorm2d(in_ch),
                nn.ReLU()
            ))
            self.alphas.append(nn.Parameter(torch.tensor(0.5)))  # 初始α=0.5
            in_ch = out_ch

    def forward(self, x):
        residuals = []

        # 编码阶段
        for i in range(self.layers):
            residual = x
            x = self.encoders[i](x)
            residuals.append(residual)

        # 解码阶段 + 自适应融合
        for i in range(self.layers - 1, -1, -1):
            decoded = self.decoders[i](x)
            alpha = torch.sigmoid(self.alphas[i])  # 约束到[0,1]
            x = alpha * decoded + (1 - alpha) * residuals[i]

        return x


class AdaptiveCAPTCHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # AFFN模块
        self.affn = AFFN(layers=config.affn_layers)

        # 卷积主干 (Deep CAPTCHA)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # CRNN模块
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # LSTM序列建模
        self.dropout1 = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=256 * 2,  # 特征图高度=2
            hidden_size=config.lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(config.dropout)

        # 分类头 (4个独立分类器)
        self.char_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.lstm_hidden, config.num_classes),
                nn.LogSoftmax(dim=1)
            ) for _ in range(config.num_chars)
        ])

        # 残差连接
        # self.residual = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.residual = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2)  # 添加池化层匹配尺寸
        )  # 尺寸不匹配，需要下采样

    def forward(self, x):
        # 输入处理: RGB转灰度
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # RGB转灰度

        # AFFN滤波
        x = self.affn(x)

        # 卷积主干 + 残差连接
        residual = self.residual(x)
        x = self.conv1(x)
        x = x + residual  # 残差连接 (T→1)

        x = self.conv2(x)
        x = self.conv3(x)

        # CRNN模块
        x = self.conv4(x)
        x = self.conv5(x)  # 输出: [B, 256, 2, 6]

        # 序列拆分与LSTM处理
        B, C, H, W = x.shape
        x = x.view(B, C * H, W)  # [B, 512, 6]
        x = x.permute(0, 2, 1)  # [B, 6, 512]

        # 均匀分割为4段 (每段1.5列 -> 取整处理)
        segments = []
        segment_size = W // self.config.num_chars
        for i in range(self.config.num_chars):
            start = i * segment_size
            end = (i + 1) * segment_size
            segments.append(x[:, start:end, :])

        # 合并段并处理
        x = torch.cat(segments, dim=0)  # [4B, seg_len, 512]
        x = self.dropout1(x)
        x, _ = self.lstm(x)
        x = self.dropout2(x)

        # 获取每段最后一个时间步的输出
        x = x[:, -1, :]  # [4B, hidden_size]

        # 分字符分类
        outputs = []
        chunk_size = B
        for i in range(self.config.num_chars):
            chunk = x[i * chunk_size: (i + 1) * chunk_size]
            outputs.append(self.char_classifiers[i](chunk))

        return outputs