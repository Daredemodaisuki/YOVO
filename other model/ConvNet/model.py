import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiKernelConv(nn.Module):
    """多核卷积模块"""

    def __init__(self, in_channels, out_channels, ratio=(1, 4, 1)):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * ratio[0], kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels * ratio[1], kernel_size=3, padding=1)

        # 5x5卷积分解为两个3x3卷积
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * ratio[2], kernel_size=3, padding=1),
            nn.Conv2d(out_channels * ratio[2], out_channels * ratio[2], kernel_size=3, padding=1)
        )

        self.post_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels * sum(ratio)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * sum(ratio), out_channels, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.post_conv(out)


class CAPTCHAModel(nn.Module):
    """验证码识别模型"""

    def __init__(self, num_chars=62, num_positions=4):
        super().__init__()
        self.num_positions = num_positions

        # 主干网络
        self.stem = nn.Sequential(
            MultiKernelConv(3, 48),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2 -> 32x80
            MultiKernelConv(48, 96),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2 -> 16x40
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2h -> 8x40
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2h -> 4x40
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 预测网络
        self.predictive_conv = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),  # 通道压缩
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 修复通道问题：输入应为 [B, 64, 4, 40]
        # 分组卷积 (每个字符独立处理)
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                # 输入通道应为64，而不是10
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for _ in range(num_positions)
        ])

        # 字符分类器
        self.char_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, num_chars, kernel_size=1)
            ) for _ in range(num_positions)
        ])

    def forward(self, x):
        # 主干网络
        x = self.stem(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # [B, 256, 4, 40]

        # 预测网络
        x = self.predictive_conv(x)  # [B, 64, 4, 40]

        # 维度转置: [B, C, H, W] -> [B, W, H, C]
        x = x.permute(0, 3, 2, 1)  # [B, 40, 4, 64]

        # 按宽度分组 (每个字符10个宽度单元)
        group_size = x.size(1) // self.num_positions  # 40/4=10
        groups = [x[:, i * group_size:(i + 1) * group_size] for i in range(self.num_positions)]

        # 处理每个字符组
        outputs = []
        for i, group in enumerate(groups):
            # 分组卷积: [B, 10, 4, 64]
            # 注意：现在输入形状是 [B, 10, 4, 64]
            # 我们需要将其转换为 [B, 64, 4, 10] 以适应卷积层
            group = group.permute(0, 3, 2, 1)  # [B, 64, 4, 10]

            # 应用分组卷积
            group = self.group_convs[i](group)  # [B, 64, 4, 10]

            # 全局平均池化: [B, 64, 4, 10] -> [B, 64, 1, 1]
            group = F.adaptive_avg_pool2d(group, (1, 1))

            # 分类: [B, 64, 1, 1] -> [B, num_chars]
            outputs.append(self.char_classifiers[i](group).squeeze())

        # 组合所有字符输出 [B, num_positions, num_chars]
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    # 测试模型
    model = CAPTCHAModel(num_chars=62, num_positions=4)
    dummy = torch.randn(2, 3, 64, 160)  # [B, C, H, W]
    output = model(dummy)
    print(f"输出形状: {output.shape}")  # 应为 [2, 4, 62]