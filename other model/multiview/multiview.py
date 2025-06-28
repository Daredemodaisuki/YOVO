import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试复现Mukhtar Opeyemi Yusuf的Multiview deep learning‑based attack to break text‑CAPTCHAs
class MultiViewCAPTCHANet(nn.Module):
    def __init__(self, num_views, num_chars, img_size=(100, 200), hidden_size=128):
        super().__init__()
        self.num_views = num_views
        self.img_size = img_size

        conv_output_h = img_size[0] // 4
        conv_output_w = img_size[1] // 4
        flattened_dim = 64 * conv_output_h

        # 每个视图独立的 CNN、FC 和 GRU
        self.cnn_modules = nn.ModuleList()
        self.fc_modules = nn.ModuleList()
        self.gru_modules = nn.ModuleList()

        for _ in range(num_views):
            self.cnn_modules.append(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ))
            self.fc_modules.append(nn.Linear(flattened_dim, hidden_size))
            self.gru_modules.append(nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.25
            ))

        # 融合 + 分类
        self.view_fusion = nn.Sequential(
            nn.Linear(2 * hidden_size * num_views, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.ctc_classifier = nn.Linear(256, num_chars)
        self.ctc_loss = nn.CTCLoss(blank=0)

    def forward(self, x, targets=None, target_lengths=None):
        batch_size, num_views, C, H, W = x.shape
        view_outputs = []

        for view_idx in range(num_views):
            view = x[:, view_idx, :, :, :]  # [B, 1, H, W]
            cnn_out = self.cnn_modules[view_idx](view)  # [B, 64, H/4, W/4]

            # 转换为序列 [B, W/4, H/4, 64]
            cnn_out = cnn_out.permute(0, 3, 2, 1)
            batch_size, seq_len, _, channels = cnn_out.shape
            spatial_features = cnn_out.reshape(batch_size, seq_len, -1)

            fc_out = self.fc_modules[view_idx](spatial_features)
            fc_out = F.relu(fc_out)

            gru_out, _ = self.gru_modules[view_idx](fc_out)
            view_outputs.append(gru_out)

        fused = torch.cat(view_outputs, dim=-1)  # [B, T, 2H*num_views]
        fused = self.view_fusion(fused)  # [B, T, 256]

        logits = self.ctc_classifier(fused)  # [B, T, num_chars]
        log_probs = F.log_softmax(logits, dim=2)

        if targets is not None:
            input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
            loss = self.ctc_loss(
                log_probs.permute(1, 0, 2),  # [T, B, C]
                targets,
                input_lengths,
                target_lengths
            )
            return logits, loss

        return logits


def ctc_decode(log_probs, blank=0):
    """CTC贪婪解码"""
    _, max_index = torch.max(log_probs, 2)
    return max_index  # [batch_size, seq_len]


def decode_predictions(predictions, idx2char, blank=0):
    """将模型输出转换为字符串"""
    decoded_strings = []
    for pred in predictions:
        # 移除空白符和重复字符
        current_char = None
        decoded_chars = []
        for idx in pred:
            if idx != blank and idx != current_char:
                decoded_chars.append(idx2char.get(idx.item(), ''))
            current_char = idx

        decoded_strings.append(''.join(decoded_chars))
    return decoded_strings