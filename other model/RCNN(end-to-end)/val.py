import torch
from model import CRNN
from dataset import create_data_loader
from utils import CHARSET, NUM_CHARS, ctc_decode, calculate_accuracy
import os

# 配置参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
VAL_DIR = 'data/val'
MODEL_PATH = 'checkpoints/crnn_epoch_50.pth'  # 使用最终训练的模型

# 加载模型
model = CRNN(NUM_CHARS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 数据加载
val_loader = create_data_loader(VAL_DIR, CHARSET, BATCH_SIZE, shuffle=False)

# 验证循环
all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets, target_lengths in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)

        # 解码预测
        pred_texts = ctc_decode(outputs, CHARSET)
        true_texts = [''.join([CHARSET[i] for i in t[:l]])
                      for t, l in zip(targets.cpu(), target_lengths.cpu())]

        all_preds.extend(pred_texts)
        all_targets.extend(true_texts)

# 计算准确率
accuracy = calculate_accuracy(all_preds, all_targets)
print(f'Validation Accuracy: {accuracy:.4f}')

# 打印部分样本结果
print("\nSample Predictions:")
for i in range(min(10, len(all_targets))):
    print(f'Target: {all_targets[i]} | Predicted: {all_preds[i]}')