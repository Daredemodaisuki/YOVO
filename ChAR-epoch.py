import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'runs/remote/detect/' \
           'yolo_origin去小核C2fFaster（0.985-0.986, 69 layers, 825,372 parameters, 0 gradients, 5.2 GFLOPs）/' \
           'results.csv'
df = pd.read_csv(csv_path)

# 提取指标
epochs = df['epoch']
precision = df['metrics/precision(B)']
recall = df['metrics/recall(B)']
map50 = df['metrics/mAP50(B)']
map50_95 = df['metrics/mAP50-95(B)']
train_box_loss = df['train/box_loss']
train_cls_loss = df['train/cls_loss']
val_box_loss = df['val/box_loss']
val_cls_loss = df['val/cls_loss']

# 绘图
plt.subplot(2, 2, 1)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, train_box_loss, label='Train bbox loss', marker='o')
plt.plot(epochs, val_box_loss, label='Val bbox loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.grid(True)
plt.legend()
# plt.tight_layout()

plt.subplot(2, 2, 2)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, train_cls_loss, label='Train class loss', marker='o')
plt.plot(epochs, val_cls_loss, label='Train class loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.grid(True)
plt.legend()
# plt.tight_layout()

plt.subplot(2, 2, 3)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, precision, label='ChAR (Precision)', marker='o')
plt.plot(epochs, recall, label='Recall', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.grid(True)
plt.legend()
# plt.tight_layout()

plt.subplot(2, 2, 4)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, map50, label='mAP50', marker='^')
plt.plot(epochs, map50_95, label='mAP50-95', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.grid(True)
plt.legend()

plt.tight_layout()


plt.savefig('YOVO_training_metrics.png', dpi=300)
plt.show()
