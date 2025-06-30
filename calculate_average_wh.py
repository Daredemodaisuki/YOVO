import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 配置参数
data_folder = "dataset/4char/train/labels"  # 替换为你的数据集路径
image_width = 200  # 图片宽度（像素）
image_height = 100  # 图片高度（像素）

# 存储所有检测框的宽高
widths = []
heights = []

# 遍历数据集文件夹
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(data_folder, filename)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                if len(data) < 5:  # 确保有足够的字段
                    continue

                # 解析YOLO格式的宽高（归一化值）
                w_norm = float(data[3])
                h_norm = float(data[4])

                # 转换为绝对像素尺寸
                w_abs = w_norm * image_width
                h_abs = h_norm * image_height

                widths.append(w_abs)
                heights.append(h_abs)

# 转换为NumPy数组
widths = np.array(widths)
heights = np.array(heights)

# 1. 创建点状分布图
plt.figure(figsize=(10, 8))
plt.scatter(widths, heights, s=2, alpha=0.5, c='blue')
plt.title('Bounding Box Size Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True, alpha=0.3)
plt.xlim(0, image_width)
plt.ylim(0, image_height)

plt.tight_layout()
plt.savefig('bbox_distribution.png', dpi=150)
plt.show()

# 2. 对高度进行聚类（k=3）
height_cluster = KMeans(n_clusters=3, random_state=0, n_init=10)
height_labels = height_cluster.fit_predict(heights.reshape(-1, 1))
height_centers = height_cluster.cluster_centers_.flatten()

# 3. 对宽度进行聚类（k=2）
width_cluster = KMeans(n_clusters=2, random_state=0, n_init=10)
width_labels = width_cluster.fit_predict(widths.reshape(-1, 1))
width_centers = width_cluster.cluster_centers_.flatten()

# 打印聚类结果
print("=" * 50)
print("高度聚集值（像素）:")
for i, h in enumerate(np.sort(height_centers)):
    print(f" 高度中心 #{i + 1}: {h:.2f} px")

print("\n宽度聚集值（像素）:")
for i, w in enumerate(np.sort(width_centers)):
    print(f" 宽度中心 #{i + 1}: {w:.2f} px")

# 计算每个聚类中的目标数量
print("\n聚类分布统计:")
height_cluster_counts = np.bincount(height_labels)
for i, count in enumerate(height_cluster_counts):
    print(f" 高度聚类 #{i + 1}: {count} 个目标 ({count / len(heights) * 100:.1f}%)")

width_cluster_counts = np.bincount(width_labels)
for i, count in enumerate(width_cluster_counts):
    print(f" 宽度聚类 #{i + 1}: {count} 个目标 ({count / len(widths) * 100:.1f}%)")

# 输出统计摘要
print("\n" + "=" * 50)
print(f"总检测框数量: {len(widths)}")
print(f"宽度范围: {widths.min():.1f} - {widths.max():.1f} px")
print(f"高度范围: {heights.min():.1f} - {heights.max():.1f} px")
print(f"平均宽度: {widths.mean():.1f} px")
print(f"平均高度: {heights.mean():.1f} px")
print(f"宽度标准差: {widths.std():.1f} px")
print(f"高度标准差: {heights.std():.1f} px")