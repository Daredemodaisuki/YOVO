import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# (2): C2f_Faster、(4): C2f_Faster、(7): SPPF、(8): C2f_Faster、(11): C2f_Faster
layers = {0: 2, 1: 4, 2: 7, 3: 8, 4: 11}

# 真实数据集：
# 无背景"ncg2""mewy""dtng""krhw""gn5s"
# 有背景"u46t""gemd""7plc""hwns""m4nv""6vdf""blr3""gczn"
# 真实数据集下，有背景时，2层提取了边缘，4层提取了主体，7层仍能聚焦主体，但8层其只能聚焦没有干扰的地方，信息丢失了，11叠加回4层强表征，有一定改善
captcha_name = "gn5s"
ori_img_dir = "../fonts/ganji_train_x3/"
feature_dir = f"maps (ganji_train_x3_224x224)/{captcha_name}/"

# 伪数据集
# "DSW4_1751897986""0YaP_1751897999""UdQC_1751898025""cbY8_1751898038""eW6j_1751898006""iOEg_1751898040"
# "hQS6_1751898040""Nv0Z_1751897999""WeMp_1751897985""PVHV_1751897948""NpJj_1751898007""lo1O_1751897997""G5ae_1751897933""h7sv_1751898018"
# captcha_name = "iOEg_1751898040"
# ori_img_dir = "../dataset/Pseudo_Ganji_4char_2/train/images/"
# feature_dir = f"maps (Pganji_train_224x224)/{captcha_name}/"

save_path = f"compare_output (ganji_rain_x3)/{captcha_name}_compare.png"
os.makedirs("compare_output (ganji_rain_x3)", exist_ok=True)

# 收集所有通道图
channel_imgs = [[] for i in range(5)]
mean_activation = [None for i in range(5)]
for layer in layers:
    prefix = f"{layers[layer]}_"
    for fname in os.listdir(feature_dir):
        if fname.startswith(prefix) and fname.endswith(".png"):
            path = os.path.join(feature_dir, fname)
            img = Image.open(path).convert("L")
            channel_imgs[layer].append(np.array(img, dtype=np.float32))

    # 堆叠为 (C, H, W) 并求平均
    fmap = np.stack(channel_imgs[layer], axis=0)
    mean_activation[layer] = fmap.mean(axis=0)

    # Normalize to [0, 1]
    mean_activation[layer] -= mean_activation[layer].min()
    mean_activation[layer] /= (mean_activation[layer].max() + 1e-6)

# 加载原图并 resize 到 224x224
orig_img = Image.open(os.path.join(ori_img_dir, f"{captcha_name}.png")).convert("RGB").resize((224, 224))

# 叠加绘图
plt.figure(figsize=(10, 4))

# 原图
plt.subplot(1, 6, 1)
plt.imshow(orig_img)
# plt.title("Original Image")
plt.axis("off")

for i in range(2, 7):
    # 叠加图
    plt.subplot(1, 6, i)
    plt.imshow(orig_img)
    plt.imshow(mean_activation[i - 2], cmap='jet', alpha=0.5)
    # plt.title(f"Layer {layers[i - 2]} Mean Activation")
    plt.axis("off")

plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()


