# from ultralytics import YOLO
# import torch
#
# model = YOLO("runs/remote/detect/yolo_origin去小核C2fFasrer-PGanji_1_80/weights/best.pt")
#
# print(model.model)
import random

import torch
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO
import argparse


def main(model_path, layers, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 读取图像，缩放至224*224
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = to_tensor(resized).unsqueeze(0)
    # 加载模型
    model = YOLO(model_path)
    model.eval()

    features = {}

    # 注册 forward hook
    for idx in layers:
        def get_hook(i):
            def hook_fn(module, input, output):
                features[i] = output.detach().cpu().squeeze(0)
            return hook_fn
        model.model.model[idx].register_forward_hook(get_hook(idx))

    # 推理
    with torch.no_grad():
        model.predict(img_tensor, verbose=False)

    # 保存特征图
    for layer_idx, fmap in features.items():
        for ch in range(fmap.shape[0]):
            fmap_np = fmap[ch].numpy()
            save_path = os.path.join(save_dir, f"{layer_idx}_{ch}.png")
            plt.imsave(save_path, fmap_np, cmap='viridis')

    print(f"保存成功：{save_dir}")


if __name__ == "__main__":
    MODEL = "../runs/remote/detect/yolo_origin去小核C2fFasrer-PGanji_1_80/weights/best.pt"
    # IMAGE_DIR = "../fonts/ganji_train_x3"  # 原图缩放3倍
    IMAGE_DIR = "../dataset/Pseudo_Ganji_4char_2/train/images"  # 伪数据集
    TARGET_LAYERS = [2, 4, 7, 8, 11]  # 保存(2): C2f_Faster、(4): C2f_Faster、(7): SPPF、(8): C2f_Faster、(11): C2f_Faster

    for image in os.listdir(IMAGE_DIR):
        if random.random() > 0.025:  # 随机取一部分图片
            continue
        img_path = os.path.join(IMAGE_DIR, image)
        img_name = os.path.splitext(os.path.basename(image))[0]  # 获取图像名作为输出文件夹名
        save_dir = f"maps (Pganji_train_224x224)/{img_name}"
        main(MODEL, TARGET_LAYERS, img_path, save_dir)
