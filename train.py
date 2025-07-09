import os
from sklearn.model_selection import train_test_split
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from tqdm import tqdm
import shutil


# 训练
def train_yolo_model(dataset_dir, dataset_yaml, model_path, name, epochs=50, imgsz=640, batch=8, remote=True):
    # model = YOLO('yolov8n.pt')  # 加载预训练模型（yolov8n.pt是小模型，适合快速训练）
    model = YOLO(model_path)
    # 训练模型
    results = model.train(
        data=os.path.join(dataset_dir, dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project="runs/{addr}/detect".format(addr="remote" if remote else "local")
    )

    # print(model1.info)
    # DetectionModel(cfg='yolov8n (nc=62) - 去小核.yaml')
    return model


def main():
    root = os.path.abspath(os.path.dirname(__file__))

    train_yolo_model(dataset_dir=os.path.join(root, "dataset/Pseudo_Ganji_4char_2(after_round2)"),
                     dataset_yaml="data_Pseudo_Ganji.yaml",
                     model_path='yaml/yolov8n (nc=62) - 去小核 - C2fFaster.yaml',
                     name='yolo_origin去小核C2fFasrer-PGanji_2(after_round2)_80只区分是不是',  # 文件夹名称
                     epochs=80,
                     imgsz=196,
                     batch=16)
                     # YOLOv8n (nc=62) - 去小核 - C2fFaster summary: 104 layers, 820,274 parameters, 820,258 gradients, 5.3 GFLOPs


if __name__ == '__main__':
    main()
