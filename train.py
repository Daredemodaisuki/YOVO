import os
from sklearn.model_selection import train_test_split
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from tqdm import tqdm
import shutil


# 训练
def train_yolo_model(dataset_dir, name, epochs=50, imgsz=640, batch=8, remote=True):
    # model = YOLO('yolov8n.pt')  # 加载预训练模型（yolov8n.pt是小模型，适合快速训练）
    model = YOLO('yaml/yolov8n (nc=62) - 去小核 - C2fFaster.yaml')
    # 训练模型
    results = model.train(
        data=os.path.join(dataset_dir, 'data.yaml'),
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

    train_yolo_model(dataset_dir=os.path.join(root, "dataset"),
                     name='yolo_origin去小核C2fFasrer-150',
                     epochs=150,
                     imgsz=180,
                     batch=8)


if __name__ == '__main__':
    main()
