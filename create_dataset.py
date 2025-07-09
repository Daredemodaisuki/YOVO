import os
from sklearn.model_selection import train_test_split
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from tqdm import tqdm
import shutil


# 创建数据集
def create_dataset(dataset_dir, original_img_dir):
    # 创建数据集所需目录结构
    # dataset_dir                                   # ./dataset
    train_dir = os.path.join(dataset_dir, "train")  # ./dataset/train
    val_dir = os.path.join(dataset_dir, "val")      # ./dataset/val

    os.makedirs(dataset_dir, exist_ok=True)
    for tv in [train_dir, val_dir]:
        os.makedirs(tv, exist_ok=True)
        os.makedirs(os.path.join(tv, "images"), exist_ok=True)
        os.makedirs(os.path.join(tv, "labels"), exist_ok=True)

    # 遍历获取文件、分训练和测试集
    files = os.listdir(original_img_dir)
    imgs = [img for img in files if os.path.splitext(img)[1].lower() in [".png", ".jpg", ".jpeg"]]
    print("{amount} images at {dir} got.".format(amount=len(imgs), dir=original_img_dir))
    train_imgs, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
    print("There will be {train} train images, and {val} val images"
          .format(train=str(len(train_imgs)), val=str(len(val_imgs))))

    with tqdm(total=len(train_imgs), ncols=150) as _tqdm:
        _tqdm.set_description("Processing train images")
        failed = 0
        for img in train_imgs:
            img_path_name, _ = os.path.splitext(img)
            txt = img_path_name + ".txt"
            if os.path.exists(os.path.join(original_img_dir, txt)):
                shutil.copy2(os.path.join(original_img_dir, img), os.path.join(train_dir, "images", img))
                shutil.copy2(os.path.join(original_img_dir, txt), os.path.join(train_dir, "labels", txt))
            else:
                print("label for image '{img}' in train set is not found. Image skipped.".format(img=img))
                failed += 1
            _tqdm.set_postfix({"now": img, "failed": failed})
            _tqdm.update(1)
    with tqdm(total=len(val_imgs), ncols=150) as _tqdm:
        _tqdm.set_description("Processing val images  ")
        failed = 0
        for img in val_imgs:
            img_path_name, _ = os.path.splitext(img)
            txt = img_path_name + ".txt"
            if os.path.exists(os.path.join(original_img_dir, txt)):
                shutil.copy2(os.path.join(original_img_dir, img), os.path.join(val_dir, "images", img))
                shutil.copy2(os.path.join(original_img_dir, txt), os.path.join(val_dir, "labels", txt))
            else:
                print("label for image '{img}' in val set is not found. Image skipped.".format(img=img))
                failed += 1
            _tqdm.set_postfix({"now": img, "failed": failed})
            _tqdm.update(1)

    def read_classes_in_txt(txt_path):
        with open(txt_path, "r") as f_t:
            classes_txt = f_t.read()
            classes_in_txt = classes_txt.split("\n")
            return classes_in_txt

    # data.yaml
    with open(os.path.join(dataset_dir, "data_Pseudo_Ganji.yaml"), "w") as yaml:
        yaml.write("train: ../train/images\n")
        yaml.write("val: ../val/images\n")

        classes = read_classes_in_txt(os.path.join(original_img_dir, 'classes.txt'))  # 读取class里面的类
        yaml.write("nc: {0}\n".format(len(classes)))  # 类别数
        yaml.write("names: [")  # 类别名称
        for cls in classes:
            yaml.write("'{object}', ".format(object=cls))  # 类别名称
        yaml.write("]\n")


def main():
    root = os.path.abspath(os.path.dirname(__file__))
    create_dataset(dataset_dir=os.path.join(root, "dataset", "Pseudo_Ganji_4char_2(after_round2)"),
                   original_img_dir=os.path.join(root, "PseudoGanjiToReal/rounds/2/final_new_dataset"))


if __name__ == '__main__':
    main()
