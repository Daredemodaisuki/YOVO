import os
import shutil
import glob

# 配置常量
ITERATION = 3

SOURCE_ROOT_DIR = "../runs/local/test/" \
                  "realGanji（【选定】只区分是不是，ganji_train_x3，after_round2，cof0.25iou0.5）" \
                  "ganji_train_x3-yolo_origin去小核C2fFasrer-PGanji_2(after_round2)_80只区分是不是1752059721.0/"   # 结果根目录
LOG_FILE_PATH = os.path.join(SOURCE_ROOT_DIR, "result.txt")  # 当前识别结果日志
# IMG_SOURCE_DIR = os.path.join(SOURCE_ROOT_DIR, "predictions/")  # 原始图片目录
IMG_SOURCE_DIR = "../captcha_img/dataset_semi-supervised-for-captcha/dataset/ganji-1/train"  # 原始图片目录
TXT_SOURCE_DIR = os.path.join(SOURCE_ROOT_DIR, "predictions/", "labels/")  # 原始标注目录
CLASS_FILE = "./classes_only_char.txt"  # 类别文件

HISTORY_LOG_DIR = f"./rounds/"  # 历史记录目录
COPY_CORRECT_CHAR_DIR = f"./rounds/{ITERATION}/correct_char_num/"  # 副本目录1
COPY_FINE_TUNE_DIR = f"./rounds/{ITERATION}/after_fine_tune/"  # 副本目录2


def main():
    # 确保目标目录存在
    os.makedirs(COPY_CORRECT_CHAR_DIR, exist_ok=True)
    os.makedirs(COPY_FINE_TUNE_DIR, exist_ok=True)
    os.makedirs(HISTORY_LOG_DIR, exist_ok=True)

    # 读取历史记录
    history_log = os.path.join(HISTORY_LOG_DIR, "log.txt")
    processed_ids = set()
    if os.path.exists(history_log):
        with open(history_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "图片已处理：" in line:  # 图片已处理：「{img_id}」\n
                    processed_ids.add(line.split("「")[1].split("」")[0])

    # 处理当前日志
    new_ids = []
    with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if "识别为「0000」" in line and "图片「" in line:
                img_id = line.split("「")[1].split("」")[0]
                if img_id not in processed_ids:
                    new_ids.append(img_id)

    # 复制文件和更新日志
    for img_id in new_ids:
        # 复制图片文件（支持多种格式）
        for img_path in glob.glob(os.path.join(IMG_SOURCE_DIR, f"{img_id}.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(img_path, COPY_CORRECT_CHAR_DIR)
                shutil.copy2(img_path, COPY_FINE_TUNE_DIR)  # 图片复制到两个目录

        # 复制标注文件
        txt_file = f"{img_id}.txt"
        src_txt = os.path.join(TXT_SOURCE_DIR, txt_file)
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, COPY_CORRECT_CHAR_DIR)
            shutil.copy2(src_txt, COPY_FINE_TUNE_DIR)
            for dir in [COPY_CORRECT_CHAR_DIR, COPY_FINE_TUNE_DIR]:
                data = []
                # 移除最后一项置信度
                with open(os.path.join(dir, txt_file), 'r') as f:
                    for line in f:
                        data.append(line.split(" "))
                with open(os.path.join(dir, txt_file), 'w') as f:
                    for data_list in data:
                        f.write(' '.join(data_list[0:5]) + "\n")  # 只要01234项，不要5（[0:5]）


        # 复制类别文件
        # class_src = os.path.join(TXT_SOURCE_DIR, CLASS_FILE)
        if os.path.exists(CLASS_FILE):
            shutil.copy2(CLASS_FILE, os.path.join(COPY_CORRECT_CHAR_DIR, "classes.txt"))
            shutil.copy2(CLASS_FILE, os.path.join(COPY_FINE_TUNE_DIR, "classes.txt"))

        # 更新历史日志
        with open(history_log, 'a', encoding='utf-8') as f:
            f.write(f"图片已处理：「{img_id}」\n")

    print(f"第{ITERATION}轮，新增样本: {len(new_ids)}个")
    with open(history_log, 'a', encoding='utf-8') as f:
        f.write(f"第{ITERATION}轮，新增4字符样本: {len(new_ids)}个\n")


if __name__ == "__main__":
    main()
