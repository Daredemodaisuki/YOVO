import os
import shutil
import glob

# 配置常量
ITERATION = 2  # 当前迭代次数

CLASSES_TXT_PATH = "classes_only_char.txt"
HISTORY_LOG_DIR = f"./rounds/"  # 历史记录目录
COPY_CORRECT_CHAR_DIR = f"./rounds/{ITERATION}/correct_char_num/"  # 副本目录1（调整前）
COPY_FINE_TUNE_DIR = f"./rounds/{ITERATION}/after_fine_tune/"  # 副本目录2（调整后）

PREV_TRAIN_SET = f"./rounds/{ITERATION - 1}/final_new_dataset/"  # 上次数据集
# PREV_TRAIN_SET = "../captcha_img/PseudoGanji10000pic_180x54_4char_at1751897926"  # 上次训练集（第1轮为原始数据集）
NEW_TRAIN_SET = f"./rounds/{ITERATION}/final_new_dataset/"  # 新数据集

CLASS_INDEX = '0'  # 字符类别索引


def calculate_iou(box1, box2):
    """计算两个YOLO格式边界框的IoU"""

    def to_coords(box):
        x, y, w, h = map(float, box)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    box1_coords = to_coords(box1)
    box2_coords = to_coords(box2)
    # print(box1_coords)
    # print(box2_coords)

    # 计算交集区域
    xi1 = max(box1_coords[0], box2_coords[0])
    yi1 = max(box1_coords[1], box2_coords[1])
    xi2 = min(box1_coords[2], box2_coords[2])
    yi2 = min(box1_coords[3], box2_coords[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    # print(inter_area)

    # 计算并集区域
    box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
    box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
    union_area = box1_area + box2_area - inter_area
    # print(union_area)

    return inter_area / union_area if union_area > 0 else 0


def sort_boxes_by_x_center(boxes):
    """按中心x坐标对边界框进行排序"""
    return sorted(boxes, key=lambda box: float(box[0]))


def main():
    # 创建新训练集
    os.makedirs(NEW_TRAIN_SET, exist_ok=True)

    # 复制上次训练集（其中有classes.txt）
    if os.path.exists(PREV_TRAIN_SET):
        for item in os.listdir(PREV_TRAIN_SET):
            src = os.path.join(PREV_TRAIN_SET, item)
            if os.path.isfile(src):
                shutil.copy2(src, NEW_TRAIN_SET)
                print(f"已复制上一轮数据集「{src}」到「{NEW_TRAIN_SET}」。")

    # 复制调整后的数据，计算交并比
    iou_dict = {}
    for txt_file in os.listdir(COPY_FINE_TUNE_DIR):
        if txt_file == "classes.txt":
            continue  # 跳过类别文件

        img_id, ext = os.path.splitext(txt_file)
        if ext == ".txt":
            # 复制调整后的标注
            src_txt = os.path.join(COPY_FINE_TUNE_DIR, txt_file)
            dst_txt = os.path.join(NEW_TRAIN_SET, txt_file)
            shutil.copy2(src_txt, dst_txt)
            print(f"已复制上一轮数据集「{txt_file}」到{dst_txt}。")

            # 复制对应图片（支持多种格式）
            img_pattern = os.path.join(COPY_CORRECT_CHAR_DIR, f"{img_id}.*")
            for img_path in glob.glob(img_pattern):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(img_path, NEW_TRAIN_SET)
                    print(f"已复制上一轮数据集「{img_path}」到「{NEW_TRAIN_SET}」。")

            # 计算IoU
            # 读取调整前的标注
            orig_txt_path = os.path.join(COPY_CORRECT_CHAR_DIR, txt_file)
            orig_boxes = []
            if os.path.exists(orig_txt_path):
                with open(orig_txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0] == CLASS_INDEX and len(parts) >= 5:
                            orig_boxes.append(parts[1:5])
                        else:
                            print(f"错误：「{img_id}」微调前标注文件格式有误，跳过（{orig_txt_path}）")
                            continue
            else:
                print(f"错误：「{img_id}」微调前标注文件不存在，跳过（{orig_txt_path}）")
                continue

            # 读取调整后的标注
            adj_txt_path = os.path.join(COPY_FINE_TUNE_DIR, txt_file)
            adj_boxes = []
            if os.path.exists(adj_txt_path):
                with open(adj_txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0] == CLASS_INDEX and len(parts) >= 5:
                            adj_boxes.append(parts[1:5])
            else:
                print(f"错误：「{img_id}」微调后标注文件不存在，跳过（{adj_txt_path}）")
                continue

            # 确保边界框数量一致，且不为0
            if len(orig_boxes) != len(adj_boxes):
                print(f"错误：「{img_id}」微调前后边界框数量不一致，跳过（前{len(orig_boxes)}，后{len(adj_boxes)}）")
                continue
            if len(orig_boxes) == 0:
                print(f"错误：「{img_id}」不存在边界框，跳过（{orig_txt_path}，{adj_txt_path}）")
                continue

            # 按中心x坐标排序
            orig_boxes_sorted = sort_boxes_by_x_center(orig_boxes)
            adj_boxes_sorted = sort_boxes_by_x_center(adj_boxes)

            # 计算每个边界框的IoU
            iou = 0
            iou_list = []
            for orig_box, adj_box in zip(orig_boxes_sorted, adj_boxes_sorted):
                char_iou = calculate_iou(orig_box, adj_box)
                iou += char_iou
                iou_list.append(str(char_iou))
            iou /= len(orig_boxes)
            if img_id in iou_dict:
                print(f"警告：「{img_id}」已存在，覆盖iou信息。")
            iou_dict.update({img_id: [iou, iou_list]})
            print(f"图片「{img_id}」的交并比为{iou}。")

    # 更新历史日志
    history_log = os.path.join(HISTORY_LOG_DIR, "log.txt")
    orig_log = []  # 读取原始数据
    with open(history_log, 'r', encoding='utf-8') as f:
        for line in f:
            if "图片已处理：" in line:  # 图片已处理：「{img_id}」，iou=xxx\n
                id = line.split("「")[1].split("」")[0]
                line = line.split("\n")[0]  # 去除换行
                for log in orig_log:
                    if "id" in log and log["id"] == id:
                        print(f"警告：「{id}」在原始记录中出现多次，iou信息添加可能追加至错误的行。")
                if id in iou_dict:  # 如果id在本次计算的iou里面，则在改行末尾记录IoU
                    if "，iou=" in line:
                        line = line.split("，iou=")[0]
                        print(f"警告：「{id}」原始记录已包含iou信息，覆盖。")
                    line += f"，iou={iou_dict[id][0]:.6f}（{','.join(iou_dict[id][1])}）"
                line += "\n"
                orig_log.append({"type": "sample", "id": id, "content": line})
            else:
                orig_log.append({"type": "info", "content": line + "\n"})

    # 重写更新历史日志
    with open(history_log, 'w', encoding='utf-8') as f:
        for line in orig_log:
            f.write(line["content"])

        total_iou = 0
        for iou in iou_dict:
            total_iou += iou_dict[iou][0]
        avg_iou = total_iou / len(iou_dict)

        f.write(f"第{ITERATION}轮：平均IoU为{avg_iou:.6f}（{len(iou_dict)}个样本）。\n")
        f.write("----------------------------------------\n")
    print(f"处理完成！平均IoU: {avg_iou:.4f}（{len(iou_dict)}个样本）。")


if __name__ == "__main__":
    main()
    # iou = calculate_iou([0.13315622508525848, 0.49610739946365356, 0.18654948472976685, 0.7866997718811035],
    #                     [0.125000, 0.500000, 0.166667, 0.777778])
    # print(iou)
