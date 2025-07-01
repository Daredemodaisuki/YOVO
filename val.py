import time
from ultralytics import YOLO
import os
import difflib


def detect_images(
        weights_path,  # 权重文件路径
        source_dir,  # 包含PNG图像的源文件夹
        output_dir,  # 输出目录
        conf_thresh=0.25,  # 置信度阈值
        iou_thresh=0.5  # IOU阈值
):
    # 加载模型
    model = YOLO(weights_path)
    output_label_txt_dir = os.path.join(output_dir, 'predictions', "labels")

    # 确保输出目录存在
    os.makedirs(output_label_txt_dir, exist_ok=True)

    # 处理每张图片
    for img_name in os.listdir(source_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, img_name)

            # 预测并获取原始结果
            results = model.predict(img_path, save=True, save_txt=False,
                                    project=output_dir,
                                    name='predictions',  # 子目录名称
                                    exist_ok=True,  # 允许覆盖现有文件
                                    conf=conf_thresh,
                                    agnostic_nms=True,  # 非极大值抑制
                                    iou=iou_thresh,
                                    )

            # 生成自定义TXT输出
            txt_path = os.path.join(output_label_txt_dir, os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    conf = float(box.conf)  # 置信度
                    xywh = box.xywhn[0].tolist()  # 归一化xywh

                    # 格式: class x_center y_center width height confidence
                    line = f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]} {conf:.6f}\n"
                    f.write(line)


def calculate_acc(results_dir, class_file, output_file='结果.txt'):
    # 类别文件
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # yolo结果
    labels_dir = os.path.join(results_dir, 'predictions', 'labels')
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"未找到标签目录: {labels_dir}")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    correct_chars = 0
    total_chars = 0
    correct_texts = 0
    incorrect_char_num_texts = 0
    total_texts = len(label_files)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        def lcs_length(a, b):
            """计算两个字符串的最长公共子序列长度"""
            matcher = difflib.SequenceMatcher(None, a, b)
            return matcher.find_longest_match(0, len(a), 0, len(b)).size

        # 处理每个标签文件
        for label_file in label_files:
            img_name = os.path.splitext(label_file)[0]  # 原图片文件名（不带扩展名的label txt名）
            # 按照x中心排序结果
            detections = []
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        detections.append((x_center, class_id))
            detections.sort(key=lambda x: x[0])
            # 匹配预测的类别、对比真实值
            predicted_seq = ''.join([classes[class_id] for _, class_id in detections])
            ground_truth = img_name.split('_')[0]
            # 是否完全正确？
            text_correct = predicted_seq == ground_truth
            if text_correct:
                correct_chars += len(ground_truth)
                total_chars += len(ground_truth)
                correct_texts += 1
            else:
                if len(predicted_seq) != len(ground_truth):
                    incorrect_char_num_texts += 1
                    # 因为识别字符数目不正确导致的False
                    # min_len = min(len(predicted_seq), len(ground_truth))
                correct = lcs_length(ground_truth, predicted_seq)  # 取最大公共子序列长度
                correct_chars += correct
                total_chars += len(ground_truth)
                # 并非完全正确，correct_texts不增加

            # 写入结果行
            out_f.write(f"图片「{img_name}」：识别为「{predicted_seq}」，{text_correct}\n")

        # 计算并写入正确率
        if total_chars > 0:
            char_accuracy = correct_chars / total_chars
            out_f.write(f"字符准确率（char_accuracy）：{char_accuracy:.6f} | ")
        else:
            out_f.write("字符准确率（char_accuracy）：0.000000 | ")
        if incorrect_char_num_texts > 0:
            incorrect_char_num_texts_ = incorrect_char_num_texts / total_texts
            out_f.write(f"字符数识别不准确率：{incorrect_char_num_texts_:.6f} | ")
        else:
            out_f.write("字符准确率（char_accuracy）：0.000000 | ")
        if total_texts > 0:
            text_accuracy = correct_texts / total_texts
            out_f.write(f"全图准确率（text_accuracy）：{text_accuracy:.6f}\n")
        else:
            out_f.write("全图准确率（text_accuracy）：0.000000\n")


if __name__ == "__main__":
    # 检测
    weights_path = "runs/remote/detect/" \
                   "yolo_origin去小核C2fFaster-B100（0.985-0.986, 69 layers, 825,372 parameters, 0 gradients, 5.2 GFLOPs）/" \
                   "weights/best.pt"
    img_dir = "images"
    source_dir = "dataset/4-6char/val/" + img_dir + "/"
    output_dir = "runs/local/test/" + img_dir + "-" + str(time.mktime(time.localtime())) + "/"
    detect_images(weights_path, source_dir, output_dir)

    # 准确率
    yolo_results_dir = output_dir
    class_file = "runs/local/test/classes.txt"
    output_file = output_dir + "result.txt"
    calculate_acc(yolo_results_dir, class_file, output_file)
    print(f"处理完成，结果已保存到 {output_file}")
