import difflib


def analyze_log(log_file):
    total_samples = 0  # 符合条件（预测值长度>4）的样本数
    total_chars = 0  # 所有样本的真实值总字符数
    total_correct_chars = 0  # 所有样本的正确字符数之和
    total_correct_pics = 0

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '图片「' not in line or '识别为「' not in line:
                continue

            # 解析图片名（真实值），如 "zWwfA_1751183113.png" → "zWwfA"
            img_part = line.split('图片「')[1].split('」')[0]
            true_value = img_part.split('_')[0]

            # 解析预测值，如 "识别为「zWwA」" → "zWwA"
            recognized_part = line.split('识别为「')[1].split('」')[0]

            # 字符数筛选
            if len(recognized_part) < 3:
                total_samples += 1
                total_chars += len(true_value)

                # 使用 difflib 计算匹配字符数
                matcher = difflib.SequenceMatcher(None, true_value, recognized_part)
                match_blocks = matcher.get_matching_blocks()
                correct_chars = sum(block.size for block in match_blocks)
                total_correct_chars += correct_chars

                if true_value == recognized_part:
                    total_correct_pics += 1


    if total_chars == 0:
        return 0, 0.0

    # 字符级准确率 = 总正确字符数 / 总真实值字符数
    char_accuracy = total_correct_chars / total_chars * 100
    pic_accuracy = total_correct_pics / total_samples * 100
    return total_samples, pic_accuracy, char_accuracy


# 使用示例
log_file = 'runs/local/test/3-4训测6/结果.txt'
total, pic_accuracy, char_accuracy = analyze_log(log_file)

print(f"当前筛选的样本数量: {total}")
print(f"这些样本的准确率: {pic_accuracy:.2f}%")
print(f"字符级准确率: {char_accuracy:.2f}%")