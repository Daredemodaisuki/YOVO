import torch

# 定义字符集 (数字+大写字母)
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
NUM_CHARS = len(CHARSET)


# CTC解码函数
def ctc_decode(probs, charset, blank_index=None):
    if blank_index is None:
        blank_index = len(charset)

    # 贪婪解码: 取每个时间步最大概率的字符
    _, max_indices = torch.max(probs, dim=2)
    max_indices = max_indices.cpu().numpy()

    decoded_texts = []
    for i in range(max_indices.shape[0]):
        raw_pred = []
        for t in range(max_indices.shape[1]):
            if max_indices[i, t] != blank_index and (t == 0 or max_indices[i, t] != max_indices[i, t - 1]):
                raw_pred.append(max_indices[i, t])
        decoded = ''.join([charset[idx] for idx in raw_pred])
        decoded_texts.append(decoded)

    return decoded_texts


# 计算准确率
def calculate_accuracy(pred_texts, true_texts):
    correct = 0
    for pred, true in zip(pred_texts, true_texts):
        if pred == true:
            correct += 1
    return correct / len(true_texts) if true_texts else 0