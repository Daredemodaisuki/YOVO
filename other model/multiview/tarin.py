import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from multiview import MultiViewCAPTCHANet, ctc_decode, decode_predictions
from creat_dataset import CAPTCHADataset
import os
from tqdm import tqdm  # 导入tqdm库


def custom_collate_fn(batch):
    """自定义collate函数，正确处理多视图数据"""
    # 解压批处理数据
    views_list, labels_list, label_lengths = zip(*batch)  # 新增获取标签长度

    # 确定视图数量和图像尺寸
    num_views = views_list[0].shape[0]  # 每个样本有num_views个视图
    C, H, W = views_list[0].shape[1:]  # 通道、高度、宽度

    # 初始化五维张量 [batch_size, num_views, C, H, W]
    batch_size = len(views_list)
    views_tensor = torch.zeros((batch_size, num_views, C, H, W))

    # 填充视图张量
    for i, views in enumerate(views_list):
        # views的形状是 (num_views, C, H, W)
        views_tensor[i] = views

    # 填充标签序列
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=0
    )

    # 将标签长度转换为Tensor
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return views_tensor, labels, label_lengths


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, char_set):
    model.to(device)
    best_val_acc = 0.0

    # 创建字符映射
    char2idx = {char: idx + 1 for idx, char in enumerate(char_set)}
    char2idx[''] = 0
    idx2char = {idx: char for char, idx in char2idx.items()}

    # 使用tqdm包装epoch循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        # 使用tqdm包装训练数据加载器
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=True)
        for views, labels, label_lengths in train_loop:  # 接收标签长度
            views = views.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)  # 实际标签长度

            # 标签长度固定为4改为基于模型输出
            input_lengths = torch.full((views.size(0),), model.get_seq_length(), dtype=torch.long).to(device)
            # 前向传播
            _, loss = model(views, labels, input_lengths, label_lengths)  # 传递两个长度参数

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 更新训练进度条描述
            avg_loss = train_loss / len(train_loop)
            train_loop.set_postfix(loss=f"{avg_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm包装验证数据加载器
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=True)
        with torch.no_grad():
            for views, labels, label_lengths in val_loop:  # 接收标签长度
                views = views.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                # 输入长度
                input_lengths = torch.full((views.size(0),), model.get_seq_length(), dtype=torch.long).to(device)

                # 计算损失
                logits, loss = model(views, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                # 解码预测
                log_probs = F.log_softmax(logits, dim=2)
                predictions = ctc_decode(log_probs)

                # 转换为字符串
                pred_strings = decode_predictions(predictions.cpu(), idx2char)

                # 计算准确率
                for i in range(views.size(0)):
                    true_label = labels[i]
                    # 只取实际长度部分（非填充部分）
                    true_chars = true_label[:label_lengths[i]]
                    true_string = ''.join([idx2char[idx.item()] for idx in true_chars])

                    if pred_strings[i] == true_string:
                        correct += 1
                    total += 1

                # 更新验证进度条描述
                val_loop.set_postfix(acc=f"{correct / total:.2%}" if total else "N/A")

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        # 打印、保存统计信息
        def append_text_to_file(text, file):
            """
            将给定的文本追加到文件的末尾新一行。如果文件或目录不存在，则创建它们。
            """
            os.makedirs(os.path.dirname(file), exist_ok=True)  # 确保目录存在
            with open(file, 'a', encoding='utf-8') as f:
                f.write(text + '\n')

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        append_text_to_file("Epoch{epoch}/{num_epochs}："
                            "Train Loss: {train_loss} | Val Loss: {val_loss}"
                            " | Val Acc: {val_acc}".format(epoch=epoch + 1, num_epochs=num_epochs,
                                                           train_loss=train_loss, val_loss=val_loss, val_acc=val_acc),
                            "other model/multiview/runs/remote/4-real_Ganji_mixed/recording.txt")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       'other model/multiview/runs/remote/4-real_Ganji_mixed/best_model_{epoch}_val-acc{acc}.pth'.format(epoch=epoch + 1,
                                                                                               acc=val_acc))
            print('Saved best model!')

    print(f'\nTraining complete. Best validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    train_dir = "dataset/annotated_Ganji_remixed/train/images"
    val_dir =   "dataset/annotated_Ganji_remixed/val/images"
    batch_size = 16
    epochs = 250
    learning_rate = 0.001

    # 检查目录是否存在
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证目录不存在: {val_dir}")
    print("配置参数:")
    print(f"训练目录: {train_dir}")
    print(f"验证目录: {val_dir}")
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")

    # char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    char_set = "23456789abcdefghklnmprstuvwyz"  # Ganji
    # img_size = (100, 200)  # 图像尺寸
    img_size = (36, 120)  # 图像尺寸
    num_views = 3  # 三种处理视图

    # 准备数据集
    train_dataset = CAPTCHADataset(
        root_dir=train_dir,
        char_set=char_set,
        img_size=img_size
    )

    val_dataset = CAPTCHADataset(
        root_dir=val_dir,
        char_set=char_set,
        img_size=img_size
    )

    # 创建数据加载器 - 使用自定义collate函数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn  # 关键修改
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn  # 关键修改
    )

    # 初始化模型
    model = MultiViewCAPTCHANet(
        num_views=num_views,
        num_chars=train_dataset.num_chars,  # 包括空白符
        img_size=img_size
    )

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=epochs,
        device=device,
        char_set=char_set
    )
