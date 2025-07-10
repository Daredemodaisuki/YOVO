import os

img_dir = "/workspace/dataset/annotated_Pseudo_Ganji_4char_2(after_round3_final)/train/images"
txt_dir = "/workspace/dataset/annotated_Pseudo_Ganji_4char_2(after_round3_final)/train/labels"
# classes = ['2', '3', '4', '5', '6', '7', '8', '9',
#            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', ]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]
imgs = os.listdir(img_dir)
imgs = [name for name in imgs if any(name.lower().endswith(f".{ext}") for ext in ["png", "jpg"])]

for img in imgs:
    txt_name, _ = os.path.splitext(img)
    label = txt_name.split("_")[0].lower()
    if os.path.exists(os.path.join(txt_dir, txt_name + ".txt")):
        # 读取bbox
        bboxes = []
        with open(os.path.join(txt_dir, txt_name + ".txt"), 'r') as txt:
            for line in txt:
                data = line.strip().split()
                if len(data) >= 5:
                    bboxes.append(data[1:5])
                else:
                    print(txt_dir, txt_name + ".txt" + "：长度不足")
                    continue
            if len(bboxes) == 0:
                raise KeyError(txt_dir, txt_name + ".txt" + "：文件bbox无效")
        # 写clss
        with open(os.path.join(txt_dir, txt_name + ".txt"), 'w') as txt:
            annotation = []
            for i in range(len(label)):
                print(label[i] + "：" + str(classes.index(label[i])) + " " + " ".join(bboxes[i]))
                txt.write(str(classes.index(label[i])) + " " + " ".join(bboxes[i]) + "\n")

        print(txt_name, label)
    