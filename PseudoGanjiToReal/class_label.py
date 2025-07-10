import os

img_dir = "mix_set"
txt_dir = "mix_set"
classes = ['2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', ]
imgs = os.listdir(img_dir)
imgs = [name for name in imgs if any(name.lower().endswith(f".{ext}") for ext in ["png", "jpg"])]

for img in imgs:
    label, _ = os.path.splitext(img)
    if os.path.exists(os.path.join(txt_dir, label + ".txt")):
        # 读取bbox
        bboxes = []
        with open(os.path.join(txt_dir, label + ".txt"), 'r') as txt:
            for line in txt:
                data = line.strip().split()
                if len(data) >= 5:
                    bboxes.append(data[1:5])
                else:
                    print(txt_dir, label + ".txt" + "：长度不足")
                    continue
            if len(bboxes) == 0:
                raise KeyError(txt_dir, label + ".txt" + "：文件bbox无效")
        # 写clss
        with open(os.path.join(txt_dir, label + ".txt"), 'w') as txt:
            annotation = []
            for i in range(len(label)):
                print(label[i] + "：" + str(classes.index(label[i])) + " " + " ".join(bboxes[i]) + "\n")
                txt.write(str(classes.index(label[i])) + " " + " ".join(bboxes[i]) + "\n")

        print(label)
    