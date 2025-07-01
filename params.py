from ultralytics import YOLO

# 加载 YAML 结构并初始化（YOLOv8支持直接用YAML构建）
model = YOLO('yaml/yolov8n (nc=62) - 去大核（更正）.yaml')  # 或你的自定义yaml路径

# 打印参数总量
n_params = sum(p.numel() for p in model.model.parameters())
print(f'Total parameters: {n_params / 1e6:.6f}M')
