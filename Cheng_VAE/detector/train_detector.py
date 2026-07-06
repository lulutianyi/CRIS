"""
第一步：在 D-Fire 上快速微调 YOLOv8n 检测器
目的：得到一个"能用就行"的检测器，用于后续对比原图/压缩重建图的检测精度差异。
不追求绝对 mAP 数值，只追求三种图像之间误差趋势的相对可比性。

D-Fire 标注格式确认：YOLO txt，class 0=Fire, 1=Smoke
目录结构：/content/D-Fire/{train,test}/{images,labels}
"""

import os
import yaml
from ultralytics import YOLO

DATA_ROOT = "/content/D-Fire"
EPOCHS    = 8          # 快速微调，重点是"能用"而非"最优"
IMG_SIZE  = 384         # 和压缩模型训练时保持一致，三方对比口径统一
BATCH     = 16

# ---------------------------------------------------------------
# D-Fire 官方约定：0=Fire, 1=Smoke
# 如果训练效果异常（比如所有框被识别成同一类），
# 优先检查这里的映射是否与实际标注一致。
# ---------------------------------------------------------------
data_yaml = {
    "path":  DATA_ROOT,
    "train": "train/images",
    "val":   "test/images",     # D-Fire 只有 train/test，没有单独 val，这里复用 test
    "names": {0: "fire", 1: "smoke"},
}

yaml_path = os.path.join(DATA_ROOT, "dfire.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)
print(f"已生成 {yaml_path}")
print(yaml.dump(data_yaml))

# ---------------------------------------------------------------
# 微调（从 COCO 预训练权重开始，而不是随机初始化）
# ---------------------------------------------------------------
model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project="dfire_detector",
    name="yolov8n_dfire_quick",
    patience=3,         # 快速实验，早停更激进一点
    save=True,
    plots=True,
    verbose=True,
)

print("\n训练完成，最优权重路径：")
print("dfire_detector/yolov8n_dfire_quick/weights/best.pt")

# ---------------------------------------------------------------
# 快速验证一下检测器本身的 mAP（在原图上），
# 这个数字作为后续对比实验的"地板参照"
# ---------------------------------------------------------------
best_model = YOLO("dfire_detector/yolov8n_dfire_quick/weights/best.pt")
metrics = best_model.val(data=yaml_path, imgsz=IMG_SIZE)

print(f"\n检测器在原图(D-Fire test集)上的表现：")
print(f"  mAP50    : {metrics.box.map50:.4f}")
print(f"  mAP50-95 : {metrics.box.map:.4f}")
