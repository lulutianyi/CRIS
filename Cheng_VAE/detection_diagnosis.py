"""
检测框可视化诊断脚本
在 visual_diagnosis.py 基础上，加上 YOLOv8 检测器的实际推理结果，
把预测框和置信度画在图上，三联对比：原图 / task-aware重建图 / low-bpp重建图

目的：区分两种情况——
  (a) 完全漏检（框消失）       → 检测器对压缩失真完全失效，问题更严重
  (b) 检测到但置信度降低/框变松 → 压缩只是削弱了信号，RDDM补细节的思路更有希望见效

运行前提：
  - 已经跑过 train_detector.py，得到检测器权重
  - 已经跑过两个压缩模型的训练，得到 task-aware / low-bpp 权重
"""

import os
import glob
import random

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms

from compressai.models import Cheng2020Anchor
from ultralytics import YOLO

# =====================================================================
# 配置
# =====================================================================
DATA_ROOT  = "/content/D-Fire"
IMG_SIZE   = 384
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTOR_CKPT   = "/content/runs/detect/dfire_detector/yolov8n_dfire_quick/weights/best.pt"
TASK_AWARE_CKPT = "/content/cheng2020_task_aware_best.pth"
LOW_BPP_CKPT    = "/content/cheng2020_low_bpp_best.pth"

N_PER_CATEGORY  = 2
CONF_THRESHOLD  = 0.25     # 和 YOLO val 默认一致，低于此置信度的框不显示
CLASS_NAMES     = {0: "fire", 1: "smoke"}
CLASS_COLORS    = {0: "red", 1: "gray"}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
])


# =====================================================================
# 按类别分类图片（与 visual_diagnosis.py 相同逻辑）
# =====================================================================
def categorize_images(root, split="test"):
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.png"))
    )

    categories = {"fire": [], "smoke": [], "both": [], "none": []}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        classes = set()
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        classes.add(int(parts[0]))
        if not classes:
            categories["none"].append(p)
        elif classes == {0}:
            categories["fire"].append(p)
        elif classes == {1}:
            categories["smoke"].append(p)
        else:
            categories["both"].append(p)
    return categories


print("正在扫描并分类图片...", flush=True)
categories = categorize_images(DATA_ROOT)
for k, v in categories.items():
    print(f"  {k}: {len(v)} 张")

random.seed(42)   # 固定种子，和上次可视化挑同一批图，方便对照
selected_paths = []
selected_labels = []
for cat, paths in categories.items():
    if not paths:
        continue
    chosen = random.sample(paths, min(N_PER_CATEGORY, len(paths)))
    selected_paths.extend(chosen)
    selected_labels.extend([cat] * len(chosen))

print(f"\n共挑选 {len(selected_paths)} 张图片")


# =====================================================================
# 加载模型：检测器 + 两个压缩模型
# =====================================================================
print("加载检测器...", flush=True)
detector = YOLO(DETECTOR_CKPT)

print("加载压缩模型...", flush=True)
model_ta = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_ta = torch.load(TASK_AWARE_CKPT, map_location=DEVICE)
model_ta.load_state_dict(ckpt_ta["model"])
model_ta.eval()

model_lb = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_lb = torch.load(LOW_BPP_CKPT, map_location=DEVICE)
model_lb.load_state_dict(ckpt_lb["model"])
model_lb.eval()


@torch.no_grad()
def reconstruct(model, img_tensor):
    out = model(img_tensor.unsqueeze(0).to(DEVICE))
    return out["x_hat"].clamp(0, 1).squeeze(0).cpu()


def to_numpy_img(tensor):
    return tensor.permute(1, 2, 0).numpy()


def draw_detections(ax, img_np, img_tensor, title):
    """在给定 axes 上画图，并叠加检测器的预测框"""
    ax.imshow(img_np)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    # YOLO 推理：直接传 tensor 即可，内部会处理归一化等
    results = detector.predict(
        img_tensor.unsqueeze(0).to(DEVICE),
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    n_boxes = 0
    if results.boxes is not None and len(results.boxes) > 0:
        h, w = img_np.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 已经是像素坐标（基于输入图尺寸）

            color = CLASS_COLORS.get(cls_id, "yellow")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(y1 - 4, 0),
                f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}",
                color=color, fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
            )
            n_boxes += 1

    return n_boxes


# =====================================================================
# 主循环：每张图三联画图 + 打印检测框数量对比
# =====================================================================
n = len(selected_paths)
fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
if n == 1:
    axes = axes.reshape(1, 3)

print("\n" + "=" * 70)
print(f"{'图片':<14}{'类别':<8}{'原图框数':>10}{'task-aware框数':>16}{'low-bpp框数':>14}")
print("=" * 70)

for row, (path, label) in enumerate(zip(selected_paths, selected_labels)):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)

    recon_ta = reconstruct(model_ta, img_tensor)
    recon_lb = reconstruct(model_lb, img_tensor)

    stem = os.path.splitext(os.path.basename(path))[0]

    n_orig = draw_detections(
        axes[row, 0], to_numpy_img(img_tensor), img_tensor,
        f"原图 [{label}]\n{stem}"
    )
    n_ta = draw_detections(
        axes[row, 1], to_numpy_img(recon_ta), recon_ta,
        f"task-aware (bpp=0.42)"
    )
    n_lb = draw_detections(
        axes[row, 2], to_numpy_img(recon_lb), recon_lb,
        f"low-bpp (bpp=0.18)"
    )

    print(f"{stem:<14}{label:<8}{n_orig:>10}{n_ta:>16}{n_lb:>14}")

print("=" * 70)

plt.tight_layout()
out_path = "/content/detection_diagnosis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n已保存检测框对比图到: {out_path}")
print("\n判断参考：")
print("  框数量相同但置信度明显降低 → 信号被削弱但结构保留，RDDM思路有希望")
print("  框数量减少（漏检）        → 部分目标特征已被压缩破坏到检测器无法识别")
print("  框数量增加（误检）        → 压缩引入的伪影被误判为目标，需警惕")
