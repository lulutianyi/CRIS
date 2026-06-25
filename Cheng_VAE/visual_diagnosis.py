"""
可视化诊断脚本
从 D-Fire test 集里按类别挑几张代表性图片，
并排显示 原图 / task-aware重建图(bpp=0.42) / low-bpp重建图(bpp=0.18)
用于肉眼判断：mAP 大幅下降是"细节模糊"还是"语义结构丢失"。

运行前提：
  - outputs_task_aware / outputs_low_bpp 权重已加载（沿用之前评估脚本的变量），
    如果是新会话，先重新跑权重加载部分。
  - D-Fire 标注在 train/test 的 labels 目录下（YOLO格式：class cx cy w h）
"""

import os
import glob
import random

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from compressai.models import Cheng2020Anchor

# =====================================================================
# 配置
# =====================================================================
DATA_ROOT  = "/content/D-Fire"
IMG_SIZE   = 384
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_AWARE_CKPT = "/content/cheng2020_task_aware_best.pth"
LOW_BPP_CKPT    = "/content/cheng2020_low_bpp_best.pth"

N_PER_CATEGORY = 2   # 每类挑几张

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
])


# =====================================================================
# 按类别分类图片：fire / smoke / both / none
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

# 随机挑选（固定种子，方便复现）
random.seed(42)
selected_paths = []
selected_labels = []
for cat, paths in categories.items():
    if not paths:
        continue
    chosen = random.sample(paths, min(N_PER_CATEGORY, len(paths)))
    selected_paths.extend(chosen)
    selected_labels.extend([cat] * len(chosen))

print(f"\n共挑选 {len(selected_paths)} 张图片用于可视化对比")


# =====================================================================
# 加载两个压缩模型
# =====================================================================
print("加载压缩模型...", flush=True)
model_ta = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_ta = torch.load(TASK_AWARE_CKPT, map_location=DEVICE)
model_ta.load_state_dict(ckpt_ta["model"])
model_ta.eval()

model_lb = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_lb = torch.load(LOW_BPP_CKPT, map_location=DEVICE)
model_lb.load_state_dict(ckpt_lb["model"])
model_lb.eval()


# =====================================================================
# 对每张图生成两个重建版本，并三联画图
# =====================================================================
@torch.no_grad()
def reconstruct(model, img_tensor):
    out = model(img_tensor.unsqueeze(0).to(DEVICE))
    return out["x_hat"].clamp(0, 1).squeeze(0).cpu()


def to_numpy_img(tensor):
    return tensor.permute(1, 2, 0).numpy()


n = len(selected_paths)
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
if n == 1:
    axes = axes.reshape(1, 3)

for row, (path, label) in enumerate(zip(selected_paths, selected_labels)):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)

    recon_ta = reconstruct(model_ta, img_tensor)
    recon_lb = reconstruct(model_lb, img_tensor)

    stem = os.path.splitext(os.path.basename(path))[0]

    axes[row, 0].imshow(to_numpy_img(img_tensor))
    axes[row, 0].set_title(f"原图 [{label}]\n{stem}", fontsize=9)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(to_numpy_img(recon_ta))
    axes[row, 1].set_title(f"task-aware (bpp=0.42)\nPSNR={ckpt_ta['val_psnr']:.1f}", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(to_numpy_img(recon_lb))
    axes[row, 2].set_title(f"low-bpp (bpp=0.18)\nPSNR={ckpt_lb['val_psnr']:.1f}", fontsize=9)
    axes[row, 2].axis("off")

plt.tight_layout()
out_path = "/content/visual_diagnosis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n已保存对比图到: {out_path}")
print("\n看图时重点关注：")
print("  1. 火焰边缘是否模糊/锯齿化（细节模糊，RDDM能解决）")
print("  2. 烟雾的羽状边界是否变成色块（细节模糊，RDDM能解决）")
print("  3. 是否有目标位置错位、目标消失、或凭空出现伪影（语义结构丢失，RDDM未必能解决）")
print("  4. 'none'类别图片是否被压缩引入了类似烟雾/火焰的伪影（误检风险信号）")
