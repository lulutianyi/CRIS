"""
RDDM 数据对生成脚本 —— 第二步

用 task-aware (bpp≈0.42) 和 low-bpp (bpp≈0.18) 两个压缩模型，
分别在同一批图片上做重建，保存 (原图, 重建图) 配对，作为 RDDM 的训练数据。

RDDM 学习目标是从重建图预测残差 r = 原图 - 重建图，
所以这里把两者都原样保存（不计算残差），残差留给 RDDM 训练脚本自己算，
这样数据对更通用，也方便后续检查重建质量本身。

输出目录结构：
  /content/rddm_data/
    task_aware/
      train/orig/xxx.png      train/recon/xxx.png
      val/orig/xxx.png        val/recon/xxx.png
    low_bpp/
      train/orig/xxx.png      train/recon/xxx.png
      val/orig/xxx.png        val/recon/xxx.png

样本范围：与之前训练压缩模型时一致——
  train: D-Fire train/images 前 2048 张
  val:   D-Fire test/images  前 512 张
"""

import os
import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image

from compressai.models import Cheng2020Anchor

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# =====================================================================
# 配置（与之前压缩模型训练保持一致，方便结果可比）
# =====================================================================
DATA_ROOT  = "/content/D-Fire"
IMG_SIZE   = 384
BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TRAIN_SAMPLES = 2048
MAX_VAL_SAMPLES   = 512

TASK_AWARE_CKPT = "/content/cheng2020_task_aware_best.pth"
LOW_BPP_CKPT    = "/content/cheng2020_low_bpp_best.pth"

OUT_ROOT = "/content/rddm_data"

# 两个模型分别要跑一遍，配置放一起方便遍历
MODEL_CONFIGS = [
    {"name": "task_aware", "ckpt": TASK_AWARE_CKPT},
    {"name": "low_bpp",    "ckpt": LOW_BPP_CKPT},
]


# =====================================================================
# 数据集：不做随机增强（生成数据对不需要翻转/裁剪），保证和原图严格对应
# =====================================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
])


class DFireImageOnly(Dataset):
    """与之前 DFireDataset 一致的目录假设，这里只需要图片，不需要标注"""
    def __init__(self, root, split):
        img_dir = os.path.join(root, split, "images")
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        if not self.paths:
            raise FileNotFoundError(f"在 {img_dir} 下找不到图片")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = transform(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        return img, stem


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


print(f"Device: {DEVICE}", flush=True)

print("加载数据集...", flush=True)
train_db = maybe_subset(DFireImageOnly(DATA_ROOT, "train"), MAX_TRAIN_SAMPLES)
val_db   = maybe_subset(DFireImageOnly(DATA_ROOT, "test"),  MAX_VAL_SAMPLES)

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_db,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train: {len(train_db)} 张 | Val: {len(val_db)} 张", flush=True)


# =====================================================================
# 核心函数：用给定压缩模型对一个 loader 跑重建，保存 (orig, recon) 配对
# =====================================================================
@torch.no_grad()
def generate_pairs(model, loader, out_dir, split_name):
    orig_dir  = os.path.join(out_dir, split_name, "orig")
    recon_dir = os.path.join(out_dir, split_name, "recon")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    model.eval()
    saved = 0
    for imgs, stems in tqdm(loader, desc=f"生成 {split_name} 数据对"):
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        x_hat = out["x_hat"].clamp(0, 1)

        for i, stem in enumerate(stems):
            # 原图和重建图用同名文件保存在不同目录，方便 RDDM 训练时按文件名配对
            save_image(imgs[i].cpu(),  os.path.join(orig_dir,  f"{stem}.png"))
            save_image(x_hat[i].cpu(), os.path.join(recon_dir, f"{stem}.png"))
            saved += 1

    print(f"  已保存 {saved} 对图片到 {out_dir}/{split_name}/", flush=True)
    return saved


# =====================================================================
# 主流程：对两个压缩模型分别生成 train/val 数据对
# =====================================================================
for cfg in MODEL_CONFIGS:
    name = cfg["name"]
    ckpt_path = cfg["ckpt"]

    print(f"\n{'=' * 60}")
    print(f"模型: {name}  ({ckpt_path})")
    print(f"{'=' * 60}", flush=True)

    if not os.path.exists(ckpt_path):
        print(f"  [跳过] 找不到权重文件: {ckpt_path}", flush=True)
        continue

    model = Cheng2020Anchor(N=128).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"  已加载权重 (PSNR={ckpt['val_psnr']:.2f} dB, "
          f"bpp={ckpt.get('val_bpp', float('nan')):.4f})", flush=True)

    out_dir = os.path.join(OUT_ROOT, name)

    generate_pairs(model, train_loader, out_dir, "train")
    generate_pairs(model, val_loader,   out_dir, "val")

    del model
    torch.cuda.empty_cache()

print("\n全部完成。数据对目录结构：")
for cfg in MODEL_CONFIGS:
    name = cfg["name"]
    base = os.path.join(OUT_ROOT, name)
    if os.path.exists(base):
        for split in ["train", "val"]:
            orig_dir = os.path.join(base, split, "orig")
            if os.path.exists(orig_dir):
                print(f"  {base}/{split}/  — {len(os.listdir(orig_dir))} 对")

print(f"\n下一步：用 {OUT_ROOT}/<task_aware|low_bpp>/train/{{orig,recon}} "
      f"替换 RDDM 仓库（nachifur/RDDM）的数据接口，"
      f"训练目标是从 recon 预测残差 r = orig - recon。")
