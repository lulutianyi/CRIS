"""
低 bpp 微调脚本 —— RDDM 实验第一步
在已训练好的 task-aware Cheng2020 权重基础上，调小 λ 继续微调，
把率失真平衡点推到 bpp≈0.15~0.2 的极低码率区间。

相较于 train_task_aware.py 的改动：
  1. LAMBDA 调小（0.05 → 0.008），目标 bpp 落在 0.15~0.2
  2. 加载 outputs_task_aware/best_model.pth 作为起点（而非从头训练）
  3. 修复了 detection_feature_loss 的 hook bug
     （原版原图/重建图特征被混在同一个 list 里，导致 det_loss 恒为 0）
  4. EPOCHS 调小（微调不需要 30 epoch 那么久）
  其余（数据集、优化器结构、绘图）原样保留
"""

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from compressai.models import Cheng2020Anchor

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ===== 依赖检查 =====
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先运行: pip install ultralytics")

# =====================================================================
# 配置
# =====================================================================
IMG_SIZE          = 384       # Cheng2020 要求 64 的倍数
BATCH_SIZE        = 8         # 加了 YOLO 后显存增加，从 16 降到 8
NUM_WORKERS       = 0
MAX_TRAIN_SAMPLES = 2048
MAX_VAL_SAMPLES   = 512
LOG_EVERY         = 20
DATA_ROOT         = "/content/D-Fire"
EPOCHS            = 15   # 微调不需要从头训练那么久
EARLY_STOP_PATIENCE = 5

# 目标：把 bpp 从 0.42 推到 0.15~0.2。
# 参考你之前的曲线：λ=0.05→bpp 0.42，λ=0.01→bpp~0.3，
# 所以这次需要比 0.01 更小，从 0.008 起步，如果第一轮 bpp 仍 >0.2 可再调小到 0.004。
LAMBDA  = 0.008
MU      = 0.1     # 检测损失权重不变，继续保留检测感知能力

LR_MAIN = 2e-5    # 微调用更小的学习率，避免把已学到的特征破坏掉
LR_AUX  = 5e-4
GRAD_CLIP_NORM = 1.0

save_dir = "outputs_low_bpp"
os.makedirs(save_dir, exist_ok=True)

history = {
    "train_loss": [], "val_loss": [],
    "train_psnr": [], "val_psnr": [],
    "train_bpp":  [], "val_bpp":  [],
    "train_det":  [], "val_det":  [],   # 新增：检测损失追踪
}

# =====================================================================
# 设备
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# =====================================================================
# 数据集
# 改动：__getitem__ 增加 bbox 读取（YOLO 格式 txt）
# =====================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
])


class DFireDataset(Dataset):
    """
    D-Fire: root/{train,test}/images/*.jpg
           root/{train,test}/labels/*.txt   ← YOLO 格式：class cx cy w h（归一化）
    若某张图没有对应 label 文件（负样本），boxes 返回空 tensor。
    """
    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test")
        img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        if not self.paths:
            raise FileNotFoundError(f"在 {img_dir} 下找不到图片")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 读取对应 label（可能不存在）
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, stem + ".txt")
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # YOLO格式：class cx cy w h，全部归一化到 [0,1]
                        boxes.append([float(p) for p in parts])
        boxes = torch.tensor(boxes, dtype=torch.float32)  # [N, 5] 或 [0, 5]
        return img, boxes


def collate_fn(batch):
    """
    DataLoader 默认 collate 无法处理变长 bbox，需要自定义。
    imgs:  [B, 3, H, W]
    boxes: list of [N_i, 5]（每张图的 bbox 数量不同）
    """
    imgs, boxes = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(boxes)


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


@torch.no_grad()
def batch_psnr(recon, target):
    mse = (recon - target).pow(2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-8))).mean().item()


print("Loading dataset...", flush=True)
train_db = maybe_subset(
    DFireDataset(DATA_ROOT, split="train", transform=train_transform),
    MAX_TRAIN_SAMPLES,
)
val_db = maybe_subset(
    DFireDataset(DATA_ROOT, split="test", transform=val_transform),
    MAX_VAL_SAMPLES,
)

pin_memory = device.type == "cuda"
train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory,
                          collate_fn=collate_fn)
val_loader   = DataLoader(val_db,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory,
                          collate_fn=collate_fn)

print(
    f"Train: {len(train_db)} imgs | Val: {len(val_db)} imgs | "
    f"{IMG_SIZE}x{IMG_SIZE} bs={BATCH_SIZE}",
    flush=True,
)

# =====================================================================
# 压缩模型
# =====================================================================
print("Building compression model...", flush=True)
model = Cheng2020Anchor(N=128).to(device)

# ⚠️ 重点：必须加载 task-aware 权重（带检测损失训练出来的），
# 不是纯压缩基线 outputs_compressai/best_model.pth。
# 如果你按之前的建议把权重存到了 Drive，先挂载 Drive 再改这里的路径，例如：
#   from google.colab import drive; drive.mount('/content/drive')
#   PRETRAINED_CKPT = "/content/drive/MyDrive/dfire_checkpoints/cheng2020_task_aware_best.pth"
PRETRAINED_CKPT = "outputs_task_aware/best_model.pth"
if os.path.exists(PRETRAINED_CKPT):
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载 task-aware 权重：{PRETRAINED_CKPT}（来自 epoch {ckpt['epoch']}，"
          f"PSNR={ckpt['val_psnr']:.2f} dB, bpp={ckpt.get('val_bpp', float('nan')):.4f}）",
          flush=True)
else:
    raise FileNotFoundError(
        f"找不到 {PRETRAINED_CKPT}。这一步要求必须基于 task-aware 权重微调，"
        f"请确认路径，或从 Drive 拷回 /content 后再运行。"
    )

# =====================================================================
# YOLOv8 感知损失计算器
# 冻结权重，只用中间特征来计算检测损失信号
# =====================================================================
print("Loading YOLOv8n feature extractor...", flush=True)
yolo = YOLO("yolov8n.pt")          # 首次运行会自动下载，约 6 MB
yolo_model = yolo.model.to(device)
yolo_model.eval()
for p in yolo_model.parameters():  # 完全冻结，不参与梯度更新
    p.requires_grad = False


def detection_feature_loss(x_hat, x_orig):
    """
    用 YOLOv8 backbone 提取重建图和原图的中间特征，
    计算 L2 特征距离作为检测感知损失。

    [修复] 原版用两个 list 分别 append，但两次 forward 共用同一组 hook，
    导致 feats_hat 实际存的是两次 forward 的混合结果，距离恒为 0。
    这里改成按层索引存入字典，并将两次 forward 的 hook 完全分开注册/移除。

    只取第 2、4 层（浅层特征）：对应边缘/纹理等低级特征，
    正好是火焰/烟雾检测所需要的，计算量也更小。
    """
    target_layers = [2, 4]

    def make_hook(store, key):
        def hook(module, inp, out):
            store[key] = out
        return hook

    # ---- 第一次 forward：重建图，需要保留梯度 ----
    feats_hat = {}
    hooks = []
    for i, layer in enumerate(yolo_model.model):
        if i in target_layers:
            hooks.append(layer.register_forward_hook(make_hook(feats_hat, i)))
    _ = yolo_model(x_hat)
    for h in hooks:
        h.remove()

    # ---- 第二次 forward：原图，仅作参考，不需要梯度 ----
    feats_orig = {}
    hooks = []
    for i, layer in enumerate(yolo_model.model):
        if i in target_layers:
            hooks.append(layer.register_forward_hook(make_hook(feats_orig, i)))
    with torch.no_grad():
        _ = yolo_model(x_orig)
    for h in hooks:
        h.remove()

    feat_loss = torch.tensor(0.0, device=x_hat.device)
    for k in target_layers:
        feat_loss = feat_loss + F.mse_loss(feats_hat[k], feats_orig[k].detach())

    return feat_loss / len(target_layers)


# =====================================================================
# 优化器（原样保留双优化器结构）
# =====================================================================
optimizer     = optim.Adam(model.parameters(), lr=LR_MAIN)
aux_optimizer = optim.Adam(model.entropy_bottleneck.parameters(), lr=LR_AUX)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
)

# =====================================================================
# 训练循环
# =====================================================================
best_psnr         = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    t_epoch = time.time()

    # ---------- Train ----------
    model.train()
    total_loss = total_bpp = total_psnr = total_det = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False)
    for step, (imgs, boxes) in enumerate(pbar, 1):
        imgs = imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        out   = model(imgs)
        x_hat = out["x_hat"].clamp(0, 1)

        num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        bpp = sum(
            (-torch.log2(lk).sum() / num_pixels)
            for lk in out["likelihoods"].values()
        )

        distortion  = F.mse_loss(x_hat, imgs)
        det_loss    = detection_feature_loss(x_hat, imgs)

        # 联合损失：率失真 + 检测感知
        loss = bpp + LAMBDA * 255**2 * distortion + MU * det_loss

        loss.backward()
        if GRAD_CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        aux_loss = model.entropy_bottleneck.loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += loss.item()
        total_bpp  += bpp.item()
        total_det  += det_loss.item()
        with torch.no_grad():
            total_psnr += batch_psnr(x_hat, imgs)
        n_batches += 1

        if step % LOG_EVERY == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bpp=f"{bpp.item():.3f}",
                det=f"{det_loss.item():.4f}",
            )

    avg_train_loss = total_loss / n_batches
    avg_train_bpp  = total_bpp  / n_batches
    avg_train_psnr = total_psnr / n_batches
    avg_train_det  = total_det  / n_batches

    # ---------- Validation ----------
    model.eval()
    val_loss = val_bpp = val_psnr = val_det = 0.0
    n_val = 0

    with torch.no_grad():
        for imgs, boxes in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            out   = model(imgs)
            x_hat = out["x_hat"].clamp(0, 1)

            num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
            bpp = sum(
                (-torch.log2(lk).sum() / num_pixels)
                for lk in out["likelihoods"].values()
            )
            distortion = F.mse_loss(x_hat, imgs)
            det_loss   = detection_feature_loss(x_hat, imgs)
            loss       = bpp + LAMBDA * 255**2 * distortion + MU * det_loss

            val_loss += loss.item()
            val_bpp  += bpp.item()
            val_psnr += batch_psnr(x_hat, imgs)
            val_det  += det_loss.item()
            n_val    += 1

    avg_val_loss = val_loss / n_val
    avg_val_bpp  = val_bpp  / n_val
    avg_val_psnr = val_psnr / n_val
    avg_val_det  = val_det  / n_val
    elapsed = time.time() - t_epoch

    for k, v in zip(
        ["train_loss","val_loss","train_psnr","val_psnr",
         "train_bpp","val_bpp","train_det","val_det"],
        [avg_train_loss, avg_val_loss, avg_train_psnr, avg_val_psnr,
         avg_train_bpp,  avg_val_bpp,  avg_train_det,  avg_val_det],
    ):
        history[k].append(v)

    scheduler.step(avg_val_psnr)

    print(
        f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) | "
        f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
        f"Train PSNR {avg_train_psnr:.2f} | Val PSNR {avg_val_psnr:.2f} dB | "
        f"Val bpp {avg_val_bpp:.4f} | Val det {avg_val_det:.4f}",
        flush=True,
    )

    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        epochs_no_improve = 0
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_psnr": avg_val_psnr,
                "val_bpp":  avg_val_bpp,
                "val_det":  avg_val_det,
            },
            f"{save_dir}/best_model.pth",
        )
        print(f"  → 保存最优模型 PSNR={best_psnr:.2f} dB  "
              f"bpp={avg_val_bpp:.4f}  det={avg_val_det:.4f}", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stop: Val PSNR 连续 {EARLY_STOP_PATIENCE} epoch "
                f"未超过 {best_psnr:.2f} dB",
                flush=True,
            )
            break

# =====================================================================
# 绘图（在原有三张图基础上增加检测损失曲线）
# =====================================================================
epochs_ran = range(1, len(history["train_loss"]) + 1)

plt.figure()
plt.plot(epochs_ran, history["train_loss"], label="Train Loss")
plt.plot(epochs_ran, history["val_loss"],   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curve"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/loss_curve.png"); plt.show()

plt.figure()
plt.plot(epochs_ran, history["train_psnr"], label="Train PSNR")
plt.plot(epochs_ran, history["val_psnr"],   label="Val PSNR")
plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
plt.title("PSNR Curve"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/psnr_curve.png"); plt.show()

plt.figure()
plt.plot(epochs_ran, history["train_bpp"], label="Train bpp")
plt.plot(epochs_ran, history["val_bpp"],   label="Val bpp")
plt.xlabel("Epoch"); plt.ylabel("bpp")
plt.title("Bitrate Curve"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/bpp_curve.png"); plt.show()

# 新增：检测感知损失曲线
plt.figure()
plt.plot(epochs_ran, history["train_det"], label="Train Det Loss")
plt.plot(epochs_ran, history["val_det"],   label="Val Det Loss")
plt.xlabel("Epoch"); plt.ylabel("Detection Feature Loss")
plt.title("Detection Loss Curve"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/det_loss_curve.png"); plt.show()

plt.figure()
plt.scatter(history["val_bpp"], history["val_psnr"],
            c=list(epochs_ran), cmap="viridis")
plt.colorbar(label="Epoch")
plt.xlabel("bpp"); plt.ylabel("PSNR (dB)")
plt.title("Rate-Distortion (Val)"); plt.grid()
plt.savefig(f"{save_dir}/rd_curve.png"); plt.show()

print(f"\n训练完成。最优 Val PSNR: {best_psnr:.2f} dB", flush=True)
