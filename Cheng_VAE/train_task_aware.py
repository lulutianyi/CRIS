"""
任务联合压缩训练脚本
在 Cheng2020Anchor 率失真损失基础上，加入 YOLOv8 检测特征损失
损失函数：L = bpp + λ × 255² × MSE + μ × detection_loss

相较于 train_compressai.py 的改动：
  1. DFireDataset.__getitem__ 增加 bbox 读取
  2. collate_fn 处理变长 bbox
  3. 加载冻结的 YOLOv8n 作为感知损失计算器
  4. 训练循环加入 detection_loss 项
  5. history / 打印 / 绘图增加 det_loss 追踪
  其余（优化器、scheduler、早停、绘图）原样保留
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
EPOCHS            = 30
EARLY_STOP_PATIENCE = 5

LAMBDA  = 0.05    # 率失真权衡，沿用上一次最优配置
MU      = 0.1     # 检测任务损失权重，从 0.1 开始；若 PSNR 下降过多可调小到 0.05

LR_MAIN = 5e-5    # 沿用上一次最优配置
LR_AUX  = 1e-3
GRAD_CLIP_NORM = 1.0

save_dir = "outputs_task_aware"
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

# 加载上一步训练好的最优权重，在此基础上微调
PRETRAINED_CKPT = "outputs_compressai/best_model.pth"
if os.path.exists(PRETRAINED_CKPT):
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载预训练权重：{PRETRAINED_CKPT}（来自 epoch {ckpt['epoch']}，"
          f"PSNR={ckpt['val_psnr']:.2f} dB）", flush=True)
else:
    print(f"未找到预训练权重 {PRETRAINED_CKPT}，从头训练。", flush=True)

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

    这样压缩模型会被引导：在重建图上保留对检测有用的特征，
    而不只是优化像素级 MSE。

    只取前几层特征（浅层）：浅层对应边缘/纹理等低级特征，
    正好是火焰/烟雾检测所需要的，计算量也更小。
    """
    feat_loss = torch.tensor(0.0, device=x_hat.device)
    hooks = []
    feats_hat  = []
    feats_orig = []

    # 注册 hook 捕获第 2、4 层输出（浅层特征）
    def make_hook(store):
        def hook(module, inp, out):
            store.append(out)
        return hook

    target_layers = [2, 4]
    for i, layer in enumerate(yolo_model.model):
        if i in target_layers:
            hooks.append(layer.register_forward_hook(make_hook(feats_hat)))

    with torch.no_grad():
        # 先跑原图，捕获参考特征
        for i, layer in enumerate(yolo_model.model):
            if i in target_layers:
                hooks.append(layer.register_forward_hook(make_hook(feats_orig)))
        _ = yolo_model(x_orig)
        # 移除原图 hook
        for h in hooks[len(target_layers):]:
            h.remove()
        hooks = hooks[:len(target_layers)]

    # 跑重建图
    _ = yolo_model(x_hat)
    for h in hooks:
        h.remove()

    # 计算特征 L2 距离
    for f_hat, f_orig in zip(feats_hat, feats_orig):
        feat_loss = feat_loss + F.mse_loss(f_hat, f_orig.detach())

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
