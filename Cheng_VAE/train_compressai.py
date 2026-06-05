import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

try:
    from compressai.models import FactorizedPrior
except ImportError:
    raise ImportError("请先运行: pip install compressai")

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ===== 配置（与你原来保持一致）=====
IMG_SIZE        = 416
BATCH_SIZE      = 16
NUM_WORKERS     = 0
MAX_TRAIN_SAMPLES = 2048
MAX_VAL_SAMPLES   = 512
LOG_EVERY       = 20
DATA_ROOT       = "/content/D-Fire"
EPOCHS          = 30
EARLY_STOP_PATIENCE = 5

# 率失真权衡：λ 越大越重视失真（PSNR），越小越重视压缩率（bpp）
# 推荐先用 0.01 跑通，再试 0.001（更低比特率）或 0.05（更高PSNR）
LAMBDA          = 0.01

LR_MAIN         = 1e-4   # 主网络（encoder + decoder + hyperprior）
LR_AUX          = 1e-3   # 熵模型辅助参数，CompressAI 官方推荐比主网络大一个量级
GRAD_CLIP_NORM  = 1.0

save_dir = "outputs_compressai"
os.makedirs(save_dir, exist_ok=True)

history = {
    "train_loss": [], "val_loss": [],
    "train_psnr": [], "val_psnr": [],
    "train_bpp":  [], "val_bpp":  [],
}

# ===== 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


# ===== 数据集（完全复用你原来的 DFireDataset）=====
# 注意：去掉了 Normalize，CompressAI 期望输入在 [0, 1]
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),          # 输出已在 [0, 1]，不再 Normalize
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
])


class DFireDataset(Dataset):
    """D-Fire: root/{train,test}/images/*.jpg  （原样复用）"""
    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test")
        img_dir = os.path.join(root, split, "images")
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
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


@torch.no_grad()
def batch_psnr(recon, target):
    """原样复用"""
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
                          num_workers=NUM_WORKERS, pin_memory=pin_memory)
val_loader   = DataLoader(val_db,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory)

print(
    f"Train: {len(train_db)} imgs, {len(train_loader)} batches | "
    f"Val: {len(val_db)} imgs, {len(val_loader)} batches | "
    f"{IMG_SIZE}x{IMG_SIZE} bs={BATCH_SIZE}",
    flush=True,
)

# ===== 模型 =====
# quality=3 对应中等比特率，范围 1~8；T4 显存跑 quality<=5 比较稳
# pretrained=False 从头在 D-Fire 上训练
print("Building model...", flush=True)
# N=128 是超先验通道数，M=192 是主潜变量通道数
# 对应原来 quality=3 的参数量，T4 显存完全够用
model = FactorizedPrior(N=128, M=192).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

# ===== 优化器（CompressAI 标准双优化器写法）=====
# main optimizer: encoder + decoder + hyperprior 主体参数
# aux  optimizer: 熵模型内部的 CDF 参数，单独更新
optimizer     = optim.Adam(model.parameters(), lr=LR_MAIN)
aux_optimizer = optim.Adam(model.aux_parameters(), lr=LR_AUX)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
)

# ===== 训练循环 =====
best_psnr      = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    t_epoch = time.time()

    # ---------- Train ----------
    model.train()
    total_loss = total_bpp = total_psnr = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False)
    for step, (imgs, _) in enumerate(pbar, 1):
        imgs = imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        # forward：返回 {"x_hat", "likelihoods": {"y":…, "z":…}}
        out = model(imgs)
        x_hat = out["x_hat"].clamp(0, 1)

        # 比特率估算（bpp）
        num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        bpp = sum(
            (-torch.log2(lk).sum() / num_pixels)
            for lk in out["likelihoods"].values()
        )

        # 失真（MSE，与你原来保持一致；若要换 MS-SSIM 在此处替换）
        distortion = torch.nn.functional.mse_loss(x_hat, imgs)

        # 率失真联合损失
        loss = bpp + LAMBDA * 255**2 * distortion
        # 注：乘以 255^2 是将 MSE 从 [0,1] 域折算到 [0,255] 域，
        #     让 LAMBDA 的量级和文献中保持一致

        loss.backward()
        if GRAD_CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        # 辅助损失（熵模型 CDF 参数），独立 backward + step
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += loss.item()
        total_bpp  += bpp.item()
        total_psnr += batch_psnr(x_hat, imgs)
        n_batches  += 1

        if step % LOG_EVERY == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", bpp=f"{bpp.item():.3f}")

    avg_train_loss = total_loss / n_batches
    avg_train_bpp  = total_bpp  / n_batches
    avg_train_psnr = total_psnr / n_batches

    # ---------- Validation ----------
    model.eval()
    val_loss = val_bpp = val_psnr = 0.0
    n_val = 0

    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            out  = model(imgs)
            x_hat = out["x_hat"].clamp(0, 1)

            num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
            bpp = sum(
                (-torch.log2(lk).sum() / num_pixels)
                for lk in out["likelihoods"].values()
            )
            distortion = torch.nn.functional.mse_loss(x_hat, imgs)
            loss = bpp + LAMBDA * 255**2 * distortion

            val_loss += loss.item()
            val_bpp  += bpp.item()
            val_psnr += batch_psnr(x_hat, imgs)
            n_val    += 1

    avg_val_loss = val_loss / n_val
    avg_val_bpp  = val_bpp  / n_val
    avg_val_psnr = val_psnr / n_val
    elapsed = time.time() - t_epoch

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_psnr"].append(avg_train_psnr)
    history["val_psnr"].append(avg_val_psnr)
    history["train_bpp"].append(avg_train_bpp)
    history["val_bpp"].append(avg_val_bpp)

    scheduler.step(avg_val_psnr)

    print(
        f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) | "
        f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
        f"Train PSNR {avg_train_psnr:.2f} | Val PSNR {avg_val_psnr:.2f} dB | "
        f"Val bpp {avg_val_bpp:.4f}",
        flush=True,
    )

    # ---------- Checkpoint ----------
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
            },
            f"{save_dir}/best_model.pth",
        )
        print(f"  → 保存最优模型 PSNR={best_psnr:.2f} dB  bpp={avg_val_bpp:.4f}", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stop: Val PSNR 连续 {EARLY_STOP_PATIENCE} epoch "
                f"未超过 {best_psnr:.2f} dB",
                flush=True,
            )
            break

# ===== 曲线绘图（原样复用）=====
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

# ===== 率失真散点（每 epoch 一个点）=====
plt.figure()
plt.scatter(history["val_bpp"], history["val_psnr"], c=list(epochs_ran), cmap="viridis")
plt.colorbar(label="Epoch")
plt.xlabel("bpp"); plt.ylabel("PSNR (dB)")
plt.title("Rate-Distortion (Val)"); plt.grid()
plt.savefig(f"{save_dir}/rd_curve.png"); plt.show()

print(f"\n训练完成。最优 Val PSNR: {best_psnr:.2f} dB", flush=True)
