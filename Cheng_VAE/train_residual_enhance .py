"""
简化版残差增强网络 —— RDDM 思路的单步实现

不是完整复现 RDDM 论文（不含多步扩散采样、不含噪声扩散分支），
只保留核心思想：用一个网络直接学习 r = orig - recon，
训练目标：L1(预测残差, 真实残差)
推理：enhanced = recon + 预测残差

这样做的好处：训练快、参数少、不依赖原仓库复杂的参数耦合，
代价：失去了扩散模型多步迭代精化和生成多样性的能力，
本质上等价于一个轻量级图像增强/去伪影网络。

数据来源：generate_rddm_pairs.py 生成的 (orig, recon) 配对
  /content/rddm_data/<task_aware|low_bpp>/train/{orig,recon}
  /content/rddm_data/<task_aware|low_bpp>/val/{orig,recon}
"""

import os
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# =====================================================================
# 配置
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择对哪个压缩模型的重建图做残差增强：
# "low_bpp" 问题更严重(bpp≈0.18)，"task_aware" 损失较小(bpp≈0.42)
# 建议先用 low_bpp，因为这是检测精度损失最大、RDDM最有价值的场景
TARGET = "low_bpp"   # 改成 "task_aware" 可以训另一套

DATA_ROOT  = f"/content/rddm_data/{TARGET}"
IMG_SIZE   = 384
BATCH_SIZE = 8        # UNet 比压缩模型更吃显存，T4 上从 8 起步
NUM_WORKERS = 0
EPOCHS     = 30
EARLY_STOP_PATIENCE = 6

LR = 1e-4
GRAD_CLIP_NORM = 1.0

save_dir = f"outputs_residual_enhance_{TARGET}"
os.makedirs(save_dir, exist_ok=True)

history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}


# =====================================================================
# 数据集：直接读 generate_rddm_pairs.py 产出的 orig/recon 配对
# =====================================================================
to_tensor = transforms.ToTensor()  # 图片已经是 384x384 PNG，不需要再 Resize


class ResidualPairDataset(Dataset):
    def __init__(self, root, split):
        self.orig_dir  = os.path.join(root, split, "orig")
        self.recon_dir = os.path.join(root, split, "recon")
        self.stems = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.orig_dir, "*.png"))
        )
        if not self.stems:
            raise FileNotFoundError(f"在 {self.orig_dir} 下找不到图片，请先运行 generate_rddm_pairs.py")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        orig  = to_tensor(Image.open(os.path.join(self.orig_dir,  f"{stem}.png")).convert("RGB"))
        recon = to_tensor(Image.open(os.path.join(self.recon_dir, f"{stem}.png")).convert("RGB"))
        return orig, recon


@torch.no_grad()
def batch_psnr(recon, target):
    mse = (recon - target).pow(2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-8))).mean().item()


print(f"Device: {DEVICE}", flush=True)
print(f"目标压缩模型: {TARGET}", flush=True)

train_db = ResidualPairDataset(DATA_ROOT, "train")
val_db   = ResidualPairDataset(DATA_ROOT, "val")

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
val_loader   = DataLoader(val_db,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

print(f"Train: {len(train_db)} 对 | Val: {len(val_db)} 对", flush=True)


# =====================================================================
# 模型：轻量 UNet，输入 recon，输出残差预测
# 比压缩模型的 encoder/decoder 更浅，因为任务更简单（只是局部纹理修复，
# 不需要重新学习整个压缩/解压映射）
# =====================================================================
# 模型定义已抽到独立文件，避免被其他脚本 import 时触发重复训练
from residual_unet_model import ResidualUNet


print("Building model...", flush=True)
model = ResidualUNet(base_ch=32).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_params / 1e6:.1f}M", flush=True)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
)

# =====================================================================
# 训练循环
# =====================================================================
best_psnr = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    t_epoch = time.time()

    # ---------- Train ----------
    model.train()
    total_loss = total_psnr = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False)
    for orig, recon in pbar:
        orig, recon = orig.to(DEVICE), recon.to(DEVICE)
        target_residual = orig - recon

        optimizer.zero_grad(set_to_none=True)
        pred_residual = model(recon)

        loss = F.l1_loss(pred_residual, target_residual)
        loss.backward()
        if GRAD_CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        with torch.no_grad():
            enhanced = (recon + pred_residual).clamp(0, 1)
            psnr = batch_psnr(enhanced, orig)

        total_loss += loss.item()
        total_psnr += psnr
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")

    avg_train_loss = total_loss / n_batches
    avg_train_psnr = total_psnr / n_batches

    # ---------- Validation ----------
    model.eval()
    val_loss = val_psnr = 0.0
    val_psnr_baseline = 0.0   # 不经过增强网络，recon本身相对orig的PSNR，作为对照
    n_val = 0

    with torch.no_grad():
        for orig, recon in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
            orig, recon = orig.to(DEVICE), recon.to(DEVICE)
            target_residual = orig - recon

            pred_residual = model(recon)
            loss = F.l1_loss(pred_residual, target_residual)

            enhanced = (recon + pred_residual).clamp(0, 1)

            val_loss += loss.item()
            val_psnr += batch_psnr(enhanced, orig)
            val_psnr_baseline += batch_psnr(recon, orig)
            n_val += 1

    avg_val_loss = val_loss / n_val
    avg_val_psnr = val_psnr / n_val
    avg_val_psnr_baseline = val_psnr_baseline / n_val
    elapsed = time.time() - t_epoch

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_psnr"].append(avg_train_psnr)
    history["val_psnr"].append(avg_val_psnr)

    scheduler.step(avg_val_psnr)

    print(
        f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) | "
        f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
        f"Train PSNR {avg_train_psnr:.2f} | Val PSNR(enhanced) {avg_val_psnr:.2f} dB | "
        f"Val PSNR(recon,baseline) {avg_val_psnr_baseline:.2f} dB | "
        f"提升 {avg_val_psnr - avg_val_psnr_baseline:+.2f} dB",
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
                "val_psnr_baseline": avg_val_psnr_baseline,
                "target": TARGET,
            },
            f"{save_dir}/best_model.pth",
        )
        print(f"  → 保存最优模型 PSNR={best_psnr:.2f} dB "
              f"(相比未增强提升 {best_psnr - avg_val_psnr_baseline:+.2f} dB)", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop: Val PSNR 连续 {EARLY_STOP_PATIENCE} epoch 未超过 {best_psnr:.2f} dB", flush=True)
            break

# =====================================================================
# 绘图
# =====================================================================
epochs_ran = range(1, len(history["train_loss"]) + 1)

plt.figure()
plt.plot(epochs_ran, history["train_loss"], label="Train L1 Loss")
plt.plot(epochs_ran, history["val_loss"],   label="Val L1 Loss")
plt.xlabel("Epoch"); plt.ylabel("L1 Loss")
plt.title(f"Residual Loss ({TARGET})"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/loss_curve.png"); plt.show()

plt.figure()
plt.plot(epochs_ran, history["train_psnr"], label="Train PSNR (enhanced)")
plt.plot(epochs_ran, history["val_psnr"],   label="Val PSNR (enhanced)")
plt.axhline(y=avg_val_psnr_baseline, color="gray", linestyle="--",
            label=f"Baseline (no enhance) = {avg_val_psnr_baseline:.2f}dB")
plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
plt.title(f"PSNR Improvement ({TARGET})"); plt.legend(); plt.grid()
plt.savefig(f"{save_dir}/psnr_curve.png"); plt.show()

print(f"\n训练完成。")
print(f"  未增强 baseline PSNR : {avg_val_psnr_baseline:.2f} dB")
print(f"  增强后最优 PSNR      : {best_psnr:.2f} dB")
print(f"  提升                : {best_psnr - avg_val_psnr_baseline:+.2f} dB")
