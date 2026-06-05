import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from vqvae import VectorQuantizer

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ===== 快速实验配置（Colab 目标：约 1 分钟 / epoch）=====
IMG_SIZE = 416
BATCH_SIZE = 32
NUM_WORKERS = 0          # Colab/Jupyter 勿用多进程，否则易卡死
MAX_TRAIN_SAMPLES = 2048 # None = 全量训练集；调小可更快
MAX_VAL_SAMPLES = 512      # None = 全量验证集
LOG_EVERY = 20
DATA_ROOT = "/content/D-Fire"
EPOCHS = 30

# 稳定性：缓解 epoch 10+ 的 loss/PSNR 回落（VQ 码本塌陷 + encoder 漂移）
FREEZE_ENCODER_EPOCHS = 8   # 前 N 个 epoch 只训 decoder/quantizer；0 = 不冻结
VQ_LOSS_WEIGHT = 0.1        # 原 0.25 易与 quantizer 内部 commitment 叠加过猛
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 5     # 验证 PSNR 连续 N epoch 不提升则停止
LR_ENCODER = 5e-6
LR_QUANT_DEC = 5e-5

# ===== 记录 =====
history = {
    "train_loss": [],
    "val_loss": [],
    "train_psnr": [],
    "val_psnr": [],
}


# ---------------- Decoder ----------------
class PretrainedDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Model ----------------
class VQVAE_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=128)
        self.decoder = PretrainedDecoder(128)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


@torch.no_grad()
def batch_psnr(recon, target):
    """GPU 上快速 PSNR，避免 skimage 逐张 CPU 计算。"""
    mse = (recon - target).pow(2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-8))).mean().item()


# ===== 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

save_dir = "outputs_improved"
os.makedirs(save_dir, exist_ok=True)

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
_mean = torch.tensor(img_mean, device=device).view(1, 3, 1, 1)
_std = torch.tensor(img_std, device=device).view(1, 3, 1, 1)


def denormalize(tensor):
    return torch.clamp(tensor * _std + _mean, 0, 1)


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(img_mean, img_std),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.ToTensor(),
    transforms.Normalize(img_mean, img_std),
])


class DFireDataset(Dataset):
    """D-Fire：root/{train,test}/images/*.jpg"""

    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test")
        img_dir = os.path.join(root, split, "images")
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在 {img_dir} 下找不到图片，请确认 D-Fire 已解压到 {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


print("Loading dataset...", flush=True)
train_db = maybe_subset(
    DFireDataset(root=DATA_ROOT, split="train", transform=train_transform),
    MAX_TRAIN_SAMPLES,
)
val_db = maybe_subset(
    DFireDataset(root=DATA_ROOT, split="test", transform=val_transform),
    MAX_VAL_SAMPLES,
)

pin_memory = device.type == "cuda"
train_loader = DataLoader(
    train_db,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)
val_loader = DataLoader(
    val_db,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)

print(
    f"Train: {len(train_db)} imgs, {len(train_loader)} batches | "
    f"Val: {len(val_db)} imgs, {len(val_loader)} batches | "
    f"{IMG_SIZE}x{IMG_SIZE} bs={BATCH_SIZE}",
    flush=True,
)

print("Building model...", flush=True)
model = VQVAE_ResNet().to(device)

optimizer = optim.Adam([
    {"params": model.encoder.parameters(), "lr": LR_ENCODER},
    {"params": model.quantizer.parameters(), "lr": LR_QUANT_DEC},
    {"params": model.decoder.parameters(), "lr": LR_QUANT_DEC},
], weight_decay=1e-5)

# 按验证 loss 降 LR，比固定 Cosine 更不易在中后期把训练“冲垮”
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
)
criterion = nn.MSELoss()


def set_encoder_trainable(trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def codebook_perplexity(model, sample_batch):
    """码本困惑度越低说明越多少码未使用；< ~20 时常伴随塌陷。"""
    z = model.encoder(sample_batch)
    b, c, h, w = z.shape
    flat = z.permute(0, 2, 3, 1).reshape(-1, c)
    emb = model.quantizer.embedding.weight
    dist = (
        flat.pow(2).sum(1, keepdim=True)
        + emb.pow(2).sum(1)
        - 2 * flat @ emb.t()
    )
    idx = dist.argmin(1)
    counts = torch.bincount(idx, minlength=model.quantizer.num_embeddings).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return torch.exp(-(probs * probs.log()).sum()).item()

# 预热：避免第一个 batch 把计时算进去
_warm = next(iter(train_loader))[0][:2].to(device, non_blocking=True)
with torch.no_grad():
    model(_warm)
del _warm
if device.type == "cuda":
    torch.cuda.synchronize()
print("Warmup done. Start training.\n", flush=True)

best_psnr = 0
epochs_no_improve = 0
set_encoder_trainable(FREEZE_ENCODER_EPOCHS <= 0)

for epoch in range(EPOCHS):
    t_epoch = time.time()

    if FREEZE_ENCODER_EPOCHS > 0 and epoch == FREEZE_ENCODER_EPOCHS:
        set_encoder_trainable(True)
        print(f"Epoch {epoch + 1}: 解冻 encoder，以 lr={LR_ENCODER} 微调", flush=True)

    model.train()
    total_loss = 0.0
    train_psnr_sum = 0.0
    train_psnr_n = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False)
    for step, (imgs, _) in enumerate(pbar, start=1):
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        recon, vq_loss = model(imgs)
        target = denormalize(imgs)
        recon_loss = criterion(recon, target)
        loss = recon_loss + VQ_LOSS_WEIGHT * vq_loss

        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            train_psnr_sum += batch_psnr(recon, target)
            train_psnr_n += 1
        if step % LOG_EVERY == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_psnr = train_psnr_sum / max(train_psnr_n, 1)

    model.eval()
    total_val_loss = 0.0
    psnrs_val = []

    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            recon, _ = model(imgs)
            target = denormalize(imgs)
            total_val_loss += criterion(recon, target).item()
            psnrs_val.append(batch_psnr(recon, target))

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_psnr = float(np.mean(psnrs_val))
    elapsed = time.time() - t_epoch

    sample = next(iter(val_loader))[0][:8].to(device)
    perplexity = codebook_perplexity(model, sample)
    lrs = [pg["lr"] for pg in optimizer.param_groups]

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_psnr"].append(avg_train_psnr)
    history["val_psnr"].append(avg_val_psnr)
    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} ({elapsed:.1f}s) | "
        f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
        f"Train PSNR {avg_train_psnr:.2f} | Val PSNR {avg_val_psnr:.2f} | "
        f"Codebook perplexity {perplexity:.1f} | LR enc/dec {lrs[0]:.1e}/{lrs[1]:.1e}",
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
            },
            f"{save_dir}/best_model.pth",
        )
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stop: Val PSNR 连续 {EARLY_STOP_PATIENCE} epoch 未超过 {best_psnr:.2f} dB",
                flush=True,
            )
            break

epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/loss_curve.png")
plt.show()

plt.figure()
plt.plot(epochs, history["train_psnr"], label="Train PSNR")
plt.plot(epochs, history["val_psnr"], label="Validation PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Training vs Validation PSNR")
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/psnr_curve.png")
plt.show()