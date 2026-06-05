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
BATCH_SIZE = 16
NUM_WORKERS = 0          # Colab/Jupyter 勿用多进程，否则易卡死
MAX_TRAIN_SAMPLES = 2048 # None = 全量训练集；调小可更快
MAX_VAL_SAMPLES = 512      # None = 全量验证集
LOG_EVERY = 20
DATA_ROOT = "/content/D-Fire"
EPOCHS = 30

# 稳定性：缓解 epoch 10+ 的 loss/PSNR 回落（VQ 码本塌陷 + encoder 漂移）
FREEZE_ENCODER_EPOCHS = 0   # 前 N 个 epoch 只训 decoder/quantizer；0 = 不冻结
VQ_LOSS_WEIGHT = 0.25        # 原 0.25 易与 quantizer 内部 commitment 叠加过猛
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 5     # 验证 PSNR 连续 N epoch 不提升则停止
LR_ENCODER = 5e-6
LR_QUANT_DEC = 2e-4

# ===== 记录 =====
history = {
    "train_loss": [],
    "val_loss": [],
    "train_psnr": [],
    "val_psnr": [],
}


# ---------------- Decoder ----------------
# ---------------- 改进后的 Decoder（6层，对称+BN）----------------
class ImprovedDecoder(nn.Module):
    """
    原来只有 3 层 ConvTranspose，与 9 层 Encoder 严重不对称。
    改进：6 层 ConvTranspose + BatchNorm，中间先升维再降维，
    给解码器更大的"表达宽度"来还原细节。
    16x16 -> 32 -> 64 -> 128 (output)，每步都有 BN 稳定梯度。
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # 16x16 -> 16x16，先升维，增加表达容量
            nn.ConvTranspose2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64x64 -> 128x128，最后输出 3 通道
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
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
        self.quantizer = VectorQuantizer(
            num_embeddings=1024,        # 扩容，416 尺寸特征图更大需要更多码字
            embedding_dim=128,
            commitment_cost=0.25,
            decay=0.99,                 # EMA 衰减系数
            dead_code_threshold=1.0,    # 使用次数低于此值的码字会被重置
        )
        self.decoder = ImprovedDecoder(128)

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
def combined_loss(recon, target):
    mse = nn.functional.mse_loss(recon, target)
    # 手动 SSIM（避免额外依赖）
    def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
        mu_x = nn.functional.avg_pool2d(x, 3, 1, 1)
        mu_y = nn.functional.avg_pool2d(y, 3, 1, 1)
        s_x  = nn.functional.avg_pool2d(x*x,3,1,1) - mu_x**2
        s_y  = nn.functional.avg_pool2d(y*y,3,1,1) - mu_y**2
        s_xy = nn.functional.avg_pool2d(x*y,3,1,1) - mu_x*mu_y
        num  = (2*mu_x*mu_y+C1)*(2*s_xy+C2)
        den  = (mu_x**2+mu_y**2+C1)*(s_x+s_y+C2)
        return num.mean() / den.mean().clamp(min=1e-8)
    return 0.7 * mse + 0.3 * (1 - ssim_simple(recon, target))

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
        recon_loss = combined_loss(recon, target)
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
    # 新增：打印码本实际使用情况，方便确认 EMA 是否生效
    with torch.no_grad():
        counts = torch.zeros(model.quantizer.num_embeddings, device=device)
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            z = model.encoder(imgs)
            b, c, h, w = z.shape
            flat = z.permute(0,2,3,1).reshape(-1, c)
            emb = model.quantizer.embedding.weight
            dist = (flat.pow(2).sum(1, keepdim=True)
                    + emb.pow(2).sum(1)
                    - 2 * flat @ emb.t())
            idx = dist.argmin(1)
            counts += torch.bincount(idx, minlength=model.quantizer.num_embeddings).float()
            break  # 只用第一个 batch 估计
        active = (counts > 0).sum().item()
# 把这行加进已有的 print 里
    print(
        f"Epoch {epoch + 1}/{EPOCHS} ({elapsed:.1f}s) | "
        f"Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | "
        f"Train PSNR {avg_train_psnr:.2f} | Val PSNR {avg_val_psnr:.2f} | "
        f"Codebook perplexity {perplexity:.1f} | Active codes {active}/{model.quantizer.num_embeddings} | "
        f"LR enc/dec {lrs[0]:.1e}/{lrs[1]:.1e}",
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

epochs = range(1, len(history["train_loss"]) + 1)
# ===== 推理速度测试 =====
import time

def benchmark_inference(model, device, img_size=IMG_SIZE, n_warmup=10, n_runs=100):
    """
    测试单张图片的推理延迟。
    n_warmup: 预热次数，不计入统计（GPU 需要预热才能达到稳定频率）
    n_runs:   正式测试次数，取均值和分位数
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # --- 预热 ---
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # GPU 和 CPU 的计时方式不同：
    # CPU 用 time.perf_counter() 即可；
    # GPU 需要用 CUDA Event，否则异步执行会让计时严重偏低
    latencies = []

    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(n_runs):
                starter.record()
                model(dummy)
                ender.record()
                torch.cuda.synchronize()          # 等 GPU 完成再读时间
                latencies.append(starter.elapsed_time(ender))  # 单位：毫秒

    else:
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # 转毫秒

    latencies = torch.tensor(latencies)
    mean_ms   = latencies.mean().item()
    std_ms    = latencies.std().item()
    p50_ms    = latencies.median().item()
    p95_ms    = latencies.quantile(0.95).item()
    p99_ms    = latencies.quantile(0.99).item()
    fps       = 1000.0 / mean_ms

    print("\n===== 推理速度测试结果 =====")
    print(f"  设备        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  输入尺寸    : 1 × 3 × {img_size} × {img_size}")
    print(f"  测试轮数    : {n_runs} 次（预热 {n_warmup} 次）")
    print(f"  平均延迟    : {mean_ms:.3f} ms  ±{std_ms:.3f} ms")
    print(f"  P50 延迟    : {p50_ms:.3f} ms")
    print(f"  P95 延迟    : {p95_ms:.3f} ms")
    print(f"  P99 延迟    : {p99_ms:.3f} ms")
    print(f"  吞吐量      : {fps:.1f} FPS")
    print("============================\n")

    return {"mean_ms": mean_ms, "std_ms": std_ms,
            "p50_ms": p50_ms, "p95_ms": p95_ms, "p99_ms": p99_ms, "fps": fps}


# 加载最优权重再测（而不是用训练结束时可能已经过拟合的权重）
ckpt = torch.load(f"{save_dir}/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model"])
print(f"已加载最优模型（来自 epoch {ckpt['epoch']}，Val PSNR {ckpt['val_psnr']:.2f} dB）")

bench = benchmark_inference(model, device, img_size=IMG_SIZE)
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