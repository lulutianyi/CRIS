import glob
import os
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

try:
    from compressai.models import Cheng2020Anchor
except ImportError:
    raise ImportError("请先运行: pip install compressai")

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ===== 配置 =====
IMG_SIZE            = 384
BATCH_SIZE          = 16
NUM_WORKERS         = 0
MAX_TRAIN_SAMPLES   = 2048
MAX_VAL_SAMPLES     = 512
LOG_EVERY           = 20
DATA_ROOT           = "/content/D-Fire"
EPOCHS              = 30
EARLY_STOP_PATIENCE = 5
LAMBDA              = 0.05
LR_MAIN             = 5e-5
LR_AUX              = 1e-3
GRAD_CLIP_NORM      = 1.0

save_dir = "outputs_compressai"
os.makedirs(save_dir, exist_ok=True)

# ===== 自动下载 D-Fire 数据集 =====
def download_dfire(data_root=DATA_ROOT):
    """
    从 GitHub Release 下载 D-Fire 数据集并解压。
    如果数据目录已存在且包含图片则跳过。
    """
    train_img_dir = os.path.join(data_root, "train", "images")
    test_img_dir  = os.path.join(data_root, "test",  "images")

    already_exists = (
        os.path.isdir(train_img_dir) and len(glob.glob(os.path.join(train_img_dir, "*.jpg"))) > 0
        and os.path.isdir(test_img_dir)  and len(glob.glob(os.path.join(test_img_dir,  "*.jpg"))) > 0
    )
    if already_exists:
        n_train = len(glob.glob(os.path.join(train_img_dir, "*.jpg")))
        n_test  = len(glob.glob(os.path.join(test_img_dir,  "*.jpg")))
        print(f"D-Fire 已存在：train={n_train} 张, test={n_test} 张，跳过下载。", flush=True)
        return

    # ---------- 尝试用 kaggle API 下载（Colab 推荐方式）----------
    try:
        import kaggle  # noqa: F401
        print("检测到 kaggle，使用 Kaggle API 下载 D-Fire ...", flush=True)
        os.makedirs(data_root, exist_ok=True)
        os.system(
            f'kaggle datasets download -d phylake1337/dfire-dataset -p "{data_root}" --unzip'
        )
        print("Kaggle 下载完成。", flush=True)
        return
    except Exception:
        pass

    # ---------- 尝试从 Hugging Face Hub 下载 ----------
    try:
        from huggingface_hub import snapshot_download
        print("尝试从 Hugging Face Hub 下载 D-Fire ...", flush=True)
        snapshot_download(
            repo_id="pyronear/d-fire",
            repo_type="dataset",
            local_dir=data_root,
        )
        print("Hugging Face 下载完成。", flush=True)
        return
    except Exception:
        pass

    # ---------- 最后兜底：从 GitHub Release 下载 zip ----------
    import urllib.request

    GITHUB_URL = (
        "https://github.com/gaiasd/DFireDataset/releases/download/v1.0/"
        "D-Fire.zip"
    )
    zip_path = os.path.join(data_root, "D-Fire.zip")
    os.makedirs(data_root, exist_ok=True)

    print(f"正在从 GitHub 下载 D-Fire（~3 GB），请耐心等待...", flush=True)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = block_num * block_size / total_size * 100
            print(f"\r  下载进度: {min(pct, 100):.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(GITHUB_URL, zip_path, reporthook=_progress)
        print("\n下载完成，正在解压...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        os.remove(zip_path)
        print("解压完成。", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"自动下载失败：{e}\n"
            "请手动下载 D-Fire 数据集并解压到 /content/D-Fire，\n"
            "目录结构应为：\n"
            "  /content/D-Fire/train/images/*.jpg\n"
            "  /content/D-Fire/test/images/*.jpg"
        )

download_dfire()

# ===== 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ===== 数据集 =====
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
print("Building model...", flush=True)
model = Cheng2020Anchor(N=128).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

# ===== 优化器 =====
optimizer     = optim.Adam(model.parameters(), lr=LR_MAIN)
aux_optimizer = optim.Adam(model.entropy_bottleneck.parameters(), lr=LR_AUX)
scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
)

# ===== 断点续训 =====
# history 先初始化空
history = {
    "train_loss": [], "val_loss": [],
    "train_psnr": [], "val_psnr": [],
    "train_bpp":  [], "val_bpp":  [],
}

best_psnr         = 0.0
start_epoch       = 0
epochs_no_improve = 0

CKPT_PATH = f"{save_dir}/best_model.pth"

if os.path.isfile(CKPT_PATH):
    print(f"发现已有 checkpoint：{CKPT_PATH}，正在恢复...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # aux_optimizer 状态（旧版 checkpoint 可能没有，兼容处理）
    if "aux_optimizer" in ckpt:
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])

    # scheduler 状态
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    # 训练历史
    if "history" in ckpt:
        history = ckpt["history"]

    best_psnr         = ckpt.get("val_psnr", 0.0)
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    start_epoch       = ckpt.get("epoch", 0)   # 已完成的 epoch 数

    print(
        f"已恢复到 Epoch {start_epoch}，最优 PSNR={best_psnr:.2f} dB，"
        f"从 Epoch {start_epoch + 1} 继续训练。",
        flush=True,
    )
else:
    print("未找到 checkpoint，从头开始训练。", flush=True)

# ===== 训练循环 =====
for epoch in range(start_epoch, EPOCHS):
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

        out   = model(imgs)
        x_hat = out["x_hat"].clamp(0, 1)

        num_pixels  = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        bpp         = sum((-torch.log2(lk).sum() / num_pixels) for lk in out["likelihoods"].values())
        distortion  = torch.nn.functional.mse_loss(x_hat, imgs)
        loss        = bpp + LAMBDA * 255**2 * distortion

        loss.backward()
        if GRAD_CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        aux_loss = model.entropy_bottleneck.loss()
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
            imgs  = imgs.to(device, non_blocking=True)
            out   = model(imgs)
            x_hat = out["x_hat"].clamp(0, 1)

            num_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
            bpp        = sum((-torch.log2(lk).sum() / num_pixels) for lk in out["likelihoods"].values())
            distortion = torch.nn.functional.mse_loss(x_hat, imgs)
            loss       = bpp + LAMBDA * 255**2 * distortion

            val_loss += loss.item()
            val_bpp  += bpp.item()
            val_psnr += batch_psnr(x_hat, imgs)
            n_val    += 1

    avg_val_loss = val_loss / n_val
    avg_val_bpp  = val_bpp  / n_val
    avg_val_psnr = val_psnr / n_val
    elapsed      = time.time() - t_epoch

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
        best_psnr         = avg_val_psnr
        epochs_no_improve = 0

        torch.save(
            {
                "epoch":           epoch + 1,          # 已完成的 epoch 数
                "model":           model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "aux_optimizer":   aux_optimizer.state_dict(),  # ← 新增
                "scheduler":       scheduler.state_dict(),      # ← 新增
                "history":         history,                     # ← 新增
                "val_psnr":        avg_val_psnr,
                "val_bpp":         avg_val_bpp,
                "epochs_no_improve": epochs_no_improve,         # ← 新增
            },
            CKPT_PATH,
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

# ===== 曲线绘图 =====
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

plt.figure()
plt.scatter(history["val_bpp"], history["val_psnr"], c=list(epochs_ran), cmap="viridis")
plt.colorbar(label="Epoch")
plt.xlabel("bpp"); plt.ylabel("PSNR (dB)")
plt.title("Rate-Distortion (Val)"); plt.grid()
plt.savefig(f"{save_dir}/rd_curve.png"); plt.show()

print(f"\n训练完成。最优 Val PSNR: {best_psnr:.2f} dB", flush=True)