import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchvision.utils import save_image

from vqvae import VQVAE

# ===== 参数 =====
batch_size = 16
epochs = 30
lr = 1e-4
image_size = 128
save_dir = "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(save_dir, exist_ok=True)

# ===== 数据增强 =====
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: img.convert("RGB")),  # ⭐ 关键修复
    transforms.ToTensor(),
])
# ===== 下载 Caltech101 =====
dataset = datasets.Caltech101(
    root="./data",
    download=True,
    transform=transform
)

# ===== 划分 train / val =====
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ===== 模型 =====
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ===== 记录 =====
train_loss_list = []
val_loss_list = []
psnr_list = []

best_psnr = 0

# ===== 训练 =====
for epoch in range(epochs):

    # ========= Train =========
    model.train()
    train_loss = 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        recon, _ = model(imgs)
        loss = criterion(recon, imgs)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ========= Validation =========
    model.eval()
    val_loss = 0
    psnr_epoch = []

    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recon, _ = model(imgs)

            loss = criterion(recon, imgs)
            val_loss += loss.item()

            recon_np = recon.cpu().numpy()
            imgs_np = imgs.cpu().numpy()

            for i in range(recon_np.shape[0]):
                img1 = np.transpose(imgs_np[i], (1, 2, 0))
                img2 = np.transpose(recon_np[i], (1, 2, 0))

                psnr_val = sk_psnr(img1, img2, data_range=1.0)
                psnr_epoch.append(psnr_val)

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = np.mean(psnr_epoch)

    # ===== 记录 =====
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    psnr_list.append(avg_psnr)

    # ===== 保存最好模型 =====
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"{save_dir}/best_model.pth")

    # ===== 保存重建图 =====
    save_image(recon[:4], f"{save_dir}/recon_epoch_{epoch}.png")

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"PSNR: {avg_psnr:.2f}")

    scheduler.step()

# ===== 保存指标 =====
np.save(f"{save_dir}/train_loss.npy", train_loss_list)
np.save(f"{save_dir}/val_loss.npy", val_loss_list)
np.save(f"{save_dir}/psnr.npy", psnr_list)

print("训练完成！")