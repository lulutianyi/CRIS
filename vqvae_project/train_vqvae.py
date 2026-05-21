import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchvision.utils import save_image

from vqvae import VQVAE

# ===== 参数 =====
batch_size = 16
epochs = 20
lr = 1e-3
image_size = 32
save_dir = "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(save_dir, exist_ok=True)

# ===== 数据 =====
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# 👉 小规模快速验证（可删）
dataset.data = dataset.data[:5000]
dataset.targets = dataset.targets[:5000]

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 模型 =====
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# ===== 记录 =====
loss_list = []
psnr_list = []
mse_list = []
fps_list = []
decode_time_list = []

# ===== 训练 =====
for epoch in range(epochs):
    model.train()

    epoch_loss = 0
    epoch_psnr = 0
    epoch_mse = 0
    count = 0

    start_time = time.time()

    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        optimizer.zero_grad()

        recon, _ = model(imgs)
        loss = criterion(recon, imgs)

        loss.backward()
        optimizer.step()

        # ===== 转 numpy =====
        recon_np = recon.detach().cpu().numpy()
        imgs_np = imgs.detach().cpu().numpy()

        # ===== ✅ 正确PSNR计算（逐张图）=====
        psnr_batch = []
        mse_batch = []

        for j in range(recon_np.shape[0]):
            img1 = np.transpose(imgs_np[j], (1, 2, 0))  # (H, W, C)
            img2 = np.transpose(recon_np[j], (1, 2, 0))

            psnr_val = sk_psnr(img1, img2, data_range=1.0)
            mse_val = np.mean((img1 - img2) ** 2)

            psnr_batch.append(psnr_val)
            mse_batch.append(mse_val)

        psnr = np.mean(psnr_batch)
        mse = np.mean(mse_batch)

        # ===== 累加 =====
        epoch_loss += loss.item()
        epoch_psnr += psnr
        epoch_mse += mse
        count += 1

        if i % 50 == 0:
            print(f"Epoch[{epoch}] Step[{i}] Loss:{loss.item():.4f} PSNR:{psnr:.2f}")

    # ===== epoch统计 =====
    avg_loss = epoch_loss / count
    avg_psnr = epoch_psnr / count
    avg_mse = epoch_mse / count

    epoch_time = time.time() - start_time
    fps = len(dataset) / epoch_time

    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    mse_list.append(avg_mse)
    fps_list.append(fps)
    decode_time_list.append(epoch_time)

    print(f"\nEpoch {epoch} Summary:")
    print(f"Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | MSE: {avg_mse:.4f} | FPS: {fps:.2f}\n")

    # 保存重建图
    save_image(recon[:4], f"{save_dir}/recon_epoch_{epoch}.png")

# ===== 保存指标 =====
np.save(f"{save_dir}/loss.npy", loss_list)
np.save(f"{save_dir}/psnr.npy", psnr_list)
np.save(f"{save_dir}/mse.npy", mse_list)
np.save(f"{save_dir}/fps.npy", fps_list)
np.save(f"{save_dir}/decode_time.npy", decode_time_list)

print("训练完成，所有指标已保存！")