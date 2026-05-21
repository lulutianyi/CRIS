import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import matplotlib.pyplot as plt

from vqvae import VectorQuantizer

IMG_SIZE = 416
# ===== 记录 =====
history = {
    'train_loss': [],
    'val_loss': [],
    'train_psnr': [],
    'val_psnr': []
}

# ---------------- Decoder ----------------
class PretrainedDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
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
            resnet.layer2
        )
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=128)
        self.decoder = PretrainedDecoder(128)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# ===== 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "outputs_improved"
os.makedirs(save_dir, exist_ok=True)

# ===== 数据增强 =====
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # ⭐ 修复灰度图问题
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),

    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),

    transforms.ToTensor(),
    transforms.Normalize(img_mean, img_std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # ⭐ 同样必须加
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),

    transforms.ToTensor(),
    transforms.Normalize(img_mean, img_std)
])

def denormalize(tensor):
    mean = torch.tensor(img_mean).view(1,3,1,1).to(device)
    std = torch.tensor(img_std).view(1,3,1,1).to(device)
    return torch.clamp(tensor * std + mean, 0, 1)

# ===== 数据集 =====
from torch.utils.data import Dataset
from PIL import Image
import glob

class DFireDataset(Dataset):
    """
    D-Fire 数据集加载器（仅用图像，忽略标注，用于无监督图像重建）
    目录结构：
        root/
          train/images/*.jpg
          test/images/*.jpg
    """
    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test")
        img_dir = os.path.join(root, split, "images")
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                            glob.glob(os.path.join(img_dir, "*.png")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在 {img_dir} 下找不到图片，请确认 D-Fire 已解压到 {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0   # 标签占位，与原代码 for imgs, _ 兼容

train_db = DFireDataset(root="/content/D-Fire", split="train", transform=train_transform)
val_db   = DFireDataset(root="/content/D-Fire", split="test",  transform=val_transform)

train_loader = DataLoader(train_db, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_db, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# ===== 模型 =====
model = VQVAE_ResNet().to(device)

optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.quantizer.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-4}
])

# ⭐ Cosine 学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
    eta_min=1e-6
)

criterion = nn.MSELoss()

# ===== 训练 =====
best_psnr = 0
EPOCHS = 30

for epoch in range(EPOCHS):

    # ===== Train =====
    model.train()
    total_loss = 0
    psnrs_train = []

    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()

        recon, vq_loss = model(imgs)
        target = denormalize(imgs)

        recon_loss = criterion(recon, target)
        loss = recon_loss + 0.25 * vq_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ===== Train PSNR =====
        res_np = recon.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        for i in range(res_np.shape[0]):
            p = sk_psnr(
                tgt_np[i].transpose(1,2,0),
                res_np[i].transpose(1,2,0),
                data_range=1.0
            )
            psnrs_train.append(p)

    avg_train_loss = total_loss / len(train_loader)
    avg_train_psnr = np.mean(psnrs_train)

    # ===== Validation =====
    model.eval()
    total_val_loss = 0
    psnrs_val = []

    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            target = denormalize(imgs)

            recon_loss = criterion(recon, target)
            total_val_loss += recon_loss.item()

            res_np = recon.cpu().numpy()
            tgt_np = target.cpu().numpy()

            for i in range(res_np.shape[0]):
                p = sk_psnr(
                    tgt_np[i].transpose(1,2,0),
                    res_np[i].transpose(1,2,0),
                    data_range=1.0
                )
                psnrs_val.append(p)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_psnr = np.mean(psnrs_val)

    # ===== 记录 =====
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_psnr'].append(avg_train_psnr)
    history['val_psnr'].append(avg_val_psnr)

    scheduler.step()

    print(f"Epoch {epoch+1} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Train PSNR {avg_train_psnr:.2f} | Val PSNR {avg_val_psnr:.2f}")

    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save(model.state_dict(), f"{save_dir}/best_model.pth")

# ===== 画图 =====
epochs = range(1, EPOCHS+1)

plt.figure()
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/loss_curve.png")
plt.show()

plt.figure()
plt.plot(epochs, history['train_psnr'], label='Train PSNR')
plt.plot(epochs, history['val_psnr'], label='Validation PSNR')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Training vs Validation PSNR")
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/psnr_curve.png")
plt.show()