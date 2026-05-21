import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# ===== 在训练循环前初始化记录器 =====
history = {
    'train_loss': [],
    'val_loss': [],
    'psnr': []
}
# 导入你原有的组件
from vqvae import VectorQuantizer 

# ---------------- 重新定义的 Decoder ----------------
# 因为预训练 Encoder 下采样倍数不同，我们需要专门配一个 Decoder
class PretrainedDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # 输入: [B, 128, 16, 16]
            nn.ConvTranspose2d(embedding_dim, 128, 4, 2, 1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),           # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),             # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ---------------- 集成后的 VQVAE ----------------
class VQVAE_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 预训练 Encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,   # 1/4
            resnet.layer1,    # 64 channels
            resnet.layer2     # 1/8, 输出通道 128, 尺寸 16x16
        )
        
        # 2. 你的 VectorQuantizer (embedding_dim 需与 Encoder 输出一致，即 128)
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=128)
        
        # 3. 适配的 Decoder
        self.decoder = PretrainedDecoder(embedding_dim=128)

    def forward(self, x):
        z = self.encoder(x)
        # ⭐ 关键修复：根据你的 vqvae.py，返回顺序是 (quantized, loss)
        z_q, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# ---------------- 训练配置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "outputs_resnet"
os.makedirs(save_dir, exist_ok=True)

# 标准化参数 (ImageNet)
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    mean = torch.tensor(img_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(img_std).view(1, 3, 1, 1).to(device)
    return torch.clamp(tensor * std + mean, 0, 1)

# 加载 Caltech101
dataset = datasets.Caltech101(root="./data", download=True, transform=transform)
train_size = int(0.8 * len(dataset))
train_db, val_db = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_db, batch_size=16, shuffle=True)
val_loader = DataLoader(val_db, batch_size=16, shuffle=False)

# 初始化模型
model = VQVAE_ResNet().to(device)

# 优化器：Backbone 用小学习率，其他用正常学习率
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.quantizer.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-4}
])

criterion = nn.MSELoss()

# ---------------- 训练循环 ----------------
best_psnr = 0
for epoch in range(30):
    model.train()
    total_loss = 0
    total_val_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        
        recon, vq_loss = model(imgs)
        # 重建损失对比的是还原后的原图
        recon_loss = criterion(recon, denormalize(imgs))
        
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss/len(train_loader)
    # 验证
    model.eval()
    psnrs = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            target = denormalize(imgs)
            total_val_loss += loss.item()
            # 计算 PSNR
            res_np = recon.cpu().numpy()
            tgt_np = target.cpu().numpy()
            for i in range(res_np.shape[0]):
                p = sk_psnr(tgt_np[i].transpose(1,2,0), res_np[i].transpose(1,2,0), data_range=1.0)
                psnrs.append(p)
    avg_val_loss = total_val_loss/len(val_loader)

    avg_psnr = np.mean(psnrs)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['psnr'].append(avg_psnr)

    print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | PSNR: {avg_psnr:.2f}")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
    
    save_image(recon[:4], f"{save_dir}/epoch_{epoch}.png")

# ===== 2. 增加绘图函数 =====
def plot_metrics(history, save_dir):
    epochs_range = range(len(history['train_loss']))
    
    plt.figure(figsize=(12, 5))

    # 子图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.ylabel('dB')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png")
    plt.show() # 如果在 Jupyter/Colab 中可以直接显示

# 训练结束后调用
plot_metrics(history, save_dir)