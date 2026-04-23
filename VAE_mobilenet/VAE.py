import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.models import mobilenet_v2
from torchvision.utils import save_image

# 设置随机种子
torch.manual_seed(22)
np.random.seed(22)

# =========================
# 创建输出文件夹
# =========================
save_dir = "VAE_images"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 超参数
# =========================
latent_dim = 128
batch_size = 256
lr = 1e-3
epochs =50
img_size = 32
channels = 3
beta = 0.25  # KL损失系数(笔记本先小一点更容易看到重建提升)

# 编码器模式: "cnn" 或 "mobilenet"
# 电脑资源紧张时，建议继续使用默认的 mobilenet
encoder_mode = "cnn"
mobilenet_width_mult = 0.35

# 笔记本友好：可选只用部分数据做快速训练
# 1.0 表示全量，0.5 表示只用50%
dataset_fraction = 1.0

# 降低磁盘I/O与测试负担
test_every = 2
save_every = 5

# 数据集划分比例 6:2:2
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# =========================
# 数据加载与预处理
# =========================
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ]
)

# 将CIFAR-10的train和test合并，再做6:2:2
dataset_train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
dataset_test = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
full_dataset = ConcatDataset([dataset_train, dataset_test])

total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

if dataset_fraction < 1.0:
    def downsample_dataset(ds, frac, seed):
        keep_len = max(1, int(len(ds) * frac))
        drop_len = len(ds) - keep_len
        sub_ds, _ = random_split(
            ds,
            [keep_len, drop_len],
            generator=torch.Generator().manual_seed(seed),
        )
        return sub_ds

    train_dataset = downsample_dataset(train_dataset, dataset_fraction, 101)
    val_dataset = downsample_dataset(val_dataset, dataset_fraction, 102)
    test_dataset = downsample_dataset(test_dataset, dataset_fraction, 103)

is_cuda = torch.cuda.is_available()
loader_kwargs = {
    "num_workers": 0,  # Windows笔记本通常0更稳
    "pin_memory": is_cuda,
}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_channels = 128
        self.out_spatial = 4
        flat_dim = self.out_channels * self.out_spatial * self.out_spatial
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, start_dim=1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class MobileNetEncoder(nn.Module):
    def __init__(self, width_mult=0.35):
        super().__init__()
        backbone = mobilenet_v2(weights=None, width_mult=width_mult)
        self.features = backbone.features

        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.features(dummy)
        self.features.train(was_training)
        self.out_channels = out.shape[1]
        self.out_spatial = out.shape[2]
        flat_dim = self.out_channels * self.out_spatial * self.out_spatial

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, start_dim=1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_channels, in_spatial):
        super().__init__()
        self.in_channels = in_channels
        self.in_spatial = in_spatial

        self.fc = nn.Linear(latent_dim, in_channels * in_spatial * in_spatial)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.in_channels, self.in_spatial, self.in_spatial)
        x = self.deconv(x)
        # 解码输出按输入尺寸裁剪/插值，保证始终是32x32
        if x.shape[-1] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return x


class VAE(nn.Module):
    def __init__(self, mode="mobilenet"):
        super().__init__()
        if mode == "mobilenet":
            self.encoder = MobileNetEncoder(width_mult=mobilenet_width_mult)
        elif mode == "cnn":
            self.encoder = CNNEncoder()
        else:
            raise ValueError("mode must be 'cnn' or 'mobilenet'")

        self.decoder = Decoder(self.encoder.out_channels, self.encoder.out_spatial)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    # 按batch求平均的KL
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def compute_psnr(x, y):
    mse = F.mse_loss(x, y, reduction="mean").clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(mode=encoder_mode).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print(model)
print(f"Using device: {device}")
print(f"Encoder mode: {encoder_mode}")


def train_epoch(epoch):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for data, _ in train_loader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss_fn(recon, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    print(
        f"Epoch {epoch} Train Loss: {avg_loss:.4f} "
        f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})"
    )
    return avg_loss


def validate(epoch):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_fn(recon, data, mu, logvar)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_recon = total_recon / len(val_loader)
    avg_kl = total_kl / len(val_loader)
    print(
        f"Epoch {epoch} Validation Loss: {avg_loss:.4f} "
        f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})"
    )
    return avg_loss


def test(epoch, save_images=True):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    psnr_total = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_fn(recon, data, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            psnr = compute_psnr(denorm(recon), denorm(data))
            psnr_total += psnr.item()

    avg_loss = total_loss / len(test_loader)
    avg_recon = total_recon / len(test_loader)
    avg_kl = total_kl / len(test_loader)
    avg_psnr = psnr_total / len(test_loader)

    print(
        f"Epoch {epoch} Test Loss: {avg_loss:.4f} "
        f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, PSNR: {avg_psnr:.2f})"
    )

    if (epoch + 1) % save_every == 0:
        # 保存重建图
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon, _, _ = model(data)
        vis = torch.cat([denorm(data[:16]), denorm(recon[:16])], dim=0)
        save_image(vis, os.path.join(save_dir, f"recon_epoch_{epoch}.png"), nrow=8)

        # 保存采样图 (VAE latent sampling)
        z = torch.randn(16, latent_dim, device=device)
        gen = model.decoder(z)
        save_image(denorm(gen), os.path.join(save_dir, f"sample_epoch_{epoch}.png"), nrow=4)

    return avg_loss


def train():
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print("\n============================")
        print(f"Epoch {epoch} 开始")

        train_epoch(epoch)
        val_loss = validate(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vae_model.pth"))
            print(f"模型已保存，验证损失: {best_val_loss:.4f}")

        if epoch % test_every == 0 or epoch == 1 or epoch == epochs:
            need_save = (epoch % save_every == 0) or (epoch == 1) or (epoch == epochs)
            test(epoch, save_images=need_save)


if __name__ == "__main__":
    train()
