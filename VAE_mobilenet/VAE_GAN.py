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
save_dir = "VAE_GAN_images"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 超参数 - 增加了GAN相关参数
# =========================
latent_dim = 128
batch_size = 256
lr = 1e-3
lr_disc = 1e-4  # 判别器学习率通常较小
epochs = 50
img_size = 32
channels = 3

# VAE损失权重
beta = 0.25  # KL损失系数
lambda_recon = 1.0  # 重建损失权重
lambda_kl = beta  # KL损失权重
lambda_gan = 0.1   # GAN损失权重，可以调整

# 编码器模式: "cnn" 或 "mobilenet"
encoder_mode = "cnn"
mobilenet_width_mult = 0.35

# 笔记本友好：可选只用部分数据做快速训练
dataset_fraction = 1.0

# 降低磁盘I/O与测试负担
test_every = 2
save_every = 5

# 数据集划分比例 6:2:2
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# =========================
# 数据加载与预处理 (保持不变)
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

# =========================
# 1. VAE编码器 (保持不变)
# =========================
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


# =========================
# 2. VAE解码器 (保持不变)
# =========================
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
        if x.shape[-1] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return x


# =========================
# 3. 判别器 (Discriminator) - 新增
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: 3 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            # 输出: 1 x 4 x 4
        )
        
    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1).squeeze(1)


# =========================
# 4. VAE模型 (修改为支持GAN训练)
# =========================
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
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


# =========================
# 5. 损失函数 (修改为包含GAN损失)
# =========================
def vae_loss_fn(recon_x, x, mu, logvar):
    """VAE原始损失函数"""
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def gan_loss_fn(disc_real, disc_fake, mode='lsgan'):
    """GAN损失函数"""
    if mode == 'lsgan':  # Least Squares GAN
        # 处理判别器训练（有真实数据）的情况
        if disc_real is not None:
            loss_real = F.mse_loss(disc_real, torch.ones_like(disc_real))
        else:
            loss_real = 0
        
        loss_fake = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = 0.5 * (loss_real + loss_fake) if disc_real is not None else loss_fake
        
        # 生成器损失
        loss_gen = 0.5 * F.mse_loss(disc_fake, torch.ones_like(disc_fake))
    else:  # Standard GAN
        # 处理判别器训练（有真实数据）的情况
        if disc_real is not None:
            loss_real = F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real))
        else:
            loss_real = 0
        
        loss_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = loss_real + loss_fake
        loss_gen = F.binary_cross_entropy_with_logits(disc_fake, torch.ones_like(disc_fake))
    
    return loss_disc, loss_gen



# =========================
# 辅助函数
# =========================
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def compute_psnr(x, y):
    mse = F.mse_loss(x, y, reduction="mean").clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


# =========================
# 设备设置和模型初始化
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
vae = VAE(mode=encoder_mode).to(device)
discriminator = Discriminator().to(device)

# 初始化优化器
optimizer_vae = optim.Adam(vae.parameters(), lr=lr)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)

print("VAE模型:")
print(vae)
print("\n判别器模型:")
print(discriminator)
print(f"Encoder mode: {encoder_mode}")


# =========================
# 训练函数 (修改为VAE-GAN训练)
# =========================
def train_epoch(epoch):
    vae.train()
    discriminator.train()
    
    total_vae_loss, total_recon, total_kl, total_gan_gen = 0.0, 0.0, 0.0, 0.0
    total_disc_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        batch_size = data.size(0)
        
        # ====================
        # 1. 训练判别器
        # ====================
        optimizer_disc.zero_grad()
        
        # 真实图像
        real_output = discriminator(data)
        
        # 生成图像 (来自VAE重建)
        recon, mu, logvar = vae(data)
        fake_output = discriminator(recon.detach())  # 阻止梯度传到VAE
        
        # 判别器损失
        disc_loss, _ = gan_loss_fn(real_output, fake_output, mode='lsgan')
        disc_loss.backward()
        optimizer_disc.step()
        
        # ====================
        # 2. 训练VAE (生成器)
        # ====================
        optimizer_vae.zero_grad()
        
        # 前向传播
        recon, mu, logvar = vae(data)
        
        # VAE损失
        recon_loss, kl_loss = vae_loss_fn(recon, data, mu, logvar)
        
        # GAN损失 (生成器部分)
        fake_output = discriminator(recon)  # 这次需要梯度
        _, gen_loss = gan_loss_fn(None, fake_output, mode='lsgan')
        
        # 总损失
        vae_total_loss = lambda_recon * recon_loss + lambda_kl * kl_loss + lambda_gan * gen_loss
        
        vae_total_loss.backward()
        optimizer_vae.step()
        
        # 记录损失
        total_vae_loss += vae_total_loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_gan_gen += gen_loss.item()
        total_disc_loss += disc_loss.item()
        
        # 可选：每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"VAE Loss: {vae_total_loss.item():.4f}, "
                  f"Disc Loss: {disc_loss.item():.4f}")

    # 计算平均损失
    avg_vae_loss = total_vae_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    avg_gan_gen = total_gan_gen / len(train_loader)
    avg_disc_loss = total_disc_loss / len(train_loader)
    
    print(f"Epoch {epoch} Train - "
          f"VAE Loss: {avg_vae_loss:.4f} "
          f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, GAN: {avg_gan_gen:.4f}), "
          f"Disc Loss: {avg_disc_loss:.4f}")
    
    return avg_vae_loss, avg_disc_loss


# =========================
# 补全 validate 函数
# =========================
def validate(epoch):
    vae.eval()
    discriminator.eval()
    
    total_vae_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    total_disc_real, total_disc_fake = 0.0, 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            
            # VAE前向
            recon, mu, logvar = vae(data)
            recon_loss, kl_loss = vae_loss_fn(recon, data, mu, logvar)
            vae_total_loss = lambda_recon * recon_loss + lambda_kl * kl_loss
            
            # 判别器输出
            real_output = discriminator(data)
            fake_output = discriminator(recon)
            psnr = compute_psnr(denorm(recon), denorm(data))
            
            total_vae_loss += vae_total_loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_disc_real += real_output.mean().item()
            total_disc_fake += fake_output.mean().item()
            total_psnr += psnr.item()

    # 计算平均指标
    num_batches = len(val_loader)
    avg_vae_loss = total_vae_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    avg_disc_real = total_disc_real / num_batches
    avg_disc_fake = total_disc_fake / num_batches
    avg_psnr = total_psnr / num_batches
    
    print(f"Epoch {epoch} Val - VAE Loss: {avg_vae_loss:.4f} "
          f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}), "
          f"Disc Real: {avg_disc_real:.4f}, Disc Fake: {avg_disc_fake:.4f}, "
          f"PSNR: {avg_psnr:.2f} dB")
    
    return avg_vae_loss, avg_psnr

# =========================
# 添加样本生成函数
# =========================
def generate_samples(epoch, num_samples=8):
    vae.eval()
    with torch.no_grad():
        # 重建样本
        test_batch, _ = next(iter(test_loader))
        test_batch = test_batch[:num_samples].to(device)
        recon_batch, _, _ = vae(test_batch)
        
        # 随机生成样本
        z = torch.randn(num_samples, latent_dim).to(device)
        gen_images = vae.decode(z)
        
        # 保存图像
        save_image(
            denorm(torch.cat([test_batch, recon_batch, gen_images])),
            f"{save_dir}/samples_epoch_{epoch}.png",
            nrow=num_samples
        )

# =========================
# 添加主训练循环
# =========================
def main():
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        
        # 训练
        train_vae_loss, train_disc_loss = train_epoch(epoch)
        
        # 验证
        if epoch % test_every == 0:
            val_loss, val_psnr = validate(epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(vae.state_dict(), f"{save_dir}/best_vae.pth")
                torch.save(discriminator.state_dict(), f"{save_dir}/best_disc.pth")
        
        # 保存样本和模型
        if epoch % save_every == 0:
            generate_samples(epoch)
            torch.save(vae.state_dict(), f"{save_dir}/vae_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{save_dir}/disc_epoch_{epoch}.pth")
    
    print("\nTraining completed!")

# =========================
# 主程序入口
# =========================
if __name__ == '__main__':
    main()
