import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from torchvision.utils import save_image

# 设置随机种子
torch.manual_seed(22)
np.random.seed(22)

# =========================
# 创建输出文件夹
# =========================
save_dir = "AutoEncoder_images"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 超参数
# =========================
h_dim = 256  # 隐藏层维度
batch_size = 128  # 批次大小
lr = 1e-3  # 学习率
epochs = 20  # 训练轮数
img_size = 32  # CIFAR-10图片大小
channels = 3  # RGB彩色图片

# 数据集划分比例
train_ratio = 0.8  # 训练集比例
val_ratio = 0.1    # 验证集比例
test_ratio = 0.1   # 测试集比例

# =========================
# 数据加载与预处理
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

# 加载CIFAR-10数据集
full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 划分数据集
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# =========================
# 模型定义
# =========================
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size * channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, h_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()  # 输出范围[-1,1]与归一化后的数据匹配
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图片
        h = self.encoder(x)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(x.size(0), channels, img_size, img_size)  # 恢复图片形状
        return x_hat

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
print(model)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# =========================
# 单轮训练
# =========================
def train_epoch(epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, data)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
    return avg_loss

# =========================
# 验证函数
# =========================
def validate(epoch):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    return avg_loss

# =========================
# 测试函数
# =========================
def test(epoch):
    model.eval()
    total_loss = 0
    psnr_total = 0
    ssim_total = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            
            # 计算损失
            loss = criterion(outputs, data)
            total_loss += loss.item()
            
            # 计算PSNR
            mse = torch.mean((data - outputs) ** 2)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_total += psnr.item()
            
            # 计算SSIM (简化版)
            ssim = 1 - mse  # 简化版SSIM
            ssim_total += ssim.item()
    
    # 计算平均值
    avg_loss = total_loss / len(test_loader)
    avg_psnr = psnr_total / len(test_loader)
    avg_ssim = ssim_total / len(test_loader)
    
    print(f"Epoch {epoch} → Test MSE: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    
    # 保存测试集图片复现结果
    data, _ = next(iter(test_loader))
    data = data.to(device)
    outputs = model(data)
    
    # 保存原始图片和重建图片
    save_image(torch.cat([data, outputs], dim=0), 
               os.path.join(save_dir, f"rec_epoch_{epoch}.png"), 
               nrow=10, normalize=True)
    
    return avg_loss, avg_psnr, avg_ssim

# =========================
# 训练主函数
# =========================
def train():
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print("\n============================")
        print(f"Epoch {epoch} 开始")
        
        # 训练
        train_loss = train_epoch(epoch)
        
        # 验证
        val_loss = validate(epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"模型已保存，验证损失: {best_val_loss:.4f}")
        
        # 测试并保存图片
        test(epoch)

# =========================
# 启动
# =========================
if __name__ == '__main__':
    train()
