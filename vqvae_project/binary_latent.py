import os
os.makedirs("sample", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

import glob
from tqdm import tqdm
from torch import nn, optim
import torch, argparse, math
import sys
sys.path.append('/content')
from vq_model import VQModel
from lossers.lpips import LPIPS
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as tvf
from discriminator import NLayerDiscriminator
from PIL import Image
from lossers.gan import (
    hinge_d_loss as d_loss_fn,
    vanilla_g_loss as g_loss_fn,
)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# ---------------------------------------------------------
# 自定义本地图片 Dataset，替代原来的 HF load_dataset
# 递归查找 root_dir 下所有图片文件（不依赖具体子目录结构）
# ---------------------------------------------------------
class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, exts=('jpg', 'jpeg', 'png')):
        self.transform = transform
        self.paths = []
        for ext in exts:
            self.paths += glob.glob(os.path.join(root_dir, '**', f'*.{ext}'), recursive=True)
            self.paths += glob.glob(os.path.join(root_dir, '**', f'*.{ext.upper()}'), recursive=True)
        self.paths = sorted(set(self.paths))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root_dir}, check the path.")
        print(f"Found {len(self.paths)} images under {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img}


parser = argparse.ArgumentParser(description="Train VQModel on D-Fire")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dataset", type=str, default="/content/D-Fire", help="local path to D-Fire dataset root")
parser.add_argument("--cache_dir", type=str, default="./.cache")
parser.add_argument("--iter", type=int, default=10000, help="total training iterations")
parser.add_argument("--batch", type=int, default=32, help="batch sizes for each gpus")
parser.add_argument("--size", type=int, default=64, help="image sizes for the model")
args, unknown = parser.parse_known_args()

model = VQModel().to(args.device)
lpips = LPIPS(net='vgg', cache_dir=args.cache_dir).to(args.device)
discriminator = NLayerDiscriminator().to(args.device)
vq_optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.0, 0.999))

to_tensor = transforms.Compose([
    transforms.Resize(args.size, interpolation=tvf.InterpolationMode.LANCZOS),
    transforms.CenterCrop(args.size),   # D-Fire图片长宽比不一，加CenterCrop保证方形
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])

dataset = LocalImageDataset(args.dataset, transform=to_tensor)
dataloader = DataLoader(
    dataset, batch_size=args.batch, shuffle=True,
    drop_last=True, num_workers=2, pin_memory=True,
)
dataloader = sample_data(dataloader)
pbar = tqdm(range(args.iter))

sample = next(dataloader)['image'].to(args.device)
utils.save_image(
    sample, f"sample/sample.png",
    nrow=int(math.sqrt(args.batch)), normalize=True, value_range=(-1, 1),
)

for idx in pbar:
    image = next(dataloader)['image'].to(args.device)
    # train vqmodel
    requires_grad(model, True)
    requires_grad(discriminator, False)
    rec = model(image)
    lpips_loss = lpips(image, rec).mean()
    fake_pred = discriminator(rec)
    g_loss = g_loss_fn(fake_pred)
    loss = lpips_loss + 0.001 * g_loss
    vq_optim.zero_grad()
    loss.backward()
    vq_optim.step()
    # train discriminator
    requires_grad(model, False)
    requires_grad(discriminator, True)
    rec = model(image)
    real_pred, fake_pred = discriminator(image), discriminator(rec.detach())
    d_loss = d_loss_fn(real_pred, fake_pred)
    d_optim.zero_grad()
    d_loss.backward()
    d_optim.step()
    pbar.set_description(
        f'lpips_loss: {lpips_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
        f'd_loss: {d_loss.item():.4f}'
    )

    if idx % 200 == 0:
        with torch.no_grad():
            rec = model(sample)
            utils.save_image(
                rec, f"sample/{str(idx).zfill(6)}.png",
                nrow=int(math.sqrt(args.batch)), normalize=True, value_range=(-1, 1),
            )

    if idx % 5000 == 0 and idx > 0:
        torch.save({
            'vq_model': model.state_dict(),
            'disc': discriminator.state_dict(),
            }, f"checkpoints/{str(idx).zfill(6)}.pt",
        )