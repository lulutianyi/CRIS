"""
purify_bld.py
=============
Two-stage binary diffusion purification with BLD-style autoencoder.

Stage 1 — Train autoencoder (Encoder + Decoder) on D-Fire:
  x → Encoder → continuous latent → STE binary → z ∈ {0,1}^D
  z → Decoder → x_recon
  Loss: MSE reconstruction + binary regularization

Stage 2 — Train UNet denoiser on binary latents:
  freeze Encoder, encode D-Fire → binary latents
  train UNet to denoise binary latents (same as purify_v2.py)

Inference (purification):
  x_adv → Encoder → z_adv → add binary noise t* steps
        → UNet reverse denoise → z_clean_hat → Decoder → x_clean_hat
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==============================================================================
#  CONFIG
# ==============================================================================

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DFIRE_ROOT  = "/content/drive/MyDrive/D-Fire"  # ← change to your path

IMG_SIZE    = 128          # resize D-Fire to 128x128
BATCH_SIZE  = 32
LR_AE       = 1e-4        # autoencoder learning rate
LR_DIFF     = 2e-4        # diffusion learning rate
EPOCHS_AE   = 20          # stage 1 epochs
EPOCHS_DIFF = 20          # stage 2 epochs

# autoencoder architecture
BASE_CH         = 32
CH_MULT_ENC     = (1, 2, 4)   # 128→64→32→16, latent 16x16
LATENT_CHANNELS = 8
NUM_RES_BLOCKS  = 2

# diffusion schedule
T_MAX   = 200
T_STARS = [5, 10, 20, 40, 80]
N_TEST  = 256

# binary attack strengths for evaluation
FLIP_RATES = (0.03, 0.10, 0.20, 0.30)

# checkpoint paths
CKPT_AE   = "bld_autoencoder.pt"
CKPT_DIFF = "bld_diffusion.pt"

# ==============================================================================
#  DATASET
# ==============================================================================

class DFireDataset(Dataset):
    def __init__(self, root, split="train", size=IMG_SIZE):
        self.paths = glob.glob(f"{root}/{split}/images/*.jpg")
        if len(self.paths) == 0:
            # fallback: try png
            self.paths = glob.glob(f"{root}/{split}/images/*.png")
        assert len(self.paths) > 0, f"No images found at {root}/{split}/images/"
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),             # [0,1] float
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), 0        # label=0 placeholder

def get_loaders():
    train_ds = DFireDataset(DFIRE_ROOT, split="train")
    test_ds  = DFireDataset(DFIRE_ROOT, split="test")
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
    print(f"D-Fire: {len(train_ds)} train / {len(test_ds)} test images")
    return train_ld, test_ld

# ==============================================================================
#  AUTOENCODER BUILDING BLOCKS
# ==============================================================================

class Normalize(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gn = nn.GroupNorm(min(32, ch), ch)
    def forward(self, x):
        return self.gn(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_channels=None):
        super().__init__()
        out_ch = out_channels or in_ch
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = Normalize(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = Normalize(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape
        q = self.q(h).reshape(B, C, -1).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * C**-0.5, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

# ==============================================================================
#  ENCODER  (with STE binary quantization)
# ==============================================================================

class Encoder(nn.Module):
    """
    Convolutional encoder → continuous latent → STE binary quantization → z∈{0,1}.
    Straight-through estimator allows gradients to flow through the binary step.
    """
    def __init__(self, in_channels=3, base_ch=BASE_CH,
                 num_res_blocks=NUM_RES_BLOCKS,
                 latent_channels=LATENT_CHANNELS,
                 ch_mult=CH_MULT_ENC):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_ch, 3, padding=1)]
        in_ch = base_ch
        for mul in ch_mult:
            out_ch = base_ch * mul
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            layers.append(Downsample(in_ch))
        layers += [
            ResnetBlock(in_ch),
            AttnBlock(in_ch),
            ResnetBlock(in_ch),
            Normalize(in_ch),
            nn.Conv2d(in_ch, latent_channels, 3, padding=1),
        ]
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x, hard=True):
        """
        x    : (B, 3, H, W) float [0,1]
        hard : if True, return binary {0,1} via STE
               if False, return continuous logits (for visualization)

        Returns z : (B, latent_channels, H', W') float
                    hard=True  → values in {0.,1.}, gradients via STE
                    hard=False → continuous values
        """
        logits = self.conv_layers(x)          # continuous latent

        if not hard:
            return torch.sigmoid(logits)       # continuous [0,1]

        # Straight-through estimator:
        # forward:  hard binary threshold
        # backward: gradient of sigmoid (smooth surrogate)
        z_soft = torch.sigmoid(logits)
        z_hard = (z_soft > 0.5).float()
        z_ste  = z_soft + (z_hard - z_soft).detach()   # STE
        return z_ste                           # {0,1} in forward, smooth in backward

# ==============================================================================
#  DECODER
# ==============================================================================

class Decoder(nn.Module):
    def __init__(self, out_channels=3, base_ch=BASE_CH,
                 num_res_blocks=NUM_RES_BLOCKS,
                 latent_channels=LATENT_CHANNELS,
                 ch_mult=CH_MULT_ENC):
        super().__init__()
        ch_mult_dec = tuple(reversed(ch_mult))
        in_ch = base_ch * ch_mult[-1]
        layers = [
            nn.Conv2d(latent_channels, in_ch, 3, padding=1),
            ResnetBlock(in_ch),
            AttnBlock(in_ch),
            ResnetBlock(in_ch),
        ]
        for mul in ch_mult_dec:
            out_ch = base_ch * mul
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            layers.append(Upsample(in_ch))
        layers += [
            Normalize(in_ch),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.Sigmoid(),                      # output in [0,1]
        ]
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.conv_layers(z)

# ==============================================================================
#  AUTOENCODER WRAPPER
# ==============================================================================

class BinaryAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # compute latent spatial size: IMG_SIZE / 2^len(CH_MULT_ENC)
        self.latent_h = IMG_SIZE // (2 ** len(CH_MULT_ENC))
        self.latent_w = self.latent_h
        self.D = LATENT_CHANNELS * self.latent_h * self.latent_w
        print(f"Latent space: ({LATENT_CHANNELS}, {self.latent_h}, {self.latent_w})"
              f"  D={self.D}")

    def encode_binary(self, x):
        """x → binary latent z, shape (B, D) long {0,1}"""
        with torch.no_grad():
            z = self.encoder(x, hard=True)        # (B, C, H', W')
        return z.long().view(z.shape[0], -1)       # (B, D)

    def decode(self, z_flat):
        """z_flat (B, D) float → reconstructed image (B, 3, H, W)"""
        B = z_flat.shape[0]
        z = z_flat.view(B, LATENT_CHANNELS, self.latent_h, self.latent_w)
        return self.decoder(z)

    def forward(self, x):
        """Full autoencoder forward for training."""
        z     = self.encoder(x, hard=True)         # (B, C, H', W') STE binary
        x_rec = self.decoder(z)                    # (B, 3, H, W)
        return x_rec, z

# ==============================================================================
#  STAGE 1: TRAIN AUTOENCODER
# ==============================================================================

def train_autoencoder(ae, train_loader, test_loader):
    opt = torch.optim.Adam(ae.parameters(), lr=LR_AE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS_AE)

    best_psnr = 0
    for epoch in range(EPOCHS_AE):
        ae.train()
        total_loss, n = 0.0, 0

        for x, _ in tqdm(train_loader, desc=f"AE epoch {epoch+1}", leave=False):
            x = x.to(DEVICE)
            x_rec, z = ae(x)

            # reconstruction loss
            mse  = F.mse_loss(x_rec, x)

            # binary regularization: push latent toward 0 or 1
            # entropy of Bernoulli(sigmoid(logits)) should be low
            z_soft = z.float()
            bin_reg = -(z_soft * torch.log(z_soft + 1e-6) +
                        (1 - z_soft) * torch.log(1 - z_soft + 1e-6)).mean()

            loss = mse + 0.01 * bin_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.shape[0]
            n          += x.shape[0]

        scheduler.step()
        avg_loss = total_loss / n

        # validation PSNR
        psnr = evaluate_ae(ae, test_loader)
        print(f"  AE Epoch {epoch+1}/{EPOCHS_AE}  "
              f"loss={avg_loss:.4f}  PSNR={psnr:.2f}dB")

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(ae.state_dict(), CKPT_AE)
            print(f"    ✓ checkpoint saved (best PSNR={best_psnr:.2f}dB)")

    print(f"Stage 1 done. Best PSNR: {best_psnr:.2f}dB")
    ae.load_state_dict(torch.load(CKPT_AE, map_location=DEVICE))
    return ae

@torch.no_grad()
def evaluate_ae(ae, loader, n_batches=5):
    ae.eval()
    psnrs = []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x     = x.to(DEVICE)
        x_rec, _ = ae(x)
        mse   = F.mse_loss(x_rec, x, reduction='none').mean(dim=(1,2,3))
        psnr  = (-10 * torch.log10(mse + 1e-8)).mean().item()
        psnrs.append(psnr)
    return np.mean(psnrs)

# ==============================================================================
#  BINARY DIFFUSION  (same math as purify_v2.py)
# ==============================================================================

def get_schedule(T=T_MAX, b_start=1e-4, b_end=0.02):
    betas     = torch.linspace(b_start, b_end, T, device=DEVICE)
    retention = torch.cumprod(1.0 - 2.0 * betas, dim=0)
    flip_prob = 0.5 * (1.0 - retention)
    return betas, flip_prob

def q_sample(z0, t, flip_prob):
    p    = flip_prob[t].unsqueeze(1).expand_as(z0.float())
    mask = torch.bernoulli(p)
    return ((z0.float() + mask) % 2).long()

def q_posterior_prob(z0_pred, z_t, t, betas, flip_prob):
    beta_t = betas[t - 1].unsqueeze(1)
    fp_tm1 = torch.where(
        t > 1,
        flip_prob[t - 2],
        torch.zeros(1, device=DEVICE).expand(t.shape[0])
    ).unsqueeze(1)
    zt  = z_t.float()
    z0p = z0_pred
    q1  = (1 - beta_t) * zt       + beta_t * (1 - zt)
    q0  = beta_t       * zt       + (1 - beta_t) * (1 - zt)
    e1  = (1 - fp_tm1) * z0p + fp_tm1 * (1 - z0p)
    n1  = q1 * e1
    n0  = q0 * (1 - e1)
    return n1 / (n1 + n0 + 1e-8)

# ==============================================================================
#  UNET DENOISER  (same as purify_v2.py, adapted for D here)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1  = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.drop   = nn.Dropout(dropout)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.drop(self.conv2(F.silu(self.norm2(h))))
        return h + self.skip(x)

class SelfAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H*W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k) * C**-0.5, dim=-1)
        out  = torch.bmm(v, attn.transpose(1,2)).reshape(B, C, H, W)
        return x + self.proj(out)

class DiffUNet(nn.Module):
    """
    UNet denoiser operating on binary latent space.
    Input/output: (B, LATENT_CHANNELS, H', W')
    Timestep via sinusoidal embedding.
    """
    def __init__(self, in_ch=LATENT_CHANNELS, base_ch=64,
                 ch_mult=(1,2,2), T=T_MAX,
                 latent_size=None, attn_res=(4,8), dropout=0.1):
        super().__init__()
        if latent_size is None:
            latent_size = IMG_SIZE // (2 ** len(CH_MULT_ENC))
        t_dim = base_ch * 4

        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        chs = [base_ch * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # encoder
        self.downs       = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = base_ch
        self.enc_chs = [ch_in]

        for i, ch_out in enumerate(chs):
            res = latent_size // (2 ** i)
            self.downs.append(nn.ModuleList([
                ResidualBlock(ch_in, ch_out, t_dim, dropout),
                ResidualBlock(ch_out, ch_out, t_dim, dropout),
                SelfAttn(ch_out) if res in attn_res else nn.Identity(),
            ]))
            self.enc_chs.append(ch_out)
            ds = nn.Conv2d(ch_out, ch_out, 4, 2, 1) if i < len(chs)-1 else nn.Identity()
            self.downsamples.append(ds)
            ch_in = ch_out

        # bottleneck
        self.mid1     = ResidualBlock(chs[-1], chs[-1], t_dim, dropout)
        self.mid_attn = SelfAttn(chs[-1])
        self.mid2     = ResidualBlock(chs[-1], chs[-1], t_dim, dropout)

        # decoder
        self.ups       = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_chs = list(reversed(chs))
        rev_enc = list(reversed(self.enc_chs[:-1]))

        for i, (ch_out, skip_ch) in enumerate(zip(rev_chs, rev_enc + [base_ch])):
            res = latent_size // (2 ** (len(chs) - 1 - i))
            self.ups.append(nn.ModuleList([
                ResidualBlock(ch_in + skip_ch, ch_out, t_dim, dropout),
                ResidualBlock(ch_out, ch_out, t_dim, dropout),
                SelfAttn(ch_out) if res in attn_res else nn.Identity(),
            ]))
            us = (nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(ch_out, ch_out, 3, padding=1))
                  if i < len(rev_chs)-1 else nn.Identity())
            self.upsamples.append(us)
            ch_in = ch_out

        self.out_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    @staticmethod
    def sin_embed(t, dim):
        half  = dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device).float() *
                          (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t):
        t_emb = self.t_mlp(self.sin_embed(t, self.t_mlp[0].in_features))
        h     = self.in_conv(x)
        skips = [h]

        for (r1, r2, attn), ds in zip(self.downs, self.downsamples):
            h = attn(r2(r1(h, t_emb), t_emb))
            skips.append(h)
            h = ds(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for (r1, r2, attn), us in zip(self.ups, self.upsamples):
            h = us(attn(r2(r1(torch.cat([h, skips.pop()], 1), t_emb), t_emb)))

        return self.out_conv(F.silu(self.out_norm(h)))   # logits (B, C, H', W')

class ReverseModel(nn.Module):
    """
    Wrapper: (B,D) long ↔ (B,C,H',W') float for DiffUNet.
    Keeps same interface as purify_v2.py so all downstream functions work unchanged.
    """
    def __init__(self, D, latent_ch=LATENT_CHANNELS,
                 latent_h=None, latent_w=None):
        super().__init__()
        if latent_h is None:
            latent_h = IMG_SIZE // (2 ** len(CH_MULT_ENC))
        self.C, self.H, self.W = latent_ch, latent_h, latent_h
        self.unet = DiffUNet(in_ch=latent_ch, latent_size=latent_h)

    def forward(self, zt, t):
        B  = zt.shape[0]
        x  = zt.float().view(B, self.C, self.H, self.W)
        return self.unet(x, t).view(B, -1)   # (B, D) logits

# ==============================================================================
#  STAGE 2: TRAIN DIFFUSION ON BINARY LATENTS
# ==============================================================================

def compute_loss(model, z0, betas, flip_prob):
    B, D = z0.shape
    t    = torch.randint(1, T_MAX, (B,), device=DEVICE)
    z_t  = q_sample(z0, t - 1, flip_prob)
    logits   = model(z_t, t)
    ce_loss  = F.binary_cross_entropy_with_logits(logits, z0.float())
    z0_pred  = torch.sigmoid(logits)
    pt  = q_posterior_prob(z0_pred, z_t, t, betas, flip_prob)
    tt  = q_posterior_prob(z0.float(), z_t, t, betas, flip_prob)
    eps = 1e-6
    kl  = (tt * (torch.log(tt+eps) - torch.log(pt+eps)) +
           (1-tt) * (torch.log(1-tt+eps) - torch.log(1-pt+eps))).mean()
    return ce_loss + 0.001 * kl, ce_loss.item(), kl.item()

def train_diffusion(diff_model, ae, train_loader, betas, flip_prob):
    ae.eval()   # encoder frozen
    opt = torch.optim.Adam(diff_model.parameters(), lr=LR_DIFF)

    for epoch in range(EPOCHS_DIFF):
        diff_model.train()
        total, n = 0.0, 0
        for x, _ in tqdm(train_loader, desc=f"Diff epoch {epoch+1}", leave=False):
            x  = x.to(DEVICE)
            z0 = ae.encode_binary(x)              # (B, D) long {0,1}
            opt.zero_grad()
            loss, ce, kl = compute_loss(diff_model, z0, betas, flip_prob)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.shape[0]
            n     += x.shape[0]
        print(f"  Diff Epoch {epoch+1}/{EPOCHS_DIFF}  loss={total/n:.4f}")

    torch.save(diff_model.state_dict(), CKPT_DIFF)
    print(f"Diffusion checkpoint saved to {CKPT_DIFF}")

# ==============================================================================
#  REVERSE SAMPLING
# ==============================================================================

@torch.no_grad()
def reverse(model, z_t_start, t_start, betas, flip_prob):
    z = z_t_start.clone()
    for t_val in range(t_start, 0, -1):
        tb      = torch.full((z.shape[0],), t_val, device=DEVICE, dtype=torch.long)
        z0_pred = torch.sigmoid(model(z, tb))
        if t_val == 1:
            z = (z0_pred > 0.5).long()
        else:
            post = q_posterior_prob(z0_pred, z, tb, betas, flip_prob)
            z    = torch.bernoulli(post).long()
    return z

# ==============================================================================
#  EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(diff_model, ae, loader, betas, flip_prob,
             flip_rates=FLIP_RATES):
    diff_model.eval()
    ae.eval()

    # collect test binary latents
    z_list = []
    for x, _ in loader:
        z_list.append(ae.encode_binary(x.to(DEVICE)))
        if sum(b.shape[0] for b in z_list) >= N_TEST:
            break
    z_clean = torch.cat(z_list)[:N_TEST]   # (N, D)

    # clean upper bound
    results_clean = {}
    for t_star in T_STARS:
        t_vec    = torch.full((z_clean.shape[0],), t_star-1,
                              device=DEVICE, dtype=torch.long)
        z_noised = q_sample(z_clean, t_vec, flip_prob)
        z_rec    = reverse(diff_model, z_noised, t_star, betas, flip_prob)
        results_clean[t_star] = (z_rec == z_clean).float().mean().item()

    print("\n  [clean upper bound]")
    for t, acc in results_clean.items():
        print(f"    t*={t:3d}  bit_acc={acc:.4f}")

    # binary-space attacks at multiple strengths
    all_adv, baselines = {}, {}
    for fr in flip_rates:
        z_adv = ((z_clean.float() +
                  torch.bernoulli(torch.full_like(z_clean.float(), fr))) % 2).long()
        base  = (z_adv == z_clean).float().mean().item()
        baselines[fr] = base

        res = {}
        for t_star in T_STARS:
            t_vec    = torch.full((z_adv.shape[0],), t_star-1,
                                  device=DEVICE, dtype=torch.long)
            z_noised = q_sample(z_adv, t_vec, flip_prob)
            z_rec    = reverse(diff_model, z_noised, t_star, betas, flip_prob)
            acc      = (z_rec == z_clean).float().mean().item()
            res[t_star] = acc
        all_adv[fr] = res
        print(f"\n  [flip={fr:.0%}  baseline={base:.4f}]")
        for t, acc in res.items():
            print(f"    t*={t:3d}  bit_acc={acc:.4f}  delta={acc-base:+.4f}")

    plot_results(results_clean, all_adv, baselines)
    return results_clean, all_adv, baselines

# ==============================================================================
#  PLOT
# ==============================================================================

def plot_results(results_clean, all_adv, baselines):
    ts     = list(results_clean.keys())
    colors = ['#E8593C', '#F5A623', '#7B68EE', '#1D9E75']
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ts, list(results_clean.values()), marker='o', color='#3B8BD4',
            linewidth=2.5, label='clean upper bound', zorder=5)

    for (fr, res), col in zip(all_adv.items(), colors):
        base = baselines[fr]
        ax.plot(ts, list(res.values()), marker='s', color=col,
                linewidth=1.8, label=f'flip={fr:.0%} → purif')
        ax.axhline(base, color=col, linestyle='--', alpha=0.4,
                   label=f'flip={fr:.0%} baseline: {base:.3f}')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='random (0.50)')
    ax.set_xlabel("t*", fontsize=12)
    ax.set_ylabel("Bit accuracy (vs z_clean)", fontsize=12)
    ax.set_title("BLD Purification — Binary Latent Space", fontsize=13)
    ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig("bld_purify_curve.png", dpi=150)
    plt.show()
    print("Saved bld_purify_curve.png")

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print(f"Device: {DEVICE}")

    betas, flip_prob = get_schedule()
    train_loader, test_loader = get_loaders()

    # ── Stage 1: autoencoder ──────────────────────────────────────────────────
    ae = BinaryAutoEncoder().to(DEVICE)
    D  = ae.D

    if os.path.exists(CKPT_AE):
        print(f"\nLoading AE checkpoint {CKPT_AE} ...")
        ae.load_state_dict(torch.load(CKPT_AE, map_location=DEVICE))
        psnr = evaluate_ae(ae, test_loader)
        print(f"  AE PSNR: {psnr:.2f}dB")
    else:
        print("\n=== Stage 1: Training autoencoder ===")
        ae = train_autoencoder(ae, train_loader, test_loader)

    # ── Stage 2: diffusion ────────────────────────────────────────────────────
    diff_model = ReverseModel(D=D).to(DEVICE)
    total_params = sum(p.numel() for p in diff_model.parameters())
    print(f"\nDiffusion UNet params: {total_params:,}")

    if os.path.exists(CKPT_DIFF):
        print(f"Loading diffusion checkpoint {CKPT_DIFF} ...")
        diff_model.load_state_dict(torch.load(CKPT_DIFF, map_location=DEVICE))
    else:
        print("\n=== Stage 2: Training diffusion on binary latents ===")
        train_diffusion(diff_model, ae, train_loader, betas, flip_prob)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n=== Evaluation ===")
    evaluate(diff_model, ae, test_loader, betas, flip_prob)


if __name__ == "__main__":
    main()