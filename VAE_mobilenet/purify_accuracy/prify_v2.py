"""
purify_v2.py
============
fixes vs previous version:
  1. q_sample: closed-form one-shot forward (no loop over t steps)
  2. reverse:  samples z_{t-1} via posterior q(z_{t-1}|z_t, z0_hat), not z0 directly
  3. betas:    smaller range so useful t* window is wider
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import torchattacks
    HAS_TORCHATTACKS = True
except ImportError:
    HAS_TORCHATTACKS = False
    print("[warn] torchattacks not found. pip install torchattacks")

# ======================
# CONFIG
# ======================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 2e-4
T_MAX      = 200
T_STARS    = [5, 10, 20, 40, 80]
N_TEST     = 256
IMG_SIZE = 64
DFIRE_ROOT = "./D-Fire" # 你的D-Fire路径，改成实际路径
# adversarial attack
EPS        = 8 / 255
ALPHA      = 2 / 255
PGD_STEPS  = 20

# ======================
# BETA SCHEDULE
# ======================
def get_schedule(T=T_MAX, b_start=1e-4, b_end=0.02):
    """
    Linear beta schedule.
    Also precomputes cumulative flip probability:
      flip_prob[t] = 0.5 * (1 - prod_{s=0}^{t}(1 - 2*beta_s))
    This is the closed-form marginal: q(z_t=1 | z_0=0) = flip_prob[t]
    """
    betas     = torch.linspace(b_start, b_end, T, device=DEVICE)
    retention = torch.cumprod(1.0 - 2.0 * betas, dim=0)
    flip_prob = 0.5 * (1.0 - retention)   # shape (T,)
    return betas, flip_prob

# ======================
# FORWARD PROCESS  [FIX 1]
# ======================
def q_sample(z0, t, flip_prob):
    """
    One-shot closed-form forward: sample z_t from z_0 at timestep t.
    No loop needed — uses cumulative flip probability directly.

    z0        : (B, D) long {0,1}
    t         : (B,)   long, 0-indexed timestep per sample
    flip_prob : (T,)   float, precomputed cumulative flip probs

    Returns z_t : (B, D) long {0,1}
    """
    # gather per-sample flip probability
    p = flip_prob[t].unsqueeze(1)                        # (B, 1)
    p = p.expand_as(z0.float())                          # (B, D)
    noise = torch.bernoulli(p)                           # 1 = flip this bit
    z_t   = (z0.float() + noise) % 2
    return z_t.long()

# ======================
# POSTERIOR  [KEY FOR FIX 2]
# ======================
def q_posterior_prob(z0_hat, z_t, t, betas, flip_prob):
    """
    Compute P(z_{t-1} = 1 | z_t, z0_hat) analytically.

    For binary flip diffusion the posterior factors as:
      q(z_{t-1} | z_t, z0) ∝ q(z_t | z_{t-1}) * q(z_{t-1} | z0)

    All terms are Bernoulli, so this has a closed form.

    z0_hat    : (B, D) float, predicted prob(z0=1) from model
    z_t       : (B, D) long  {0,1}
    t         : (B,)   long  (1-indexed, t >= 1)
    betas     : (T,)   float
    flip_prob : (T,)   float, cumulative

    Returns post_1 : (B, D) float, prob(z_{t-1}=1)
    """
    beta_t  = betas[t - 1].unsqueeze(1)                 # (B,1) one-step flip prob
    # cumulative flip prob at t-1  (t is 1-indexed, so t-1 maps to index t-2)
    fp_tm1  = torch.where(
        t > 1,
        flip_prob[t - 2],
        torch.zeros_like(betas[0]).expand(t.shape[0])
    ).unsqueeze(1)                                       # (B,1)

    zt  = z_t.float()
    z0p = z0_hat                                         # prob(z0=1), (B,D)

    # q(z_t | z_{t-1} = v):
    #   if v=1: prob of seeing z_t =  (1-beta)*z_t + beta*(1-z_t)
    #   if v=0: prob of seeing z_t =  beta*z_t + (1-beta)*(1-z_t)
    q_zt_given_1 = (1 - beta_t) * zt       + beta_t * (1 - zt)
    q_zt_given_0 = beta_t       * zt       + (1 - beta_t) * (1 - zt)

    # q(z_{t-1} = v | z0):
    #   prob(z_{t-1}=1 | z0=1) = 1 - fp_tm1
    #   prob(z_{t-1}=1 | z0=0) = fp_tm1
    # marginalising over predicted z0:
    q_zm1_eq_1 = (1 - fp_tm1) * z0p + fp_tm1 * (1 - z0p)

    num_1 = q_zt_given_1 * q_zm1_eq_1
    num_0 = q_zt_given_0 * (1 - q_zm1_eq_1)

    post_1 = num_1 / (num_1 + num_0 + 1e-8)
    return post_1                                        # (B,D) float in [0,1]

# ======================
# MODEL
# ======================
# ======================
# UNET BUILDING BLOCKS
# adapted from cloneofsimo/d3pm
# ======================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.dropout(self.conv2(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn  = torch.softmax(torch.bmm(q.transpose(1,2), k) * scale, dim=-1)
        out   = torch.bmm(v, attn.transpose(1,2)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    UNet denoiser for binary images.
    Input/output: (B, C, H, W) float logits.
    Timestep embedded via sinusoidal + MLP.

    For CIFAR-10: C=3, H=W=32.
    ch_mult controls channel widths at each resolution level.
    """
    def __init__(self, in_ch=3, base_ch=32, ch_mult=(1,2), T=T_MAX,
                 attn_resolutions=(8,), dropout=0.1):
        super().__init__()
        t_dim = base_ch * 4

        # timestep embedding: sinusoidal → MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        chs = [base_ch * m for m in ch_mult]  # channel sizes per level

        # input projection
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # encoder
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = base_ch
        self.enc_chs = [ch_in]

        for i, ch_out in enumerate(chs):
            res = 32 // (2 ** i)
            self.downs.append(nn.ModuleList([
                ResidualBlock(ch_in, ch_out, t_dim, dropout),
                ResidualBlock(ch_out, ch_out, t_dim, dropout),
                AttentionBlock(ch_out) if res in attn_resolutions else nn.Identity(),
            ]))
            self.enc_chs.append(ch_out)
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch_out, ch_out, 4, 2, 1))
            else:
                self.downsamples.append(nn.Identity())
            ch_in = ch_out

        # bottleneck
        self.mid1 = ResidualBlock(chs[-1], chs[-1], t_dim, dropout)
        self.mid_attn = AttentionBlock(chs[-1])
        self.mid2 = ResidualBlock(chs[-1], chs[-1], t_dim, dropout)

        # decoder
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_chs = list(reversed(chs))
        rev_enc = list(reversed(self.enc_chs[1:]))

        for i, (ch_out, skip_ch) in enumerate(zip(rev_chs, rev_enc + [base_ch])):
            res = 32 // (2 ** (len(chs) - 1 - i))
            self.ups.append(nn.ModuleList([
                ResidualBlock(ch_in + skip_ch, ch_out, t_dim, dropout),
                ResidualBlock(ch_out, ch_out, t_dim, dropout),
                AttentionBlock(ch_out) if res in attn_resolutions else nn.Identity(),
            ]))
            if i < len(rev_chs) - 1:
                self.upsamples.append(
                    nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(ch_out, ch_out, 3, padding=1)))
            else:
                self.upsamples.append(nn.Identity())
            ch_in = ch_out

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    @staticmethod
    def sinusoidal_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args  = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t):
        """
        x : (B, C, H, W) float — binary image as float {0.,1.}
        t : (B,) long — timestep

        Returns logits (B, C, H, W), same shape as x.
        """
        # timestep embedding
        t_emb = self.sinusoidal_embedding(t, self.t_mlp[0].in_features)
        t_emb = self.t_mlp(t_emb)

        h = self.in_conv(x)

        # encoder with skip connections
        skips = [h]
        for (r1, r2, attn), ds in zip(self.downs, self.downsamples):
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = ds(h)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # decoder
        for (r1, r2, attn), us in zip(self.ups, self.upsamples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            h = attn(h)
            h = us(h)

        return self.out_conv(F.silu(self.out_norm(h)))   # (B, C, H, W) logits


class ReverseModel(nn.Module):
    """
    Wrapper around UNet that handles the (B,D) ↔ (B,C,H,W) reshape.
    Keeps the same forward signature as the MLP:
      forward(zt: (B,D) long, t: (B,) long) → logits (B,D)
    so that compute_loss, reverse, evaluate are unchanged.
    """
    def __init__(self, C=3, H=32, W=32, T=T_MAX,
                 base_ch=32, ch_mult=(1,2)):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.unet = UNet(in_ch=C, base_ch=base_ch,
                         ch_mult=ch_mult, T=T)

    def forward(self, zt, t):
        """
        zt : (B, D) long {0,1}  where D = C*H*W
        t  : (B,) long
        Returns logits (B, D)
        """
        B = zt.shape[0]
        # reshape flat binary → image float
        x = zt.float().view(B, self.C, self.H, self.W)
        # UNet forward
        logits_img = self.unet(x, t)                    # (B, C, H, W)
        # flatten back
        return logits_img.view(B, -1)                   # (B, D)

# ======================
# LOSS
# ======================
def compute_loss(model, z0, betas, flip_prob):
    """
    Hybrid D3PM loss:
      L = CE(z0_hat, z0)  +  lambda * KL(q_posterior || p_theta_posterior)

    The CE term is the dominant signal for the MLP.
    KL term adds structural consistency with the diffusion posterior.
    """
    B, D = z0.shape
    # sample random timestep per sample (1-indexed)
    t = torch.randint(1, T_MAX, (B,), device=DEVICE)

    # closed-form forward sample  [FIX 1 in use]
    z_t = q_sample(z0, t - 1, flip_prob)                # t-1 because flip_prob is 0-indexed

    # model predicts z0 logits
    logits  = model(z_t, t)
    z0_pred = torch.sigmoid(logits)                      # prob(z0=1)

    # CE loss: predict z0 from z_t
    ce_loss = F.binary_cross_entropy_with_logits(logits, z0.float())

    # KL loss: match posterior distributions
    post_true = q_posterior_prob(z0.float(), z_t, t, betas, flip_prob)
    post_pred = q_posterior_prob(z0_pred,    z_t, t, betas, flip_prob)

    eps = 1e-6
    kl = (post_true * (torch.log(post_true + eps) - torch.log(post_pred + eps)) +
          (1 - post_true) * (torch.log(1 - post_true + eps) -
                             torch.log(1 - post_pred + eps)))
    kl_loss = kl.mean()

    return ce_loss + 0.001 * kl_loss, ce_loss.item(), kl_loss.item()

# ======================
# REVERSE PROCESS  [FIX 2]
# ======================
@torch.no_grad()
def reverse(model, z_t_start, t_start, betas, flip_prob):
    """
    Correct ancestral sampling:
      for t = t_start down to 1:
        1. predict z0 from (z_t, t)
        2. compute posterior q(z_{t-1} | z_t, z0_hat)
        3. sample z_{t-1} from that posterior   ← KEY FIX
    """
    z = z_t_start.clone()

    for t_val in range(t_start, 0, -1):
        t_batch = torch.full((z.shape[0],), t_val,
                             device=DEVICE, dtype=torch.long)

        logits  = model(z, t_batch)
        z0_pred = torch.sigmoid(logits)                  # prob(z0=1), (B,D)

        if t_val == 1:
            # final step: just take the argmax prediction
            z = (z0_pred > 0.5).long()
        else:
            # sample z_{t-1} via the true posterior  [FIX 2]
            post_1 = q_posterior_prob(z0_pred, z, t_batch, betas, flip_prob)
            z      = torch.bernoulli(post_1).long()

    return z

# ======================
# DATA
# ======================
from torch.utils.data import Dataset
from PIL import Image
import glob

class DFireDataset(Dataset):
    def __init__(self, root, split="train", size=IMG_SIZE):
        # D-Fire目录结构: root/train/images/*.jpg, root/test/images/*.jpg
        self.paths = glob.glob(f"{root}/{split}/images/*.jpg")
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), 0   # 标签用0占位，训练不需要label

def get_loaders():
    train_ds = DFireDataset(DFIRE_ROOT, split="train")
    test_ds  = DFireDataset(DFIRE_ROOT, split="test")
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_ld  = torch.utils.data.DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"D-Fire: {len(train_ds)} train / {len(test_ds)} test images")
    return train_ld, test_ld

# ======================
# ADVERSARIAL ATTACK
# ======================
def get_attacker(classifier):
    if HAS_TORCHATTACKS:
        return torchattacks.PGD(classifier, eps=EPS, alpha=ALPHA, steps=PGD_STEPS)
    # manual PGD fallback
    def pgd(x, y):
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-EPS, EPS)
        x_adv = x_adv.clamp(0, 1)
        for _ in range(PGD_STEPS):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(classifier(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + ALPHA * grad.sign()
            x_adv = x + (x_adv - x).clamp(-EPS, EPS)
            x_adv = x_adv.clamp(0, 1).detach()
        return x_adv
    return pgd

# ======================
# TRAIN
# ======================
def train_epoch(model, loader, betas, flip_prob, opt):
    model.train()
    total, n = 0.0, 0
    for x, _ in tqdm(loader, desc="train", leave=False):
        x  = x.to(DEVICE)
        z0 = binarize(x)
        opt.zero_grad()
        loss, ce, kl = compute_loss(model, z0, betas, flip_prob)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
        n     += x.size(0)
    return total / n

# ======================
# EVALUATE
# ======================
@torch.no_grad()
def evaluate(model, loader, betas, flip_prob):
    model.eval()

    # collect N_TEST clean samples
    x_list = []
    for x, _ in loader:
        x_list.append(x.to(DEVICE))
        if sum(b.size(0) for b in x_list) >= N_TEST:
            break
    x_all   = torch.cat(x_list)[:N_TEST]
    z_clean = binarize(x_all)                            # (N, D)

    # baseline: reconstruction from t=0 (no noise added, just model passthrough)
    t0 = torch.zeros(z_clean.size(0), device=DEVICE, dtype=torch.long) + 1
    logits_t0 = model(z_clean, t0)
    z_rec_t0  = (torch.sigmoid(logits_t0) > 0.5).long()
    baseline_self = (z_rec_t0 == z_clean).float().mean().item()
    print(f"\n  [sanity] self-reconstruction acc at t=1: {baseline_self:.4f}"
          f"  (should be high if model learned anything)")

    results = {}
    for t_star in T_STARS:
        # forward noise z_clean → z_t*
        t_vec = torch.full((z_clean.size(0),), t_star - 1,
                           device=DEVICE, dtype=torch.long)
        z_noised = q_sample(z_clean, t_vec, flip_prob)

        # reverse denoise z_t* → z_rec
        z_rec = reverse(model, z_noised, t_star, betas, flip_prob)

        acc = (z_rec == z_clean).float().mean().item()
        results[t_star] = acc
        print(f"  t*={t_star:3d}  bit_acc={acc:.4f}")

    return results

# ======================
# BINARY-SPACE ATTACK
# ======================
def binary_attack(z_clean, flip_rate):
    """
    Simulate adversarial attack directly in binary space
    by randomly flipping a fraction of bits.

    flip_rate : float, fraction of bits to flip
                0.03  ≈ mild  (analogous to PGD 8/255 effect after binarization)
                0.10  ≈ medium
                0.20  ≈ strong
                0.30  ≈ very strong
    """
    mask = torch.bernoulli(
        torch.full_like(z_clean.float(), flip_rate)
    )
    return (z_clean + mask.long()) % 2

# ======================
# EVALUATE ADVERSARIAL (binary-space attack)
# ======================
@torch.no_grad()
def evaluate_adv(model, loader, betas, flip_prob,
                 flip_rates=(0.03, 0.10, 0.20, 0.30)):
    """
    For each flip_rate (attack strength), measure bit accuracy
    after purification across all T_STARS.

    Returns:
      all_results : {flip_rate: {t_star: bit_acc}}
      baselines   : {flip_rate: bit_acc_without_purification}
    """
    model.eval()

    # collect N_TEST clean samples
    x_list = []
    for x, _ in loader:
        x_list.append(x.to(DEVICE))
        if sum(b.size(0) for b in x_list) >= N_TEST:
            break
    z_clean = binarize(torch.cat(x_list)[:N_TEST])   # (N, D)

    all_results = {}
    baselines   = {}

    for flip_rate in flip_rates:
        print(f"\n  Attack flip_rate={flip_rate:.2f} "
              f"({flip_rate*100:.0f}% bits flipped)")

        z_adv = binary_attack(z_clean, flip_rate)

        # baseline: no purification
        base = (z_adv == z_clean).float().mean().item()
        baselines[flip_rate] = base
        print(f"    baseline bit_acc (no purif): {base:.4f}")

        results = {}
        for t_star in T_STARS:
            t_vec    = torch.full((z_adv.size(0),), t_star - 1,
                                  device=DEVICE, dtype=torch.long)
            z_noised = q_sample(z_adv, t_vec, flip_prob)
            z_rec    = reverse(model, z_noised, t_star, betas, flip_prob)
            acc      = (z_rec == z_clean).float().mean().item()
            results[t_star] = acc
            print(f"    t*={t_star:3d}  bit_acc_after_purif={acc:.4f}  "
                  f"delta={acc-base:+.4f}")

        all_results[flip_rate] = results

    return all_results, baselines

# ======================
# COMPARISON PLOT
# ======================
def plot_comparison(results_clean, all_results_adv, baselines):
    """
    One plot with multiple curves:
      blue              — clean upper bound
      colored lines     — purification at different attack strengths
      dashed same color — baseline (no purification) for each strength
    """
    ts         = list(results_clean.keys())
    accs_clean = list(results_clean.values())

    colors = ['#E8593C', '#F5A623', '#7B68EE', '#1D9E75']
    fig, ax = plt.subplots(figsize=(9, 5))

    # clean upper bound
    ax.plot(ts, accs_clean, marker='o', color='#3B8BD4',
            linewidth=2.5, label='clean → noise → denoise (upper bound)', zorder=5)

    for (flip_rate, results), color in zip(all_results_adv.items(), colors):
        accs = list(results.values())
        base = baselines[flip_rate]
        label = f'flip={flip_rate:.0%} → purif'

        ax.plot(ts, accs, marker='s', color=color,
                linewidth=1.8, label=label)
        ax.axhline(base, color=color, linestyle='--', alpha=0.4,
                   label=f'flip={flip_rate:.0%} baseline (no purif): {base:.3f}')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6,
               label='random (0.50)')

    ax.set_xlabel("t*", fontsize=12)
    ax.set_ylabel("Bit accuracy (vs z_clean)", fontsize=12)
    ax.set_title("Binary Diffusion Purification — Attack Strength vs Recovery", fontsize=13)
    ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig("purify_comparison.png", dpi=150)
    plt.show()
    print("Saved purify_comparison.png")

    # summary table
    print(f"\n{'flip_rate':<12} {'t*':<6} {'base':<8} {'purif':<8} {'delta':<8}")
    print("-" * 46)
    for flip_rate, results in all_results_adv.items():
        base = baselines[flip_rate]
        for t_star, acc in results.items():
            print(f"{flip_rate:<12.2f} {t_star:<6} {base:<8.4f} "
                  f"{acc:<8.4f} {acc-base:+.4f}")

# ======================
# MAIN
# ======================
def main():
    print(f"Device: {DEVICE}")
    betas, flip_prob = get_schedule()

    print(f"flip_prob at t=10:  {flip_prob[9]:.4f}")
    print(f"flip_prob at t=50:  {flip_prob[49]:.4f}")
    print(f"flip_prob at t=100: {flip_prob[99]:.4f}")
    print(f"flip_prob at t=200: {flip_prob[199]:.4f}  (→ ~0.5)")

    train_loader, test_loader = get_loaders()

    # UNet: base_ch=64, ch_mult=(1,2,2) → ~6M params, fits on CPU/GPU
    # For faster CPU runs: base_ch=32, ch_mult=(1,2)
    model = ReverseModel(C=3, H=IMG_SIZE, W=IMG_SIZE, T=T_MAX,
                         base_ch=32, ch_mult=(1,2,2)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet param count: {total_params:,}")

    import os
    CKPT = "purify_unet.pt"
    if os.path.exists(CKPT):
        print(f"\nLoading existing checkpoint {CKPT} ...")
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    else:
        print("\nTraining UNet reverse model on clean CIFAR-10...")
        for ep in range(EPOCHS):
            loss = train_epoch(model, train_loader, betas, flip_prob, opt)
            print(f"  Epoch {ep+1}/{EPOCHS}  loss={loss:.4f}")
        torch.save(model.state_dict(), CKPT)
        print(f"Checkpoint saved to {CKPT}.")

    # evaluate on clean images (upper bound)
    print("\n--- Clean image purification (upper bound) ---")
    results_clean = evaluate(model, test_loader, betas, flip_prob)

    # evaluate with binary-space attacks at multiple strengths
    print("\n--- Binary-space attack purification ---")
    all_results_adv, baselines = evaluate_adv(
        model, test_loader, betas, flip_prob,
        flip_rates=(0.03, 0.10, 0.20, 0.30)
    )

    plot_comparison(results_clean, all_results_adv, baselines)

if __name__ == "__main__":
    main()