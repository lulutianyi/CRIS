"""
diff_vs_t.py
============
Measures the bit-level distribution gap between clean and adversarial images
in binary latent space across different diffusion timesteps t.

Pipeline:
  x_clean / x_adv
      ↓  BLD Encoder (Bernoulli autoencoder) — or pixel-level binarization fallback
      ↓  Binary forward process (bit-flip, BLD-style)
      ↓  Measure (z_t_clean != z_t_adv).mean() at each t

Usage (Colab):
  !pip install torch torchvision torchattacks matplotlib

If you have BLD pretrained weights:
  Set USE_BLD_ENCODER = True and point BLD_CKPT_PATH to your checkpoint.
  The encoder is loaded from JiauZhang/binary-latent-diffusion (model/vq.py).

Fallback (no pretrained weights):
  USE_BLD_ENCODER = False  →  raw pixel binarization (threshold at 0.5)
  Good enough for the gap-measurement experiment.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# ── optional: adversarial attack library ──────────────────────────────────────
try:
    import torchattacks
    HAS_TORCHATTACKS = True
except ImportError:
    HAS_TORCHATTACKS = False
    print("[warn] torchattacks not found. Install with: pip install torchattacks")
    print("       Falling back to manual PGD implementation.")

# ==============================================================================
#  CONFIG
# ==============================================================================

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATASET         = "cifar10"          # "cifar10" | "imagenet" (needs manual path)
N_SAMPLES       = 256                # how many images to average over
BATCH_SIZE      = 64

# Diffusion schedule
T_MAX           = 200                # total timesteps
BETA_START      = 1e-4
BETA_END        = 0.02

# Adversarial attack (PGD)
EPS             = 8 / 255
ALPHA           = 2 / 255
STEPS           = 20

# BLD encoder toggle
USE_BLD_ENCODER = False              # set True if you have pretrained BLD weights
BLD_REPO_PATH   = "./binary-latent-diffusion"   # path to cloned JiauZhang repo
BLD_CKPT_PATH   = "./bld_vq.pt"                  # path to VQ checkpoint

# ==============================================================================
#  BINARY FORWARD PROCESS  (BLD-style Bernoulli bit-flip)
# ==============================================================================

def make_beta_schedule(T, beta_start, beta_end):
    """Linear beta schedule, same convention as BLD / D3PM."""
    return torch.linspace(beta_start, beta_end, T, device=DEVICE)


def compute_cumulative_flip_prob(betas):
    """
    Cumulative flip probability at each timestep t.
    For binary flip:  q(x_t | x_0) = Bernoulli(alpha_bar_t)
    where alpha_bar_t = 0.5 * (1 - prod_{s=1}^{t} (1 - 2*beta_s))
    This is the probability that a single bit has been flipped relative to x_0.
    """
    # (1 - 2*beta) is the "retention" factor per step
    retention = torch.cumprod(1.0 - 2.0 * betas, dim=0)   # shape: (T,)
    flip_prob  = 0.5 * (1.0 - retention)                   # shape: (T,)
    return flip_prob


def q_sample(z0, t_idx, flip_probs):
    """
    Forward diffusion: sample z_t given z_0 at timestep index t_idx.

    z0         : binary tensor {0,1}, shape (B, D)
    t_idx      : scalar int, 0-indexed
    flip_probs : cumulative flip probabilities, shape (T,)

    Returns z_t of same shape as z0.
    """
    p = flip_probs[t_idx]                        # scalar probability
    noise_mask = torch.bernoulli(
        torch.full_like(z0.float(), p.item())
    )
    z_t = (z0.float() + noise_mask) % 2          # XOR via modular arithmetic
    return z_t.long()


# ==============================================================================
#  ENCODER  (BLD or pixel-level fallback)
# ==============================================================================

def load_bld_encoder():
    """
    Load the binary autoencoder from JiauZhang/binary-latent-diffusion.
    Adjust import paths if your clone has a different structure.
    """
    import sys
    sys.path.insert(0, BLD_REPO_PATH)
    from vq_model import VQModel          # adjust if class name differs
    model = VQModel()
    ckpt = torch.load(BLD_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval().to(DEVICE)
    return model


def encode_to_binary(images, encoder=None):
    """
    images  : float tensor (B, C, H, W) in [0, 1]
    encoder : BLD VQModel or None (pixel fallback)

    Returns binary tensor (B, D) with values in {0, 1}.
    """
    if encoder is not None:
        with torch.no_grad():
            # BLD encoder returns Bernoulli logits; sample binary code
            z = encoder.encode(images)          # returns binary latent
            # If the encoder returns continuous logits, binarize:
            if z.dtype != torch.long:
                z = (z > 0.5).long()
        return z.view(z.shape[0], -1)           # flatten spatial dims
    else:
        # Pixel-level fallback: threshold at 0.5, flatten
        z = (images > 0.5).long()
        return z.view(images.shape[0], -1)


# ==============================================================================
#  ADVERSARIAL ATTACK
# ==============================================================================

class SimplePGD:
    """Manual PGD, used when torchattacks is not available."""
    def __init__(self, model, eps, alpha, steps):
        self.model = model
        self.eps   = eps
        self.alpha = alpha
        self.steps = steps

    def __call__(self, x, y):
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.eps, self.eps)
        x_adv = x_adv.clamp(0, 1)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss   = loss_fn(logits, y)
            grad   = torch.autograd.grad(loss, x_adv)[0]
            x_adv  = x_adv.detach() + self.alpha * grad.sign()
            delta  = (x_adv - x).clamp(-self.eps, self.eps)
            x_adv  = (x + delta).clamp(0, 1).detach()
        return x_adv


def get_attacker(classifier):
    if HAS_TORCHATTACKS:
        return torchattacks.PGD(classifier, eps=EPS, alpha=ALPHA, steps=STEPS)
    else:
        return SimplePGD(classifier, EPS, ALPHA, STEPS)


# ==============================================================================
#  DATASET & CLASSIFIER
# ==============================================================================

def get_cifar10_loader():
    tf = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False,
                                       download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return loader


def get_pretrained_classifier():
    """
    Simple ResNet18 pretrained on CIFAR-10.
    Using torchvision's ResNet18 with imagenet weights as a proxy classifier
    (replace with a CIFAR-10 finetuned model for more realistic attacks).
    """
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.eval().to(DEVICE)
    return model


# ==============================================================================
#  MAIN EXPERIMENT
# ==============================================================================

def run_experiment():
    print(f"Device: {DEVICE}")

    # 1. Schedule
    betas      = make_beta_schedule(T_MAX, BETA_START, BETA_END)
    flip_probs = compute_cumulative_flip_prob(betas)
    print(f"Flip prob at t=10:  {flip_probs[9]:.4f}")
    print(f"Flip prob at t=100: {flip_probs[99]:.4f}")
    print(f"Flip prob at t=200: {flip_probs[199]:.4f}  (should be ~0.5)")

    # 2. Encoder
    encoder = None
    if USE_BLD_ENCODER:
        print("Loading BLD encoder...")
        encoder = load_bld_encoder()
    else:
        print("Using pixel-level binarization (no BLD encoder).")

    # 3. Classifier + attacker
    classifier = get_pretrained_classifier()
    attacker   = get_attacker(classifier)

    # 4. Data
    loader = get_cifar10_loader()

    # 5. Collect samples
    all_z_clean = []
    all_z_adv   = []
    n_collected  = 0

    for x, y in loader:
        if n_collected >= N_SAMPLES:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Generate adversarial examples
        x_adv = attacker(x, y)

        # Encode to binary latent
        z_clean = encode_to_binary(x,     encoder)   # (B, D)
        z_adv   = encode_to_binary(x_adv, encoder)   # (B, D)

        all_z_clean.append(z_clean.cpu())
        all_z_adv.append(z_adv.cpu())
        n_collected += x.shape[0]
        print(f"  collected {n_collected}/{N_SAMPLES} samples")

    z_clean = torch.cat(all_z_clean, dim=0)[:N_SAMPLES]   # (N, D)
    z_adv   = torch.cat(all_z_adv,   dim=0)[:N_SAMPLES]   # (N, D)

    # Baseline: bit difference before any diffusion noise
    baseline_diff = (z_clean != z_adv).float().mean().item()
    print(f"\nBaseline bit diff (t=0): {baseline_diff:.6f}")

    # 6. Measure diff-vs-t
    t_values    = list(range(0, T_MAX, 5))   # sample every 5 steps
    mean_diffs  = []
    std_diffs   = []

    z_clean = z_clean.to(DEVICE)
    z_adv   = z_adv.to(DEVICE)
    flip_probs = flip_probs.to(DEVICE)

    print("\nRunning forward diffusion across timesteps...")
    for t in t_values:
        diffs = []
        for _ in range(5):    # average over 5 stochastic forward samples
            zt_clean = q_sample(z_clean, t, flip_probs)
            zt_adv   = q_sample(z_adv,   t, flip_probs)
            diff = (zt_clean != zt_adv).float().mean().item()
            diffs.append(diff)
        mean_diffs.append(np.mean(diffs))
        std_diffs.append(np.std(diffs))

    mean_diffs = np.array(mean_diffs)
    std_diffs  = np.array(std_diffs)

    # 7. Find t* where gap drops below threshold
    threshold = 0.01
    below = np.where(mean_diffs < threshold)[0]
    t_star = t_values[below[0]] if len(below) > 0 else None
    if t_star is not None:
        print(f"\nt* (diff < {threshold}): t = {t_star}")
    else:
        print(f"\nDiff never drops below {threshold} within T={T_MAX}. "
              "Consider increasing T_MAX or beta schedule.")

    # 8. Plot
    plt.figure(figsize=(9, 5))
    plt.plot(t_values, mean_diffs, label="mean bit diff", color="#3B8BD4", linewidth=2)
    plt.fill_between(t_values,
                     mean_diffs - std_diffs,
                     mean_diffs + std_diffs,
                     alpha=0.2, color="#3B8BD4", label="±1 std")
    plt.axhline(baseline_diff, color="#E8593C", linestyle="--",
                label=f"baseline (t=0): {baseline_diff:.4f}")
    plt.axhline(threshold, color="gray", linestyle=":", label=f"threshold={threshold}")
    if t_star is not None:
        plt.axvline(t_star, color="#1D9E75", linestyle="--",
                    label=f"t* = {t_star}")
    plt.xlabel("Timestep t", fontsize=12)
    plt.ylabel("Bit difference rate", fontsize=12)
    plt.title("Binary Forward Process: Clean vs Adversarial Gap", fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("diff_vs_t.png", dpi=150)
    plt.show()
    print("\nPlot saved to diff_vs_t.png")

    # 9. Print summary table
    print("\n{:<8} {:<12} {:<10}".format("t", "mean_diff", "std_diff"))
    print("-" * 32)
    for t, m, s in zip(t_values, mean_diffs, std_diffs):
        print(f"{t:<8} {m:<12.6f} {s:<10.6f}")

    return t_values, mean_diffs, std_diffs


if __name__ == "__main__":
    run_experiment()