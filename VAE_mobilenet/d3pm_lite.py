import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-4

T_MAX = 100
T_STARS = [5, 10, 20, 40, 80]

N_TEST = 256

# ======================
# DATA
# ======================
def get_loaders():
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def binarize(x):
    return (x > 0.5).long().view(x.size(0), -1)


# ======================
# D3PM FORWARD PROCESS (TRUE MULTI-STEP)
# ======================
def sample_q(x0, t, betas):
    """
    true D3PM forward: x0 -> xt via t-step Markov chain
    """
    x = x0.clone()

    for i in range(t):
        flip = torch.bernoulli(betas[i] * torch.ones_like(x.float()))
        x = (x + flip.long()) % 2

    return x


# ======================
# MODEL (lightweight reverse model)
# ======================
class ReverseModel(nn.Module):
    def __init__(self, D=3072, T=100):
        super().__init__()
        self.t_emb = nn.Embedding(T, 64)

        self.net = nn.Sequential(
            nn.Linear(D + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D)
        )

    def forward(self, x, t):
        x = x.float()
        t = torch.clamp(t, 0, self.t_emb.num_embeddings - 1)
        te = self.t_emb(t)
        x = torch.cat([x, te], dim=-1)
        return self.net(x)


# ======================
# LOSS (D3PM-lite: predict x0)
# ======================
def compute_loss(model, x0, betas):
    B = x0.size(0)

    t = torch.randint(1, T_MAX, (B,), device=x0.device)

    xt_list = []
    for i in range(B):
        xt_list.append(sample_q(x0[i], t[i].item(), betas))
    xt = torch.stack(xt_list)

    logits = model(xt, t)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        x0.float()
    )

    return loss


# ======================
# TRAIN
# ======================
def train(model, loader, betas, opt):
    model.train()
    total = 0
    n = 0

    for x, _ in tqdm(loader, desc="train"):
        x = x.to(DEVICE)
        x0 = binarize(x)

        opt.zero_grad()
        loss = compute_loss(model, x0, betas)

        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / n


# ======================
# REVERSE PROCESS
# ======================
@torch.no_grad()
def reverse(model, xt, t_start):
    x = xt.clone()

    for t in reversed(range(1, t_start + 1)):
        tb = torch.full((x.size(0),), t, device=x.device)

        logits = model(x, tb)
        p = torch.sigmoid(logits)

        x = torch.bernoulli(p).long()

    return x


# ======================
# EVALUATION
# ======================
@torch.no_grad()
def evaluate(model, loader, betas):
    model.eval()

    x_list = []
    n = 0

    for x, _ in loader:
        x = x.to(DEVICE)
        x_list.append(x)
        n += x.size(0)
        if n >= N_TEST:
            break

    x = torch.cat(x_list)[:N_TEST]
    z_clean = binarize(x)

    results = {}

    for t_star in T_STARS:
        zt_list = []

        for i in range(N_TEST):
            zt_list.append(sample_q(z_clean[i], t_star, betas))

        zt = torch.stack(zt_list)

        z_rec = reverse(model, zt, t_star)

        acc = (z_rec == z_clean).float().mean().item()
        results[t_star] = acc

        print(f"t*={t_star}  bit_acc={acc:.4f}")

    return results


# ======================
# PLOT
# ======================
def plot(results):
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("t*")
    plt.ylabel("bit accuracy")
    plt.title("D3PM purification curve (true multi-step)")
    plt.show()


# ======================
# BETAS (IMPORTANT)
# ======================
def get_betas():
    return torch.linspace(0.01, 0.2, T_MAX, device=DEVICE)


# ======================
# MAIN
# ======================
def main():
    print("device:", DEVICE)

    betas = get_betas()

    train_loader, test_loader = get_loaders()

    model = ReverseModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("training...")

    for ep in range(EPOCHS):
        loss = train(model, train_loader, betas, opt)
        print(f"epoch {ep}: loss={loss:.4f}")

    print("\n evaluating...")
    results = evaluate(model, test_loader, betas)

    plot(results)


if __name__ == "__main__":
    main()