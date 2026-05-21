import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, hidden_channels=128, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),  # 256x256
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Vector Quantizer ----------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, H, W]
        B, C, H, W = inputs.shape

        # reshape → [B*H*W, C]
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # 计算距离
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # 最近编码
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(B, H, W, C)

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss

# ---------------- VQ-VAE ----------------
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss