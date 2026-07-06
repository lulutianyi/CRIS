import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Encoder ----------------

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 4, 2, 1),   # 416→208
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1), # 208→104
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),      # 104→52
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),      # 52→26
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),      # 26×26 输出
        )

class Decoder(nn.Module):
    def __init__(self, hidden_channels=128, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 256, 3, 1, 1),  # 26×26
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),              # 26→52
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),              # 52→104
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),               # 104→208
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                # 208→416
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

# ---------------- Vector Quantizer ----------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.25,
                 decay=0.99, dead_code_threshold=1.0):  # [改] 新增 EMA 参数
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # [改] EMA 统计量，用于码本 EMA 更新，解决 perplexity=1 的塌陷问题
        # 不参与梯度，用 register_buffer 跟随模型 .to(device)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_sum', self.embedding.weight.data.clone())

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]

        # 计算每个向量到所有码字的距离
        distances = (
            flat_input.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * flat_input @ self.embedding.weight.t()
        )

        encoding_indices = distances.argmin(1)                              # [B*H*W]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()# [B*H*W, K]
        quantized = self.embedding(encoding_indices).view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()             # [B, C, H, W]

        # [改] 训练时用 EMA 更新码本，替代原来纯梯度更新
        # EMA 更新比梯度更新更稳定，能有效防止码本塌陷
        if self.training:
            with torch.no_grad():
                # 每个码字本次 batch 被选中的次数
                counts = encodings.sum(0)
                # EMA 平滑使用频次
                self.cluster_size.mul_(self.decay).add_(counts * (1 - self.decay))
                # EMA 平滑码字对应的输入向量之和
                dw = encodings.t() @ flat_input
                self.embed_sum.mul_(self.decay).add_(dw * (1 - self.decay))
                # 用平滑后的均值更新码字权重（加 1e-5 防除零）
                self.embedding.weight.data.copy_(
                    self.embed_sum / (self.cluster_size.unsqueeze(1) + 1e-5)
                )

                # [改] 死码字重置：使用次数低于阈值的码字用随机输入向量替换
                # 这是解决 perplexity=1.0 最直接的手段
                dead_mask = self.cluster_size < self.dead_code_threshold
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    n_vectors = flat_input.size(0)
                    perm = torch.randperm(n_vectors, device=flat_input.device)[:n_dead]
                    self.embedding.weight.data[dead_mask] = flat_input[perm].detach()
                    # 重置对应的 EMA 统计量，让新码字从干净状态开始积累
                    self.cluster_size[dead_mask] = self.dead_code_threshold
                    self.embed_sum[dead_mask] = flat_input[perm].detach()

        # VQ loss：只保留 commitment loss（码本已由 EMA 维护，不再需要 q_latent_loss）
        # [改] 原来 loss = q_latent_loss + commitment * e_latent_loss
        #      EMA 模式下 q_latent_loss 的梯度会和 EMA 更新冲突，去掉
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # straight-through estimator：梯度直通，让 encoder 能收到信号
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