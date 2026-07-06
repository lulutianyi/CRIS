任务：

构建一个基于CompressAI的火灾检测联合压缩系统。

数据集：

D-Fire

目录结构：

dataset/
train/
images/
labels/
val/
images/
labels/

类别：

0 smoke
1 fire

训练目标：

总损失：

L_total

=

λ_rate * R
+
λ_dist * D
+
λ_det * L_detection

其中：

R = bpp

D = MSE

L_detection = YOLO Detection Loss

训练阶段：

Stage-1

仅训练压缩器

Loss：

R + λD

Stage-2

冻结检测器

训练压缩器

Loss：

R + λD + λdet

Stage-3

联合微调

Loss：

R + λD + λdet

训练参数：

epoch = 100

batch_size = 8

optimizer = AdamW

lr = 1e-4

scheduler = CosineAnnealingLR

保存：

best_psnr.pth
best_msssim.pth
best_map.pth
last.pth

日志：

TensorBoard

保存：

loss
bpp
psnr
ms_ssim
map50
map50_95

每个epoch打印：

Epoch
Loss
BPP
PSNR
MS-SSIM
mAP50
mAP50-95

使用CompressAI中的：

cheng2020-anchor

模型：

from compressai.zoo import cheng2020_anchor

quality=3

模型结构说明：

Encoder:
Conv
ResidualBlock
Downsample

Hyper Encoder

Entropy Bottleneck

Hyper Decoder

Decoder:
ResidualBlock
Upsample

要求：

生成网络结构图：

Input
→
Encoder
→
Latent z
→
Hyperprior
→
Entropy Model
→
Decoder
→
Reconstruction

并打印：

模型参数量

FLOPs

Latent尺寸

压缩率估计
