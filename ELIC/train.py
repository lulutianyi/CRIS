# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from ELICUtilis.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile
import numpy as np
from tqdm.auto import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ELICUtilis.utilis import DelfileList, load_checkpoint
from Network import TestModel


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def compute_psnr(a, b):
    """Compute PSNR between two image tensors (values in [0, 1])."""
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10 * math.log10(mse)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant=True,
    log_every=10,
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    start = time.time()

    # ── 分阶段计时累加器，用于定位卡顿到底发生在哪个阶段 ──
    t_data = 0.0
    t_forward = 0.0
    t_backward = 0.0
    t_step = 0.0
    t_mark = time.time()

    total_batches = len(train_dataloader)
    pbar = tqdm(
        total=total_batches,
        desc=f"Epoch {epoch} [train]",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
    )

    for i, d in enumerate(train_dataloader):
        # ---- 数据加载完成 ----
        now = time.time()
        t_data += now - t_mark
        t_mark = now

        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d, noisequant)
        out_criterion = criterion(out_net, d)

        # ---- forward 完成 ----
        now = time.time()
        t_forward += now - t_mark
        t_mark = now

        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        # ---- backward 完成 ----
        now = time.time()
        t_backward += now - t_mark
        t_mark = now

        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # ---- optimizer step 完成 ----
        now = time.time()
        t_step += now - t_mark
        t_mark = now

        # 进度条后缀实时显示关键指标
        pbar.set_postfix({
            "loss": f'{out_criterion["loss"].item():.3f}',
            "mse": f'{out_criterion["mse_loss"].item():.2f}',
            "bpp": f'{out_criterion["bpp_loss"].item():.3f}',
        })
        pbar.update(1)

        # 周期性详细日志（含阶段耗时占比），第一个batch也打印方便确认没卡死
        if i == 0 or (i + 1) % log_every == 0:
            elapsed = time.time() - start
            avg_per_batch = elapsed / (i + 1)
            eta_sec = avg_per_batch * (total_batches - i - 1)
            total_t = max(t_data + t_forward + t_backward + t_step, 1e-6)
            print(
                f"\nTrain epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * (i + 1) / total_batches:.1f}%)] "
                f"已用时 {elapsed:.1f}s | 预计剩余 {eta_sec:.1f}s\n"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}\n"
                f"\t阶段耗时占比 -> data: {100*t_data/total_t:.1f}% | "
                f"forward: {100*t_forward/total_t:.1f}% | "
                f"backward: {100*t_backward/total_t:.1f}% | "
                f"optim_step: {100*t_step/total_t:.1f}%",
                flush=True,
            )

    pbar.close()
    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f} |"
          f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f} |"
          f"\tTime (s) : {time.time()-start:.4f} |"
          )

    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    # ↓ 新增 PSNR 统计
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

            # 逐样本计算 PSNR 后取平均，输入已经是 [0,1] 范围
            batch_psnr = sum(
                compute_psnr(d[i], out_net["x_hat"][i].clamp(0, 1))
                for i in range(d.size(0))
            ) / d.size(0)
            psnr_meter.update(batch_psnr, d.size(0))

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
        f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f} |"
        f"\tPSNR: {psnr_meter.avg:.4f} dB\n"   # ← 新增打印
    )

    return loss.avg, bpp_loss.avg, mse_loss.avg, psnr_meter.avg  # ← 新增返回值


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--N",
        default=192,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=320,
        type=int,
        help="Number of channels of latent",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=4000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=15e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--log-every",
        default=10,
        type=int,
        help="打印训练日志的batch间隔 (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    # 先用 Resize 把图片短边缩放到 patch_size，保证不会出现
    # "Required crop size is larger than input image size" 的报错
    # （D-Fire数据集里混有部分尺寸小于256x256的图片，例如 360x198）
    train_transforms = transforms.Compose(
        [transforms.Resize(min(args.patch_size)),
         transforms.RandomCrop(args.patch_size),
         transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.Resize(min(args.patch_size)),
         transforms.CenterCrop(args.patch_size),
         transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TestModel(N=args.N, M=args.M)
    net = net.to(device)
    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3800], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    stemode = False  # set the pretrained flag
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
        last_epoch = 0
        stemode = True

    noisequant = True
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if epoch > 3800 or stemode:
            noisequant = False
        print("noisequant: {}, stemode:{}".format(noisequant, stemode))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp, train_mse = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            noisequant,
            log_every=args.log_every,
        )
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        writer.add_scalar('Train/bpp', train_bpp, epoch)

        # ↓ test_epoch 现在返回 4 个值，多了 psnr
        loss, bpp, mse, psnr = test_epoch(epoch, test_dataloader, net, criterion)
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        writer.add_scalar('Test/bpp', bpp, epoch)
        writer.add_scalar('Test/psnr', psnr, epoch)   # ← 新增 PSNR 写入 TensorBoard
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )

    # ──────────────────────────────────────────────────────────────────
    # 训练结束后：从 TensorBoard 日志读取数据并绘制各指标随 epoch 变化图
    # ──────────────────────────────────────────────────────────────────
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import matplotlib
        matplotlib.use("Agg")           # 无显示器环境下使用非交互后端
        import matplotlib.pyplot as plt

        print("\n[Visualization] Loading TensorBoard logs from:", args.savepath)
        ea = EventAccumulator(args.savepath)
        ea.Reload()

        # 需要可视化的所有标量 tag
        tags_config = {
            "Train/loss":  ("Train Loss",  "Loss"),
            "Train/mse":   ("Train MSE",   "MSE (×255²)"),
            "Train/bpp":   ("Train Bpp",   "Bpp"),
            "Test/loss":   ("Test Loss",   "Loss"),
            "Test/mse":    ("Test MSE",    "MSE (×255²)"),
            "Test/bpp":    ("Test Bpp",    "Bpp"),
            "Test/psnr":   ("Test PSNR",   "PSNR (dB)"),
        }

        # 按共享 Y 轴含义分组，每组画一张图
        groups = {
            "loss":  ["Train/loss",  "Test/loss"],
            "mse":   ["Train/mse",   "Test/mse"],
            "bpp":   ["Train/bpp",   "Test/bpp"],
            "psnr":  ["Test/psnr"],
        }

        available_tags = ea.Tags().get("scalars", [])
        vis_dir = os.path.join(args.savepath, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

        for group_name, tag_list in groups.items():
            fig, ax = plt.subplots(figsize=(9, 5))
            plotted = False
            for tag in tag_list:
                if tag not in available_tags:
                    continue
                events = ea.Scalars(tag)
                epochs_list = [e.step for e in events]
                values     = [e.value for e in events]
                label, ylabel = tags_config[tag]
                ax.plot(epochs_list, values, label=label, linewidth=1.5)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{ylabel} vs Epoch", fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            save_fig_path = os.path.join(vis_dir, f"{group_name}_curve.png")
            fig.savefig(save_fig_path, dpi=150)
            plt.close(fig)
            print(f"[Visualization] Saved: {save_fig_path}")

        # 额外：Rate-Distortion 曲线（Bpp vs PSNR），若两者均存在
        if "Test/bpp" in available_tags and "Test/psnr" in available_tags:
            bpp_events  = ea.Scalars("Test/bpp")
            psnr_events = ea.Scalars("Test/psnr")
            bpp_vals  = [e.value for e in bpp_events]
            psnr_vals = [e.value for e in psnr_events]
            min_len = min(len(bpp_vals), len(psnr_vals))

            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(bpp_vals[:min_len], psnr_vals[:min_len],
                            c=range(min_len), cmap="viridis", s=10)
            plt.colorbar(sc, ax=ax, label="Epoch")
            ax.set_xlabel("Bpp", fontsize=12)
            ax.set_ylabel("PSNR (dB)", fontsize=12)
            ax.set_title("Rate-Distortion Curve (Test)", fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            rd_path = os.path.join(vis_dir, "rd_curve.png")
            fig.savefig(rd_path, dpi=150)
            plt.close(fig)
            print(f"[Visualization] Saved: {rd_path}")

        print(f"[Visualization] All figures saved to: {vis_dir}")

    except ImportError as e:
        print(f"[Visualization] Skipped (missing dependency: {e}). "
              f"Install with: pip install tensorboard matplotlib")


if __name__ == "__main__":
    main(sys.argv[1:])