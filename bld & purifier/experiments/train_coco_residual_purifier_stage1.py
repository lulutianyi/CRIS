"""Stage-1 reconstruction training for a native-resolution COCO purifier."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.utils import save_image
from tqdm.auto import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return x + 0.1 * self.body(x)


class NativeResidualPurifier(nn.Module):
    def __init__(self, channels=48, max_residual=12 / 255):
        super().__init__()
        self.max_residual = float(max_residual)
        self.gradient_checkpointing = False
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        dilations = (1, 2, 4, 8, 1, 2, 4, 8)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, dilation) for dilation in dilations]
        )
        self.head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image):
        features = self.stem(image)
        if self.training and self.gradient_checkpointing:
            features = checkpoint_sequential(
                self.blocks, 4, features, use_reentrant=False
            )
        else:
            features = self.blocks(features)
        raw_residual = self.head(features)
        residual = self.max_residual * torch.tanh(raw_residual)
        purified = (image + residual).clamp(0, 1)
        return purified, residual


class PairedCocoDataset(Dataset):
    def __init__(self, samples, patch_size=None, augment=False):
        self.samples = list(samples)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["original_path"]) as image:
            clean = TF.to_tensor(image.convert("RGB"))
        with Image.open(sample["adversarial_path"]) as image:
            adversarial = TF.to_tensor(image.convert("RGB"))
        if clean.shape != adversarial.shape:
            raise ValueError(f'配对尺寸不一致: image_id={sample["image_id"]}')

        if self.patch_size:
            clean, adversarial = paired_random_crop(
                clean, adversarial, self.patch_size
            )
        if self.augment and random.random() < 0.5:
            clean = clean.flip(-1)
            adversarial = adversarial.flip(-1)
        if self.augment and random.random() < 0.5:
            clean = clean.flip(-2)
            adversarial = adversarial.flip(-2)
        return adversarial, clean, int(sample["image_id"])


def paired_random_crop(clean, adversarial, size):
    height, width = clean.shape[-2:]
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    if pad_height or pad_width:
        padding = (0, pad_width, 0, pad_height)
        clean = F.pad(clean, padding, mode="replicate")
        adversarial = F.pad(adversarial, padding, mode="replicate")
        height, width = clean.shape[-2:]
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return (
        clean[:, top : top + size, left : left + size],
        adversarial[:, top : top + size, left : left + size],
    )


def charbonnier(prediction, target, epsilon=1e-3):
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def highpass(image):
    kernel = image.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=image.shape[1])


def fft_high_frequency_loss(prediction, target, cutoff=0.15):
    height, width = prediction.shape[-2:]
    fy = torch.fft.fftfreq(height, device=prediction.device).view(-1, 1)
    fx = torch.fft.rfftfreq(width, device=prediction.device).view(1, -1)
    radius = torch.sqrt(fy.square() + fx.square())
    mask = (radius >= cutoff).to(prediction.dtype)[None, None]
    prediction_spectrum = torch.fft.rfft2(prediction, norm="ortho")
    target_spectrum = torch.fft.rfft2(target, norm="ortho")
    prediction_amplitude = torch.log1p(prediction_spectrum.abs())
    target_amplitude = torch.log1p(target_spectrum.abs())
    numerator = ((prediction_amplitude - target_amplitude).abs() * mask).sum()
    denominator = mask.sum() * prediction.shape[0] * prediction.shape[1]
    return numerator / denominator.clamp_min(1)


def ssim_value(prediction, target):
    kernel_size = 11
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(prediction, kernel_size, 1, padding)
    mu_y = F.avg_pool2d(target, kernel_size, 1, padding)
    sigma_x = F.avg_pool2d(prediction.square(), kernel_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), kernel_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def reconstruction_loss(model, adversarial, clean, args):
    purified, residual = model(adversarial)
    recon = charbonnier(purified, clean)
    hp = F.l1_loss(highpass(purified), highpass(clean))
    fft = fft_high_frequency_loss(purified, clean, args.fft_cutoff)
    clean_output, clean_residual = model(clean)
    identity = F.l1_loss(clean_output, clean) + clean_residual.abs().mean()
    total = (
        args.recon_weight * recon
        + args.highpass_weight * hp
        + args.fft_weight * fft
        + args.identity_weight * identity
    )
    return total, {
        "recon": recon.detach(),
        "highpass": hp.detach(),
        "fft": fft.detach(),
        "identity": identity.detach(),
        "residual_abs": residual.detach().abs().mean(),
    }


def train_epoch(model, loader, optimizer, scaler, device, args, epoch):
    model.train()
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "highpass": 0.0,
        "fft": 0.0,
        "identity": 0.0,
        "residual_abs": 0.0,
    }
    progress = tqdm(loader, desc=f"stage1 train {epoch}/{args.epochs}")
    for adversarial, clean, _image_id in progress:
        adversarial = adversarial.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts = reconstruction_loss(
                model, adversarial, clean, args
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach())
        for name, value in parts.items():
            totals[name] += float(value)
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return {name: value / len(loader) for name, value in totals.items()}


@torch.inference_mode()
def validate(model, dataset, device, preview_path=None):
    model.eval()
    totals = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0, "highpass_psnr": 0.0}
    for index in tqdm(range(len(dataset)), desc="stage1 validation"):
        adversarial, clean, _image_id = dataset[index]
        adversarial = adversarial.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        purified, _residual = model(adversarial)
        error = purified - clean
        mse = error.square().mean().clamp_min(1e-12)
        hp_mse = (highpass(purified) - highpass(clean)).square().mean().clamp_min(1e-12)
        totals["mae"] += float(error.abs().mean())
        totals["psnr"] += float(-10 * torch.log10(mse))
        totals["ssim"] += float(ssim_value(purified, clean))
        totals["highpass_psnr"] += float(-10 * torch.log10(hp_mse))
        if index == 0 and preview_path is not None:
            difference = (purified - adversarial).abs() * 8
            preview = torch.cat(
                [adversarial, purified, clean, difference.clamp(0, 1)], dim=-1
            )
            save_image(preview.cpu(), preview_path, format="png")
    return {name: value / len(dataset) for name, value in totals.items()}


def load_samples(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if not samples:
        raise ValueError("manifest中没有配对样本")
    return samples


def make_split(samples, val_size, seed):
    if not 0 < val_size < len(samples):
        raise ValueError("val-size必须大于0且小于总样本数")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_size])
    train_samples = [sample for i, sample in enumerate(samples) if i not in val_indices]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_indices]
    return train_samples, val_samples


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_psnr, args):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "native_residual_purifier_stage1_v1",
            "epoch": epoch,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": {
                "channels": args.channels,
                "max_residual": args.max_residual,
            },
            "args": serializable_args(args),
        },
        temporary,
    )
    temporary.replace(path)


def serializable_args(args):
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.manifest)
    train_samples, val_samples = make_split(samples, args.val_size, args.seed)
    split = {
        "seed": args.seed,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "train_image_ids": [int(x["image_id"]) for x in train_samples],
        "val_image_ids": [int(x["image_id"]) for x in val_samples],
    }
    atomic_json(output_dir / "split.json", split)

    train_dataset = PairedCocoDataset(
        train_samples, patch_size=args.patch_size, augment=True
    )
    val_dataset = PairedCocoDataset(val_samples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=generator,
    )

    model = NativeResidualPurifier(
        channels=args.channels, max_residual=args.max_residual
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_psnr = -math.inf
    history = []

    last_path = output_dir / "purifier_stage1_last.pt"
    best_path = output_dir / "purifier_stage1_best.pt"
    history_path = output_dir / "history.json"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint["best_psnr"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    print("device:", device)
    print("train/val:", len(train_dataset), len(val_dataset))
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch
        )
        val_metrics = validate(
            model,
            val_dataset,
            device,
            preview_dir / f"epoch_{epoch:03d}.png",
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            scheduler,
            epoch,
            best_psnr,
            args,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_psnr,
                args,
            )
        atomic_json(history_path, history)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print("best val PSNR:", best_psnr)

    print("\nStage-1训练完成")
    print("best checkpoint:", best_path)
    print("history:", history_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--max-residual", type=float, default=12 / 255)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--highpass-weight", type=float, default=0.1)
    parser.add_argument("--fft-weight", type=float, default=0.03)
    parser.add_argument("--fft-cutoff", type=float, default=0.15)
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
