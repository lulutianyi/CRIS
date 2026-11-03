"""Task-aware Stage-2 fine-tuning with a frozen Faster R-CNN victim."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import (  # noqa: E402
    atomic_write_json,
    evaluate_coco_predictions,
    load_victim,
    predictions_to_coco,
    target_from_coco,
    target_to_device,
)
from train_coco_residual_purifier_stage1 import (  # noqa: E402
    NativeResidualPurifier,
    charbonnier,
    fft_high_frequency_loss,
    highpass,
    paired_random_crop,
    ssim_value,
)


class PairedDetectionDataset(Dataset):
    def __init__(self, samples, annotation_path):
        self.samples = list(samples)
        self.coco = COCO(str(annotation_path))

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
        image_id = int(sample["image_id"])
        target = target_from_coco(
            self.coco,
            image_id,
            width=clean.shape[-1],
            height=clean.shape[-2],
        )
        return adversarial, clean, target


def single_item_collate(batch):
    if len(batch) != 1:
        raise ValueError("Stage-2仅支持batch-size=1")
    return batch[0]


def freeze_normalization(model):
    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            module.eval()


def detection_losses(model, image, target):
    model.train()
    freeze_normalization(model)
    losses = model([image], [target])
    return sum(losses.values()), losses


def resized_fft_loss(prediction, target, cutoff, size=256):
    prediction_small = F.interpolate(
        prediction.float(), size=(size, size), mode="bilinear", align_corners=False
    )
    target_small = F.interpolate(
        target.float(), size=(size, size), mode="bilinear", align_corners=False
    )
    return fft_high_frequency_loss(
        prediction_small, target_small, cutoff=cutoff
    )


def identity_patch(clean, patch_size):
    patch, _ = paired_random_crop(
        clean[0], clean[0], min(patch_size, max(clean.shape[-2:]))
    )
    return patch.unsqueeze(0)


def stage2_loss(purifier, victim, adversarial, clean, target, det_weight, args):
    purified, residual = purifier(adversarial)
    recon = charbonnier(purified, clean)
    hp = F.l1_loss(highpass(purified), highpass(clean))
    with torch.autocast(device_type="cuda", enabled=False):
        fft = resized_fft_loss(
            purified, clean, cutoff=args.fft_cutoff, size=args.fft_size
        )

    clean_patch = identity_patch(clean, args.identity_patch_size)
    clean_output, clean_residual = purifier(clean_patch)
    identity = F.l1_loss(clean_output, clean_patch) + clean_residual.abs().mean()

    detector_total, detector_parts = detection_losses(
        victim, purified[0], target
    )
    total = (
        args.recon_weight * recon
        + args.highpass_weight * hp
        + args.fft_weight * fft
        + args.identity_weight * identity
        + det_weight * detector_total
    )
    parts = {
        "recon": recon.detach(),
        "highpass": hp.detach(),
        "fft": fft.detach(),
        "identity": identity.detach(),
        "detector": detector_total.detach(),
        "residual_abs": residual.detach().abs().mean(),
    }
    for name, value in detector_parts.items():
        parts[f"det_{name}"] = value.detach()
    return total, parts


def optimizer_step(optimizer, scaler, model, grad_clip):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(
    purifier,
    victim,
    loader,
    optimizer,
    scaler,
    device,
    det_weight,
    args,
    epoch,
):
    purifier.train()
    totals = {}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"stage2 train {epoch}/{args.epochs}")
    for step, (adversarial, clean, target) in enumerate(progress, start=1):
        adversarial = adversarial.unsqueeze(0).to(device, non_blocking=True)
        clean = clean.unsqueeze(0).to(device, non_blocking=True)
        target = target_to_device(target, device, attack_only=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts = stage2_loss(
                purifier,
                victim,
                adversarial,
                clean,
                target,
                det_weight,
                args,
            )
            scaled_loss = loss / args.grad_accumulation
        scaler.scale(scaled_loss).backward()
        if step % args.grad_accumulation == 0 or step == len(loader):
            optimizer_step(
                optimizer, scaler, purifier, args.grad_clip
            )
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            det=f"{float(parts['detector']):.3f}",
            w=f"{det_weight:.3f}",
        )
        del adversarial, clean, target, loss, scaled_loss
    return {name: value / len(loader) for name, value in totals.items()}


@torch.inference_mode()
def predict_rows(victim, image, image_id, score_threshold):
    victim.eval()
    prediction = victim([image])[0]
    return predictions_to_coco(
        image_id, prediction, score_threshold=score_threshold
    )


def image_metrics(prediction, target):
    error = prediction - target
    mse = error.square().mean().clamp_min(1e-12)
    hp_mse = (
        highpass(prediction) - highpass(target)
    ).square().mean().clamp_min(1e-12)
    return {
        "mae": float(error.abs().mean()),
        "psnr": float(-10 * torch.log10(mse)),
        "ssim": float(ssim_value(prediction, target)),
        "highpass_psnr": float(-10 * torch.log10(hp_mse)),
    }


@torch.inference_mode()
def baseline_detection(victim, dataset, coco, device, score_threshold):
    groups = {"clean": [], "adversarial": []}
    image_ids = []
    for adversarial, clean, target in tqdm(
        dataset, desc="stage2 baseline detection"
    ):
        image_id = int(target["image_id"].item())
        image_ids.append(image_id)
        clean = clean.to(device)
        adversarial = adversarial.to(device)
        groups["clean"].extend(
            predict_rows(victim, clean, image_id, score_threshold)
        )
        groups["adversarial"].extend(
            predict_rows(victim, adversarial, image_id, score_threshold)
        )
    return {
        name: evaluate_coco_predictions(coco, image_ids, rows)
        for name, rows in groups.items()
    }


@torch.inference_mode()
def validate(
    purifier,
    victim,
    dataset,
    coco,
    device,
    args,
    preview_path,
):
    purifier.eval()
    groups = {"clean_purified": [], "adversarial_purified": []}
    image_ids = []
    totals = {
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "highpass_psnr": 0.0,
        "clean_identity_mae": 0.0,
    }
    for index in tqdm(range(len(dataset)), desc="stage2 validation"):
        adversarial, clean, target = dataset[index]
        image_id = int(target["image_id"].item())
        image_ids.append(image_id)
        adversarial = adversarial.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        adversarial_purified, _ = purifier(adversarial)
        clean_purified, _ = purifier(clean)
        metrics = image_metrics(adversarial_purified, clean)
        for name, value in metrics.items():
            totals[name] += value
        totals["clean_identity_mae"] += float(
            (clean_purified - clean).abs().mean()
        )
        groups["clean_purified"].extend(
            predict_rows(
                victim, clean_purified[0], image_id, args.score_threshold
            )
        )
        groups["adversarial_purified"].extend(
            predict_rows(
                victim,
                adversarial_purified[0],
                image_id,
                args.score_threshold,
            )
        )
        if index == 0:
            correction = (adversarial_purified - adversarial).abs() * 8
            preview = torch.cat(
                [clean, adversarial, adversarial_purified, correction.clamp(0, 1)],
                dim=-1,
            )
            save_image(preview.cpu(), preview_path, format="png")
    detection = {
        name: evaluate_coco_predictions(coco, image_ids, rows)
        for name, rows in groups.items()
    }
    image_results = {
        name: value / len(dataset) for name, value in totals.items()
    }
    return image_results, detection


def load_stage1(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = NativeResidualPurifier(
        channels=int(config.get("channels", 48)),
        max_residual=float(config.get("max_residual", 12 / 255)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def load_split_samples(manifest_path, split_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    sample_by_id = {
        int(sample["image_id"]): sample for sample in manifest["samples"]
    }
    train_ids = [int(value) for value in split["train_image_ids"]]
    val_ids = [int(value) for value in split["val_image_ids"]]
    missing = [
        image_id
        for image_id in train_ids + val_ids
        if image_id not in sample_by_id
    ]
    if missing:
        raise ValueError(f"manifest缺少split中的{len(missing)}个image_id")
    return (
        [sample_by_id[image_id] for image_id in train_ids],
        [sample_by_id[image_id] for image_id in val_ids],
        split,
    )


def det_weight_for_epoch(epoch, epochs, start, end):
    if epochs <= 1:
        return end
    progress = (epoch - 1) / (epochs - 1)
    return start + progress * (end - start)


def clean_penalized_score(baseline, current, penalty):
    clean_drop = max(
        0.0,
        baseline["clean"]["map50_95"]
        - current["clean_purified"]["map50_95"],
    )
    return current["adversarial_purified"]["map50_95"] - penalty * clean_drop


def serializable_args(args):
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def save_checkpoint(
    path,
    purifier,
    optimizer,
    scaler,
    scheduler,
    epoch,
    best_score,
    best_adv_map,
    args,
):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "native_residual_purifier_stage2_v1",
            "epoch": epoch,
            "best_score": best_score,
            "best_adv_map50_95": best_adv_map,
            "model": purifier.state_dict(),
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


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Stage-2处理原尺寸可变图像，batch-size必须为1")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-2需要GPU运行时")

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, split = load_split_samples(
        args.manifest, args.split
    )
    atomic_write_json(output_dir / "split.json", split)
    train_dataset = PairedDetectionDataset(train_samples, args.annotations)
    val_dataset = PairedDetectionDataset(val_samples, args.annotations)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=single_item_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )

    purifier, stage1_checkpoint = load_stage1(
        args.stage1_checkpoint, device
    )
    purifier.gradient_checkpointing = args.gradient_checkpointing
    args.channels = int(
        stage1_checkpoint.get("model_config", {}).get("channels", 48)
    )
    args.max_residual = float(
        stage1_checkpoint.get("model_config", {}).get(
            "max_residual", 12 / 255
        )
    )
    victim, weights = load_victim(device)
    victim.transform.min_size = (args.detector_min_size,)
    victim.transform.max_size = args.detector_max_size
    for parameter in victim.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        purifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    baseline_path = output_dir / "baseline_detection.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = baseline_detection(
            victim,
            val_dataset,
            val_dataset.coco,
            device,
            args.score_threshold,
        )
        atomic_write_json(baseline_path, baseline)

    last_path = output_dir / "purifier_stage2_last.pt"
    best_path = output_dir / "purifier_stage2_best.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1
    best_score = -math.inf
    best_adv_map = -math.inf
    history = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        purifier.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_adv_map = float(checkpoint["best_adv_map50_95"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    print("device:", device)
    print("victim weights:", weights)
    print("train/val:", len(train_dataset), len(val_dataset))
    print("baseline:", json.dumps(baseline, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        det_weight = det_weight_for_epoch(
            epoch,
            args.epochs,
            args.det_weight_start,
            args.det_weight_end,
        )
        train_metrics = train_epoch(
            purifier,
            victim,
            train_loader,
            optimizer,
            scaler,
            device,
            det_weight,
            args,
            epoch,
        )
        image_metrics_result, detection = validate(
            purifier,
            victim,
            val_dataset,
            val_dataset.coco,
            device,
            args,
            preview_dir / f"epoch_{epoch:03d}.png",
        )
        scheduler.step()
        score = clean_penalized_score(
            baseline, detection, args.clean_drop_penalty
        )
        adv_map = detection["adversarial_purified"]["map50_95"]
        is_best = score > best_score
        if is_best:
            best_score = score
            best_adv_map = adv_map
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "det_weight": det_weight,
            "train": train_metrics,
            "validation_image": image_metrics_result,
            "validation_detection": detection,
            "selection_score": score,
        }
        history.append(row)
        save_checkpoint(
            last_path,
            purifier,
            optimizer,
            scaler,
            scheduler,
            epoch,
            best_score,
            best_adv_map,
            args,
        )
        if is_best:
            save_checkpoint(
                best_path,
                purifier,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_score,
                best_adv_map,
                args,
            )
        atomic_write_json(history_path, history)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print("best score:", best_score, "best adv mAP50-95:", best_adv_map)

    purifier.to("cpu")
    victim.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print("\nStage-2完成")
    print("best checkpoint:", best_path)
    print("history:", history_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--det-weight-start", type=float, default=0.05)
    parser.add_argument("--det-weight-end", type=float, default=0.10)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--highpass-weight", type=float, default=0.05)
    parser.add_argument("--fft-weight", type=float, default=0.02)
    parser.add_argument("--fft-cutoff", type=float, default=0.15)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--identity-weight", type=float, default=0.20)
    parser.add_argument("--identity-patch-size", type=int, default=256)
    parser.add_argument("--clean-drop-penalty", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--detector-min-size", type=int, default=800)
    parser.add_argument("--detector-max-size", type=int, default=1333)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(channels=48, max_residual=12 / 255)
    return parser.parse_args()


if __name__ == "__main__":
    main()
