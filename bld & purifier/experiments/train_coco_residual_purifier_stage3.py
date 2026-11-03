"""Stage-3 adaptive adversarial fine-tuning for the COCO purifier."""

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
)


class CleanCocoSubset(Dataset):
    def __init__(self, image_root, annotation_path, image_ids):
        self.image_root = Path(image_root)
        self.coco = COCO(str(annotation_path))
        self.image_ids = [int(value) for value in image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        path = self.image_root / info["file_name"]
        with Image.open(path) as image:
            clean = TF.to_tensor(image.convert("RGB"))
        target = target_from_coco(
            self.coco,
            image_id,
            width=clean.shape[-1],
            height=clean.shape[-2],
        )
        return clean, target


def single_item_collate(batch):
    return batch[0]


def freeze_normalization(model):
    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            module.eval()


def detection_loss(model, image, target):
    model.train()
    freeze_normalization(model)
    losses = model([image], [target])
    return sum(losses.values()), losses


def set_trainable(model, enabled):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def adaptive_pgd(purifier, victim, clean, target, steps, args, image_id):
    torch.manual_seed(args.seed + int(image_id))
    adversarial = clean + torch.empty_like(clean).uniform_(
        -args.epsilon, args.epsilon
    )
    adversarial = adversarial.clamp(0, 1).detach()

    set_trainable(purifier, False)
    purifier.train()
    purifier.gradient_checkpointing = args.gradient_checkpointing
    for _ in range(steps):
        adversarial.requires_grad_(True)
        purified, _ = purifier(adversarial)
        loss, _ = detection_loss(victim, purified[0], target)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + args.attack_alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-args.epsilon, args.epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
        del purified, loss, gradient
    set_trainable(purifier, True)
    return adversarial


def resized_fft_loss(prediction, target, cutoff, size):
    prediction = F.interpolate(
        prediction.float(), size=(size, size), mode="bilinear", align_corners=False
    )
    target = F.interpolate(
        target.float(), size=(size, size), mode="bilinear", align_corners=False
    )
    return fft_high_frequency_loss(prediction, target, cutoff=cutoff)


def identity_patch(clean, patch_size):
    patch, _ = paired_random_crop(
        clean[0], clean[0], min(patch_size, max(clean.shape[-2:]))
    )
    return patch.unsqueeze(0)


def stage3_loss(purifier, victim, adversarial, clean, target, det_weight, args):
    purifier.train()
    purified, residual = purifier(adversarial)
    recon = charbonnier(purified, clean)
    hp = F.l1_loss(highpass(purified), highpass(clean))
    with torch.autocast(device_type="cuda", enabled=False):
        fft = resized_fft_loss(
            purified, clean, args.fft_cutoff, args.fft_size
        )

    clean_patch = identity_patch(clean, args.identity_patch_size)
    clean_output, clean_residual = purifier(clean_patch)
    identity = F.l1_loss(clean_output, clean_patch) + clean_residual.abs().mean()
    detector, detector_parts = detection_loss(victim, purified[0], target)
    total = (
        args.recon_weight * recon
        + args.highpass_weight * hp
        + args.fft_weight * fft
        + args.identity_weight * identity
        + det_weight * detector
    )
    parts = {
        "recon": recon.detach(),
        "highpass": hp.detach(),
        "fft": fft.detach(),
        "identity": identity.detach(),
        "detector": detector.detach(),
        "residual_abs": residual.detach().abs().mean(),
    }
    for name, value in detector_parts.items():
        parts[f"det_{name}"] = value.detach()
    return total, parts


def attack_steps_for_epoch(epoch, epochs, start, end):
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return int(round(start + ratio * (end - start)))


def weight_for_epoch(epoch, epochs, start, end):
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return start + ratio * (end - start)


def optimizer_step(optimizer, scaler, purifier, grad_clip):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(purifier.parameters(), grad_clip)
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
    attack_steps,
    det_weight,
    args,
    epoch,
):
    purifier.train()
    purifier.gradient_checkpointing = args.gradient_checkpointing
    totals = {}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"stage3 train {epoch}/{args.epochs} PGD-{attack_steps}")

    for step, (clean, target) in enumerate(progress, start=1):
        image_id = int(target["image_id"].item())
        clean = clean.unsqueeze(0).to(device, non_blocking=True)
        target = target_to_device(target, device, attack_only=True)
        adversarial = adaptive_pgd(
            purifier, victim, clean, target, attack_steps, args, image_id
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts = stage3_loss(
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
            optimizer_step(optimizer, scaler, purifier, args.grad_clip)

        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            det=f"{float(parts['detector']):.3f}",
            w=f"{det_weight:.3f}",
        )
        del clean, target, adversarial, loss, scaled_loss
    return {name: value / len(loader) for name, value in totals.items()}


@torch.inference_mode()
def predict_rows(victim, image, image_id, threshold):
    victim.eval()
    prediction = victim([image])[0]
    return predictions_to_coco(image_id, prediction, threshold)


@torch.inference_mode()
def clean_baseline(victim, dataset, device, threshold):
    rows = []
    for clean, target in tqdm(dataset, desc="stage3 clean baseline"):
        image_id = int(target["image_id"].item())
        rows.extend(predict_rows(victim, clean.to(device), image_id, threshold))
    return evaluate_coco_predictions(dataset.coco, dataset.image_ids, rows)


def validate_adaptive(purifier, victim, dataset, device, args, preview_path):
    groups = {
        "clean_purified": [],
        "adaptive_adversarial": [],
        "adaptive_adversarial_purified": [],
    }
    for index in tqdm(range(len(dataset)), desc=f"stage3 val PGD-{args.val_attack_steps}"):
        clean, target = dataset[index]
        image_id = int(target["image_id"].item())
        clean = clean.unsqueeze(0).to(device)
        target_device = target_to_device(target, device, attack_only=True)

        adversarial = adaptive_pgd(
            purifier,
            victim,
            clean,
            target_device,
            args.val_attack_steps,
            args,
            image_id,
        )
        purifier.eval()
        with torch.inference_mode():
            clean_purified, _ = purifier(clean)
            adversarial_purified, _ = purifier(adversarial)
        groups["clean_purified"].extend(
            predict_rows(victim, clean_purified[0], image_id, args.score_threshold)
        )
        groups["adaptive_adversarial"].extend(
            predict_rows(victim, adversarial[0], image_id, args.score_threshold)
        )
        groups["adaptive_adversarial_purified"].extend(
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
        del clean, adversarial, clean_purified, adversarial_purified
        gc.collect()
        torch.cuda.empty_cache()

    return {
        name: evaluate_coco_predictions(dataset.coco, dataset.image_ids, rows)
        for name, rows in groups.items()
    }


def load_purifier(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    purifier = NativeResidualPurifier(
        channels=int(config.get("channels", 48)),
        max_residual=float(config.get("max_residual", 12 / 255)),
    ).to(device)
    purifier.load_state_dict(checkpoint["model"], strict=True)
    return purifier, checkpoint


def load_split(path):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        [int(value) for value in split["train_image_ids"]],
        [int(value) for value in split["val_image_ids"]],
        split,
    )


def selection_score(clean_baseline_result, detection, penalty):
    clean_drop = max(
        0.0,
        clean_baseline_result["map50_95"]
        - detection["clean_purified"]["map50_95"],
    )
    return detection["adaptive_adversarial_purified"]["map50_95"] - penalty * clean_drop


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
    best_adaptive_map,
    args,
):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(
        {
            "format": "native_residual_purifier_stage3_adaptive_v1",
            "epoch": epoch,
            "best_score": best_score,
            "best_adaptive_map50_95": best_adaptive_map,
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
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-3需要GPU运行时")

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, split = load_split(args.split)
    if args.train_limit > 0:
        train_ids = train_ids[: args.train_limit]
    if args.val_limit > 0:
        val_ids = val_ids[: args.val_limit]
    atomic_write_json(output_dir / "split.json", split)
    train_dataset = CleanCocoSubset(args.images, args.annotations, train_ids)
    val_dataset = CleanCocoSubset(args.images, args.annotations, val_ids)
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

    purifier, source_checkpoint = load_purifier(args.stage2_checkpoint, device)
    config = source_checkpoint.get("model_config", {})
    args.channels = int(config.get("channels", 48))
    args.max_residual = float(config.get("max_residual", 12 / 255))
    purifier.gradient_checkpointing = args.gradient_checkpointing
    victim, weights = load_victim(device)
    victim.transform.min_size = (args.detector_min_size,)
    victim.transform.max_size = args.detector_max_size

    optimizer = torch.optim.AdamW(
        purifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    baseline_path = output_dir / "clean_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = clean_baseline(
            victim, val_dataset, device, args.score_threshold
        )
        atomic_write_json(baseline_path, baseline)

    last_path = output_dir / "purifier_stage3_last.pt"
    best_path = output_dir / "purifier_stage3_best.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1
    best_score = -math.inf
    best_adaptive_map = -math.inf
    history = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        purifier.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_adaptive_map = float(checkpoint["best_adaptive_map50_95"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    print("device:", device)
    print("victim weights:", weights)
    print("train/val:", len(train_dataset), len(val_dataset))
    print("clean baseline:", json.dumps(baseline, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        attack_steps = attack_steps_for_epoch(
            epoch,
            args.epochs,
            args.train_attack_steps_start,
            args.train_attack_steps_end,
        )
        det_weight = weight_for_epoch(
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
            attack_steps,
            det_weight,
            args,
            epoch,
        )
        detection = validate_adaptive(
            purifier,
            victim,
            val_dataset,
            device,
            args,
            preview_dir / f"epoch_{epoch:03d}.png",
        )
        scheduler.step()
        score = selection_score(
            baseline, detection, args.clean_drop_penalty
        )
        adaptive_map = detection["adaptive_adversarial_purified"]["map50_95"]
        is_best = score > best_score
        if is_best:
            best_score = score
            best_adaptive_map = adaptive_map
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_attack_steps": attack_steps,
            "det_weight": det_weight,
            "train": train_metrics,
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
            best_adaptive_map,
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
                best_adaptive_map,
                args,
            )
        atomic_write_json(history_path, history)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print("best score:", best_score, "best adaptive mAP50-95:", best_adaptive_map)

    purifier.to("cpu")
    victim.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print("\nStage-3完成")
    print("best checkpoint:", best_path)
    print("history:", history_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-alpha", type=float, default=2 / 255)
    parser.add_argument("--train-attack-steps-start", type=int, default=1)
    parser.add_argument("--train-attack-steps-end", type=int, default=3)
    parser.add_argument("--val-attack-steps", type=int, default=3)
    parser.add_argument("--det-weight-start", type=float, default=0.05)
    parser.add_argument("--det-weight-end", type=float, default=0.10)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--highpass-weight", type=float, default=0.05)
    parser.add_argument("--fft-weight", type=float, default=0.02)
    parser.add_argument("--fft-cutoff", type=float, default=0.15)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--identity-patch-size", type=int, default=256)
    parser.add_argument("--clean-drop-penalty", type=float, default=2.0)
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
