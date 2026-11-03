"""Mixed non-adaptive/adaptive training with a frozen Stage-2 teacher."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import (  # noqa: E402
    atomic_write_json,
    evaluate_coco_predictions,
    load_victim,
    target_to_device,
)
from train_coco_residual_purifier_stage1 import (  # noqa: E402
    charbonnier,
    highpass,
)
from train_coco_residual_purifier_stage3 import (  # noqa: E402
    CleanCocoSubset,
    adaptive_pgd,
    clean_baseline,
    detection_loss,
    identity_patch,
    load_purifier,
    load_split,
    optimizer_step,
    predict_rows,
    resized_fft_loss,
    seed_everything,
    serializable_args,
    single_item_collate,
    validate_adaptive,
)


def nonadaptive_pgd(victim, clean, target, steps, args, image_id, seed_offset=0):
    torch.manual_seed(args.seed + seed_offset + int(image_id))
    adversarial = clean + torch.empty_like(clean).uniform_(
        -args.epsilon, args.epsilon
    )
    adversarial = adversarial.clamp(0, 1).detach()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss, _ = detection_loss(victim, adversarial[0], target)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + args.attack_alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-args.epsilon, args.epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
        del loss, gradient
    victim.eval()
    return adversarial


def mixed_loss(
    student,
    teacher,
    victim,
    adversarial,
    clean,
    target,
    use_distillation,
    args,
):
    student.train()
    purified, residual = student(adversarial)
    recon = charbonnier(purified, clean)
    hp = F.l1_loss(highpass(purified), highpass(clean))
    with torch.autocast(device_type="cuda", enabled=False):
        fft = resized_fft_loss(
            purified, clean, args.fft_cutoff, args.fft_size
        )

    clean_patch = identity_patch(clean, args.identity_patch_size)
    clean_output, clean_residual = student(clean_patch)
    identity = F.l1_loss(clean_output, clean_patch) + clean_residual.abs().mean()
    detector, detector_parts = detection_loss(victim, purified[0], target)

    if use_distillation:
        teacher.eval()
        with torch.inference_mode():
            teacher_output, _ = teacher(adversarial)
        distill = F.l1_loss(purified, teacher_output)
    else:
        distill = purified.new_zeros(())

    total = (
        args.recon_weight * recon
        + args.highpass_weight * hp
        + args.fft_weight * fft
        + args.identity_weight * identity
        + args.det_weight * detector
        + args.distill_weight * distill
    )
    parts = {
        "recon": recon.detach(),
        "highpass": hp.detach(),
        "fft": fft.detach(),
        "identity": identity.detach(),
        "detector": detector.detach(),
        "distill": distill.detach(),
        "residual_abs": residual.detach().abs().mean(),
    }
    for name, value in detector_parts.items():
        parts[f"det_{name}"] = value.detach()
    return total, parts


def train_epoch(
    student,
    teacher,
    victim,
    loader,
    optimizer,
    scaler,
    device,
    args,
    epoch,
):
    student.train()
    student.gradient_checkpointing = args.gradient_checkpointing
    totals = {"adaptive_count": 0.0, "nonadaptive_count": 0.0}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"stage3-mix train {epoch}/{args.epochs}")

    for step, (clean, target) in enumerate(progress, start=1):
        image_id = int(target["image_id"].item())
        clean = clean.unsqueeze(0).to(device, non_blocking=True)
        target = target_to_device(target, device, attack_only=True)
        use_adaptive = random.random() < args.adaptive_probability

        if use_adaptive:
            adversarial = adaptive_pgd(
                student,
                victim,
                clean,
                target,
                args.adaptive_train_steps,
                args,
                image_id,
            )
            attack_name = "adaptive"
        else:
            adversarial = nonadaptive_pgd(
                victim,
                clean,
                target,
                args.nonadaptive_train_steps,
                args,
                image_id,
                seed_offset=epoch * 100000,
            )
            attack_name = "nonadaptive"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts = mixed_loss(
                student,
                teacher,
                victim,
                adversarial,
                clean,
                target,
                use_distillation=not use_adaptive,
                args=args,
            )
            scaled_loss = loss / args.grad_accumulation
        scaler.scale(scaled_loss).backward()
        if step % args.grad_accumulation == 0 or step == len(loader):
            optimizer_step(optimizer, scaler, student, args.grad_clip)

        totals[f"{attack_name}_count"] += 1
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        progress.set_postfix(
            attack=attack_name[:3],
            loss=f"{float(loss.detach()):.4f}",
            det=f"{float(parts['detector']):.3f}",
        )
        del clean, target, adversarial, loss, scaled_loss

    averaged = {
        name: value / len(loader)
        for name, value in totals.items()
        if not name.endswith("_count")
    }
    averaged["adaptive_fraction"] = totals["adaptive_count"] / len(loader)
    averaged["nonadaptive_fraction"] = totals["nonadaptive_count"] / len(loader)
    return averaged


def validate_nonadaptive(student, victim, dataset, device, args, preview_path):
    groups = {"adversarial": [], "adversarial_purified": []}
    for index in tqdm(
        range(len(dataset)), desc=f"stage3-mix nonadaptive PGD-{args.val_attack_steps}"
    ):
        clean, target = dataset[index]
        image_id = int(target["image_id"].item())
        clean = clean.unsqueeze(0).to(device)
        target_device = target_to_device(target, device, attack_only=True)
        adversarial = nonadaptive_pgd(
            victim,
            clean,
            target_device,
            args.val_attack_steps,
            args,
            image_id,
        )
        student.eval()
        with torch.inference_mode():
            purified, _ = student(adversarial)
        groups["adversarial"].extend(
            predict_rows(victim, adversarial[0], image_id, args.score_threshold)
        )
        groups["adversarial_purified"].extend(
            predict_rows(victim, purified[0], image_id, args.score_threshold)
        )
        if index == 0:
            correction = (purified - adversarial).abs() * 8
            preview = torch.cat(
                [clean, adversarial, purified, correction.clamp(0, 1)], dim=-1
            )
            save_image(preview.cpu(), preview_path, format="png")
        del clean, adversarial, purified
        gc.collect()
        torch.cuda.empty_cache()
    return {
        name: evaluate_coco_predictions(dataset.coco, dataset.image_ids, rows)
        for name, rows in groups.items()
    }


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def calculate_selection(clean, adaptive, nonadaptive, args):
    clean_map = clean["map50_95"]
    clean_purified_map = adaptive["clean_purified"]["map50_95"]
    adversarial_map = nonadaptive["adversarial"]["map50_95"]
    nonadaptive_purified_map = nonadaptive["adversarial_purified"]["map50_95"]
    adaptive_map = adaptive["adaptive_adversarial_purified"]["map50_95"]
    clean_retention = safe_ratio(clean_purified_map, clean_map)
    nonadaptive_recovery = safe_ratio(
        nonadaptive_purified_map - adversarial_map,
        clean_map - adversarial_map,
    )
    feasible = (
        clean_retention >= args.minimum_clean_retention
        and nonadaptive_recovery >= args.minimum_nonadaptive_recovery
    )
    if feasible:
        score = adaptive_map
    else:
        clean_progress = min(
            clean_retention / args.minimum_clean_retention, 1.0
        )
        recovery_progress = min(
            nonadaptive_recovery / args.minimum_nonadaptive_recovery, 1.0
        )
        score = -1.0 + 0.5 * clean_progress + 0.5 * recovery_progress
    return {
        "score": score,
        "feasible": feasible,
        "clean_retention": clean_retention,
        "nonadaptive_recovery": nonadaptive_recovery,
        "adaptive_map50_95": adaptive_map,
    }


def save_checkpoint(
    path,
    student,
    optimizer,
    scaler,
    scheduler,
    epoch,
    best_score,
    args,
):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "native_residual_purifier_stage3_mix_v1",
            "epoch": epoch,
            "best_score": best_score,
            "model": student.state_dict(),
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


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage3-Mix需要GPU运行时")

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

    student, source_checkpoint = load_purifier(args.stage2_checkpoint, device)
    teacher, _ = load_purifier(args.stage2_checkpoint, device)
    config = source_checkpoint.get("model_config", {})
    args.channels = int(config.get("channels", 48))
    args.max_residual = float(config.get("max_residual", 12 / 255))
    student.gradient_checkpointing = args.gradient_checkpointing
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    victim, weights = load_victim(device)
    victim.transform.min_size = (args.detector_min_size,)
    victim.transform.max_size = args.detector_max_size
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    clean_path = output_dir / "clean_baseline.json"
    if clean_path.exists():
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
    else:
        clean = clean_baseline(victim, val_dataset, device, args.score_threshold)
        atomic_write_json(clean_path, clean)

    source_path = output_dir / "source_stage2_dual_baseline.json"
    if source_path.exists():
        source_results = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        source_adaptive = validate_adaptive(
            student,
            victim,
            val_dataset,
            device,
            args,
            preview_dir / "source_stage2_adaptive.png",
        )
        source_nonadaptive = validate_nonadaptive(
            student,
            victim,
            val_dataset,
            device,
            args,
            preview_dir / "source_stage2_nonadaptive.png",
        )
        source_results = {
            "clean": clean,
            "adaptive": source_adaptive,
            "nonadaptive": source_nonadaptive,
            "selection": calculate_selection(
                clean, source_adaptive, source_nonadaptive, args
            ),
        }
        atomic_write_json(source_path, source_results)

    last_path = output_dir / "purifier_stage3_mix_last.pt"
    best_path = output_dir / "purifier_stage3_mix_best.pt"
    history_path = output_dir / "history.json"
    start_epoch = 1
    best_score = float(source_results["selection"]["score"])
    history = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    elif not best_path.exists():
        save_checkpoint(
            best_path,
            student,
            optimizer,
            scaler,
            scheduler,
            0,
            best_score,
            args,
        )

    print("device:", device)
    print("victim weights:", weights)
    print("train/val:", len(train_dataset), len(val_dataset))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            student,
            teacher,
            victim,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
        )
        adaptive = validate_adaptive(
            student,
            victim,
            val_dataset,
            device,
            args,
            preview_dir / f"epoch_{epoch:03d}_adaptive.png",
        )
        nonadaptive = validate_nonadaptive(
            student,
            victim,
            val_dataset,
            device,
            args,
            preview_dir / f"epoch_{epoch:03d}_nonadaptive.png",
        )
        scheduler.step()
        selection = calculate_selection(clean, adaptive, nonadaptive, args)
        score = selection["score"]
        is_best = score > best_score
        if is_best:
            best_score = score
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "adaptive_validation": adaptive,
            "nonadaptive_validation": nonadaptive,
            "selection": selection,
        }
        history.append(row)
        save_checkpoint(
            last_path,
            student,
            optimizer,
            scaler,
            scheduler,
            epoch,
            best_score,
            args,
        )
        if is_best:
            save_checkpoint(
                best_path,
                student,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_score,
                args,
            )
        atomic_write_json(history_path, history)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print("best score:", best_score)

    student.to("cpu")
    teacher.to("cpu")
    victim.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print("\nStage3-Mix完成")
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
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-alpha", type=float, default=2 / 255)
    parser.add_argument("--adaptive-probability", type=float, default=0.5)
    parser.add_argument("--adaptive-train-steps", type=int, default=3)
    parser.add_argument("--nonadaptive-train-steps", type=int, default=3)
    parser.add_argument("--val-attack-steps", type=int, default=5)
    parser.add_argument("--det-weight", type=float, default=0.05)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--highpass-weight", type=float, default=0.05)
    parser.add_argument("--fft-weight", type=float, default=0.02)
    parser.add_argument("--fft-cutoff", type=float, default=0.15)
    parser.add_argument("--fft-size", type=int, default=256)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--identity-patch-size", type=int, default=256)
    parser.add_argument("--distill-weight", type=float, default=0.50)
    parser.add_argument("--minimum-clean-retention", type=float, default=0.98)
    parser.add_argument("--minimum-nonadaptive-recovery", type=float, default=0.40)
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
