"""Evaluate a COCO residual purifier with non-adaptive or adaptive PGD."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from pycocotools.coco import COCO
from tqdm.auto import tqdm
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import (  # noqa: E402
    CocoFixedSubset,
    atomic_write_json,
    evaluate_coco_predictions,
    load_victim,
    predictions_to_coco,
    select_image_ids,
    target_to_device,
)
from train_coco_residual_purifier_stage1 import (  # noqa: E402
    NativeResidualPurifier,
    highpass,
    ssim_value,
)


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
    return sum(model([image], [target]).values())


def nonadaptive_pgd(victim, _purifier, clean, target, args, image_id):
    torch.manual_seed(args.seed + int(image_id))
    adversarial = clean + torch.empty_like(clean).uniform_(
        -args.epsilon, args.epsilon
    )
    adversarial = adversarial.clamp(0, 1).detach()
    initial_loss = None
    for _ in range(args.attack_steps):
        adversarial.requires_grad_(True)
        loss = detection_loss(victim, adversarial, target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + args.attack_alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-args.epsilon, args.epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
    with torch.no_grad():
        final_loss = detection_loss(victim, adversarial, target)
    victim.eval()
    return adversarial, float(initial_loss), float(final_loss)


def adaptive_pgd(victim, purifier, clean, target, args, image_id):
    torch.manual_seed(args.seed + int(image_id))
    adversarial = clean + torch.empty_like(clean).uniform_(
        -args.epsilon, args.epsilon
    )
    adversarial = adversarial.clamp(0, 1).detach()
    initial_loss = None
    purifier.train()
    purifier.gradient_checkpointing = True
    for _ in range(args.attack_steps):
        adversarial.requires_grad_(True)
        purified, _ = purifier(adversarial.unsqueeze(0))
        loss = detection_loss(victim, purified[0], target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + args.attack_alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-args.epsilon, args.epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
        del purified, loss, gradient
    with torch.no_grad():
        final_purified, _ = purifier(adversarial.unsqueeze(0))
        final_loss = detection_loss(victim, final_purified[0], target)
    purifier.eval()
    victim.eval()
    return adversarial, float(initial_loss), float(final_loss)


def load_purifier(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint.get("model_config", {})
    model = NativeResidualPurifier(
        channels=int(config.get("channels", 48)),
        max_residual=float(config.get("max_residual", 12 / 255)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def predict_one(model, image, image_id, score_threshold):
    model.eval()
    prediction = model([image])[0]
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


def add_metrics(totals, values):
    for name, value in values.items():
        totals[name] += float(value)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def recovery(clean, adversarial, purified):
    return safe_ratio(purified - adversarial, clean - adversarial)


def load_image_ids(args, coco):
    source = args.image_ids_json or args.reference_summary
    if source:
        value = json.loads(source.read_text(encoding="utf-8"))
        image_ids = value["image_ids"] if isinstance(value, dict) else value
        return [int(image_id) for image_id in image_ids]
    return select_image_ids(coco, args.subset_size, args.seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PGD评测需要GPU运行时")

    adaptive = args.attack_mode == "adaptive"
    attack = adaptive_pgd if adaptive else nonadaptive_pgd
    adversarial_key = "adaptive_adversarial" if adaptive else "adversarial"
    purified_key = f"{adversarial_key}_purified"

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.annotations))
    image_ids = load_image_ids(args, coco)
    dataset = CocoFixedSubset(args.images, args.annotations, image_ids)
    victim, weights = load_victim(device)
    purifier, checkpoint = load_purifier(args.checkpoint, device)

    groups = {
        "clean": [],
        "clean_purified": [],
        adversarial_key: [],
        purified_key: [],
    }
    metric_names = ("mae", "psnr", "ssim", "highpass_psnr")
    metric_totals = {
        name: {key: 0.0 for key in metric_names}
        for name in (adversarial_key, purified_key, "clean_purified")
    }
    attack_totals = {
        "linf": 0.0,
        "mae": 0.0,
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "loss_increased": 0,
    }

    label = "adaptive" if adaptive else "nonadaptive"
    progress = tqdm(dataset, desc=f"{label} PGD-{args.attack_steps}")
    for index, (image, target, _path, _name) in enumerate(progress):
        image_id = int(target["image_id"].item())
        clean = image.to(device)
        attack_target = target_to_device(target, device, attack_only=True)

        groups["clean"].extend(
            predict_one(victim, clean, image_id, args.score_threshold)
        )
        with torch.inference_mode():
            clean_purified, _ = purifier(clean.unsqueeze(0))
        groups["clean_purified"].extend(
            predict_one(victim, clean_purified[0], image_id, args.score_threshold)
        )

        adversarial, initial_loss, final_loss = attack(
            victim, purifier, clean, attack_target, args, image_id
        )
        groups[adversarial_key].extend(
            predict_one(victim, adversarial, image_id, args.score_threshold)
        )
        with torch.inference_mode():
            adversarial_purified, _ = purifier(adversarial.unsqueeze(0))
        groups[purified_key].extend(
            predict_one(
                victim,
                adversarial_purified[0],
                image_id,
                args.score_threshold,
            )
        )

        difference = adversarial - clean
        attack_totals["linf"] += float(difference.abs().max())
        attack_totals["mae"] += float(difference.abs().mean())
        attack_totals["initial_loss"] += initial_loss
        attack_totals["final_loss"] += final_loss
        attack_totals["loss_increased"] += int(final_loss > initial_loss)

        clean_batch = clean.unsqueeze(0)
        add_metrics(
            metric_totals[adversarial_key],
            image_metrics(adversarial.unsqueeze(0), clean_batch),
        )
        add_metrics(
            metric_totals[purified_key],
            image_metrics(adversarial_purified, clean_batch),
        )
        add_metrics(
            metric_totals["clean_purified"],
            image_metrics(clean_purified, clean_batch),
        )

        if index < args.preview_count:
            correction = (adversarial_purified[0] - adversarial).abs() * 8
            preview = torch.cat(
                [
                    clean.unsqueeze(0),
                    adversarial.unsqueeze(0),
                    adversarial_purified,
                    correction.unsqueeze(0).clamp(0, 1),
                ],
                dim=-1,
            )
            save_image(
                preview.cpu(),
                preview_dir / f"{image_id:012d}.png",
                format="png",
            )

        del clean, clean_purified, adversarial, adversarial_purified
        torch.cuda.empty_cache()

    detection = {}
    for name, predictions in groups.items():
        atomic_write_json(output_dir / f"{name}_predictions.json", predictions)
        detection[name] = evaluate_coco_predictions(coco, image_ids, predictions)

    count = len(dataset)
    clean = detection["clean"]
    clean_purified = detection["clean_purified"]
    adversarial = detection[adversarial_key]
    purified = detection[purified_key]
    attack_summary = {
        "type": (
            "end_to_end_adaptive_detection_loss_pgd"
            if adaptive
            else "non_adaptive_detection_loss_pgd"
        ),
        "epsilon": args.epsilon,
        "alpha": args.attack_alpha,
        "steps": args.attack_steps,
        "random_start": True,
        "mean_linf": attack_totals["linf"] / count,
        "mean_mae": attack_totals["mae"] / count,
        "loss_increased_fraction": attack_totals["loss_increased"] / count,
    }
    if adaptive:
        attack_summary["mean_initial_defended_loss"] = (
            attack_totals["initial_loss"] / count
        )
        attack_summary["mean_final_defended_loss"] = (
            attack_totals["final_loss"] / count
        )
        derived = {
            "adaptive_map50_95_recovery_vs_direct": recovery(
                clean["map50_95"], adversarial["map50_95"], purified["map50_95"]
            ),
            "adaptive_map50_recovery_vs_direct": recovery(
                clean["map50"], adversarial["map50"], purified["map50"]
            ),
            "defended_map50_95_retention": safe_ratio(
                purified["map50_95"], clean_purified["map50_95"]
            ),
            "defended_map50_retention": safe_ratio(
                purified["map50"], clean_purified["map50"]
            ),
        }
    else:
        attack_summary["mean_initial_loss"] = attack_totals["initial_loss"] / count
        attack_summary["mean_final_loss"] = attack_totals["final_loss"] / count
        derived = {
            "map50_95_recovery": recovery(
                clean["map50_95"], adversarial["map50_95"], purified["map50_95"]
            ),
            "map50_recovery": recovery(
                clean["map50"], adversarial["map50"], purified["map50"]
            ),
        }
    derived.update(
        {
            "clean_map50_95_retention": safe_ratio(
                clean_purified["map50_95"], clean["map50_95"]
            ),
            "clean_map50_retention": safe_ratio(
                clean_purified["map50"], clean["map50"]
            ),
        }
    )

    summary = {
        "dataset": "coco_val2017",
        "subset_size": count,
        "seed": args.seed,
        "image_ids": image_ids,
        "victim": "fasterrcnn_resnet50_fpn_v2",
        "victim_weights": str(weights),
        "purifier_checkpoint": str(args.checkpoint),
        "purifier_epoch": int(checkpoint.get("epoch", -1)),
        "attack": attack_summary,
        "image_metrics": {
            name: {key: value / count for key, value in totals.items()}
            for name, totals in metric_totals.items()
        },
        "detection": detection,
        "derived": derived,
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"\n========== COCO {label} PGD评测 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果:", summary_path)

    victim.to("cpu")
    purifier.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-mode",
        choices=("nonadaptive", "adaptive"),
        default="nonadaptive",
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids-json", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-alpha", type=float, default=2 / 255)
    parser.add_argument("--attack-steps", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
