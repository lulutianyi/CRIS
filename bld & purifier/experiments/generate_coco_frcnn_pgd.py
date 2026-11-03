"""Generate resumable untargeted PGD pairs from official Faster R-CNN v2."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import (  # noqa: E402
    CocoFixedSubset,
    atomic_write_json,
    load_manifest,
    load_victim,
    select_image_ids,
    target_to_device,
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
    losses = model([image], [target])
    return sum(losses.values()), {
        name: float(value.detach()) for name, value in losses.items()
    }


def attack_one(model, clean, target, args, image_id):
    torch.manual_seed(args.seed + int(image_id))
    adversarial = clean + torch.empty_like(clean).uniform_(
        -args.epsilon, args.epsilon
    )
    adversarial = adversarial.clamp(0, 1).detach()
    initial_loss = None
    final_components = None
    for _ in range(args.steps):
        adversarial.requires_grad_(True)
        loss, components = detection_loss(model, adversarial, target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + args.alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-args.epsilon, args.epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
        final_components = components
    with torch.no_grad():
        final_loss, final_components = detection_loss(
            model, adversarial, target
        )
    adversarial = adversarial.detach()
    return adversarial, {
        "initial_loss": initial_loss,
        "final_loss": float(final_loss.detach()),
        "final_loss_components": final_components,
    }


def save_png(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = TF.to_pil_image(tensor.detach().cpu().clamp(0, 1))
    image.save(path, format="PNG", compress_level=3)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    adversarial_dir = output_dir / "adversarial_png"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.annotations))
    selected_ids = select_image_ids(coco, args.subset_size, args.seed)
    dataset = CocoFixedSubset(
        args.images, args.annotations, selected_ids
    )
    model, weights = load_victim(device)

    manifest = {
        "dataset": "coco",
        "images": str(args.images),
        "annotations": str(args.annotations),
        "victim": "fasterrcnn_resnet50_fpn_v2",
        "weights": str(weights),
        "attack": {
            "type": "untargeted_detection_loss_pgd",
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "random_start": True,
            "seed": args.seed,
        },
        "requested_subset_size": args.subset_size,
        "selected_image_ids": selected_ids,
        "samples": [],
    }
    if args.resume and manifest_path.exists():
        old = load_manifest(manifest_path)
        if old.get("selected_image_ids") != selected_ids:
            raise ValueError("已有manifest的image_ids与本次seed/subset不一致")
        if old.get("attack") != manifest["attack"]:
            raise ValueError("已有manifest的PGD参数与本次运行不一致")
        manifest = old
    completed = {int(sample["image_id"]) for sample in manifest["samples"]}

    for image, target, original_path, file_name in tqdm(
        dataset, desc=f"Faster R-CNN PGD-{args.steps}"
    ):
        image_id = int(target["image_id"].item())
        if image_id in completed:
            continue
        clean = image.to(device)
        attack_target = target_to_device(target, device, attack_only=True)
        adversarial, loss_info = attack_one(
            model, clean, attack_target, args, image_id
        )
        output_path = adversarial_dir / f"{image_id:012d}.png"
        save_png(adversarial, output_path)
        quantized = TF.to_tensor(Image.open(output_path).convert("RGB")).to(device)
        difference = quantized - clean
        sample = {
            "image_id": image_id,
            "file_name": file_name,
            "original_path": original_path,
            "adversarial_path": str(output_path),
            "width": int(clean.shape[-1]),
            "height": int(clean.shape[-2]),
            "linf_after_png": float(difference.abs().max().cpu()),
            "mae_after_png": float(difference.abs().mean().cpu()),
            **loss_info,
        }
        manifest["samples"].append(sample)
        completed.add(image_id)
        atomic_write_json(manifest_path, manifest)
        del clean, adversarial, quantized, difference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["completed_samples"] = len(manifest["samples"])
    atomic_write_json(manifest_path, manifest)
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nPGD生成完成:", manifest_path)
    print("样本数:", len(manifest["samples"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
