"""Evaluate official ZeroPur with AutoAttack on a fixed CIFAR-10 subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def accuracy(model, images, labels, batch_size, device):
    correct = 0
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        with torch.inference_mode():
            correct += int((model(x).argmax(1) == y).sum())
    return correct / len(images)


def purify_and_measure(purifier, model, images, labels, args, device):
    coarse_correct = 0
    fine_correct = 0
    for start in range(0, len(images), args.purify_batch_size):
        x = images[start : start + args.purify_batch_size].to(device)
        y = labels[start : start + args.purify_batch_size].to(device)
        fine, coarse = purifier(
            x,
            eps=args.purify_epsilon,
            arch="r18",
            num_steps=args.coarse_steps,
            ila_step=args.fine_steps,
        )
        with torch.inference_mode():
            coarse_correct += int((model(coarse).argmax(1) == y).sum())
            fine_correct += int((model(fine).argmax(1) == y).sum())
        del x, y, fine, coarse
    return coarse_correct / len(images), fine_correct / len(images)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("请使用GPU运行时")
    device = torch.device("cuda")

    official = args.official_root.resolve()
    zero_pur_dir = official / "zero_pur"
    sys.path.insert(0, str(official))
    sys.path.insert(0, str(zero_pur_dir))

    import helper
    from attack import run_autoattack
    from models.preact_resnet import PreActResNet18
    from zeropur import ZeroPur

    checkpoint_path = (
        zero_pur_dir / "checkpoint" / "resnet18_cifar10_all" / "ckpt.pth"
    )
    classifier = PreActResNet18().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    classifier.load_state_dict(checkpoint["net"], strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    normalizer = helper.Normalize(mean=MEAN, std=STD).to(device)
    model = torch.nn.Sequential(normalizer, classifier).eval()
    purifier = ZeroPur(
        classifier,
        transform=helper.WeakTransform("gaussian"),
        normalizer=normalizer,
        arch="r18",
    ).to(device)

    dataset = CIFAR10(
        root=str(args.data_root), train=False, transform=ToTensor(), download=True
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.subset_size]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.subset_size,
        shuffle=False,
        num_workers=0,
    )
    clean, labels = next(iter(loader))

    start = time.time()
    clean_accuracy = accuracy(
        model, clean, labels, args.attack_batch_size, device
    )
    adversarial = run_autoattack(
        model, clean, labels, batch_size=args.attack_batch_size
    ).detach().cpu()
    adversarial_accuracy = accuracy(
        model, adversarial, labels, args.attack_batch_size, device
    )
    coarse_accuracy, fine_accuracy = purify_and_measure(
        purifier, model, adversarial, labels, args, device
    )
    clean_coarse_accuracy, clean_fine_accuracy = purify_and_measure(
        purifier, model, clean, labels, args, device
    )

    summary = {
        "dataset": "cifar10_test",
        "subset_size": len(indices),
        "seed": args.seed,
        "classifier": "PreActResNet18",
        "classifier_type": "all",
        "checkpoint": str(checkpoint_path),
        "attack": {
            "type": "autoattack_standard_linf",
            "epsilon": 8 / 255,
            "batch_size": args.attack_batch_size,
        },
        "purifier": {
            "type": "official_zeropur",
            "epsilon": args.purify_epsilon,
            "coarse_steps": args.coarse_steps,
            "fine_steps": args.fine_steps,
        },
        "accuracy": {
            "clean": clean_accuracy,
            "adversarial": adversarial_accuracy,
            "zeropur_coarse": coarse_accuracy,
            "zeropur_fine": fine_accuracy,
            "clean_after_coarse": clean_coarse_accuracy,
            "clean_after_fine": clean_fine_accuracy,
        },
        "elapsed_seconds": time.time() - start,
        "indices": indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果：", args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/content/cifar10"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--attack-batch-size", type=int, default=100)
    parser.add_argument("--purify-batch-size", type=int, default=8)
    parser.add_argument("--purify-epsilon", type=float, default=10 / 255)
    parser.add_argument("--coarse-steps", type=int, default=10)
    parser.add_argument("--fine-steps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
