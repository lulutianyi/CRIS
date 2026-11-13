"""Reproduce a controllable ZeroPur CIFAR-10 baseline on a fixed PGD subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pgd_linf(model, images, labels, epsilon, alpha, steps, seed):
    generator = torch.Generator(device=images.device).manual_seed(seed)
    delta = torch.empty_like(images).uniform_(
        -epsilon, epsilon, generator=generator
    )
    adversarial = (images + delta).clamp(0, 1).detach()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - images).clamp(-epsilon, epsilon)
        adversarial = (images + delta).clamp(0, 1).detach()
    return adversarial


@torch.inference_mode()
def correct(model, images, labels):
    return int((model(images).argmax(dim=1) == labels).sum())


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("ZeroPur包含输入梯度迭代，请使用GPU运行时")
    device = torch.device("cuda")

    official = args.official_root.resolve()
    zero_pur_dir = official / "zero_pur"
    sys.path.insert(0, str(official))
    sys.path.insert(0, str(zero_pur_dir))

    import helper  # noqa: E402
    from models.preact_resnet import PreActResNet18  # noqa: E402
    from zeropur import ZeroPur  # noqa: E402

    checkpoint_path = (
        zero_pur_dir
        / "checkpoint"
        / f"resnet18_cifar10_{args.classifier_type}"
        / "ckpt.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少官方分类器权重：{checkpoint_path}")

    classifier = PreActResNet18().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    classifier.load_state_dict(checkpoint["net"], strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    normalizer = helper.Normalize(mean=MEAN, std=STD).to(device)
    defended_model = torch.nn.Sequential(normalizer, classifier).eval()
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    totals = {
        "count": 0,
        "clean_correct": 0,
        "adversarial_correct": 0,
        "blur_correct": 0,
        "zeropur_coarse_correct": 0,
        "zeropur_fine_correct": 0,
        "clean_coarse_correct": 0,
        "clean_fine_correct": 0,
        "linf_sum": 0.0,
    }
    start = time.time()
    for batch_index, (clean, labels) in enumerate(
        tqdm(loader, desc=f"ZeroPur CIFAR-10 n={args.subset_size}")
    ):
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adversarial = pgd_linf(
            defended_model,
            clean,
            labels,
            args.epsilon,
            args.attack_alpha,
            args.attack_steps,
            args.seed + batch_index,
        )
        blurred = purifier.transform(adversarial.detach())
        fine, coarse = purifier(
            adversarial,
            eps=args.purify_epsilon,
            arch="r18",
            num_steps=args.coarse_steps,
            ila_step=args.fine_steps,
        )
        clean_fine, clean_coarse = purifier(
            clean,
            eps=args.purify_epsilon,
            arch="r18",
            num_steps=args.coarse_steps,
            ila_step=args.fine_steps,
        )

        batch_size = clean.shape[0]
        totals["count"] += batch_size
        totals["clean_correct"] += correct(defended_model, clean, labels)
        totals["adversarial_correct"] += correct(
            defended_model, adversarial, labels
        )
        totals["blur_correct"] += correct(defended_model, blurred, labels)
        totals["zeropur_coarse_correct"] += correct(
            defended_model, coarse, labels
        )
        totals["zeropur_fine_correct"] += correct(defended_model, fine, labels)
        totals["clean_coarse_correct"] += correct(
            defended_model, clean_coarse, labels
        )
        totals["clean_fine_correct"] += correct(
            defended_model, clean_fine, labels
        )
        totals["linf_sum"] += float(
            (adversarial - clean).abs().flatten(1).amax(dim=1).sum()
        )

        del clean, labels, adversarial, blurred, fine, coarse
        del clean_fine, clean_coarse
        torch.cuda.empty_cache()

    count = totals["count"]
    summary = {
        "dataset": "cifar10_test",
        "subset_size": count,
        "seed": args.seed,
        "classifier": "PreActResNet18",
        "classifier_type": args.classifier_type,
        "official_root": str(official),
        "checkpoint": str(checkpoint_path),
        "attack": {
            "type": "nonadaptive_pgd_linf",
            "epsilon": args.epsilon,
            "alpha": args.attack_alpha,
            "steps": args.attack_steps,
            "mean_linf": totals["linf_sum"] / count,
        },
        "purifier": {
            "type": "official_zeropur",
            "blur": "gaussian_sigma_1.8",
            "epsilon": args.purify_epsilon,
            "coarse_steps": args.coarse_steps,
            "fine_steps": args.fine_steps,
        },
        "accuracy": {
            "clean": totals["clean_correct"] / count,
            "adversarial": totals["adversarial_correct"] / count,
            "blur_only": totals["blur_correct"] / count,
            "zeropur_coarse": totals["zeropur_coarse_correct"] / count,
            "zeropur_fine": totals["zeropur_fine_correct"] / count,
            "clean_after_coarse": totals["clean_coarse_correct"] / count,
            "clean_after_fine": totals["clean_fine_correct"] / count,
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
    parser.add_argument("--classifier-type", choices=("none", "baseline", "all"), default="all")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-alpha", type=float, default=2 / 255)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--purify-epsilon", type=float, default=10 / 255)
    parser.add_argument("--coarse-steps", type=int, default=10)
    parser.add_argument("--fine-steps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
