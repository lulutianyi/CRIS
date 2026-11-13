"""Reproduce and ablate training-free BiGuidePur on a fixed CIFAR-10 subset."""

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
from tqdm.auto import tqdm

from biguide_purifier import BiGuidePurifier


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def load_models(args, device):
    official = args.official_root.resolve()
    zero_pur_dir = official / "zero_pur"
    sys.path.insert(0, str(official))
    sys.path.insert(0, str(zero_pur_dir))
    import helper
    from models.preact_resnet import PreActResNet18
    from zeropur import ZeroPur

    checkpoint_path = (
        zero_pur_dir / "checkpoint" / "resnet18_cifar10_all" / "ckpt.pth"
    )
    classifier = PreActResNet18().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["net"], strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    normalizer = helper.Normalize(mean=MEAN, std=STD).to(device)
    model = torch.nn.Sequential(normalizer, classifier).eval()
    official_purifier = ZeroPur(
        classifier,
        transform=helper.WeakTransform("gaussian"),
        normalizer=normalizer,
        arch="r18",
    ).to(device)
    proposed = BiGuidePurifier(classifier, normalizer).to(device).eval()
    return model, official_purifier, proposed, helper, checkpoint_path


def make_attack_cache(args, model, device):
    from attack import run_autoattack

    dataset = CIFAR10(
        root=str(args.data_root), train=False, transform=ToTensor(), download=True
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.subset_size]
    clean, labels = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
        )
    )
    adversarial = run_autoattack(
        model, clean, labels, batch_size=args.attack_batch_size
    ).detach().cpu()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clean": clean.cpu(),
            "adversarial": adversarial,
            "labels": labels.cpu(),
            "indices": indices,
            "seed": args.seed,
            "attack": "official_custom_apgd_ce_dlr_linf_8_255",
        },
        args.cache,
    )
    print("攻击缓存：", args.cache)


def add_correct(totals, name, images, labels, model):
    with torch.inference_mode():
        totals[name] += int((model(images).argmax(1) == labels).sum())


def evaluate(args, model, official, proposed, helper, device):
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    clean = cache["clean"]
    adversarial = cache["adversarial"]
    labels = cache["labels"]
    names = (
        "clean_input",
        "adversarial_input",
        "blur_only_adv",
        "blur_only_clean",
        "official_coarse_adv",
        "official_coarse_clean",
        "local_graph_adv",
        "local_graph_clean",
        "hf_consensus_adv",
        "hf_consensus_clean",
        "mutual_fusion_adv",
        "mutual_fusion_clean",
    )
    totals = {name: 0 for name in names}
    start_time = time.time()

    for start in tqdm(
        range(0, len(labels), args.batch_size), desc="BiGuidePur ablation"
    ):
        end = start + args.batch_size
        x_clean = clean[start:end].to(device)
        x_adv = adversarial[start:end].to(device)
        y = labels[start:end].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + start)

        blur_adv = official.transform(x_adv)
        blur_clean = official.transform(x_clean)
        official_adv = official.coarse_shifting(
            x_adv, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()
        official_clean = official.coarse_shifting(
            x_clean, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()
        local_adv = proposed.local_graph(
            x_adv,
            generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )
        local_clean = proposed.local_graph(
            x_clean,
            generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )
        hf_adv = proposed.high_frequency_consensus(
            x_adv,
            generator,
            eps=args.epsilon,
            steps=args.steps,
            noise_scale=args.noise_scale,
        )
        hf_clean = proposed.high_frequency_consensus(
            x_clean,
            generator,
            eps=args.epsilon,
            steps=args.steps,
            noise_scale=args.noise_scale,
        )
        fused_adv = proposed.mutual_fusion(
            x_adv,
            local_adv,
            hf_adv,
            eps=args.epsilon,
            steps=args.fusion_steps,
            local_weight=args.fusion_local_weight,
        )
        fused_clean = proposed.mutual_fusion(
            x_clean,
            local_clean,
            hf_clean,
            eps=args.epsilon,
            steps=args.fusion_steps,
            local_weight=args.fusion_local_weight,
        )

        values = {
            "clean_input": x_clean,
            "adversarial_input": x_adv,
            "blur_only_adv": blur_adv,
            "blur_only_clean": blur_clean,
            "official_coarse_adv": official_adv,
            "official_coarse_clean": official_clean,
            "local_graph_adv": local_adv,
            "local_graph_clean": local_clean,
            "hf_consensus_adv": hf_adv,
            "hf_consensus_clean": hf_clean,
            "mutual_fusion_adv": fused_adv,
            "mutual_fusion_clean": fused_clean,
        }
        for name, images in values.items():
            add_correct(totals, name, images, y, model)
        del values, x_clean, x_adv, y

    count = len(labels)
    accuracy = {name: value / count for name, value in totals.items()}
    clean_base = accuracy["clean_input"]
    summary = {
        "dataset": "cifar10_test",
        "subset_size": count,
        "seed": cache["seed"],
        "attack": cache["attack"],
        "accuracy": accuracy,
        "derived": {
            name.removesuffix("_adv"): {
                "robust_accuracy": accuracy[name],
                "robust_retention_vs_clean_input": accuracy[name] / clean_base,
                "clean_retention": accuracy[name.replace("_adv", "_clean")]
                / clean_base,
            }
            for name in accuracy
            if name.endswith("_adv") and name != "adversarial_input"
        },
        "settings": vars_for_json(args),
        "elapsed_seconds": time.time() - start_time,
        "indices": cache["indices"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果：", args.output)


def vars_for_json(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("attack", "evaluate", "all"), default="all")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/content/cifar10"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--attack-batch-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=10 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-guides", type=int, default=3)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--patch-ratio", type=float, default=0.25)
    parser.add_argument("--graph-steps", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=4 / 255)
    parser.add_argument("--fusion-steps", type=int, default=3)
    parser.add_argument("--fusion-local-weight", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("请使用GPU运行时")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, official, proposed, helper, checkpoint = load_models(args, device)
    print("checkpoint:", checkpoint)
    if args.stage in ("attack", "all"):
        make_attack_cache(args, model, device)
    if args.stage in ("evaluate", "all"):
        evaluate(args, model, official, proposed, helper, device)


if __name__ == "__main__":
    main()
