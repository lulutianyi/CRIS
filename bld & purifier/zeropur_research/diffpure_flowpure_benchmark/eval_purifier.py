"""Evaluate DiffPure/FlowPure with ZeroPur's ResNet18 on one fixed cache."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from benchmark_common import (
    evaluate_accuracy,
    generate_bpda,
    load_fixed_cache,
    load_zeropur_resnet18,
    seed_everything,
)


METHOD_ALIASES = {
    "diffpure": "diffpure",
    "flowpure": "pgd_flowpure",
    "pgd_flowpure": "pgd_flowpure",
    "gauss_flowpure_0.15": "gauss_flowpure_0.15",
    "gauss_flowpure_0.2": "gauss_flowpure_0.2",
    "flowpure_stoch": "flowpure_stoch",
}


def expected_checkpoint(root, method):
    paths = {
        "diffpure": "resources/checkpoints/score_sde/checkpoint_c10.pth",
        "pgd_flowpure": "resources/checkpoints/flowpure_pgd/checkpoint_c10.pth",
        "gauss_flowpure_0.15": "resources/checkpoints/flowpure_gauss/checkpoint_c10.pth",
        "gauss_flowpure_0.2": "resources/checkpoints/flowpure_gauss/checkpoint_c10.pth",
        "flowpure_stoch": "resources/checkpoints/flowpure_from_zero/checkpoint_c10.pth",
    }
    return root / paths[method]


def load_official_defense(args, victim, count):
    root = args.flowpure_root.resolve()
    method = METHOD_ALIASES[args.method]
    checkpoint = expected_checkpoint(root, method)
    if not checkpoint.exists():
        raise FileNotFoundError(f"缺少净化器权重: {checkpoint}")

    sys.path.insert(0, str(root))
    os.chdir(root)
    from config import get_config
    import defense  # noqa: F401 - registers the official defenses
    import utils as flowpure_utils
    from utils import get_defense, get_model

    # FlowPure uses random.randint(0, 1e8); Python 3.13 rejects the float bound.
    def compatible_gen_seed(seed, offset):
        for _ in range(offset):
            flowpure_utils.set_seed(seed)
            seed = random.randint(0, 100_000_000)
        return seed

    flowpure_utils.gen_seed = compatible_gen_seed
    diffpure_module = importlib.import_module("defense.diffpure")
    diffpure_module.gen_seed = compatible_gen_seed

    cfg = get_config(
        method,
        "CIFAR10",
        "class_pgd",
        args.seed,
        args.batch_size,
        count,
    )
    diffusion = get_model(cfg.DEFENSE.DIFFUSION_NAME).cuda().eval()
    defense_config = getattr(cfg.DEFENSE, cfg.DEFENSE.METHOD.upper())
    defense = get_defense(cfg.DEFENSE.METHOD)(
        diffusion, victim, defense_config
    ).cuda().eval()
    defense.requires_grad_(False)
    return defense, method, checkpoint, cfg


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DiffPure/FlowPure与BPDA评测需要GPU运行时")
    args.flowpure_root = args.flowpure_root.resolve()
    args.zeropur_root = args.zeropur_root.resolve()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    seed_everything(args.seed)
    started = time.time()

    clean, autoattack, labels, autoattack_metadata = load_fixed_cache(
        args.cache, args.limit
    )
    victim, victim_checkpoint = load_zeropur_resnet18(
        args.zeropur_root, torch.device("cuda")
    )
    defense, method, purifier_checkpoint, cfg = load_official_defense(
        args, victim, len(labels)
    )

    clean_accuracy = evaluate_accuracy(
        defense,
        clean,
        labels,
        args.batch_size,
        args.seed,
        args.eval_eot,
        f"{method} clean",
    )
    autoattack_accuracy = evaluate_accuracy(
        defense,
        autoattack,
        labels,
        args.batch_size,
        args.seed,
        args.eval_eot,
        f"{method} AutoAttack",
    )

    bpda = generate_bpda(
        defense,
        clean,
        labels,
        args.bpda_batch_size,
        args.epsilon,
        args.bpda_alpha,
        args.bpda_steps,
        args.bpda_eot,
        args.seed + 100_000,
    )
    bpda_accuracy = evaluate_accuracy(
        defense,
        bpda,
        labels,
        args.batch_size,
        args.seed + 200_000,
        args.eval_eot,
        f"{method} BPDA",
    )

    linf = (bpda - clean).abs().flatten(1).amax(1)
    summary = {
        "dataset": "cifar10_test",
        "count": len(labels),
        "seed": args.seed,
        "method": method,
        "victim": {
            "model": "ZeroPur PreActResNet18",
            "checkpoint": str(victim_checkpoint),
        },
        "purifier_checkpoint": str(purifier_checkpoint),
        "accuracy": {
            "clean_after_purification": clean_accuracy,
            "autoattack_after_purification": autoattack_accuracy,
            "bpda20_after_purification": bpda_accuracy,
        },
        "attacks": {
            "autoattack": {
                "source": str(args.cache),
                "mode": "fixed preprocessor-blind adversarial cache",
                "epsilon": args.epsilon,
                "cache_metadata": autoattack_metadata,
            },
            "bpda": {
                "mode": "adaptive BPDA-PGD Linf",
                "epsilon": args.epsilon,
                "alpha": args.bpda_alpha,
                "steps": args.bpda_steps,
                "gradient_eot": args.bpda_eot,
                "mean_linf": float(linf.mean()),
                "max_linf": float(linf.max()),
            },
        },
        "evaluation_eot": args.eval_eot,
        "official_defense_config": str(
            getattr(cfg.DEFENSE, cfg.DEFENSE.METHOD.upper())
        ),
        "elapsed_seconds": time.time() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bpda_path = args.output.with_suffix(".bpda.pt")
    torch.save(
        {"clean": clean, "adversarial": bpda, "labels": labels}, bpda_path
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果:", args.output)
    print("BPDA样本:", bpda_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_ALIASES),
        required=True,
    )
    parser.add_argument("--flowpure-root", type=Path, required=True)
    parser.add_argument("--zeropur-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bpda-batch-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--bpda-alpha", type=float, default=2 / 255)
    parser.add_argument("--bpda-steps", type=int, default=20)
    parser.add_argument("--bpda-eot", type=int, default=1)
    parser.add_argument("--eval-eot", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()

