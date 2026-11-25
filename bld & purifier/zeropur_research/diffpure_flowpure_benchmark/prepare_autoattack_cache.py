"""Generate standard AutoAttack examples on an existing fixed CIFAR subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark_common import load_zeropur_resnet18, seed_everything


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AutoAttack需要GPU运行时")
    seed_everything(args.seed)

    source = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    stop = len(source["labels"]) if args.limit is None else args.limit
    clean = source["clean"][:stop]
    labels = source["labels"][:stop]
    victim, checkpoint = load_zeropur_resnet18(
        args.zeropur_root, torch.device("cuda")
    )

    from autoattack import AutoAttack

    adversary = AutoAttack(
        victim,
        norm="Linf",
        eps=args.epsilon,
        version="standard",
        seed=args.seed,
    )
    adversarial = adversary.run_standard_evaluation(
        clean.cuda(), labels.cuda(), bs=args.batch_size
    ).cpu()

    result = {
        "clean": clean,
        "adversarial": adversarial,
        "labels": labels,
        "indices": source.get("indices"),
        "attack": {
            "name": "AutoAttack standard",
            "mode": "preprocessor-blind attack against victim only",
            "norm": "Linf",
            "epsilon": args.epsilon,
            "seed": args.seed,
            "victim_checkpoint": str(checkpoint),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print("样本数:", len(labels))
    print("平均Linf:", float((adversarial - clean).abs().flatten(1).amax(1).mean()))
    print("结果:", args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeropur-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    return parser.parse_args()


if __name__ == "__main__":
    main()

