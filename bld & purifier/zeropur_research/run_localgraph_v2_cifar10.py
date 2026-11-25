"""Paired evaluation of detail-preserving LocalGraph-V2 on cached attacks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from run_biguided_cifar10 import load_models, vars_for_json


def predict_correct(model, images, labels):
    with torch.inference_mode():
        return (model(images).argmax(1) == labels).cpu()


def mcnemar(reference, candidate):
    reference = torch.as_tensor(reference, dtype=torch.bool)
    candidate = torch.as_tensor(candidate, dtype=torch.bool)
    gained = int((~reference & candidate).sum())
    lost = int((reference & ~candidate).sum())
    discordant = gained + lost
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(gained, lost) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "reference_wrong_candidate_right": gained,
        "reference_right_candidate_wrong": lost,
        "net_correct_samples": gained - lost,
        "exact_two_sided_p": p_value,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("请使用GPU运行时")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, official, proposed, _helper, checkpoint = load_models(args, device)
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
        "local_graph_v2_adv",
        "local_graph_v2_clean",
    )
    per_sample = {name: [] for name in names}
    started = time.time()

    for start in tqdm(
        range(0, len(labels), args.batch_size), desc="LocalGraph-V2 paired eval"
    ):
        end = start + args.batch_size
        x_clean = clean[start:end].to(device)
        x_adv = adversarial[start:end].to(device)
        y = labels[start:end].to(device)

        blur_adv = official.transform(x_adv)
        blur_clean = official.transform(x_clean)
        official_adv = official.coarse_shifting(
            x_adv, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()
        official_clean = official.coarse_shifting(
            x_clean, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()

        paired_seed = args.seed + start
        adv_generator = torch.Generator(device=device).manual_seed(paired_seed)
        clean_generator = torch.Generator(device=device).manual_seed(paired_seed)
        local_adv = proposed.local_graph_v2(
            x_adv,
            adv_generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )
        local_clean = proposed.local_graph_v2(
            x_clean,
            clean_generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )

        values = {
            "clean_input": x_clean,
            "adversarial_input": x_adv,
            "blur_only_adv": blur_adv,
            "blur_only_clean": blur_clean,
            "official_coarse_adv": official_adv,
            "official_coarse_clean": official_clean,
            "local_graph_v2_adv": local_adv,
            "local_graph_v2_clean": local_clean,
        }
        for name, images in values.items():
            per_sample[name].extend(predict_correct(model, images, y).tolist())

    count = len(labels)
    accuracy = {
        name: sum(values) / count for name, values in per_sample.items()
    }
    clean_base = accuracy["clean_input"]
    summary = {
        "dataset": "cifar10_test",
        "subset_size": count,
        "seed": cache["seed"],
        "attack": cache["attack"],
        "checkpoint": str(checkpoint),
        "accuracy": accuracy,
        "derived": {
            "official_coarse": {
                "robust_retention": accuracy["official_coarse_adv"] / clean_base,
                "clean_retention": accuracy["official_coarse_clean"] / clean_base,
            },
            "local_graph_v2": {
                "robust_retention": accuracy["local_graph_v2_adv"] / clean_base,
                "clean_retention": accuracy["local_graph_v2_clean"] / clean_base,
                "robust_gain_pp_vs_official": 100
                * (accuracy["local_graph_v2_adv"] - accuracy["official_coarse_adv"]),
                "clean_gain_pp_vs_official": 100
                * (accuracy["local_graph_v2_clean"] - accuracy["official_coarse_clean"]),
            },
        },
        "paired_tests": {
            "adversarial_local_v2_vs_official": mcnemar(
                per_sample["official_coarse_adv"],
                per_sample["local_graph_v2_adv"],
            ),
            "clean_local_v2_vs_official": mcnemar(
                per_sample["official_coarse_clean"],
                per_sample["local_graph_v2_clean"],
            ),
        },
        "settings": vars_for_json(args),
        "elapsed_seconds": time.time() - started,
        "indices": cache["indices"],
        "per_sample_correct": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("indices", "per_sample_correct")}, ensure_ascii=False, indent=2))
    print("结果：", args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=10 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-guides", type=int, default=3)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--patch-ratio", type=float, default=0.25)
    parser.add_argument("--graph-steps", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
