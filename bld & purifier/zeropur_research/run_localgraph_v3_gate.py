"""Prediction-consistency gates for LocalGraph-V2 on cached CIFAR-10 attacks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from run_biguided_cifar10 import load_models, vars_for_json


RULES = (
    "soft_ratio",
    "hard_any",
    "hard_majority",
    "hard_global_or_local_majority",
)


def predict(model, images):
    with torch.inference_mode():
        return model(images).argmax(1)


def make_gates(proposed, model, images, generator, args):
    guides = proposed._local_guides_v2(
        images,
        args.local_guides,
        args.patch_grid,
        args.patch_ratio,
        args.graph_steps,
        generator,
    )
    input_prediction = predict(model, images)
    guide_predictions = torch.stack([predict(model, guide) for guide in guides], dim=1)
    disagreement = (guide_predictions != input_prediction[:, None]).float()
    global_change = disagreement[:, 0]
    local_changes = disagreement[:, 1:].sum(1)
    total_changes = disagreement.sum(1)
    total_guides = disagreement.shape[1]
    gates = {
        "soft_ratio": total_changes / total_guides,
        "hard_any": (total_changes >= 1).float(),
        "hard_majority": (total_changes >= math.ceil(total_guides / 2)).float(),
        "hard_global_or_local_majority": (
            (global_change > 0)
            | (local_changes >= math.ceil(args.local_guides / 2))
        ).float(),
    }
    return {name: gate.view(-1, 1, 1, 1) for name, gate in gates.items()}


def mcnemar(reference, candidate):
    reference = torch.tensor(reference, dtype=torch.bool)
    candidate = torch.tensor(candidate, dtype=torch.bool)
    gained = int((~reference & candidate).sum())
    lost = int((reference & ~candidate).sum())
    discordant = gained + lost
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "reference_wrong_candidate_right": gained,
        "reference_right_candidate_wrong": lost,
        "net_correct_samples": gained - lost,
        "exact_two_sided_p": p_value,
    }


def add_correct(storage, name, output, labels, model):
    storage[name].extend((predict(model, output) == labels).cpu().tolist())


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("请使用GPU运行时")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, official, proposed, _helper, checkpoint = load_models(args, device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    if args.stage == "dev":
        begin, end = 0, min(args.dev_size, len(cache["labels"]))
        active_rules = RULES
    else:
        begin, end = min(args.dev_size, len(cache["labels"])), len(cache["labels"])
        active_rules = (args.gate_rule,)

    clean = cache["clean"][begin:end]
    adversarial = cache["adversarial"][begin:end]
    labels = cache["labels"][begin:end]
    base_names = (
        "clean_input",
        "adversarial_input",
        "official_blur_adv",
        "official_blur_clean",
        "official_coarse_adv",
        "official_coarse_clean",
        "local_v2_adv",
        "local_v2_clean",
    )
    per_sample = {name: [] for name in base_names}
    for rule in active_rules:
        per_sample[f"{rule}_adv"] = []
        per_sample[f"{rule}_clean"] = []
    gate_values = {
        rule: {"adv": [], "clean": []} for rule in active_rules
    }

    for local_start in tqdm(
        range(0, len(labels), args.batch_size), desc=f"LocalGraph-V3 gate {args.stage}"
    ):
        local_end = local_start + args.batch_size
        absolute_start = begin + local_start
        x_clean = clean[local_start:local_end].to(device)
        x_adv = adversarial[local_start:local_end].to(device)
        y = labels[local_start:local_end].to(device)
        paired_seed = args.seed + absolute_start

        official_blur_adv = official.transform(x_adv)
        official_blur_clean = official.transform(x_clean)
        official_adv = official.coarse_shifting(
            x_adv, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()
        official_clean = official.coarse_shifting(
            x_clean, eps=args.epsilon, num_steps=args.steps, arch="r18"
        ).detach()

        gate_adv_generator = torch.Generator(device=device).manual_seed(paired_seed)
        gate_clean_generator = torch.Generator(device=device).manual_seed(paired_seed)
        gates_adv = make_gates(
            proposed, model, x_adv, gate_adv_generator, args
        )
        gates_clean = make_gates(
            proposed, model, x_clean, gate_clean_generator, args
        )

        local_adv_generator = torch.Generator(device=device).manual_seed(paired_seed)
        local_clean_generator = torch.Generator(device=device).manual_seed(paired_seed)
        local_adv = proposed.local_graph_v2(
            x_adv,
            local_adv_generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )
        local_clean = proposed.local_graph_v2(
            x_clean,
            local_clean_generator,
            eps=args.epsilon,
            steps=args.steps,
            guides=args.local_guides,
            grid=args.patch_grid,
            patch_ratio=args.patch_ratio,
            graph_steps=args.graph_steps,
        )

        base_outputs = {
            "clean_input": x_clean,
            "adversarial_input": x_adv,
            "official_blur_adv": official_blur_adv,
            "official_blur_clean": official_blur_clean,
            "official_coarse_adv": official_adv,
            "official_coarse_clean": official_clean,
            "local_v2_adv": local_adv,
            "local_v2_clean": local_clean,
        }
        for name, output in base_outputs.items():
            add_correct(per_sample, name, output, y, model)

        for rule in active_rules:
            gate_adv = gates_adv[rule]
            gate_clean = gates_clean[rule]
            gated_adv = x_adv + gate_adv * (local_adv - x_adv)
            gated_clean = x_clean + gate_clean * (local_clean - x_clean)
            add_correct(per_sample, f"{rule}_adv", gated_adv, y, model)
            add_correct(per_sample, f"{rule}_clean", gated_clean, y, model)
            gate_values[rule]["adv"].extend(gate_adv.flatten().cpu().tolist())
            gate_values[rule]["clean"].extend(gate_clean.flatten().cpu().tolist())

    count = len(labels)
    accuracy = {name: sum(values) / count for name, values in per_sample.items()}
    rule_results = {}
    for rule in active_rules:
        adv_name = f"{rule}_adv"
        clean_name = f"{rule}_clean"
        mean_adv_gate = sum(gate_values[rule]["adv"]) / count
        mean_clean_gate = sum(gate_values[rule]["clean"]) / count
        rule_results[rule] = {
            "adversarial_accuracy": accuracy[adv_name],
            "clean_accuracy": accuracy[clean_name],
            "mean_gate_adversarial": mean_adv_gate,
            "mean_gate_clean": mean_clean_gate,
            "gate_separation_adv_minus_clean": mean_adv_gate - mean_clean_gate,
            "gain_pp_vs_official": {
                "adversarial": 100 * (accuracy[adv_name] - accuracy["official_coarse_adv"]),
                "clean": 100 * (accuracy[clean_name] - accuracy["official_coarse_clean"]),
            },
            "paired_vs_official": {
                "adversarial": mcnemar(
                    per_sample["official_coarse_adv"], per_sample[adv_name]
                ),
                "clean": mcnemar(
                    per_sample["official_coarse_clean"], per_sample[clean_name]
                ),
            },
        }

    eligible = [
        rule
        for rule in active_rules
        if rule_results[rule]["adversarial_accuracy"] >= accuracy["official_coarse_adv"]
        and rule_results[rule]["clean_accuracy"] >= accuracy["official_coarse_clean"]
        and rule_results[rule]["gate_separation_adv_minus_clean"] > 0
    ]
    eligible.sort(
        key=lambda rule: (
            rule_results[rule]["adversarial_accuracy"],
            rule_results[rule]["clean_accuracy"],
        ),
        reverse=True,
    )
    summary = {
        "stage": args.stage,
        "range": [begin, end],
        "count": count,
        "seed": cache["seed"],
        "attack": cache["attack"],
        "checkpoint": str(checkpoint),
        "baseline_accuracy": {
            name: value for name, value in accuracy.items() if name in base_names
        },
        "gate_rules": rule_results,
        "selection": {
            "eligible": eligible,
            "recommended": eligible[0] if eligible else None,
        },
        "settings": vars_for_json(args),
        "indices": cache["indices"][begin:end],
        "per_sample_correct": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {key: value for key, value in summary.items() if key not in ("indices", "per_sample_correct")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("结果：", args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--epsilon", type=float, default=10 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-guides", type=int, default=3)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--patch-ratio", type=float, default=0.25)
    parser.add_argument("--graph-steps", type=int, default=2)
    parser.add_argument("--gate-rule", choices=RULES, default="hard_global_or_local_majority")
    return parser.parse_args()


if __name__ == "__main__":
    main()
