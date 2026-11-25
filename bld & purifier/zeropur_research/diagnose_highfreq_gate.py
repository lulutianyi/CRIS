"""Training-free high-frequency residual gate diagnosis on cached CIFAR-10 pairs."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur, pil_to_tensor, to_pil_image
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from run_biguided_cifar10 import load_models, vars_for_json


def blur(x, sigma):
    size = 2 * round(3 * sigma) + 1
    return gaussian_blur(x, [size, size], [sigma, sigma])


def median_filter(x):
    patches = F.pad(x, (1, 1, 1, 1), mode="reflect").unfold(2, 3, 1).unfold(3, 3, 1)
    return patches.contiguous().view(*x.shape, 9).median(-1).values


def jpeg90(x):
    outputs = []
    for image in x.detach().cpu():
        buffer = io.BytesIO()
        to_pil_image(image).save(buffer, format="JPEG", quality=90, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            outputs.append(pil_to_tensor(decoded.convert("RGB")).float() / 255)
    return torch.stack(outputs).to(x.device)


def sobel(x):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    channels = x.shape[1]
    gx = F.conv2d(x, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    gy = F.conv2d(x, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return gx, gy


def make_probes(x, kind, generator, noise_scale):
    if kind == "scale":
        return [blur(x, sigma) for sigma in (0.4, 0.7, 1.0)]
    if kind == "operator":
        return [blur(x, 0.5), median_filter(x), jpeg90(x)]
    if kind == "noise":
        gaussian = torch.randn(x.shape, device=x.device, generator=generator) * noise_scale
        uniform = (torch.rand(x.shape, device=x.device, generator=generator) * 2 - 1) * noise_scale
        rademacher = (
            torch.randint(0, 2, x.shape, device=x.device, generator=generator).float() * 2 - 1
        ) * noise_scale
        return [blur((x + noise).clamp(0, 1), 0.5) for noise in (gaussian, uniform, rademacher)]
    raise ValueError(kind)


def stable_components(x, kind, generator, noise_scale):
    probes = make_probes(x, kind, generator, noise_scale)
    residuals = torch.stack([probe - blur(probe, 1.0) for probe in probes])
    stable = residuals.median(0).values
    uncertainty = (residuals - stable.unsqueeze(0)).abs().median(0).values
    numerator = uncertainty.mean(1, keepdim=True)
    denominator = stable.abs().mean(1, keepdim=True) + 1 / 255
    relative_uncertainty = numerator / denominator

    gradients = [sobel(probe) for probe in probes]
    gx_stack = torch.stack([item[0] for item in gradients])
    gy_stack = torch.stack([item[1] for item in gradients])
    median_gx = gx_stack.median(0).values
    median_gy = gy_stack.median(0).values
    agreements = []
    for gx, gy in gradients:
        dot = gx * median_gx + gy * median_gy
        norm = torch.sqrt(gx.square() + gy.square() + 1e-12)
        median_norm = torch.sqrt(median_gx.square() + median_gy.square() + 1e-12)
        agreements.append(dot / (norm * median_norm + 1e-12))
    edge_agreement = torch.stack(agreements).mean(0).mean(1, keepdim=True).clamp(0, 1)
    return blur(x, 1.0), stable, relative_uncertainty, edge_agreement


def reconstruct(components, temperature, use_edge):
    low, stable, uncertainty, edge_agreement = components
    gate = torch.exp(-uncertainty / temperature)
    if use_edge:
        gate = gate * edge_agreement
    output = (low + gate * stable).clamp(0, 1)
    return output, gate


def ssim_per_image(x, target):
    mu_x = blur(x, 1.5)
    mu_y = blur(target, 1.5)
    sigma_x = blur(x.square(), 1.5) - mu_x.square()
    sigma_y = blur(target.square(), 1.5) - mu_y.square()
    sigma_xy = blur(x * target, 1.5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)


def metrics_per_image(output, target):
    mse = (output - target).square().flatten(1).mean(1).clamp_min(1e-12)
    high_output = output - blur(output, 1.0)
    high_target = target - blur(target, 1.0)
    high_mse = (high_output - high_target).square().flatten(1).mean(1).clamp_min(1e-12)
    gx, gy = sobel(output)
    tx, ty = sobel(target)
    dot = (gx * tx + gy * ty).flatten(1).sum(1)
    norm = torch.sqrt((gx.square() + gy.square()).flatten(1).sum(1))
    target_norm = torch.sqrt((tx.square() + ty.square()).flatten(1).sum(1))
    return {
        "psnr": -10 * torch.log10(mse),
        "ssim": ssim_per_image(output, target),
        "mae": (output - target).abs().flatten(1).mean(1),
        "highpass_psnr": -10 * torch.log10(high_mse),
        "sobel_cosine": dot / (norm * target_norm + 1e-12),
    }


class Accumulator:
    def __init__(self):
        self.count = 0
        self.correct = 0
        self.gate_sum = 0.0
        self.gate_count = 0
        self.metric_sums = {name: 0.0 for name in ("psnr", "ssim", "mae", "highpass_psnr", "sobel_cosine")}

    def update(self, output, target, labels, model, gate=None):
        with torch.inference_mode():
            self.correct += int((model(output).argmax(1) == labels).sum())
        values = metrics_per_image(output, target)
        for name, tensor in values.items():
            self.metric_sums[name] += float(tensor.sum())
        self.count += len(output)
        if gate is not None:
            self.gate_sum += float(gate.mean(dim=(1, 2, 3)).sum())
            self.gate_count += len(gate)

    def summary(self):
        result = {
            "accuracy": self.correct / self.count,
            **{name: total / self.count for name, total in self.metric_sums.items()},
        }
        result["mean_gate"] = self.gate_sum / self.gate_count if self.gate_count else None
        return result


def config_name(kind, temperature, edge):
    return f"{kind}_t{temperature:g}_edge{int(edge)}"


def run(args, model, official, clean, adversarial, labels, offset, device):
    if args.stage == "dev":
        configurations = [
            (kind, temperature, edge)
            for kind in ("scale", "operator", "noise")
            for temperature in (0.25, 0.5, 1.0)
            for edge in (False, True)
        ]
    else:
        configurations = [(args.probe, args.temperature, args.edge_gate)]

    accumulators = {}
    for name in ("raw", "anchor_blur_sigma1", "official_blur_sigma1_8", "official_coarse"):
        accumulators[name] = {"adv": Accumulator(), "clean": Accumulator()}
    for kind, temperature, edge in configurations:
        accumulators[config_name(kind, temperature, edge)] = {
            "adv": Accumulator(),
            "clean": Accumulator(),
        }

    preview = {kind: [] for kind in ("scale", "operator", "noise")}
    for start in tqdm(range(0, len(labels), args.batch_size), desc=f"HF gate {args.stage}"):
        end = start + args.batch_size
        x_clean = clean[start:end].to(device)
        x_adv = adversarial[start:end].to(device)
        y = labels[start:end].to(device)
        blurred_clean = blur(x_clean, 1.0)
        blurred_adv = blur(x_adv, 1.0)
        official_blur_adv = official.transform(x_adv)
        official_blur_clean = official.transform(x_clean)
        official_adv = official.coarse_shifting(
            x_adv, eps=10 / 255, num_steps=10, arch="r18"
        ).detach()
        official_clean = official.coarse_shifting(
            x_clean, eps=10 / 255, num_steps=10, arch="r18"
        ).detach()

        for condition, source, reference, blurred, official_blur, coarse in (
            ("adv", x_adv, x_clean, blurred_adv, official_blur_adv, official_adv),
            ("clean", x_clean, x_clean, blurred_clean, official_blur_clean, official_clean),
        ):
            accumulators["raw"][condition].update(source, reference, y, model)
            accumulators["anchor_blur_sigma1"][condition].update(blurred, reference, y, model)
            accumulators["official_blur_sigma1_8"][condition].update(official_blur, reference, y, model)
            accumulators["official_coarse"][condition].update(coarse, reference, y, model)

        components = {}
        for kind in {item[0] for item in configurations}:
            seed = args.seed + offset + start + {"scale": 0, "operator": 100000, "noise": 200000}[kind]
            adv_generator = torch.Generator(device=device).manual_seed(seed)
            clean_generator = torch.Generator(device=device).manual_seed(seed)
            components[(kind, "adv")] = stable_components(
                x_adv, kind, adv_generator, args.noise_scale
            )
            components[(kind, "clean")] = stable_components(
                x_clean, kind, clean_generator, args.noise_scale
            )

        for kind, temperature, edge in configurations:
            name = config_name(kind, temperature, edge)
            out_adv, gate_adv = reconstruct(components[(kind, "adv")], temperature, edge)
            out_clean, gate_clean = reconstruct(components[(kind, "clean")], temperature, edge)
            accumulators[name]["adv"].update(out_adv, x_clean, y, model, gate_adv)
            accumulators[name]["clean"].update(out_clean, x_clean, y, model, gate_clean)
            if temperature == 0.5 and edge and len(preview[kind]) < args.preview_count:
                take = min(args.preview_count - len(preview[kind]), len(x_clean))
                for index in range(take):
                    gate_image = gate_adv[index].repeat(3, 1, 1)
                    preview[kind].append(
                        torch.stack(
                            [x_clean[index], x_adv[index], blurred_adv[index], out_adv[index], gate_image]
                        ).cpu()
                    )

    results = {
        name: {condition: accumulator.summary() for condition, accumulator in pair.items()}
        for name, pair in accumulators.items()
    }
    for name, pair in results.items():
        pair["gate_clean_minus_adv"] = (
            None
            if pair["clean"]["mean_gate"] is None
            else pair["clean"]["mean_gate"] - pair["adv"]["mean_gate"]
        )

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for kind, rows in preview.items():
        if rows:
            panel = torch.cat(rows)
            save_image(make_grid(panel, nrow=5, padding=2), args.preview_dir / f"{args.stage}_{kind}_t0.5_edge1.png")
    return results


def recommend(results):
    blur_result = results["anchor_blur_sigma1"]
    eligible = []
    for name, result in results.items():
        if name in ("raw", "anchor_blur_sigma1", "official_blur_sigma1_8", "official_coarse"):
            continue
        if (
            result["adv"]["accuracy"] >= blur_result["adv"]["accuracy"]
            and result["clean"]["accuracy"] >= blur_result["clean"]["accuracy"]
            and result["adv"]["highpass_psnr"] > blur_result["adv"]["highpass_psnr"]
            and result["gate_clean_minus_adv"] > 0
        ):
            eligible.append(name)
    eligible.sort(
        key=lambda name: (
            results[name]["adv"]["accuracy"],
            results[name]["clean"]["accuracy"],
            results[name]["adv"]["highpass_psnr"],
        ),
        reverse=True,
    )
    return {"eligible": eligible, "recommended": eligible[0] if eligible else None}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("请使用GPU运行时")
    device = torch.device("cuda")
    model, official, _proposed, _helper, checkpoint = load_models(args, device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.stage == "dev":
        begin, end = 0, min(args.dev_size, len(cache["labels"]))
    else:
        begin, end = min(args.dev_size, len(cache["labels"])), len(cache["labels"])
    clean = cache["clean"][begin:end]
    adversarial = cache["adversarial"][begin:end]
    labels = cache["labels"][begin:end]
    results = run(args, model, official, clean, adversarial, labels, begin, device)
    summary = {
        "stage": args.stage,
        "range": [begin, end],
        "count": end - begin,
        "seed": cache["seed"],
        "attack": cache["attack"],
        "checkpoint": str(checkpoint),
        "results": results,
        "selection": recommend(results) if args.stage == "dev" else None,
        "settings": vars_for_json(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "stage": summary["stage"],
        "count": summary["count"],
        "selection": summary["selection"],
        "table": {
            name: {
                "adv_acc": round(result["adv"]["accuracy"], 4),
                "clean_acc": round(result["clean"]["accuracy"], 4),
                "adv_hp_psnr": round(result["adv"]["highpass_psnr"], 4),
                "gate_separation": None if result["gate_clean_minus_adv"] is None else round(result["gate_clean_minus_adv"], 6),
            }
            for name, result in results.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("完整结果：", args.output)
    print("预览目录：", args.preview_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--noise-scale", type=float, default=2 / 255)
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--probe", choices=("scale", "operator", "noise"), default="operator")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--edge-gate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
