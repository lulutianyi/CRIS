"""CIFAR-10 robustness diagnostics for Learned-v2 and HF-AE.

This script evaluates a fixed, stratified CIFAR-10 subset under:
  * clean inputs;
  * classifier-targeted non-adaptive PGD-20;
  * high-frequency noise;
  * low-frequency noise;
  * Gaussian blur;
  * JPEG compression.

It also reports per-class accuracy and clean-vs-PGD binary-code change rates
for main, skip64 and skip32. Run with ``%run -i`` after executing the model
definition cells in gcn.ipynb.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm.auto import tqdm


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NOTEBOOK_NAMES = (
    "bld",
    "BinaryAutoEncoderAttn",
    "BinaryAutoEncoderGCN",
    "GCNSkipAEConfig",
    "GCNBlockConfig",
    "auto_detect_skip_levels",
)


class CifarResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=10)
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()
        self.register_buffer(
            "mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_notebook_context():
    missing = [name for name in NOTEBOOK_NAMES if name not in globals()]
    if missing:
        raise RuntimeError(
            "缺少gcn.ipynb模型定义: " + ", ".join(missing)
            + "。请先运行模型定义单元，再用 %run -i 执行本脚本。"
        )


def unwrap_state_dict(raw):
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def load_classifier(path, device):
    raw = torch.load(path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model = CifarResNet18().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_notebook_gcn_ae(checkpoint):
    require_notebook_context()
    device = bld.DEVICE
    base = BinaryAutoEncoderAttn().to(device)
    levels = auto_detect_skip_levels(base, img_size=bld.IMG_SIZE)
    config = GCNSkipAEConfig(
        active_levels=None,
        block_cfg=GCNBlockConfig(
            zero_init=True, temperature=None, attn_dropout=0.0
        ),
        freeze_base=False,
        quantize_skip=True,
        skip_quantizer_mode="learned_v2",
        skip_level_code_shapes={"64x64": (2, 16), "32x32": (8, 8)},
    )
    model = BinaryAutoEncoderGCN(base, config, levels).to(device)
    raw = torch.load(checkpoint, map_location=device)
    model.load_state_dict(unwrap_state_dict(raw), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class IndexedCifar(Dataset):
    def __init__(self, root, indices, download):
        self.dataset = CIFAR10(
            root=root, train=False, download=download, transform=T.ToTensor()
        )
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def stratified_indices(root, size, seed, download):
    dataset = CIFAR10(root=root, train=False, download=download)
    size = min(size, len(dataset))
    rng = random.Random(seed)
    by_class = [[] for _ in range(10)]
    for index, target in enumerate(dataset.targets):
        by_class[int(target)].append(index)
    for indices in by_class:
        rng.shuffle(indices)
    base, remainder = divmod(size, 10)
    selected = []
    for class_id, indices in enumerate(by_class):
        selected.extend(indices[:base + int(class_id < remainder)])
    rng.shuffle(selected)
    return selected


def load_subset(args):
    indices = stratified_indices(
        args.data_root, args.subset_size, args.seed, args.download
    )
    loader = DataLoader(
        IndexedCifar(args.data_root, indices, args.download),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images), torch.cat(labels), indices


def average_blur(x, kernel_size):
    padding = kernel_size // 2
    return F.avg_pool2d(
        F.pad(x, (padding,) * 4, mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )


def scale_noise_linf(noise, epsilon):
    maximum = noise.abs().flatten(1).amax(1).view(-1, 1, 1, 1)
    return noise / maximum.clamp_min(1e-12) * epsilon


def jpeg_batch(images, quality):
    outputs = []
    for image in tqdm(images, desc=f"JPEG quality={quality}"):
        buffer = io.BytesIO()
        TF.to_pil_image(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            outputs.append(TF.to_tensor(decoded.convert("RGB")))
    return torch.stack(outputs)


def pgd_classifier_attack(classifier, images, labels, args, device):
    adversarial_batches = []
    for start in tqdm(
        range(0, len(images), args.attack_batch_size),
        desc=f"non-adaptive PGD-{args.pgd_steps}",
    ):
        stop = min(start + args.attack_batch_size, len(images))
        clean = images[start:stop].to(device)
        target = labels[start:stop].to(device)
        adversarial = clean + torch.empty_like(clean).uniform_(
            -args.pgd_epsilon, args.pgd_epsilon
        )
        adversarial = adversarial.clamp(0, 1)
        for _ in range(args.pgd_steps):
            adversarial.requires_grad_(True)
            loss = F.cross_entropy(classifier(adversarial), target)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + args.pgd_alpha * gradient.sign()
            delta = (adversarial - clean).clamp(
                -args.pgd_epsilon, args.pgd_epsilon
            )
            adversarial = (clean + delta).clamp(0, 1).detach()
        adversarial_batches.append(adversarial.cpu())
    return torch.cat(adversarial_batches)


def create_conditions(clean, labels, classifier, args, device):
    generator = torch.Generator().manual_seed(args.seed + 101)
    raw_noise = torch.randn(clean.shape, generator=generator)
    high_noise = raw_noise - average_blur(raw_noise, args.noise_kernel)
    low_noise = average_blur(raw_noise, args.noise_kernel)
    conditions = {
        "clean": clean,
        f"pgd{args.pgd_steps}_nonadaptive": pgd_classifier_attack(
            classifier, clean, labels, args, device
        ),
        "highfreq_noise": (
            clean + scale_noise_linf(high_noise, args.noise_epsilon)
        ).clamp(0, 1),
        "lowfreq_noise": (
            clean + scale_noise_linf(low_noise, args.noise_epsilon)
        ).clamp(0, 1),
        "gaussian_blur": TF.gaussian_blur(
            clean,
            kernel_size=[args.blur_kernel, args.blur_kernel],
            sigma=[args.blur_sigma, args.blur_sigma],
        ),
        "jpeg": jpeg_batch(clean, args.jpeg_quality),
    }
    return conditions


def perturbation_summary(images, clean):
    difference = images - clean
    mse = difference.square().mean((1, 2, 3))
    return {
        "mean_linf": difference.abs().flatten(1).amax(1).mean().item(),
        "mean_l2": difference.flatten(1).norm(p=2, dim=1).mean().item(),
        "mae": difference.abs().mean().item(),
        "psnr": (-10 * torch.log10(mse + 1e-12)).mean().item(),
    }


class AccuracyAccumulator:
    def __init__(self):
        self.count = 0
        self.correct = 0
        self.class_count = torch.zeros(10, dtype=torch.long)
        self.class_correct = torch.zeros(10, dtype=torch.long)

    def update(self, logits, labels):
        predictions = logits.argmax(1).cpu()
        labels = labels.cpu()
        self.count += labels.numel()
        self.correct += (predictions == labels).sum().item()
        self.class_count += torch.bincount(labels, minlength=10)
        correct_labels = labels[predictions == labels]
        self.class_correct += torch.bincount(correct_labels, minlength=10)

    def summary(self):
        return {
            "count": self.count,
            "accuracy": self.correct / self.count,
            "per_class_accuracy": {
                name: (self.class_correct[index] / self.class_count[index]).item()
                for index, name in enumerate(CIFAR_CLASSES)
            },
        }


def transform_for_method(images, method, ae):
    if method == "direct":
        return images
    images128 = F.interpolate(
        images, (128, 128), mode="bilinear", align_corners=False
    )
    if ae is not None:
        images128, _ = ae(images128)
        images128 = images128.clamp(0, 1)
    return F.interpolate(
        images128, (32, 32), mode="bilinear", align_corners=False
    )


@torch.inference_mode()
def evaluate_method(method, conditions, labels, classifier, args, device, ae=None):
    method_results = {}
    for condition_name, condition_images in conditions.items():
        accumulator = AccuracyAccumulator()
        for start in tqdm(
            range(0, len(labels), args.batch_size),
            desc=f"{method}/{condition_name}",
        ):
            stop = min(start + args.batch_size, len(labels))
            images = condition_images[start:stop].to(device)
            targets = labels[start:stop].to(device)
            processed = transform_for_method(images, method, ae)
            accumulator.update(classifier(processed), targets)
        method_results[condition_name] = accumulator.summary()
    return method_results


class BitStabilityAccumulator:
    COMPONENTS = ("main", "skip64", "skip32")

    def __init__(self):
        self.flips = {name: 0 for name in self.COMPONENTS}
        self.bits = {name: 0 for name in self.COMPONENTS}
        self.class_flips = {
            name: torch.zeros(10, dtype=torch.float64)
            for name in self.COMPONENTS
        }
        self.class_bits = {
            name: torch.zeros(10, dtype=torch.float64)
            for name in self.COMPONENTS
        }
        self.outcome_flips = {
            "defended_correct": 0,
            "defended_incorrect": 0,
        }
        self.outcome_bits = {
            "defended_correct": 0,
            "defended_incorrect": 0,
        }

    def update_component(self, name, clean_code, altered_code, labels):
        clean_bits = clean_code.detach().gt(0.5).flatten(1).cpu()
        altered_bits = altered_code.detach().gt(0.5).flatten(1).cpu()
        per_sample_flips = clean_bits.ne(altered_bits).sum(1)
        bits_per_sample = clean_bits.shape[1]
        self.flips[name] += int(per_sample_flips.sum())
        self.bits[name] += int(clean_bits.numel())
        labels = labels.cpu()
        for class_id in range(10):
            mask = labels == class_id
            self.class_flips[name][class_id] += per_sample_flips[mask].sum()
            self.class_bits[name][class_id] += int(mask.sum()) * bits_per_sample
        return per_sample_flips, bits_per_sample

    def update(self, clean_codes, altered_codes, labels, defended_correct):
        clean_main, clean_skip = clean_codes
        altered_main, altered_skip = altered_codes
        sample_total_flips = torch.zeros(len(labels), dtype=torch.long)
        total_bits_per_sample = 0
        for name, clean_code, altered_code in (
            ("main", clean_main, altered_main),
            ("skip64", clean_skip["64x64"], altered_skip["64x64"]),
            ("skip32", clean_skip["32x32"], altered_skip["32x32"]),
        ):
            component_flips, component_bits = self.update_component(
                name, clean_code, altered_code, labels
            )
            sample_total_flips += component_flips
            total_bits_per_sample += component_bits
        defended_correct = defended_correct.cpu()
        for outcome, mask in (
            ("defended_correct", defended_correct),
            ("defended_incorrect", ~defended_correct),
        ):
            self.outcome_flips[outcome] += int(sample_total_flips[mask].sum())
            self.outcome_bits[outcome] += int(mask.sum()) * total_bits_per_sample

    def summary(self):
        overall = {
            name + "_flip_rate": self.flips[name] / max(self.bits[name], 1)
            for name in self.COMPONENTS
        }
        total_flips = sum(self.flips.values())
        total_bits = sum(self.bits.values())
        overall["total_flip_rate"] = total_flips / max(total_bits, 1)
        per_class = {}
        for class_id, class_name in enumerate(CIFAR_CLASSES):
            class_result = {}
            class_total_flips = 0.0
            class_total_bits = 0.0
            for name in self.COMPONENTS:
                flips = float(self.class_flips[name][class_id])
                bits = float(self.class_bits[name][class_id])
                class_result[name + "_flip_rate"] = flips / max(bits, 1.0)
                class_total_flips += flips
                class_total_bits += bits
            class_result["total_flip_rate"] = (
                class_total_flips / max(class_total_bits, 1.0)
            )
            per_class[class_name] = class_result
        by_outcome = {
            name: self.outcome_flips[name] / max(self.outcome_bits[name], 1)
            for name in self.outcome_flips
        }
        return {
            "overall": overall,
            "per_class": per_class,
            "total_flip_rate_by_defended_outcome": by_outcome,
        }


@torch.inference_mode()
def evaluate_bit_stability(ae, clean, adversarial, labels, classifier, args, device):
    accumulator = BitStabilityAccumulator()
    for start in tqdm(
        range(0, len(labels), args.batch_size), desc="binary-code stability"
    ):
        stop = min(start + args.batch_size, len(labels))
        clean_batch = clean[start:stop].to(device)
        adversarial_batch = adversarial[start:stop].to(device)
        target = labels[start:stop].to(device)
        clean128 = F.interpolate(
            clean_batch, (128, 128), mode="bilinear", align_corners=False
        )
        adversarial128 = F.interpolate(
            adversarial_batch, (128, 128), mode="bilinear", align_corners=False
        )
        clean_codes = ae.encode_all_codes(clean128)
        adversarial_codes = ae.encode_all_codes(adversarial128)
        reconstructed128, _ = ae(adversarial128)
        reconstructed32 = F.interpolate(
            reconstructed128.clamp(0, 1),
            (32, 32),
            mode="bilinear",
            align_corners=False,
        )
        defended_correct = classifier(reconstructed32).argmax(1).eq(target)
        accumulator.update(
            clean_codes,
            adversarial_codes,
            target,
            defended_correct,
        )
    return accumulator.summary()


def add_comparative_metrics(results):
    clean_reference = results["direct"]["clean"]["accuracy"]
    for method_results in results.values():
        for condition_name, value in method_results.items():
            direct_accuracy = results["direct"][condition_name]["accuracy"]
            value["gain_vs_direct"] = value["accuracy"] - direct_accuracy
            value["retention_vs_clean_original"] = (
                value["accuracy"] / max(clean_reference, 1e-12)
            )
    pgd_name = next(name for name in results["direct"] if name.startswith("pgd"))
    attacked_accuracy = results["direct"][pgd_name]["accuracy"]
    lost_accuracy = max(clean_reference - attacked_accuracy, 1e-12)
    for method_results in results.values():
        method_results[pgd_name]["recovery_of_lost_accuracy"] = (
            method_results[pgd_name]["accuracy"] - attacked_accuracy
        ) / lost_accuracy


def write_csv_files(run_dir, results, bit_stability):
    with (run_dir / "aggregate_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "condition", "accuracy", "gain_vs_direct",
            "retention_vs_clean_original", "recovery_of_lost_accuracy",
        ])
        for method, method_results in results.items():
            for condition, value in method_results.items():
                writer.writerow([
                    method,
                    condition,
                    value["accuracy"],
                    value["gain_vs_direct"],
                    value["retention_vs_clean_original"],
                    value.get("recovery_of_lost_accuracy", ""),
                ])

    with (run_dir / "per_class_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "condition", "class", "accuracy"])
        for method, method_results in results.items():
            for condition, value in method_results.items():
                for class_name, accuracy in value["per_class_accuracy"].items():
                    writer.writerow([method, condition, class_name, accuracy])

    with (run_dir / "bit_stability_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "class", "main_flip_rate", "skip64_flip_rate",
            "skip32_flip_rate", "total_flip_rate",
        ])
        for model_name, model_result in bit_stability.items():
            for class_name, value in model_result["per_class"].items():
                writer.writerow([
                    model_name,
                    class_name,
                    value["main_flip_rate"],
                    value["skip64_flip_rate"],
                    value["skip32_flip_rate"],
                    value["total_flip_rate"],
                ])


def print_summary(results, bit_stability):
    print("\n========== CIFAR-10鲁棒性准确率 ==========")
    condition_names = list(results["direct"].keys())
    for condition in condition_names:
        print(f"\n[{condition}]")
        for method in ("direct", "resize_only_128", "learned_v2", "hf_ae"):
            value = results[method][condition]
            print(
                f"{method:16s} accuracy={value['accuracy']:.4f} "
                f"gain_vs_direct={value['gain_vs_direct']:+.4f}"
            )
    print("\n========== clean-vs-PGD二进制编码稳定性 ==========")
    for model_name, value in bit_stability.items():
        overall = value["overall"]
        print(
            f"{model_name:12s} main={overall['main_flip_rate']:.6f} "
            f"skip64={overall['skip64_flip_rate']:.6f} "
            f"skip32={overall['skip32_flip_rate']:.6f} "
            f"total={overall['total_flip_rate']:.6f}"
        )


def main():
    args = parse_args()
    seed_everything(args.seed)
    require_notebook_context()
    device = bld.DEVICE
    run_dir = (
        Path(args.output_dir)
        / f"cifar10_robustness_n{args.subset_size}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    clean, labels, indices = load_subset(args)
    classifier = load_classifier(args.classifier_checkpoint, device)
    conditions = create_conditions(clean, labels, classifier, args, device)
    perturbations = {
        name: perturbation_summary(images, clean)
        for name, images in conditions.items()
    }

    results = {
        "direct": evaluate_method(
            "direct", conditions, labels, classifier, args, device
        ),
        "resize_only_128": evaluate_method(
            "resize_only_128", conditions, labels, classifier, args, device
        ),
    }
    bit_stability = {}
    pgd_name = f"pgd{args.pgd_steps}_nonadaptive"
    for model_name, checkpoint in (
        ("learned_v2", args.learned_checkpoint),
        ("hf_ae", args.hf_checkpoint),
    ):
        ae = build_notebook_gcn_ae(checkpoint)
        results[model_name] = evaluate_method(
            model_name, conditions, labels, classifier, args, device, ae=ae
        )
        bit_stability[model_name] = evaluate_bit_stability(
            ae,
            clean,
            conditions[pgd_name],
            labels,
            classifier,
            args,
            device,
        )
        ae.to("cpu")
        del ae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    add_comparative_metrics(results)
    summary = {
        "dataset": "cifar10",
        "subset_size": len(indices),
        "seed": args.seed,
        "attack_scope": (
            "non-adaptive: PGD attacks the native CIFAR classifier only; "
            "all defenses receive the same adversarial images"
        ),
        "settings": {
            "pgd_epsilon": args.pgd_epsilon,
            "pgd_alpha": args.pgd_alpha,
            "pgd_steps": args.pgd_steps,
            "noise_epsilon": args.noise_epsilon,
            "noise_kernel": args.noise_kernel,
            "blur_kernel": args.blur_kernel,
            "blur_sigma": args.blur_sigma,
            "jpeg_quality": args.jpeg_quality,
        },
        "classifier_checkpoint": args.classifier_checkpoint,
        "learned_checkpoint": args.learned_checkpoint,
        "hf_checkpoint": args.hf_checkpoint,
        "perturbation_metrics": perturbations,
        "results": results,
        "bit_stability_clean_vs_pgd": bit_stability,
    }
    output = run_dir / "summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv_files(run_dir, results, bit_stability)
    print_summary(results, bit_stability)
    print(f"\n结果目录: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content/cifar10")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--learned-checkpoint", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument(
        "--output-dir", default="/content/drive/MyDrive/cifar10_robustness_eval"
    )
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--attack-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--noise-epsilon", type=float, default=16 / 255)
    parser.add_argument("--noise-kernel", type=int, default=5)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    main()
