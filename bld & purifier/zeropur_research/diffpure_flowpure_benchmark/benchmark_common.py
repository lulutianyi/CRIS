"""Shared ZeroPur-victim and BPDA utilities for purification benchmarks."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class Normalize(torch.nn.Module):
    def __init__(self, mean=MEAN, std=STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_zeropur_resnet18(official_root, device):
    """Load ZeroPur's PreActResNet18 without colliding with FlowPure.models."""
    zero_pur = Path(official_root) / "zero_pur"
    source = zero_pur / "models" / "preact_resnet.py"
    checkpoint_path = (
        zero_pur / "checkpoint" / "resnet18_cifar10_all" / "ckpt.pth"
    )

    spec = importlib.util.spec_from_file_location("zeropur_preact_resnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classifier = module.PreActResNet18().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    classifier.load_state_dict(checkpoint["net"], strict=True)
    classifier.eval().requires_grad_(False)
    victim = torch.nn.Sequential(Normalize().to(device), classifier).eval()
    return victim, checkpoint_path


def load_fixed_cache(path, limit=None):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    stop = len(cache["labels"]) if limit is None else min(limit, len(cache["labels"]))
    metadata = cache.get("attack", {"name": "legacy cached AutoAttack"})
    return (
        cache["clean"][:stop],
        cache["adversarial"][:stop],
        cache["labels"][:stop],
        metadata,
    )


@torch.no_grad()
def defended_logits(defense, x, seed, eot=1):
    logits = 0
    for repeat in range(eot):
        purified = defense.purify(x, seeds=seed + repeat)
        logits = logits + defense.classifier(purified)
    return logits / eot


@torch.no_grad()
def evaluate_accuracy(defense, images, labels, batch_size, seed, eot, desc):
    correct = 0
    for start in tqdm(range(0, len(labels), batch_size), desc=desc):
        x = images[start : start + batch_size].cuda(non_blocking=True)
        y = labels[start : start + batch_size].cuda(non_blocking=True)
        logits = defended_logits(defense, x, seed + start, eot)
        correct += int((logits.argmax(1) == y).sum())
    return correct / len(labels)


def bpda_pgd_linf(
    defense,
    clean,
    labels,
    epsilon,
    alpha,
    steps,
    eot,
    seed,
):
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    delta = torch.empty_like(clean).uniform_(
        -epsilon, epsilon, generator=generator
    )
    adversarial = (clean + delta).clamp(0, 1).detach()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for step in range(steps):
        gradient = torch.zeros_like(adversarial)
        for repeat in range(eot):
            gradient_part, _logits, _loss = defense.gradient(
                adversarial,
                labels,
                loss_fn,
                grad_mode="bpda",
                seeds=seed + step * eot + repeat,
            )
            gradient += gradient_part
        adversarial = adversarial + alpha * (gradient / eot).sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
    return adversarial


def generate_bpda(
    defense,
    clean,
    labels,
    batch_size,
    epsilon,
    alpha,
    steps,
    eot,
    seed,
):
    parts = []
    for start in tqdm(range(0, len(labels), batch_size), desc=f"BPDA-{steps}"):
        x = clean[start : start + batch_size].cuda(non_blocking=True)
        y = labels[start : start + batch_size].cuda(non_blocking=True)
        adversarial = bpda_pgd_linf(
            defense,
            x,
            y,
            epsilon,
            alpha,
            steps,
            eot,
            seed + start,
        )
        parts.append(adversarial.cpu())
    return torch.cat(parts)

