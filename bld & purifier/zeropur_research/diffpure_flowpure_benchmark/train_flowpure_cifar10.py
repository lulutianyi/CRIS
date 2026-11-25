"""Train CIFAR-10 PGD-FlowPure with ZeroPur's PreActResNet18 victim."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from benchmark_common import load_zeropur_resnet18, seed_everything


def import_flowpure_utils(root):
    """Import FlowPure's utils.py despite common top-level module collisions."""
    conflicting_prefixes = ("models", "score_sde", "flow_matching")
    for name in list(sys.modules):
        if name == "utils" or any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in conflicting_prefixes
        ):
            del sys.modules[name]
    importlib.invalidate_caches()
    expected = (root / "utils.py").resolve()
    if not expected.exists():
        raise FileNotFoundError(f"FlowPure源码不存在: {expected}")
    spec = importlib.util.spec_from_file_location("utils", expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建FlowPure utils加载器: {expected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)
    return module


def pgd_linf(model, clean, labels, eps_max, alpha, steps):
    batch = clean.shape[0]
    epsilon = torch.rand(batch, 1, 1, 1, device=clean.device) * eps_max
    adversarial = clean + torch.empty_like(clean).uniform_(-1, 1) * epsilon
    adversarial = adversarial.clamp(0, 1).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
    return adversarial


def conditional_flow_batch(source, target):
    """Equivalent to ConditionalFlowMatcher(sigma=0)."""
    t = torch.rand(source.shape[0], device=source.device)
    t_image = t.view(-1, 1, 1, 1)
    location = (1 - t_image) * source + t_image * target
    velocity = target - source
    return t, location, velocity


@torch.no_grad()
def purify(flow, images, steps):
    state = images.clone()
    times = torch.linspace(0, 1, steps + 1, device=images.device)
    dt = 1 / (steps - 1)
    for t in times[:-1]:
        batch_t = t.expand(images.shape[0])
        state = state + dt * flow(batch_t, state)
    return state.clamp(0, 1)


@torch.no_grad()
def classification_accuracy(flow, victim, images, labels, batch_size, steps):
    correct = 0
    for start in range(0, len(labels), batch_size):
        x = images[start : start + batch_size].cuda(non_blocking=True)
        y = labels[start : start + batch_size].cuda(non_blocking=True)
        reconstructed = purify(flow, x, steps)
        correct += int((victim(reconstructed).argmax(1) == y).sum())
    return correct / len(labels)


def prepare_validation(args, victim):
    dataset = CIFAR10(
        root=str(args.data_root), train=False, transform=ToTensor(), download=True
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.val_size]
    clean = torch.stack([dataset[index][0] for index in indices])
    labels = torch.tensor([dataset[index][1] for index in indices])

    adversarial_parts = []
    for start in tqdm(
        range(0, len(labels), args.val_batch_size), desc="fixed validation PGD"
    ):
        x = clean[start : start + args.val_batch_size].cuda()
        y = labels[start : start + args.val_batch_size].cuda()
        adversarial_parts.append(
            pgd_linf(
                victim,
                x,
                y,
                args.eval_epsilon,
                args.pgd_alpha,
                args.pgd_steps,
            ).cpu()
        )
    return clean, torch.cat(adversarial_parts), labels


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_payload(flow, optimizer, scaler, step, args, victim_checkpoint):
    # Element 0 stays compatible with FlowPure's official evaluation loader.
    metadata = {
        "format": "flowpure_cifar10_zeropur_resnet18_v1",
        "optimizer_step": step,
        "scaler_state_dict": scaler.state_dict(),
        "settings": vars(args),
        "victim_checkpoint": str(victim_checkpoint),
    }
    return [
        flow.model.state_dict(),
        optimizer.state_dict(),
        metadata,
    ]


def save_checkpoint(flow, optimizer, scaler, step, args, victim_checkpoint):
    payload = checkpoint_payload(
        flow, optimizer, scaler, step, args, victim_checkpoint
    )
    atomic_torch_save(payload, args.output)
    if args.archive_every and step % args.archive_every == 0:
        archive = args.output.with_name(
            f"{args.output.stem}_step{step:06d}{args.output.suffix}"
        )
        atomic_torch_save(payload, archive)


def resume_checkpoint(path, flow, optimizer, scaler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    flow.model.load_state_dict(checkpoint[0], strict=True)
    if len(checkpoint) > 1 and checkpoint[1]:
        optimizer.load_state_dict(checkpoint[1])
    metadata = checkpoint[2] if len(checkpoint) > 2 else {}
    if metadata.get("scaler_state_dict"):
        scaler.load_state_dict(metadata["scaler_state_dict"])
    return int(metadata.get("optimizer_step", 0))


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowPure训练需要GPU运行时")
    args.flowpure_root = args.flowpure_root.resolve()
    args.zeropur_root = args.zeropur_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    sys.path.insert(0, str(args.flowpure_root))
    os.chdir(args.flowpure_root)
    flowpure_utils = import_flowpure_utils(args.flowpure_root)
    print("FlowPure utils:", flowpure_utils.__file__)

    device = torch.device("cuda")
    victim, victim_checkpoint = load_zeropur_resnet18(
        args.zeropur_root, device
    )
    flow = flowpure_utils.get_model("new").to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_step = 0
    if args.resume:
        start_step = resume_checkpoint(args.output, flow, optimizer, scaler)
        print("恢复optimizer step:", start_step)
    elif args.output.exists():
        raise FileExistsError(
            f"权重已存在：{args.output}。续训请添加 --resume。"
        )

    train_set = CIFAR10(
        root=str(args.data_root), train=True, transform=ToTensor(), download=True
    )
    loader = DataLoader(
        train_set,
        batch_size=args.physical_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    iterator = iter(loader)
    val_clean, val_adversarial, val_labels = prepare_validation(args, victim)
    log_path = args.output.with_suffix(".jsonl")

    print("victim:", victim_checkpoint)
    print("train images:", len(train_set))
    print("physical/effective batch:", args.physical_batch_size,
          args.physical_batch_size * args.gradient_accumulation)
    print("optimizer steps:", start_step, "->", args.max_steps)
    print("output:", args.output)

    optimizer_step = start_step
    last_saved_step = start_step if args.resume else -1
    started = time.time()
    progress = tqdm(
        total=args.max_steps,
        initial=start_step,
        desc="FlowPure CIFAR-10",
    )
    while optimizer_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for _ in range(args.gradient_accumulation):
            (clean, labels), iterator = next_batch(iterator, loader)
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            adversarial = pgd_linf(
                victim,
                clean,
                labels,
                args.train_epsilon_max,
                args.pgd_alpha,
                args.pgd_steps,
            )
            time_points, locations, velocities = conditional_flow_batch(
                adversarial, clean
            )

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                predicted = flow(time_points, locations)
                loss = F.mse_loss(predicted, velocities)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach())

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), args.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer_step += 1
        mean_loss = loss_sum / args.gradient_accumulation
        progress.update(1)
        progress.set_postfix(loss=f"{mean_loss:.6f}")

        if optimizer_step % args.log_every == 0:
            row = {
                "step": optimizer_step,
                "loss": mean_loss,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.time() - started,
            }
            append_log(log_path, row)

        if args.val_every and optimizer_step % args.val_every == 0:
            flow.eval()
            clean_accuracy = classification_accuracy(
                flow,
                victim,
                val_clean,
                val_labels,
                args.val_batch_size,
                args.euler_steps,
            )
            adversarial_accuracy = classification_accuracy(
                flow,
                victim,
                val_adversarial,
                val_labels,
                args.val_batch_size,
                args.euler_steps,
            )
            flow.train()
            row = {
                "step": optimizer_step,
                "validation_clean_accuracy": clean_accuracy,
                "validation_pgd10_accuracy": adversarial_accuracy,
            }
            append_log(log_path, row)
            print("\nvalidation:", row)

        if optimizer_step % args.save_every == 0:
            save_checkpoint(
                flow,
                optimizer,
                scaler,
                optimizer_step,
                args,
                victim_checkpoint,
            )
            last_saved_step = optimizer_step

    progress.close()
    if last_saved_step != optimizer_step:
        save_checkpoint(
            flow, optimizer, scaler, optimizer_step, args, victim_checkpoint
        )
    print("训练完成，权重:", args.output)
    print("训练日志:", log_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowpure-root", type=Path, required=True)
    parser.add_argument("--zeropur-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/content/cifar10"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--physical-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-epsilon-max", type=float, default=0.05)
    parser.add_argument("--eval-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--archive-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false", default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

