"""CIFAR-10 classification-retention evaluation for notebook-defined GCN AEs.

Run with ``%run -i`` after executing the model-definition cells in gcn.ipynb.
The classifier-training stage is independent of the notebook AE definitions.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_runtime():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(args):
    device = device_from_runtime()
    train_set = CIFAR10(
        args.data_root,
        train=True,
        download=args.download,
        transform=T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]),
    )
    test_set = CIFAR10(
        args.data_root,
        train=False,
        download=args.download,
        transform=T.ToTensor(),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    model = CifarResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    checkpoint = Path(args.classifier_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        progress = tqdm(train_loader, desc=f"classifier {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
            progress.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")
        scheduler.step()

        accuracy = classification_accuracy(model, test_loader, device)
        print(f"epoch={epoch} test_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                },
                checkpoint,
            )

    print(f"best_accuracy={best_accuracy:.4f}")
    print(f"classifier_checkpoint={checkpoint}")


@torch.inference_mode()
def classification_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def load_classifier(path, device):
    raw = torch.load(path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model = CifarResNet18().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def stratified_indices(root, size, seed, download):
    dataset = CIFAR10(root=root, train=False, download=download)
    rng = random.Random(seed)
    by_class = [[] for _ in range(10)]
    for index, target in enumerate(dataset.targets):
        by_class[target].append(index)
    for indices in by_class:
        rng.shuffle(indices)
    size = min(size, len(dataset))
    base, remainder = divmod(size, 10)
    selected = []
    for class_id, indices in enumerate(by_class):
        selected.extend(indices[:base + (class_id < remainder)])
    rng.shuffle(selected)
    return selected


def highpass(x, kernel_size=5):
    padding = kernel_size // 2
    blur = F.avg_pool2d(
        F.pad(x, (padding,) * 4, mode="reflect"), kernel_size, stride=1
    )
    return x - blur


def ssim_per_image(x, y):
    dims = (1, 2, 3)
    mu_x, mu_y = x.mean(dims), y.mean(dims)
    var_x = ((x - mu_x[:, None, None, None]) ** 2).mean(dims)
    var_y = ((y - mu_y[:, None, None, None]) ** 2).mean(dims)
    covariance = (
        (x - mu_x[:, None, None, None])
        * (y - mu_y[:, None, None, None])
    ).mean(dims)
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )


class ResultAccumulator:
    def __init__(self):
        self.count = 0
        self.correct = 0
        self.class_count = torch.zeros(10, dtype=torch.long)
        self.class_correct = torch.zeros(10, dtype=torch.long)
        self.psnr = self.ssim = self.mae = self.highpass_psnr = 0.0

    def update(self, logits, labels, images, originals):
        predictions = logits.argmax(1).cpu()
        labels_cpu = labels.cpu()
        self.count += labels.numel()
        self.correct += (predictions == labels_cpu).sum().item()
        self.class_count += torch.bincount(labels_cpu, minlength=10)
        good = labels_cpu[predictions == labels_cpu]
        self.class_correct += torch.bincount(good, minlength=10)
        mse = (images - originals).square().mean((1, 2, 3))
        hp_mse = (highpass(images) - highpass(originals)).square().mean((1, 2, 3))
        self.psnr += (-10 * torch.log10(mse + 1e-12)).sum().item()
        self.highpass_psnr += (-10 * torch.log10(hp_mse + 1e-12)).sum().item()
        self.ssim += ssim_per_image(images, originals).sum().item()
        self.mae += (images - originals).abs().mean((1, 2, 3)).sum().item()

    def summary(self):
        return {
            "count": self.count,
            "accuracy": self.correct / self.count,
            "per_class_accuracy": {
                name: (self.class_correct[i] / self.class_count[i]).item()
                for i, name in enumerate(CIFAR_CLASSES)
            },
            "psnr": self.psnr / self.count,
            "ssim": self.ssim / self.count,
            "mae": self.mae / self.count,
            "highpass_psnr": self.highpass_psnr / self.count,
        }


@torch.inference_mode()
def evaluate_method(name, loader, classifier, device, ae=None):
    result = ResultAccumulator()
    for originals, labels in tqdm(loader, desc=name):
        originals = originals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if name == "original":
            images = originals
        else:
            images128 = F.interpolate(
                originals, (128, 128), mode="bilinear", align_corners=False
            )
            if ae is not None:
                images128, _ = ae(images128)
                images128 = images128.clamp(0, 1)
            images = F.interpolate(
                images128, (32, 32), mode="bilinear", align_corners=False
            )
        result.update(classifier(images), labels, images, originals)
    return result.summary()


def evaluate(args):
    require_notebook_context()
    device = bld.DEVICE
    indices = stratified_indices(
        args.data_root, args.subset_size, args.seed, args.download
    )
    loader = DataLoader(
        IndexedCifar(args.data_root, indices, args.download),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    classifier = load_classifier(args.classifier_checkpoint, device)
    results = {}
    results["original"] = evaluate_method(
        "original", loader, classifier, device
    )
    results["resize_only_128"] = evaluate_method(
        "resize_only_128", loader, classifier, device
    )

    for name, checkpoint in (
        ("learned_v2", args.learned_checkpoint),
        ("hf_ae", args.hf_checkpoint),
    ):
        ae = build_notebook_gcn_ae(checkpoint)
        results[name] = evaluate_method(
            name, loader, classifier, device, ae=ae
        )
        del ae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    original_accuracy = results["original"]["accuracy"]
    for value in results.values():
        value["accuracy_retention"] = value["accuracy"] / original_accuracy

    summary = {
        "dataset": "cifar10",
        "subset_size": len(indices),
        "seed": args.seed,
        "classifier_checkpoint": args.classifier_checkpoint,
        "results": results,
    }
    run_dir = Path(args.output_dir) / f"cifar10_n{len(indices)}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("========== CIFAR-10分类保持率 ==========")
    for name, value in results.items():
        print(
            f"{name:16s} accuracy={value['accuracy']:.4f} "
            f"retention={value['accuracy_retention']:.4f} "
            f"psnr={value['psnr']:.4f} ssim={value['ssim']:.4f} "
            f"highpass_psnr={value['highpass_psnr']:.4f}"
        )
    print(f"结果: {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train-classifier", "evaluate"), required=True)
    parser.add_argument("--data-root", default="/content/cifar10")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--learned-checkpoint")
    parser.add_argument("--hf-checkpoint")
    parser.add_argument("--output-dir", default="/content/cifar10_classification_eval")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.stage == "evaluate" and not (
        args.learned_checkpoint and args.hf_checkpoint
    ):
        parser.error("evaluate阶段必须提供 --learned-checkpoint 和 --hf-checkpoint")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.stage == "train-classifier":
        train_classifier(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
