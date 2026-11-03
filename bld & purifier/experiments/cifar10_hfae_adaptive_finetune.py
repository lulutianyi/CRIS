"""Decoder-only CIFAR-10 adversarial adaptation for HF-AE.

Run with ``%run -i`` after executing the model-definition cells in gcn.ipynb.
The main encoder and skip-code generators stay frozen. The base decoder, GCN
blocks and skip-code decoders are adapted with end-to-end PGD examples while a
pretrained CIFAR-10 classifier remains fixed.
"""

from __future__ import annotations

import argparse
import csv
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
    return model


class CifarSubset(Dataset):
    def __init__(self, root, train, indices, download, augment):
        transforms = []
        if augment:
            transforms.extend([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
            ])
        transforms.append(T.ToTensor())
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=T.Compose(transforms),
        )
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def stratified_indices(root, train, size, seed, download):
    dataset = CIFAR10(root=root, train=train, download=download)
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


def make_loaders(args):
    train_indices = stratified_indices(
        args.data_root, True, args.train_size, args.seed, args.download
    )
    val_indices = stratified_indices(
        args.data_root, False, args.val_size, args.seed, args.download
    )
    train_loader = DataLoader(
        CifarSubset(
            args.data_root, True, train_indices, args.download, augment=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CifarSubset(
            args.data_root, False, val_indices, args.download, augment=False
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_indices, val_indices


def freeze_for_decoder_adaptation(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if not name.startswith("decoder."):
            continue
        is_skip_code_generator = (
            "decoder.skip_quantizers" in name
            and (
                ".encoder_net." in name
                or ".encode_conv." in name
            )
        )
        if not is_skip_code_generator:
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("没有找到可训练的解码端参数")
    return trainable


def multiscale_highpass_loss(prediction, target, kernel_sizes=(3, 5, 9)):
    losses = []
    for kernel_size in kernel_sizes:
        padding = kernel_size // 2
        prediction_smooth = F.avg_pool2d(
            F.pad(prediction, (padding,) * 4, mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )
        target_smooth = F.avg_pool2d(
            F.pad(target, (padding,) * 4, mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )
        losses.append(
            F.l1_loss(
                prediction - prediction_smooth,
                target - target_smooth,
            )
        )
    return torch.stack(losses).mean()


def high_frequency_fft_loss(prediction, target, cutoff):
    prediction_fft = torch.fft.fft2(prediction.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    difference = (
        torch.log1p(prediction_fft.abs())
        - torch.log1p(target_fft.abs())
    ).abs()
    height, width = prediction.shape[-2:]
    fy = torch.fft.fftfreq(height, device=prediction.device)
    fx = torch.fft.fftfreq(width, device=prediction.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square()) / (0.5**0.5)
    return difference[..., radius >= cutoff].mean()


def native_to_ae(images):
    return F.interpolate(
        images, (128, 128), mode="bilinear", align_corners=False
    )


def ae_to_classifier(images128):
    return F.interpolate(
        images128.clamp(0, 1),
        (32, 32),
        mode="bilinear",
        align_corners=False,
    )


def pipeline_logits(model, classifier, native_images):
    reconstructed128, _ = model(native_to_ae(native_images))
    return classifier(ae_to_classifier(reconstructed128))


def adaptive_pgd(
    model,
    classifier,
    clean,
    labels,
    epsilon,
    alpha,
    steps,
):
    adversarial = clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
    adversarial = adversarial.clamp(0, 1).detach()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(
            pipeline_logits(model, classifier, adversarial), labels
        )
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
    return adversarial


def encode_detached(model, images128):
    with torch.no_grad():
        latent, features = model.encoder(images128, hard=True)
        latent = latent.detach()
        features = {name: value.detach() for name, value in features.items()}
    return latent, features


@torch.inference_mode()
def evaluate_clean(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = pipeline_logits(model, classifier, images).argmax(1)
        correct += predictions.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def evaluate_adaptive(
    model,
    classifier,
    loader,
    device,
    epsilon,
    alpha,
    steps,
    description,
):
    model.eval()
    classifier.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adversarial = adaptive_pgd(
            model,
            classifier,
            images,
            labels,
            epsilon,
            alpha,
            steps,
        )
        with torch.inference_mode():
            predictions = pipeline_logits(
                model, classifier, adversarial
            ).argmax(1)
        correct += predictions.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def save_checkpoint(path, model, epoch, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "training_args": vars(args),
        },
        path,
    )


def train_one_epoch(
    model,
    classifier,
    loader,
    optimizer,
    trainable,
    args,
    device,
    epoch,
):
    model.train()
    model.encoder.eval()
    classifier.eval()
    totals = {
        "loss": 0.0,
        "clean_cls": 0.0,
        "adv_cls": 0.0,
        "clean_l1": 0.0,
        "adv_l1": 0.0,
        "consistency": 0.0,
        "highpass": 0.0,
        "fft": 0.0,
    }
    seen = 0

    progress = tqdm(loader, desc=f"adaptive finetune {epoch}/{args.epochs}")
    for clean, labels in progress:
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.eval()
        adversarial = adaptive_pgd(
            model,
            classifier,
            clean,
            labels,
            args.epsilon,
            args.attack_alpha,
            args.train_attack_steps,
        )
        model.train()
        model.encoder.eval()

        clean128 = native_to_ae(clean)
        adversarial128 = native_to_ae(adversarial)
        clean_latent, clean_features = encode_detached(model, clean128)
        adv_latent, adv_features = encode_detached(model, adversarial128)
        clean_reconstructed = model.decoder(clean_latent, clean_features)
        adv_reconstructed = model.decoder(adv_latent, adv_features)

        clean_logits = classifier(ae_to_classifier(clean_reconstructed))
        adv_logits = classifier(ae_to_classifier(adv_reconstructed))
        clean_cls = F.cross_entropy(clean_logits, labels)
        adv_cls = F.cross_entropy(adv_logits, labels)
        clean_l1 = F.l1_loss(clean_reconstructed, clean128)
        clean_mse = F.mse_loss(clean_reconstructed, clean128)
        clean_reconstruction = (
            args.l1_mix * clean_l1
            + (1.0 - args.l1_mix) * clean_mse
        )
        adv_l1 = F.l1_loss(adv_reconstructed, clean128)
        consistency = F.l1_loss(
            adv_reconstructed,
            clean_reconstructed.detach(),
        )
        highpass = multiscale_highpass_loss(
            clean_reconstructed, clean128
        )
        fft = high_frequency_fft_loss(
            clean_reconstructed,
            clean128,
            cutoff=args.fft_cutoff,
        )

        loss = (
            args.clean_cls_weight * clean_cls
            + args.adv_cls_weight * adv_cls
            + args.clean_recon_weight * clean_reconstruction
            + args.adv_recon_weight * adv_l1
            + args.consistency_weight * consistency
            + args.highpass_weight * highpass
            + args.fft_weight * fft
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        batch_size = labels.numel()
        seen += batch_size
        values = {
            "loss": loss,
            "clean_cls": clean_cls,
            "adv_cls": adv_cls,
            "clean_l1": clean_l1,
            "adv_l1": adv_l1,
            "consistency": consistency,
            "highpass": highpass,
            "fft": fft,
        }
        for name, value in values.items():
            totals[name] += value.detach().item() * batch_size
        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            clean_ce=f"{clean_cls.item():.3f}",
            adv_ce=f"{adv_cls.item():.3f}",
        )

    return {name: value / seen for name, value in totals.items()}


def append_history_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(args.seed)
    require_notebook_context()
    device = bld.DEVICE
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "hf_ae_cifar_adaptive_decoder_best.pt"
    final_path = run_dir / "hf_ae_cifar_adaptive_decoder_final.pt"
    history_path = run_dir / "training_history.csv"
    summary_path = run_dir / "summary.json"
    if history_path.exists():
        history_path.unlink()

    train_loader, val_loader, train_indices, val_indices = make_loaders(args)
    classifier = load_classifier(args.classifier_checkpoint, device)
    model = build_notebook_gcn_ae(args.hf_checkpoint)
    trainable = freeze_for_decoder_adaptation(model)
    print(
        f"训练集={len(train_indices)}, 验证集={len(val_indices)}, "
        f"可训练解码端参数={sum(p.numel() for p in trainable):,}"
    )
    print("主编码器和skip码字生成器保持冻结；原HF-AE权重不会被覆盖。")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    baseline_clean = evaluate_clean(model, classifier, val_loader, device)
    baseline_adaptive = evaluate_adaptive(
        model,
        classifier,
        val_loader,
        device,
        args.epsilon,
        args.eval_alpha,
        args.final_attack_steps,
        f"baseline adaptive PGD-{args.final_attack_steps}",
    )
    print(
        f"微调前: clean={baseline_clean:.4f}, "
        f"adaptive_PGD{args.final_attack_steps}={baseline_adaptive:.4f}"
    )

    best_score = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            classifier,
            train_loader,
            optimizer,
            trainable,
            args,
            device,
            epoch,
        )
        scheduler.step()
        clean_accuracy = evaluate_clean(model, classifier, val_loader, device)
        adaptive_accuracy = evaluate_adaptive(
            model,
            classifier,
            val_loader,
            device,
            args.epsilon,
            args.eval_alpha,
            args.select_attack_steps,
            f"epoch {epoch} adaptive PGD-{args.select_attack_steps}",
        )
        score = 0.5 * (clean_accuracy + adaptive_accuracy)
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_clean_accuracy": clean_accuracy,
            "val_adaptive_accuracy": adaptive_accuracy,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_history_csv(history_path, row)
        print(
            f"epoch={epoch} clean={clean_accuracy:.4f} "
            f"adaptive_PGD{args.select_attack_steps}={adaptive_accuracy:.4f} "
            f"score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(best_path, model, epoch, row, args)

    save_checkpoint(final_path, model, args.epochs, row, args)
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    freeze_for_decoder_adaptation(model)
    final_clean = evaluate_clean(model, classifier, val_loader, device)
    final_adaptive = evaluate_adaptive(
        model,
        classifier,
        val_loader,
        device,
        args.epsilon,
        args.eval_alpha,
        args.final_attack_steps,
        f"best adaptive PGD-{args.final_attack_steps}",
    )

    summary = {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "seed": args.seed,
        "source_hf_checkpoint": args.hf_checkpoint,
        "classifier_checkpoint": args.classifier_checkpoint,
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "baseline": {
            "clean_accuracy": baseline_clean,
            f"adaptive_pgd{args.final_attack_steps}_accuracy": baseline_adaptive,
        },
        "adapted_best": {
            "clean_accuracy": final_clean,
            f"adaptive_pgd{args.final_attack_steps}_accuracy": final_adaptive,
        },
        "delta": {
            "clean_accuracy": final_clean - baseline_clean,
            f"adaptive_pgd{args.final_attack_steps}_accuracy": (
                final_adaptive - baseline_adaptive
            ),
        },
        "settings": vars(args),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== HF-AE CIFAR自适应解码端微调 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {run_dir}")

    model.to("cpu")
    classifier.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content/cifar10")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default=(
            "/content/drive/MyDrive/gcn_ae_checkpoints/"
            "skip_learned_v2_hf_cifar_adaptive"
        ),
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-alpha", type=float, default=2 / 255)
    parser.add_argument("--eval-alpha", type=float, default=2 / 255)
    parser.add_argument("--train-attack-steps", type=int, default=5)
    parser.add_argument("--select-attack-steps", type=int, default=5)
    parser.add_argument("--final-attack-steps", type=int, default=20)
    parser.add_argument("--clean-cls-weight", type=float, default=0.10)
    parser.add_argument("--adv-cls-weight", type=float, default=0.20)
    parser.add_argument("--clean-recon-weight", type=float, default=1.0)
    parser.add_argument("--adv-recon-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--l1-mix", type=float, default=0.7)
    parser.add_argument("--highpass-weight", type=float, default=0.10)
    parser.add_argument("--fft-weight", type=float, default=0.05)
    parser.add_argument("--fft-cutoff", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    main()
