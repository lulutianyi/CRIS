"""
Standalone shared-PGD comparison for Learned-v2 GCN-Skip and FlowPure on D-Fire.

Colab usage
------------
1. Put this file under /content (or run it from Drive).
2. Make sure D-Fire has already been downloaded to /content/D-Fire.
3. Edit the paths in CompareConfig or pass CLI arguments.

On Colab, the script installs Ultralytics when missing and mounts Drive when
needed. It does not download D-Fire because Kaggle authentication is interactive.

Run all stages:
    %run /content/shared_pgd_gcn_flowpure_compare.py --stage all

Or resume stage by stage:
    %run /content/shared_pgd_gcn_flowpure_compare.py --stage generate
    %run /content/shared_pgd_gcn_flowpure_compare.py --stage purify
    %run /content/shared_pgd_gcn_flowpure_compare.py --stage map

This script performs evaluation only. It never trains or overwrites model weights.
The checkpoints are state_dict files, so the exact architecture definitions are
still required; the minimal GCN/Attn/binary-diffusion/FlowPure definitions are
embedded below so no notebook definition cell is needed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from google.colab import drive as colab_drive

    IN_COLAB = True
except ImportError:
    colab_drive = None
    IN_COLAB = False

try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
except ModuleNotFoundError:
    if not IN_COLAB:
        raise
    print("Ultralytics is missing; installing it in this Colab runtime...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class CompareConfig:
    dfire_root: str = "/content/D-Fire"
    yolo_ckpt: str = (
        "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    )
    gcn_ckpt: str = (
        "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2/"
        "gcn_skip_learned_v2_taskaware_best.pt"
    )
    binary_diffusion_ckpt: str = (
        "/content/drive/MyDrive/bld_purify_checkpoints/bld_diffusion_v3.pt"
    )
    flowpure_ckpt: str = (
        "/content/drive/MyDrive/flowpure_dfire_checkpoints/"
        "flowpure_pgd_2000steps.pt"
    )

    shared_root: str = (
        "/content/drive/MyDrive/dfire_shared_pgd_png_eps8_seed20260822"
    )
    output_root: str = (
        "/content/drive/MyDrive/dfire_shared_pgd_gcn_flowpure_comparison"
    )

    split: str = "test"
    expected_images: int = 4306
    max_images: Optional[int] = None
    detector_size: int = 384
    purifier_size: int = 128

    eps_px: int = 8
    alpha_px: int = 2
    pgd_steps: int = 10
    attack_seed: int = 20260822
    attack_batch_size: int = 8

    t_star: int = 3
    purification_seed: int = 20260904
    flowpure_nfe: int = 10
    purify_batch_size: int = 8

    num_workers: int = 2
    use_amp: bool = True
    det_box_gain: float = 7.5
    det_cls_gain: float = 0.5
    det_dfl_gain: float = 1.5
    adopt_existing_shared: bool = False


IMG_SIZE = 128
BASE_CH = 32
CH_MULT_ENC = (1, 2, 4)
LATENT_CHANNELS = 8
NUM_RES_BLOCKS = 2
T_MAX = 200


# -----------------------------------------------------------------------------
# General helpers and D-Fire dataset
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def checkpoint_fingerprint(path: str, length: int = 12) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state_dict(raw):
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def list_images(directory: str) -> List[str]:
    paths: List[str] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(directory, extension)))
    return sorted(set(paths))


class DFireDetDataset(Dataset):
    def __init__(self, root: str, split: str = "test", size: int = 384):
        self.root = root
        self.split = split
        self.img_paths = list_images(os.path.join(root, split, "images"))
        self.label_dir = os.path.join(root, split, "labels")
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
        stems = [Path(path).stem for path in self.img_paths]
        if len(stems) != len(set(stems)):
            raise RuntimeError("D-Fire contains duplicate image stems")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        path = self.img_paths[index]
        image = self.tf(Image.open(path).convert("RGB"))
        label_path = os.path.join(self.label_dir, Path(path).stem + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 5:
                        boxes.append([float(value) for value in parts])
        target = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        return image, target, path


def det_collate(batch):
    images, boxes_list, paths = zip(*batch)
    classes, boxes, batch_indices = [], [], []
    for index, target in enumerate(boxes_list):
        if target.shape[0] > 0:
            classes.append(target[:, 0:1])
            boxes.append(target[:, 1:5])
            batch_indices.append(torch.full((target.shape[0],), index))
    targets = {
        "cls": torch.cat(classes, 0) if classes else torch.zeros((0, 1)),
        "bboxes": torch.cat(boxes, 0) if boxes else torch.zeros((0, 4)),
        "batch_idx": (
            torch.cat(batch_indices, 0) if batch_indices else torch.zeros((0,))
        ),
    }
    return torch.stack(images), targets, paths


class SharedPairDataset(Dataset):
    def __init__(self, shared_root: str, names: Sequence[str]):
        self.shared_root = shared_root
        self.names = list(names)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        clean_path = os.path.join(self.shared_root, "clean", "images", name + ".png")
        adv_path = os.path.join(self.shared_root, "adv", "images", name + ".png")
        clean = self.to_tensor(Image.open(clean_path).convert("RGB"))
        adv = self.to_tensor(Image.open(adv_path).convert("RGB"))
        return clean, adv, name


def shared_collate(batch):
    clean, adv, names = zip(*batch)
    return torch.stack(clean), torch.stack(adv), list(names)


def copy_label(source: str, destination: str) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(source):
        shutil.copyfile(source, destination)
    elif os.path.exists(destination):
        os.remove(destination)


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    return images.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()


def save_uint8_png(image: torch.Tensor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    array = image.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=3)


def save_output_batch(
    images: torch.Tensor,
    names: Sequence[str],
    output_dir: str,
    shared_label_dir: str,
) -> None:
    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    images_u8 = tensor_to_uint8(images)
    for image, name in zip(images_u8, names):
        save_uint8_png(image, os.path.join(image_dir, name + ".png"))
        copy_label(
            os.path.join(shared_label_dir, name + ".txt"),
            os.path.join(label_dir, name + ".txt"),
        )


class LossAttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def prepare_attack_yolo(cfg: CompareConfig, device: torch.device):
    attack_yolo = YOLO(cfg.yolo_ckpt)
    attack_yolo.model.to(device)
    attack_yolo.model.train()
    for parameter in attack_yolo.model.parameters():
        parameter.requires_grad_(False)
    for module in attack_yolo.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = v8DetectionLoss(attack_yolo.model)
    defaults = {
        "box": cfg.det_box_gain,
        "cls": cfg.det_cls_gain,
        "dfl": cfg.det_dfl_gain,
    }
    if isinstance(criterion.hyp, dict):
        hyp = LossAttrDict(criterion.hyp)
        for key, value in defaults.items():
            hyp.setdefault(key, value)
        criterion.hyp = hyp
    else:
        for key, value in defaults.items():
            if not hasattr(criterion.hyp, key):
                setattr(criterion.hyp, key, value)
    return attack_yolo, criterion


def sanitize_targets(targets, device: torch.device, num_classes: int = 2):
    clean = {key: value.to(device) for key, value in targets.items()}
    boxes = clean["bboxes"]
    if boxes.numel() == 0:
        return clean, 0
    classes = clean["cls"].flatten()
    valid = torch.isfinite(boxes).all(dim=1)
    valid &= ((boxes >= 0.0) & (boxes <= 1.0)).all(dim=1)
    valid &= torch.isfinite(classes)
    valid &= (classes >= 0) & (classes < num_classes)
    dropped = int((~valid).sum().item())
    if dropped:
        clean["bboxes"] = clean["bboxes"][valid]
        clean["cls"] = clean["cls"][valid]
        clean["batch_idx"] = clean["batch_idx"][valid]
    return clean, dropped


def deterministic_random_start(
    images: torch.Tensor,
    names: Sequence[str],
    eps: float,
    attack_seed: int,
) -> torch.Tensor:
    noise = []
    for image, name in zip(images, names):
        seed = (attack_seed + zlib.crc32(name.encode("utf-8"))) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample = torch.empty(image.shape, dtype=torch.float32, device="cpu")
        sample.uniform_(-eps, eps, generator=generator)
        noise.append(sample)
    return torch.stack(noise).to(device=images.device, dtype=images.dtype)


def pgd_attack(
    attack_yolo,
    criterion,
    images: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    names: Sequence[str],
    cfg: CompareConfig,
) -> torch.Tensor:
    eps = cfg.eps_px / 255.0
    alpha = cfg.alpha_px / 255.0
    lower = (images - eps).clamp(0.0, 1.0)
    upper = (images + eps).clamp(0.0, 1.0)
    adv = (images + deterministic_random_start(images, names, eps, cfg.attack_seed)).clamp(
        0.0, 1.0
    )
    with torch.enable_grad():
        for _ in range(cfg.pgd_steps):
            adv = adv.detach().requires_grad_(True)
            predictions = attack_yolo.model(adv)
            loss_vector, _ = criterion(predictions, targets)
            loss = loss_vector.sum()
            gradient = torch.autograd.grad(loss, adv, only_inputs=True)[0]
            adv = adv.detach() + alpha * gradient.sign()
            adv = torch.maximum(torch.minimum(adv, upper), lower).clamp(0.0, 1.0)
    return adv.detach()


def attack_signature(cfg: CompareConfig) -> Dict:
    return {
        "schema": "dfire_shared_pgd_uint8_png_v1",
        "split": cfg.split,
        "detector_size": cfg.detector_size,
        "eps_px": cfg.eps_px,
        "alpha_px": cfg.alpha_px,
        "pgd_steps": cfg.pgd_steps,
        "attack_seed": cfg.attack_seed,
        "attack_batch_size": cfg.attack_batch_size,
        "det_box_gain": cfg.det_box_gain,
        "det_cls_gain": cfg.det_cls_gain,
        "det_dfl_gain": cfg.det_dfl_gain,
        "yolo_sha256_12": checkpoint_fingerprint(cfg.yolo_ckpt),
        "max_images": cfg.max_images,
    }


def generate_shared_pgd(cfg: CompareConfig, device: torch.device) -> Dict:
    dataset = DFireDetDataset(cfg.dfire_root, split=cfg.split, size=cfg.detector_size)
    if cfg.max_images is not None:
        dataset.img_paths = dataset.img_paths[: cfg.max_images]
    expected = cfg.max_images if cfg.max_images is not None else cfg.expected_images
    if len(dataset) != expected:
        raise RuntimeError(f"D-Fire images={len(dataset)}, expected={expected}")

    selected_names = [Path(path).stem for path in dataset.img_paths]
    signature = attack_signature(cfg)
    manifest_path = Path(cfg.shared_root) / "manifest.json"
    adopted_existing = False
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in signature.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"Shared PGD manifest mismatch for {key}: "
                    f"existing={existing.get(key)!r}, current={value!r}. "
                    "Use a new shared_root instead of mixing experiments."
                )
        if existing.get("names") != selected_names:
            raise RuntimeError("Shared PGD manifest image order does not match D-Fire")
    else:
        existing_png = list(Path(cfg.shared_root).glob("**/*.png"))
        if existing_png:
            if not cfg.adopt_existing_shared:
                raise RuntimeError(
                    "shared_root contains PNG files but no manifest. If these files "
                    "were produced by an interrupted run with exactly the same PGD "
                    "settings, rerun with --adopt-existing-shared; otherwise use a "
                    "new shared_root."
                )
            allowed_parents = {
                Path(cfg.shared_root, "clean", "images").resolve(),
                Path(cfg.shared_root, "adv", "images").resolve(),
            }
            unexpected_paths = [
                str(path)
                for path in existing_png
                if path.parent.resolve() not in allowed_parents
            ]
            existing_names = {path.stem for path in existing_png}
            unexpected_names = existing_names - set(selected_names)
            if unexpected_paths or unexpected_names:
                raise RuntimeError(
                    "Cannot adopt shared_root: it contains unexpected PNG paths or names"
                )
            adopted_existing = True
            print(
                f"Adopting {len(existing_png)} PNG files from the interrupted run; "
                "the caller confirms that the PGD settings and YOLO weight are unchanged."
            )

    for split_name in ("clean", "adv"):
        Path(cfg.shared_root, split_name, "images").mkdir(parents=True, exist_ok=True)
        Path(cfg.shared_root, split_name, "labels").mkdir(parents=True, exist_ok=True)

    # Write the experiment identity before the expensive loop.  An interrupted
    # run can therefore validate its settings and continue instead of becoming
    # an orphaned directory with PNG files but no manifest.
    progress_manifest = {
        **signature,
        "count": len(selected_names),
        "names": selected_names,
        "status": "in_progress",
        "adopted_existing_without_manifest": adopted_existing,
    }
    manifest_path.write_text(
        json.dumps(progress_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.attack_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=det_collate,
    )
    attack_yolo, criterion = prepare_attack_yolo(cfg, device)
    dropped_total = 0
    for clean, targets, paths in tqdm(loader, desc="Generate shared PGD PNG"):
        names = [Path(path).stem for path in paths]
        clean_paths = [
            Path(cfg.shared_root, "clean", "images", name + ".png") for name in names
        ]
        adv_paths = [
            Path(cfg.shared_root, "adv", "images", name + ".png") for name in names
        ]
        if all(path.exists() for path in clean_paths + adv_paths):
            continue

        clean = clean.to(device, non_blocking=True)
        targets_gpu, dropped = sanitize_targets(targets, device)
        dropped_total += dropped
        adv = pgd_attack(attack_yolo, criterion, clean, targets_gpu, names, cfg)

        clean_u8 = tensor_to_uint8(clean)
        adv_u8_raw = tensor_to_uint8(adv)
        delta = (adv_u8_raw.to(torch.int16) - clean_u8.to(torch.int16)).clamp(
            -cfg.eps_px, cfg.eps_px
        )
        adv_u8 = (clean_u8.to(torch.int16) + delta).clamp(0, 255).to(torch.uint8)

        for clean_image, adv_image, source_path, name in zip(
            clean_u8, adv_u8, paths, names
        ):
            save_uint8_png(
                clean_image,
                os.path.join(cfg.shared_root, "clean", "images", name + ".png"),
            )
            save_uint8_png(
                adv_image,
                os.path.join(cfg.shared_root, "adv", "images", name + ".png"),
            )
            source_label = str(source_path).replace("/images/", "/labels/")
            source_label = str(Path(source_label).with_suffix(".txt"))
            for split_name in ("clean", "adv"):
                copy_label(
                    source_label,
                    os.path.join(cfg.shared_root, split_name, "labels", name + ".txt"),
                )

    del attack_yolo, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_names = {Path(path).stem for path in list_images(os.path.join(cfg.shared_root, "clean", "images"))}
    adv_names = {Path(path).stem for path in list_images(os.path.join(cfg.shared_root, "adv", "images"))}
    if clean_names != set(selected_names) or adv_names != set(selected_names):
        raise RuntimeError(
            f"Shared PGD incomplete: clean={len(clean_names)}, adv={len(adv_names)}, "
            f"expected={len(selected_names)}"
        )

    manifest = {
        **signature,
        "count": len(selected_names),
        "names": selected_names,
        "status": "complete",
        "adopted_existing_without_manifest": adopted_existing,
        "dropped_invalid_boxes_during_this_run": dropped_total,
        "note": "PNG clean/adv are uint8 and pixel delta is clamped to eps_px.",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Shared PGD ready: {manifest_path}")
    return manifest


def load_shared_manifest(cfg: CompareConfig) -> Dict:
    path = Path(cfg.shared_root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run --stage generate first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise RuntimeError(
            f"Shared PGD generation is incomplete; resume --stage generate first: {path}"
        )
    for key, value in attack_signature(cfg).items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Shared PGD manifest mismatch for {key}")
    return manifest


# -----------------------------------------------------------------------------
# Attn autoencoder and Learned-v2 GCN-Skip (matches gcn.ipynb checkpoint)
# -----------------------------------------------------------------------------


class Normalize(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x):
        return self.gn(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_channels: Optional[int] = None):
        super().__init__()
        out_ch = out_channels or in_ch
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = Normalize(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        batch, channels, height, width = h.shape
        q = self.q(h).reshape(batch, channels, -1).permute(0, 2, 1)
        k = self.k(h).reshape(batch, channels, -1)
        v = self.v(h).reshape(batch, channels, -1).permute(0, 2, 1)
        attention = torch.softmax(torch.bmm(q, k) * channels**-0.5, dim=-1)
        out = torch.bmm(attention, v).permute(0, 2, 1).reshape(
            batch, channels, height, width
        )
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class EncoderAttn(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = BASE_CH,
        num_res_blocks: int = NUM_RES_BLOCKS,
        latent_channels: int = LATENT_CHANNELS,
        ch_mult: Tuple[int, ...] = CH_MULT_ENC,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_channels, base_ch, 3, padding=1)]
        in_ch = base_ch
        for stage_index, multiplier in enumerate(ch_mult):
            out_ch = base_ch * multiplier
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            if stage_index == len(ch_mult) - 1:
                layers.append(AttnBlock(in_ch))
            layers.append(Downsample(in_ch))
        layers += [
            ResnetBlock(in_ch),
            AttnBlock(in_ch),
            ResnetBlock(in_ch),
            Normalize(in_ch),
            nn.Conv2d(in_ch, latent_channels, 3, padding=1),
        ]
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x, hard: bool = True):
        logits = self.conv_layers(x)
        if not hard:
            return torch.sigmoid(logits)
        soft = torch.sigmoid(logits)
        hard_value = (soft > 0.5).float()
        return soft + (hard_value - soft).detach()


class DecoderAttn(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_ch: int = BASE_CH,
        num_res_blocks: int = NUM_RES_BLOCKS,
        latent_channels: int = LATENT_CHANNELS,
        ch_mult: Tuple[int, ...] = CH_MULT_ENC,
    ):
        super().__init__()
        ch_mult_dec = tuple(reversed(ch_mult))
        in_ch = base_ch * ch_mult[-1]
        layers: List[nn.Module] = [
            nn.Conv2d(latent_channels, in_ch, 3, padding=1),
            ResnetBlock(in_ch),
            AttnBlock(in_ch),
            ResnetBlock(in_ch),
        ]
        for stage_index, multiplier in enumerate(ch_mult_dec):
            out_ch = base_ch * multiplier
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            layers.append(Upsample(in_ch))
            if stage_index == 0:
                layers.append(AttnBlock(in_ch))
        layers += [Normalize(in_ch), nn.Conv2d(in_ch, out_channels, 3, padding=1), nn.Sigmoid()]
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.conv_layers(z)


class BinaryAutoEncoderAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderAttn()
        self.decoder = DecoderAttn()
        self.latent_h = IMG_SIZE // (2 ** len(CH_MULT_ENC))
        self.latent_w = self.latent_h
        self.D = LATENT_CHANNELS * self.latent_h * self.latent_w


def out_channels(module) -> Optional[int]:
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    if isinstance(module, ResnetBlock):
        return module.conv2.out_channels
    if isinstance(module, AttnBlock):
        return module.proj.out_channels
    return None


def inspect_stages(sequence, resize_class) -> List[Dict]:
    stages, start, channels = [], 0, None
    modules = list(sequence)
    for index, module in enumerate(modules):
        detected = out_channels(module)
        if detected is not None:
            channels = detected
        if isinstance(module, resize_class):
            stages.append({"slice": (start, index), "channels": channels})
            start = index + 1
    stages.append({"slice": (start, len(modules)), "channels": channels})
    return stages


@dataclass
class SkipLevelSpec:
    name: str
    channels: int
    enc_slice: Tuple[int, int]
    dec_insert_after: int
    spatial_size: Optional[int] = None


def auto_detect_skip_levels(base_ae, img_size: int = IMG_SIZE) -> List[SkipLevelSpec]:
    encoder_stages = inspect_stages(base_ae.encoder.conv_layers, Downsample)
    decoder_stages = inspect_stages(base_ae.decoder.conv_layers, Upsample)
    number_down = len(encoder_stages) - 1
    if number_down != len(decoder_stages) - 1:
        raise ValueError("Encoder and decoder are not symmetric")
    levels = []
    for index in range(1, number_down):
        decoder_index = number_down - index
        encoder_segment = encoder_stages[index]
        decoder_input_channels = decoder_stages[decoder_index - 1]["channels"]
        upsample_index = decoder_stages[decoder_index]["slice"][0] - 1
        if encoder_segment["channels"] != decoder_input_channels:
            continue
        resolution = img_size // (2**index)
        levels.append(
            SkipLevelSpec(
                name=f"{resolution}x{resolution}",
                channels=encoder_segment["channels"],
                enc_slice=encoder_segment["slice"],
                dec_insert_after=upsample_index,
                spatial_size=resolution,
            )
        )
    return levels


@dataclass
class GCNBlockConfig:
    zero_init: bool = True
    temperature: Optional[float] = None
    attn_dropout: float = 0.0


@dataclass
class GCNSkipAEConfig:
    active_levels: Optional[List[str]] = None
    block_cfg: GCNBlockConfig = field(default_factory=GCNBlockConfig)
    freeze_base: bool = False
    quantize_skip: bool = True
    skip_quantizer_mode: str = "learned_v2"
    skip_code_channels: int = 8
    skip_code_spatial: int = 8
    skip_level_code_shapes: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"64x64": (2, 16), "32x32": (8, 8)}
    )

    def code_shape_for(self, level_name: str) -> Tuple[int, int]:
        shape = self.skip_level_code_shapes.get(
            level_name, (self.skip_code_channels, self.skip_code_spatial)
        )
        return int(shape[0]), int(shape[1])


class GraphConvSkipBlock(nn.Module):
    def __init__(self, channels: int, cfg: GCNBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.temperature = (
            cfg.temperature if cfg.temperature is not None else channels**-0.5
        )
        self.norm_enc = Normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.gcn_w = nn.Linear(channels, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.attn_dropout = (
            nn.Dropout(cfg.attn_dropout) if cfg.attn_dropout > 0 else nn.Identity()
        )
        if cfg.zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x_dec, feat_enc):
        batch, channels, height, width = x_dec.shape
        normalized = self.norm_enc(feat_enc)
        q = self.q(normalized).reshape(batch, channels, -1).permute(0, 2, 1)
        k = self.k(normalized).reshape(batch, channels, -1)
        adjacency = torch.softmax(
            torch.bmm(q, k) * self.temperature, dim=-1
        )
        adjacency = self.attn_dropout(adjacency)
        flat = x_dec.reshape(batch, channels, -1).permute(0, 2, 1)
        aggregate = self.gcn_w(torch.bmm(adjacency, flat))
        aggregate = aggregate.permute(0, 2, 1).reshape(
            batch, channels, height, width
        )
        return x_dec + self.proj(aggregate)


def skip_group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SkipQuantizer(nn.Module):
    def __init__(
        self,
        channels: int,
        code_channels: int = 8,
        code_spatial: int = 8,
        source_spatial: Optional[int] = None,
        mode: str = "learned_v2",
    ):
        super().__init__()
        self.channels = int(channels)
        self.code_channels = int(code_channels)
        self.code_spatial = int(code_spatial)
        self.source_spatial = int(source_spatial) if source_spatial is not None else None
        self.mode = mode
        self.n_bits = self.code_channels * self.code_spatial * self.code_spatial
        if mode != "learned_v2":
            raise ValueError("This comparison script expects learned_v2 GCN weights")
        if self.source_spatial is None:
            raise ValueError("source_spatial is required")
        ratio = self.source_spatial // self.code_spatial
        if self.source_spatial % self.code_spatial != 0 or ratio & (ratio - 1):
            raise ValueError("source/code spatial ratio must be a power of two")

        hidden = min(channels, max(32, self.code_channels * 8))
        encoder_layers: List[nn.Module] = [
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(skip_group_count(hidden), hidden),
            nn.SiLU(),
        ]
        current = self.source_spatial
        while current > self.code_spatial:
            encoder_layers += [
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                nn.GroupNorm(skip_group_count(hidden), hidden),
                nn.SiLU(),
            ]
            current //= 2
        encoder_layers.append(nn.Conv2d(hidden, self.code_channels, 1))
        self.encoder_net = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = [
            nn.Conv2d(self.code_channels, hidden, 3, padding=1),
            nn.GroupNorm(skip_group_count(hidden), hidden),
            nn.SiLU(),
        ]
        current = self.code_spatial
        while current < self.source_spatial:
            decoder_layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(skip_group_count(hidden), hidden),
                nn.SiLU(),
            ]
            current *= 2
        decoder_layers.append(nn.Conv2d(hidden, channels, 3, padding=1))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def encode(self, feat_enc):
        probability = torch.sigmoid(self.encoder_net(feat_enc))
        hard = (probability > 0.5).float()
        return probability + (hard - probability).detach()

    def decode(self, code, spatial_size: int):
        output = self.decoder_net(code)
        if output.shape[-2:] != (spatial_size, spatial_size):
            output = F.interpolate(
                output,
                size=(spatial_size, spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return output

    def forward(self, feat_enc):
        code = self.encode(feat_enc)
        return self.decode(code, feat_enc.shape[-1]), code


class EncoderGCN(nn.Module):
    def __init__(self, base_encoder, levels: List[SkipLevelSpec]):
        super().__init__()
        self.base_encoder = base_encoder
        self.levels = levels
        self._hook_at = {level.enc_slice[1] - 1: level.name for level in levels}

    def forward(self, x, hard: bool = True):
        features = {}
        hidden = x
        for index, module in enumerate(self.base_encoder.conv_layers):
            hidden = module(hidden)
            if index in self._hook_at:
                features[self._hook_at[index]] = hidden
        if hard:
            soft = torch.sigmoid(hidden)
            hard_value = (soft > 0.5).float()
            latent = soft + (hard_value - soft).detach()
        else:
            latent = torch.sigmoid(hidden)
        return latent, features


class DecoderGCN(nn.Module):
    def __init__(
        self,
        base_decoder,
        levels: List[SkipLevelSpec],
        config: GCNSkipAEConfig,
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.config = config
        active = set(config.active_levels) if config.active_levels is not None else None
        self.gcn_blocks = nn.ModuleDict()
        self.skip_quantizers = nn.ModuleDict()
        self._insert_after = {}
        for level in levels:
            if active is not None and level.name not in active:
                continue
            self.gcn_blocks[level.name] = GraphConvSkipBlock(
                level.channels, config.block_cfg
            )
            if config.quantize_skip:
                code_channels, code_spatial = config.code_shape_for(level.name)
                self.skip_quantizers[level.name] = SkipQuantizer(
                    level.channels,
                    code_channels=code_channels,
                    code_spatial=code_spatial,
                    source_spatial=level.spatial_size,
                    mode=config.skip_quantizer_mode,
                )
            self._insert_after[level.dec_insert_after] = level.name

    def forward(self, z, encoder_features: Dict[str, torch.Tensor]):
        hidden = z
        for index, module in enumerate(self.base_decoder.conv_layers):
            hidden = module(hidden)
            if index in self._insert_after:
                name = self._insert_after[index]
                feature = encoder_features[name]
                if name in self.skip_quantizers:
                    feature, _ = self.skip_quantizers[name](feature)
                hidden = self.gcn_blocks[name](hidden, feature)
        return hidden

    def skip_bit_budget(self) -> int:
        return sum(quantizer.n_bits for quantizer in self.skip_quantizers.values())


class BinaryAutoEncoderGCN(nn.Module):
    def __init__(
        self,
        base_ae,
        config: Optional[GCNSkipAEConfig] = None,
        levels: Optional[List[SkipLevelSpec]] = None,
    ):
        super().__init__()
        self.config = config or GCNSkipAEConfig()
        self.levels = levels or auto_detect_skip_levels(base_ae)
        self.encoder = EncoderGCN(base_ae.encoder, self.levels)
        self.decoder = DecoderGCN(base_ae.decoder, self.levels, self.config)
        self.latent_h = base_ae.latent_h
        self.latent_w = base_ae.latent_w
        self.D = base_ae.D

    def total_bit_budget(self) -> int:
        return self.D + self.decoder.skip_bit_budget()


def load_gcn_model(cfg: CompareConfig, device: torch.device):
    base = BinaryAutoEncoderAttn()
    levels = auto_detect_skip_levels(base, IMG_SIZE)
    model_config = GCNSkipAEConfig(
        skip_quantizer_mode="learned_v2",
        skip_level_code_shapes={"64x64": (2, 16), "32x32": (8, 8)},
    )
    model = BinaryAutoEncoderGCN(base, model_config, levels).to(device)
    raw = torch_load(cfg.gcn_ckpt, device)
    model.load_state_dict(unwrap_state_dict(raw), strict=True)
    model.eval()
    if model.total_bit_budget() != 3072:
        raise RuntimeError(f"GCN bit budget={model.total_bit_budget()}, expected=3072")
    print(f"Loaded GCN learned-v2: {cfg.gcn_ckpt}; bits=3072")
    return model


# -----------------------------------------------------------------------------
# Binary latent diffusion (matches bld_diffusion_v3.pt)
# -----------------------------------------------------------------------------


def get_schedule(device: torch.device, steps: int = T_MAX):
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    retention = torch.cumprod(1.0 - 2.0 * betas, dim=0)
    flip_probability = 0.5 * (1.0 - retention)
    return betas, flip_probability


def q_sample(zero, timestep, flip_probability):
    probability = flip_probability[timestep].unsqueeze(1).expand_as(zero.float())
    mask = torch.bernoulli(probability)
    return ((zero.float() + mask) % 2).long()


def q_posterior_probability(zero_prediction, z_t, timestep, betas, flip_probability):
    beta = betas[timestep - 1].unsqueeze(1)
    previous_flip = torch.where(
        timestep > 1,
        flip_probability[timestep - 2],
        torch.zeros(1, device=timestep.device).expand(timestep.shape[0]),
    ).unsqueeze(1)
    z_t_float = z_t.float()
    q1 = (1 - beta) * z_t_float + beta * (1 - z_t_float)
    q0 = beta * z_t_float + (1 - beta) * (1 - z_t_float)
    e1 = (1 - previous_flip) * zero_prediction + previous_flip * (1 - zero_prediction)
    numerator = q1 * e1
    denominator_part = q0 * (1 - e1)
    return numerator / (numerator + denominator_part + 1e-8)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.t_proj(F.silu(time_embedding))[:, :, None, None]
        hidden = self.drop(self.conv2(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class SelfAttn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, 3, channels, height * width)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * channels**-0.5, dim=-1
        )
        output = torch.bmm(v, attention.transpose(1, 2)).reshape(
            batch, channels, height, width
        )
        return x + self.proj(output)


class DiffUNet(nn.Module):
    def __init__(
        self,
        in_ch=LATENT_CHANNELS,
        base_ch=64,
        ch_mult=(1, 2, 2),
        latent_size=None,
        attn_res=(4, 8),
        dropout=0.1,
    ):
        super().__init__()
        latent_size = latent_size or IMG_SIZE // (2 ** len(CH_MULT_ENC))
        time_dim = base_ch * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        channels = [base_ch * multiplier for multiplier in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = base_ch
        self.enc_chs = [current_channels]
        for index, output_channels in enumerate(channels):
            resolution = latent_size // (2**index)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(current_channels, output_channels, time_dim, dropout),
                        ResidualBlock(output_channels, output_channels, time_dim, dropout),
                        SelfAttn(output_channels) if resolution in attn_res else nn.Identity(),
                    ]
                )
            )
            self.enc_chs.append(output_channels)
            downsample = (
                nn.Conv2d(output_channels, output_channels, 4, 2, 1)
                if index < len(channels) - 1
                else nn.Identity()
            )
            self.downsamples.append(downsample)
            current_channels = output_channels
        self.mid1 = ResidualBlock(channels[-1], channels[-1], time_dim, dropout)
        self.mid_attn = SelfAttn(channels[-1])
        self.mid2 = ResidualBlock(channels[-1], channels[-1], time_dim, dropout)
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        # v3 checkpoint was trained with the output of every encoder level as
        # the three decoder skips: [128, 128, 64].  The older bld draft used
        # self.enc_chs[:-1], which incorrectly produces [128, 64, 64] and makes
        # ups.1 expect 192 input channels instead of the checkpoint's 256.
        reversed_encoder = list(reversed(self.enc_chs[1:]))
        for index, (output_channels, skip_channels) in enumerate(
            zip(reversed_channels, reversed_encoder + [base_ch])
        ):
            resolution = latent_size // (2 ** (len(channels) - 1 - index))
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            current_channels + skip_channels,
                            output_channels,
                            time_dim,
                            dropout,
                        ),
                        ResidualBlock(output_channels, output_channels, time_dim, dropout),
                        SelfAttn(output_channels) if resolution in attn_res else nn.Identity(),
                    ]
                )
            )
            upsample = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                )
                if index < len(reversed_channels) - 1
                else nn.Identity()
            )
            self.upsamples.append(upsample)
            current_channels = output_channels
        self.out_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    @staticmethod
    def sin_embed(timestep, dimension):
        half = dimension // 2
        frequencies = torch.exp(
            -torch.arange(half, device=timestep.device).float()
            * (math.log(10000.0) / (half - 1))
        )
        arguments = timestep[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)

    def forward(self, x, timestep):
        time_embedding = self.t_mlp(
            self.sin_embed(timestep, self.t_mlp[0].in_features)
        )
        hidden = self.in_conv(x)
        skips = [hidden]
        for (first, second, attention), downsample in zip(
            self.downs, self.downsamples
        ):
            hidden = attention(second(first(hidden, time_embedding), time_embedding))
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.mid2(
            self.mid_attn(self.mid1(hidden, time_embedding)), time_embedding
        )
        for (first, second, attention), upsample in zip(self.ups, self.upsamples):
            hidden = upsample(
                attention(
                    second(
                        first(torch.cat([hidden, skips.pop()], 1), time_embedding),
                        time_embedding,
                    )
                )
            )
        return self.out_conv(F.silu(self.out_norm(hidden)))


class ReverseModel(nn.Module):
    def __init__(self, dimension, latent_ch=LATENT_CHANNELS, latent_h=16):
        super().__init__()
        self.C, self.H, self.W = latent_ch, latent_h, latent_h
        self.unet = DiffUNet(in_ch=latent_ch, latent_size=latent_h)

    def forward(self, z_t, timestep):
        batch = z_t.shape[0]
        image = z_t.float().view(batch, self.C, self.H, self.W)
        return self.unet(image, timestep).view(batch, -1)


@torch.no_grad()
def reverse_binary(model, start, t_start, betas, flip_probability):
    value = start.clone()
    for current in range(t_start, 0, -1):
        timestep = torch.full(
            (value.shape[0],), current, device=value.device, dtype=torch.long
        )
        zero_prediction = torch.sigmoid(model(value, timestep))
        if current == 1:
            value = (zero_prediction > 0.5).long()
        else:
            posterior = q_posterior_probability(
                zero_prediction, value, timestep, betas, flip_probability
            )
            value = torch.bernoulli(posterior).long()
    return value


def load_binary_diffusion(cfg: CompareConfig, device: torch.device):
    model = ReverseModel(2048).to(device)
    model.load_state_dict(
        unwrap_state_dict(torch_load(cfg.binary_diffusion_ckpt, device)), strict=True
    )
    model.eval()
    print(f"Loaded binary diffusion: {cfg.binary_diffusion_ckpt}")
    return model


# -----------------------------------------------------------------------------
# FlowPure model (matches flowpure_dfire.py checkpoint)
# -----------------------------------------------------------------------------


@dataclass
class FlowPureModelConfig:
    image_size: int = 128
    in_channels: int = 3
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    attention_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1


def flow_group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class FlowPureTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dim = dimension

    def forward(self, timestep):
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / (half - 1)
        )
        angles = (timestep.float() * 1000.0)[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        return F.pad(embedding, (0, 1)) if self.dim % 2 else embedding


class FlowPureResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(flow_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(flow_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class FlowPureSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(flow_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) * channels**-0.5, dim=-1)
        hidden = torch.bmm(attention, v).transpose(1, 2)
        hidden = hidden.reshape(batch, channels, height, width)
        return x + self.proj(hidden)


class FlowPureUNet(nn.Module):
    def __init__(self, config: FlowPureModelConfig):
        super().__init__()
        self.cfg = config
        time_dim = config.base_channels * 4
        self.time_embedding = nn.Sequential(
            FlowPureTimeEmbedding(config.base_channels),
            nn.Linear(config.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(
            config.in_channels, config.base_channels, 3, padding=1
        )
        self.down_levels = nn.ModuleList()
        current_channels = config.base_channels
        current_resolution = config.image_size
        skip_channels = []
        for index, multiplier in enumerate(config.channel_mult):
            output_channels = config.base_channels * multiplier
            attention = (
                FlowPureSelfAttention(output_channels)
                if current_resolution in config.attention_resolutions
                else nn.Identity()
            )
            downsample = (
                nn.Conv2d(output_channels, output_channels, 4, stride=2, padding=1)
                if index < len(config.channel_mult) - 1
                else nn.Identity()
            )
            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "block1": FlowPureResidualBlock(
                            current_channels, output_channels, time_dim, config.dropout
                        ),
                        "block2": FlowPureResidualBlock(
                            output_channels, output_channels, time_dim, config.dropout
                        ),
                        "attention": attention,
                        "downsample": downsample,
                    }
                )
            )
            skip_channels.append(output_channels)
            current_channels = output_channels
            if index < len(config.channel_mult) - 1:
                current_resolution //= 2
        self.middle1 = FlowPureResidualBlock(
            current_channels, current_channels, time_dim, config.dropout
        )
        self.middle_attention = FlowPureSelfAttention(current_channels)
        self.middle2 = FlowPureResidualBlock(
            current_channels, current_channels, time_dim, config.dropout
        )
        self.up_levels = nn.ModuleList()
        for reverse_index, (multiplier, skip_channels_at_level) in enumerate(
            zip(reversed(config.channel_mult), reversed(skip_channels))
        ):
            output_channels = config.base_channels * multiplier
            attention = (
                FlowPureSelfAttention(output_channels)
                if current_resolution in config.attention_resolutions
                else nn.Identity()
            )
            upsample = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                )
                if reverse_index < len(config.channel_mult) - 1
                else nn.Identity()
            )
            self.up_levels.append(
                nn.ModuleDict(
                    {
                        "block1": FlowPureResidualBlock(
                            current_channels + skip_channels_at_level,
                            output_channels,
                            time_dim,
                            config.dropout,
                        ),
                        "block2": FlowPureResidualBlock(
                            output_channels, output_channels, time_dim, config.dropout
                        ),
                        "attention": attention,
                        "upsample": upsample,
                    }
                )
            )
            current_channels = output_channels
            if reverse_index < len(config.channel_mult) - 1:
                current_resolution *= 2
        self.output_norm = nn.GroupNorm(
            flow_group_count(config.base_channels), config.base_channels
        )
        self.output_conv = nn.Conv2d(
            config.base_channels, config.in_channels, 3, padding=1
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, timestep):
        if timestep.ndim == 0:
            timestep = timestep.expand(x.shape[0])
        time_embedding = self.time_embedding(timestep)
        hidden = self.input_conv(x)
        skips = []
        for level in self.down_levels:
            hidden = level["block1"](hidden, time_embedding)
            hidden = level["block2"](hidden, time_embedding)
            hidden = level["attention"](hidden)
            skips.append(hidden)
            hidden = level["downsample"](hidden)
        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle2(hidden, time_embedding)
        for level in self.up_levels:
            skip = skips.pop()
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = level["block1"](hidden, time_embedding)
            hidden = level["block2"](hidden, time_embedding)
            hidden = level["attention"](hidden)
            hidden = level["upsample"](hidden)
        return self.output_conv(F.silu(self.output_norm(hidden)))


@torch.no_grad()
def flowpure_euler(model, images, steps: int):
    delta_t = 1.0 / steps
    value = images
    for index in range(steps):
        timestep = torch.full(
            (images.shape[0],),
            index * delta_t,
            device=images.device,
            dtype=images.dtype,
        )
        value = value + delta_t * model(value, timestep)
    return value.clamp(0.0, 1.0)


def load_flowpure_model(cfg: CompareConfig, device: torch.device):
    raw = torch_load(cfg.flowpure_ckpt, device)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise ValueError("FlowPure checkpoint does not contain model_state_dict")
    model_config_dict = dict(raw.get("model_config", {}))
    for name in ("channel_mult", "attention_resolutions"):
        if name in model_config_dict:
            model_config_dict[name] = tuple(model_config_dict[name])
    model_config = FlowPureModelConfig(**model_config_dict)
    model = FlowPureUNet(model_config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    if model_config.image_size != cfg.purifier_size:
        raise RuntimeError(
            f"FlowPure checkpoint size={model_config.image_size}, "
            f"configured purifier_size={cfg.purifier_size}"
        )
    print(f"Loaded FlowPure: {cfg.flowpure_ckpt}")
    return model, raw


# -----------------------------------------------------------------------------
# Shared-input purification
# -----------------------------------------------------------------------------


def comparison_run_root(cfg: CompareConfig) -> str:
    gcn_hash = checkpoint_fingerprint(cfg.gcn_ckpt)
    flow_hash = checkpoint_fingerprint(cfg.flowpure_ckpt)
    return os.path.join(
        cfg.output_root,
        f"gcn{gcn_hash}_flow{flow_hash}_t{cfg.t_star}_"
        f"pseed{cfg.purification_seed}_nfe{cfg.flowpure_nfe}",
    )


def make_shared_loader(cfg: CompareConfig, manifest: Dict, device: torch.device):
    dataset = SharedPairDataset(cfg.shared_root, manifest["names"])
    return DataLoader(
        dataset,
        batch_size=cfg.purify_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=shared_collate,
    )


def output_complete(output_dir: str, names: Sequence[str]) -> bool:
    return all(
        Path(output_dir, "images", name + ".png").exists() for name in names
    )


@torch.no_grad()
def evaluate_gcn_from_shared(
    cfg: CompareConfig, manifest: Dict, device: torch.device, run_root: str
) -> None:
    model = load_gcn_model(cfg, device)
    diffusion = load_binary_diffusion(cfg, device)
    betas, flip_probability = get_schedule(device)
    loader = make_shared_loader(cfg, manifest, device)
    shared_labels = os.path.join(cfg.shared_root, "clean", "labels")
    output_dirs = {
        "gcn_clean_reconstructed": os.path.join(run_root, "gcn_clean_reconstructed"),
        "gcn_adv_reconstructed": os.path.join(run_root, "gcn_adv_reconstructed"),
        f"gcn_adv_diffusion_t{cfg.t_star}": os.path.join(
            run_root, f"gcn_adv_diffusion_t{cfg.t_star}"
        ),
    }
    for clean_384, adv_384, names in tqdm(loader, desc="GCN shared-PGD purify"):
        if all(output_complete(directory, names) for directory in output_dirs.values()):
            continue
        clean_384 = clean_384.to(device, non_blocking=True)
        adv_384 = adv_384.to(device, non_blocking=True)
        clean_128 = F.interpolate(
            clean_384,
            size=(cfg.purifier_size, cfg.purifier_size),
            mode="bilinear",
            align_corners=False,
        )
        adv_128 = F.interpolate(
            adv_384,
            size=(cfg.purifier_size, cfg.purifier_size),
            mode="bilinear",
            align_corners=False,
        )
        clean_latent, clean_features = model.encoder(clean_128, hard=True)
        adv_latent, adv_features = model.encoder(adv_128, hard=True)
        clean_reconstructed = model.decoder(clean_latent, clean_features)
        adv_reconstructed = model.decoder(adv_latent, adv_features)

        batch_seed = (
            cfg.purification_seed
            + zlib.crc32("|".join(names).encode("utf-8"))
        )
        seed_everything(batch_seed)
        flat = adv_latent.view(adv_latent.shape[0], -1).long()
        timestep = torch.full(
            (flat.shape[0],), cfg.t_star - 1, device=device, dtype=torch.long
        )
        noised = q_sample(flat, timestep, flip_probability)
        recovered = reverse_binary(
            diffusion, noised, cfg.t_star, betas, flip_probability
        ).float().view_as(adv_latent)
        diffusion_reconstructed = model.decoder(recovered, adv_features)

        resize = dict(
            size=(cfg.detector_size, cfg.detector_size),
            mode="bilinear",
            align_corners=False,
        )
        save_output_batch(
            F.interpolate(clean_reconstructed, **resize),
            names,
            output_dirs["gcn_clean_reconstructed"],
            shared_labels,
        )
        save_output_batch(
            F.interpolate(adv_reconstructed, **resize),
            names,
            output_dirs["gcn_adv_reconstructed"],
            shared_labels,
        )
        save_output_batch(
            F.interpolate(diffusion_reconstructed, **resize),
            names,
            output_dirs[f"gcn_adv_diffusion_t{cfg.t_star}"],
            shared_labels,
        )
    del model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_flowpure_from_shared(
    cfg: CompareConfig, manifest: Dict, device: torch.device, run_root: str
) -> None:
    model, _ = load_flowpure_model(cfg, device)
    loader = make_shared_loader(cfg, manifest, device)
    shared_labels = os.path.join(cfg.shared_root, "clean", "labels")
    output_dirs = {
        "clean_resize_baseline": os.path.join(run_root, "clean_resize_baseline"),
        "adv_resize_baseline": os.path.join(run_root, "adv_resize_baseline"),
        "flowpure_clean": os.path.join(run_root, "flowpure_clean"),
        "flowpure_adv": os.path.join(run_root, "flowpure_adv"),
    }
    amp_enabled = cfg.use_amp and device.type == "cuda"
    for clean_384, adv_384, names in tqdm(loader, desc="FlowPure shared-PGD purify"):
        if all(output_complete(directory, names) for directory in output_dirs.values()):
            continue
        clean_384 = clean_384.to(device, non_blocking=True)
        adv_384 = adv_384.to(device, non_blocking=True)
        clean_128 = F.interpolate(
            clean_384,
            size=(cfg.purifier_size, cfg.purifier_size),
            mode="bilinear",
            align_corners=False,
        )
        adv_128 = F.interpolate(
            adv_384,
            size=(cfg.purifier_size, cfg.purifier_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            clean_flow = flowpure_euler(model, clean_128, cfg.flowpure_nfe)
            adv_flow = flowpure_euler(model, adv_128, cfg.flowpure_nfe)
        resize = dict(
            size=(cfg.detector_size, cfg.detector_size),
            mode="bilinear",
            align_corners=False,
        )
        save_output_batch(
            F.interpolate(clean_128, **resize),
            names,
            output_dirs["clean_resize_baseline"],
            shared_labels,
        )
        save_output_batch(
            F.interpolate(adv_128, **resize),
            names,
            output_dirs["adv_resize_baseline"],
            shared_labels,
        )
        save_output_batch(
            F.interpolate(clean_flow.float(), **resize),
            names,
            output_dirs["flowpure_clean"],
            shared_labels,
        )
        save_output_batch(
            F.interpolate(adv_flow.float(), **resize),
            names,
            output_dirs["flowpure_adv"],
            shared_labels,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_output_counts(cfg: CompareConfig, manifest: Dict, run_root: str) -> None:
    expected = manifest["count"]
    directories = [
        "gcn_clean_reconstructed",
        "gcn_adv_reconstructed",
        f"gcn_adv_diffusion_t{cfg.t_star}",
        "clean_resize_baseline",
        "adv_resize_baseline",
        "flowpure_clean",
        "flowpure_adv",
    ]
    for name in directories:
        count = len(list_images(os.path.join(run_root, name, "images")))
        print(f"{name:30s} images={count}/{expected}")
        if count != expected:
            raise RuntimeError(f"Incomplete output: {name}")


# -----------------------------------------------------------------------------
# Unified mAP and report
# -----------------------------------------------------------------------------


def eval_map(
    detector,
    split_dir: str,
    image_size: int,
    project_dir: str,
    name: str,
    device: torch.device,
) -> Tuple[float, float]:
    yaml_path = os.path.join(split_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": os.path.abspath(split_dir),
                "train": "images",
                "val": "images",
                "names": {0: "fire", 1: "smoke"},
            },
            handle,
            sort_keys=False,
        )
    result = detector.val(
        data=yaml_path,
        imgsz=image_size,
        verbose=False,
        plots=False,
        project=project_dir,
        name=name,
        exist_ok=True,
        device=0 if device.type == "cuda" else "cpu",
    )
    return float(result.box.map50), float(result.box.map)


def compute_maps(
    cfg: CompareConfig, manifest: Dict, device: torch.device, run_root: str
) -> Dict:
    verify_output_counts(cfg, manifest, run_root)
    detector = YOLO(cfg.yolo_ckpt)
    directories = {
        "original_clean": os.path.join(cfg.shared_root, "clean"),
        "adv_original": os.path.join(cfg.shared_root, "adv"),
        "clean_resize_baseline": os.path.join(run_root, "clean_resize_baseline"),
        "adv_resize_baseline": os.path.join(run_root, "adv_resize_baseline"),
        "gcn_clean_reconstructed": os.path.join(run_root, "gcn_clean_reconstructed"),
        "gcn_adv_reconstructed": os.path.join(run_root, "gcn_adv_reconstructed"),
        f"gcn_adv_diffusion_t{cfg.t_star}": os.path.join(
            run_root, f"gcn_adv_diffusion_t{cfg.t_star}"
        ),
        "flowpure_clean": os.path.join(run_root, "flowpure_clean"),
        "flowpure_adv": os.path.join(run_root, "flowpure_adv"),
    }
    results = {}
    project_dir = os.path.join(run_root, "ultralytics_val")
    for name, directory in directories.items():
        map50, map50_95 = eval_map(
            detector,
            directory,
            cfg.detector_size,
            project_dir,
            name,
            device,
        )
        results[name] = {"map50": map50, "map50_95": map50_95}
        print(f"{name:30s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")

    clean = results["original_clean"]
    adv = results["adv_original"]

    def recovery(result_name: str, metric: str) -> float:
        numerator = results[result_name][metric] - adv[metric]
        denominator = max(clean[metric] - adv[metric], 1e-8)
        return numerator / denominator

    gcn_diff_name = f"gcn_adv_diffusion_t{cfg.t_star}"
    derived = {
        "gcn_direct_recovery_map50": recovery("gcn_adv_reconstructed", "map50"),
        "gcn_direct_recovery_map50_95": recovery(
            "gcn_adv_reconstructed", "map50_95"
        ),
        "gcn_diffusion_recovery_map50": recovery(gcn_diff_name, "map50"),
        "gcn_diffusion_recovery_map50_95": recovery(gcn_diff_name, "map50_95"),
        "flowpure_recovery_map50": recovery("flowpure_adv", "map50"),
        "flowpure_recovery_map50_95": recovery("flowpure_adv", "map50_95"),
        "resize_recovery_map50": recovery("adv_resize_baseline", "map50"),
        "gcn_clean_drop_map50": clean["map50"]
        - results["gcn_clean_reconstructed"]["map50"],
        "flowpure_clean_drop_map50": clean["map50"]
        - results["flowpure_clean"]["map50"],
        "gcn_diffusion_gain_over_direct_map50": results[gcn_diff_name]["map50"]
        - results["gcn_adv_reconstructed"]["map50"],
        "flowpure_gain_over_resize_map50": results["flowpure_adv"]["map50"]
        - results["adv_resize_baseline"]["map50"],
    }
    print("\n=== Unified shared-PGD comparison ===")
    print(
        f"GCN direct recovery: mAP50={derived['gcn_direct_recovery_map50']:.1%}, "
        f"mAP50-95={derived['gcn_direct_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN T*={cfg.t_star} recovery: "
        f"mAP50={derived['gcn_diffusion_recovery_map50']:.1%}, "
        f"mAP50-95={derived['gcn_diffusion_recovery_map50_95']:.1%}"
    )
    print(
        f"FlowPure recovery: mAP50={derived['flowpure_recovery_map50']:.1%}, "
        f"mAP50-95={derived['flowpure_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN diffusion gain over direct mAP50: "
        f"{derived['gcn_diffusion_gain_over_direct_map50']:+.4f}"
    )
    print(
        f"FlowPure gain over resize mAP50: "
        f"{derived['flowpure_gain_over_resize_map50']:+.4f}"
    )

    report = {
        "schema": "shared_pgd_gcn_flowpure_comparison_v1",
        "config": asdict(cfg),
        "shared_attack_manifest": manifest,
        "checkpoint_fingerprints": {
            "yolo": checkpoint_fingerprint(cfg.yolo_ckpt),
            "gcn": checkpoint_fingerprint(cfg.gcn_ckpt),
            "binary_diffusion": checkpoint_fingerprint(cfg.binary_diffusion_ckpt),
            "flowpure": checkpoint_fingerprint(cfg.flowpure_ckpt),
        },
        "results": results,
        "derived": derived,
    }
    report_path = Path(run_root) / "metrics_shared_pgd.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Metrics saved: {report_path}")
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def validate_paths(cfg: CompareConfig, stage: str) -> None:
    required = [cfg.yolo_ckpt]
    if stage in ("purify", "map", "all"):
        required += [cfg.gcn_ckpt, cfg.binary_diffusion_ckpt, cfg.flowpure_ckpt]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if stage in ("generate", "all") and not os.path.isdir(cfg.dfire_root):
        raise FileNotFoundError(cfg.dfire_root)


def parse_args():
    defaults = CompareConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("generate", "purify", "map", "all"), default="all"
    )
    parser.add_argument("--dfire-root", default=defaults.dfire_root)
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--gcn-ckpt", default=defaults.gcn_ckpt)
    parser.add_argument(
        "--binary-diffusion-ckpt", default=defaults.binary_diffusion_ckpt
    )
    parser.add_argument("--flowpure-ckpt", default=defaults.flowpure_ckpt)
    parser.add_argument("--shared-root", default=defaults.shared_root)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--attack-seed", type=int, default=defaults.attack_seed)
    parser.add_argument("--t-star", type=int, default=defaults.t_star)
    parser.add_argument(
        "--purification-seed", type=int, default=defaults.purification_seed
    )
    parser.add_argument("--flowpure-nfe", type=int, default=defaults.flowpure_nfe)
    parser.add_argument(
        "--attack-batch-size", type=int, default=defaults.attack_batch_size
    )
    parser.add_argument(
        "--purify-batch-size", type=int, default=defaults.purify_batch_size
    )
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--adopt-existing-shared", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)
    cfg = CompareConfig(
        dfire_root=args.dfire_root,
        yolo_ckpt=args.yolo_ckpt,
        gcn_ckpt=args.gcn_ckpt,
        binary_diffusion_ckpt=args.binary_diffusion_ckpt,
        flowpure_ckpt=args.flowpure_ckpt,
        shared_root=args.shared_root,
        output_root=args.output_root,
        max_images=args.max_images,
        attack_seed=args.attack_seed,
        t_star=args.t_star,
        purification_seed=args.purification_seed,
        flowpure_nfe=args.flowpure_nfe,
        attack_batch_size=args.attack_batch_size,
        purify_batch_size=args.purify_batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        adopt_existing_shared=args.adopt_existing_shared,
    )
    return args.stage, cfg


def main() -> None:
    stage, cfg = parse_args()
    if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
        assert colab_drive is not None
        colab_drive.mount("/content/drive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.t_star < 1:
        raise ValueError("t_star must be >= 1; GCN direct is already evaluated separately")
    validate_paths(cfg, stage)
    seed_everything(cfg.attack_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if stage in ("generate", "all"):
        manifest = generate_shared_pgd(cfg, device)
    else:
        manifest = load_shared_manifest(cfg)

    # Generation is deliberately independent of either purifier checkpoint.
    # This makes the shared PGD dataset reusable for future checkpoints too.
    if stage == "generate":
        return

    run_root = comparison_run_root(cfg)
    Path(run_root).mkdir(parents=True, exist_ok=True)

    if stage in ("purify", "all"):
        evaluate_gcn_from_shared(cfg, manifest, device, run_root)
        evaluate_flowpure_from_shared(cfg, manifest, device, run_root)
        verify_output_counts(cfg, manifest, run_root)
    if stage in ("map", "all"):
        compute_maps(cfg, manifest, device, run_root)


if __name__ == "__main__":
    main()
