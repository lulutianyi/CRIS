"""Reproducible, inference-only P0/P1 audit for Robust-v3 and 6144 expansion.

The script never constructs an optimizer and never trains a model.  It can:

* archive full SHA256 fingerprints, source snapshots, Git state, environment,
  shared-PGD contents, D-Fire split manifests and invalid-label exclusions;
* generate one fixed validation PGD set from an explicit validation manifest;
* verify v3 versus expanded-checkpoint inherited tensors, main bits and old
  skip slices, then evaluate v3 / expanded-old-only / expanded-full;
* run P1 oracle stream swaps and explicitly labelled soft-code diagnostics;
* save per-image bit statistics, predictions, ground truth and deterministic
  class-aware matching outcomes.

Colab examples are documented in ``gcn_p0_p1_audit_README.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
import PIL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from . import gcn_p0_p1_models as models
    from . import shared_pgd_gcn_flowpure_compare as shared
except ImportError:
    import gcn_p0_p1_models as models
    import shared_pgd_gcn_flowpure_compare as shared


SCHEMA = "gcn_robust_v3_p0_p1_audit_v1"


@dataclass
class AuditConfig:
    robust_v3: str = (
        "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust/"
        "gcn_skip_learned_v2_robust_taskaware_final.pt"
    )
    expanded_epoch8: str = (
        "/content/drive/MyDrive/gcn_ae_checkpoints/"
        "skip6144_from_robustv3_only64/seed20260921_8x500/epoch8.pt"
    )
    yolo_ckpt: str = (
        "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    )
    p0_shared_root: str = (
        "/content/drive/MyDrive/dfire_shared_pgd_png_eps8_seed20260822"
    )
    p1_shared_root: Optional[str] = None
    dfire_root: str = "/content/D-Fire"
    output_root: str = "/content/drive/MyDrive/gcn_p0_p1_audit"
    repo_root: str = "/content"
    detector_size: int = 384
    purifier_size: int = 128
    batch_size: int = 2
    map_batch_size: int = 16
    predict_batch_size: int = 16
    num_workers: int = 2
    expected_total: int = 4306
    expected_valid: int = 4302
    expected_excluded: int = 4
    map_conf: float = 0.001
    map_iou: float = 0.7
    map_max_det: int = 300
    detail_conf: float = 0.25
    detail_iou: float = 0.50
    attack_eps_px: int = 8
    attack_alpha_px: int = 2
    attack_steps: int = 10
    attack_seed: int = 20260822
    split_seed: int = 20260928
    validation_count: int = 512
    save_code_tensors: bool = False
    save_detections: bool = True
    allow_count_mismatch: bool = False
    allow_p1_on_p0_data: bool = False
    equivalence_atol: float = 2e-4
    source_files: List[str] = field(default_factory=list)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_to: Optional[Path] = None) -> Dict:
    stat = path.stat()
    name = str(path.relative_to(relative_to)) if relative_to else str(path)
    return {
        "path": name.replace("\\", "/"),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def tree_inventory(root: Path, patterns: Sequence[str]) -> Tuple[List[Dict], str]:
    paths = sorted(
        {path for pattern in patterns for path in root.glob(pattern) if path.is_file()},
        key=lambda item: item.as_posix(),
    )
    records = [file_record(path, root) for path in tqdm(paths, desc=f"Hash {root.name}")]
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return records, digest.hexdigest()


def command_capture(command: Sequence[str], cwd: Path) -> Dict:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, capture_output=True, text=True, check=False
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (FileNotFoundError, OSError) as error:
        return {"command": list(command), "error": repr(error)}


def software_environment() -> Dict:
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception as error:
        ultralytics_version = f"unavailable: {error!r}"
    return {
        "captured_utc": now_utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "ultralytics": ultralytics_version,
    }


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def list_source_images(directory: Path) -> List[Path]:
    result = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        result.extend(directory.glob(extension))
    return sorted(result, key=lambda item: item.name)


def validate_yolo_label(path: Path, num_classes: int = 2) -> Tuple[bool, List[Dict]]:
    if not path.exists():
        return True, []  # Missing/empty labels represent background images in D-Fire.
    problems = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            problems.append({"line": line_number, "reason": "field_count", "raw": raw})
            continue
        try:
            class_value, x, y, width, height = map(float, fields)
        except ValueError:
            problems.append({"line": line_number, "reason": "non_numeric", "raw": raw})
            continue
        values = (class_value, x, y, width, height)
        if not all(math.isfinite(value) for value in values):
            problems.append({"line": line_number, "reason": "non_finite", "raw": raw})
        elif class_value != int(class_value) or not 0 <= int(class_value) < num_classes:
            problems.append({"line": line_number, "reason": "class_range", "raw": raw})
        elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            problems.append({"line": line_number, "reason": "box_range", "raw": raw})
    return not problems, problems


def read_shared_manifest(root: Path) -> Dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    names = manifest.get("names")
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError("manifest.names must be a unique ordered list")
    if manifest.get("status", "complete") != "complete":
        raise RuntimeError(f"Shared PGD is not complete: {path}")
    return manifest


def audit_shared_dataset(
    cfg: AuditConfig, root: Path, output: Path, enforce_expected_counts: bool = True
) -> Dict:
    manifest = read_shared_manifest(root)
    names = manifest["names"]
    exclusions = []
    valid_names = []
    image_rows = []
    for name in tqdm(names, desc="Audit shared PGD"):
        clean = root / "clean" / "images" / f"{name}.png"
        adv = root / "adv" / "images" / f"{name}.png"
        clean_label = root / "clean" / "labels" / f"{name}.txt"
        adv_label = root / "adv" / "labels" / f"{name}.txt"
        require_paths([clean, adv])
        with Image.open(clean) as clean_image, Image.open(adv) as adv_image:
            codec = {
                "clean_format": clean_image.format,
                "clean_mode": clean_image.mode,
                "clean_size": list(clean_image.size),
                "adv_format": adv_image.format,
                "adv_mode": adv_image.mode,
                "adv_size": list(adv_image.size),
            }
        valid, problems = validate_yolo_label(clean_label)
        if valid:
            valid_names.append(name)
        else:
            exclusions.append({"name": name, "label": str(clean_label), "problems": problems})
        image_rows.append({
            "name": name,
            "clean_sha256": sha256_file(clean),
            "adv_sha256": sha256_file(adv),
            "clean_label_sha256": sha256_file(clean_label) if clean_label.exists() else None,
            "adv_label_sha256": sha256_file(adv_label) if adv_label.exists() else None,
            **codec,
        })

    if enforce_expected_counts and not cfg.allow_count_mismatch:
        expected = (cfg.expected_total, cfg.expected_valid, cfg.expected_excluded)
        actual = (len(names), len(valid_names), len(exclusions))
        if actual != expected:
            raise RuntimeError(f"Dataset counts {actual} do not match expected {expected}")
    json_dump(output / "shared_manifest_original.json", manifest)
    json_dump(output / "valid_names.json", {"names": valid_names, "count": len(valid_names)})
    json_dump(output / "excluded_labels.json", {"items": exclusions, "count": len(exclusions)})
    hash_path = output / "shared_file_hashes.jsonl"
    if hash_path.exists():
        hash_path.unlink()
    append_jsonl(hash_path, image_rows)
    summary = {
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "total": len(names),
        "valid": len(valid_names),
        "excluded": len(exclusions),
        "content_inventory": str(hash_path),
        "codec_expectation": "uint8 RGB PNG at detector resolution",
    }
    json_dump(output / "shared_audit_summary.json", summary)
    return {"manifest": manifest, "valid_names": valid_names, "exclusions": exclusions,
            "summary": summary}


def archive_dfire_split(root: Path, split: str, output: Path) -> Dict:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    images = list_source_images(image_dir)
    rows = []
    for image in tqdm(images, desc=f"Hash D-Fire {split}"):
        label = label_dir / f"{image.stem}.txt"
        valid, problems = validate_yolo_label(label)
        rows.append({
            "name": image.stem,
            "image": file_record(image, root),
            "label": file_record(label, root) if label.exists() else None,
            "label_valid": valid,
            "label_problems": problems,
        })
    path = output / f"dfire_{split}_inventory.jsonl"
    if path.exists():
        path.unlink()
    append_jsonl(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"split": split, "count": len(rows), "inventory": str(path),
            "inventory_sha256": digest}


def default_source_files(repo_root: Path) -> List[Path]:
    relative = [
        "baselines/purify_bld_fixed.py",
        "gcn.ipynb",
        "gcn_6144bit.ipynb",
        "notebooks/cell_replacements/gcn6144_cell06.txt",
        "notebooks/cell_replacements/gcn6144_cell07.txt",
        "methods/gcn6144_new_channels.py",
        "core/shared_pgd_gcn_flowpure_compare.py",
        "core/gcn_p0_p1_models.py",
        "core/gcn_p0_p1_audit.py",
    ]
    return [repo_root / item for item in relative if (repo_root / item).is_file()]


def run_archive(cfg: AuditConfig) -> Dict:
    output = Path(cfg.output_root) / "archive"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(cfg.robust_v3), Path(cfg.expanded_epoch8), Path(cfg.yolo_ckpt)]
    require_paths(checkpoints + [Path(cfg.p0_shared_root), Path(cfg.dfire_root)])
    checkpoint_records = {path.name: file_record(path) for path in checkpoints}
    json_dump(output / "checkpoint_sha256.json", checkpoint_records)

    repo = Path(cfg.repo_root)
    source_dir = output / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    requested_sources = default_source_files(repo) + [Path(item) for item in cfg.source_files]
    missing_sources = [str(path) for path in requested_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError("Missing requested source snapshots:\n" + "\n".join(missing_sources))
    sources = []
    unique_sources = list(dict.fromkeys(path.resolve() for path in requested_sources))
    for source in unique_sources:
        try:
            relative = source.relative_to(repo.resolve())
        except ValueError:
            relative = Path("external") / source.name
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        try:
            record = file_record(source, repo.resolve())
        except ValueError:
            record = file_record(source)
        sources.append({**record, "snapshot": str(destination)})
    json_dump(output / "source_sha256.json", sources)

    git = {
        "head": command_capture(["git", "rev-parse", "HEAD"], repo),
        "status": command_capture(["git", "status", "--short"], repo),
        "diff": command_capture(["git", "diff", "--binary"], repo),
        "diff_cached": command_capture(["git", "diff", "--cached", "--binary"], repo),
    }
    json_dump(output / "git_state.json", git)
    json_dump(output / "environment.json", software_environment())
    pip_freeze = command_capture([sys.executable, "-m", "pip", "freeze"], repo)
    json_dump(output / "pip_freeze.json", pip_freeze)

    shared_audit = audit_shared_dataset(cfg, Path(cfg.p0_shared_root), output / "p0_shared")
    dfire = [
        archive_dfire_split(Path(cfg.dfire_root), split, output)
        for split in ("train", "test")
    ]
    caveats = {
        "validation_history": (
            "A validation subset newly removed from D-Fire train may already have been "
            "seen while training the frozen Robust-v3 source model. It is a future "
            "fine-tuning diagnostic split, not source-model-unseen data."
        ),
        "test_history": (
            "The 4302-image test set has already informed repeated decisions. Stopping "
            "future tuning does not retrospectively make it an untouched independent test set."
        ),
        "definition_provenance": (
            "Strict checkpoint loading verifies names and tensor shapes, not semantic identity "
            "of Python forward code. Replace/archive the source snapshot with the exact Colab "
            "cells that produced the historical results when available."
        ),
    }
    result = {
        "schema": SCHEMA,
        "captured_utc": now_utc(),
        "checkpoints": checkpoint_records,
        "sources": sources,
        "shared": shared_audit["summary"],
        "dfire": dfire,
        "caveats": caveats,
    }
    json_dump(output / "archive_index.json", result)
    return result


def deterministic_rank(name: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def run_make_splits(cfg: AuditConfig) -> Dict:
    """Create immutable name lists; never moves or edits D-Fire files."""
    output = Path(cfg.output_root) / "splits"
    output.mkdir(parents=True, exist_ok=True)
    train_images = list_source_images(Path(cfg.dfire_root) / "train" / "images")
    test_images = list_source_images(Path(cfg.dfire_root) / "test" / "images")
    train_names = [item.stem for item in train_images]
    ranked = sorted(train_names, key=lambda name: (deterministic_rank(name, cfg.split_seed), name))
    if not 0 < cfg.validation_count < len(ranked):
        raise ValueError("validation_count must be between zero and train-set size")
    validation_names = sorted(ranked[: cfg.validation_count])
    remaining_train = sorted(ranked[cfg.validation_count :])

    p0_manifest = read_shared_manifest(Path(cfg.p0_shared_root))
    shared_test_names = list(p0_manifest["names"])
    if set(shared_test_names) != {item.stem for item in test_images}:
        missing = sorted(set(shared_test_names) - {item.stem for item in test_images})
        extra = sorted({item.stem for item in test_images} - set(shared_test_names))
        raise RuntimeError(
            f"D-Fire test/shared PGD name mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    payload = {
        "schema": "dfire_fixed_splits_v1",
        "created_utc": now_utc(),
        "seed": cfg.split_seed,
        "selection": "lowest SHA256(seed:name), then lexicographic name",
        "source_train_count": len(train_names),
        "future_train_names": remaining_train,
        "future_validation_names": validation_names,
        "historical_test_names": shared_test_names,
        "caveats": [
            "Validation images may already have been seen by the frozen Robust-v3 source model.",
            "The historical test set has already influenced prior decisions and is not untouched.",
        ],
    }
    json_dump(output / "fixed_splits.json", payload)
    return payload


class NamedDFireDataset(Dataset):
    def __init__(self, dfire_root: str, split: str, names: Sequence[str], size: int):
        image_dir = Path(dfire_root) / split / "images"
        label_dir = Path(dfire_root) / split / "labels"
        available = {path.stem: path for path in list_source_images(image_dir)}
        missing = [name for name in names if name not in available]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images, first={missing[:5]}")
        self.items = [(name, available[name], label_dir / f"{name}.txt") for name in names]
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        name, image_path, label_path = self.items[index]
        image = self.transform(Image.open(image_path).convert("RGB"))
        classes, boxes = [], []
        if label_path.exists():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                class_id, x, y, width, height = map(float, raw.split())
                classes.append([class_id])
                boxes.append([x, y, width, height])
        return image, {
            "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }, name, str(label_path)


def named_collate(batch):
    images, targets, names, labels = zip(*batch)
    classes, boxes, batch_indices = [], [], []
    for index, target in enumerate(targets):
        count = target["cls"].shape[0]
        classes.append(target["cls"])
        boxes.append(target["bboxes"])
        batch_indices.append(torch.full((count,), index, dtype=torch.long))
    return torch.stack(images), {
        "cls": torch.cat(classes, 0),
        "bboxes": torch.cat(boxes, 0),
        "batch_idx": torch.cat(batch_indices, 0),
    }, list(names), list(labels)


def save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode="RGB").save(
        path, format="PNG", compress_level=3
    )


def prepare_loss_detector(yolo_ckpt: str, device: torch.device):
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    yolo = YOLO(yolo_ckpt)
    detector = yolo.model.to(device).train()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    for module in detector.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = v8DetectionLoss(detector)
    if isinstance(criterion.hyp, dict):
        criterion.hyp = shared.LossAttrDict(criterion.hyp)
    for key, value in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
        if not hasattr(criterion.hyp, key):
            setattr(criterion.hyp, key, value)
    return yolo, detector, criterion


def random_start_per_name(
    images: torch.Tensor, names: Sequence[str], eps: float, seed: int
) -> torch.Tensor:
    values = []
    for image, name in zip(images.detach().cpu(), names):
        local_seed = (seed + zlib.crc32(name.encode("utf-8"))) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        values.append(torch.empty_like(image).uniform_(-eps, eps, generator=generator))
    return torch.stack(values).to(images.device)


def pgd_attack(detector, criterion, clean, targets, names, cfg: AuditConfig):
    eps = cfg.attack_eps_px / 255.0
    alpha = cfg.attack_alpha_px / 255.0
    lower = (clean - eps).clamp(0, 1)
    upper = (clean + eps).clamp(0, 1)
    adversarial = (clean + random_start_per_name(clean, names, eps, cfg.attack_seed)).clamp(0, 1)
    with torch.enable_grad():
        for _ in range(cfg.attack_steps):
            adversarial = adversarial.detach().requires_grad_(True)
            vector, _ = criterion(detector(adversarial), targets)
            gradient = torch.autograd.grad(vector.sum(), adversarial, only_inputs=True)[0]
            adversarial = adversarial.detach() + alpha * gradient.sign()
            adversarial = torch.maximum(torch.minimum(adversarial, upper), lower).clamp(0, 1)
    return adversarial.detach()


def run_generate_validation_pgd(cfg: AuditConfig) -> Dict:
    if not cfg.p1_shared_root:
        raise ValueError("--p1-shared-root is required")
    split_path = Path(cfg.output_root) / "splits" / "fixed_splits.json"
    if not split_path.is_file():
        raise FileNotFoundError("Run --stage make-splits first")
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    names = split_data["future_validation_names"]
    root = Path(cfg.p1_shared_root)
    manifest_path = root / "manifest.json"
    attack_signature = {
        "schema": "dfire_fixed_validation_pgd_uint8_png_v1",
        "status": "complete",
        "split": "train-derived-validation",
        "count": len(names),
        "names": names,
        "detector_size": cfg.detector_size,
        "eps_px": cfg.attack_eps_px,
        "alpha_px": cfg.attack_alpha_px,
        "steps": cfg.attack_steps,
        "random_start": "uniform[-eps,+eps], deterministic per image",
        "attack_seed": cfg.attack_seed,
        "attack_loss": "Ultralytics v8DetectionLoss sum(box+cls+dfl)",
        "attack_preprocess": "PIL RGB -> Resize(384,384) -> ToTensor [0,1]",
        "yolo_path": cfg.yolo_ckpt,
        "yolo_sha256": sha256_file(Path(cfg.yolo_ckpt)),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in attack_signature.items():
            if key != "status" and existing.get(key) != value:
                raise RuntimeError(f"Existing validation PGD differs at {key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, detector, criterion = prepare_loss_detector(cfg.yolo_ckpt, device)
    dataset = NamedDFireDataset(cfg.dfire_root, "train", names, cfg.detector_size)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=named_collate,
    )
    for split in ("clean", "adv"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    progress = {**attack_signature, "status": "in_progress"}
    json_dump(manifest_path, progress)
    for clean, targets, batch_names, label_paths in tqdm(loader, desc="Validation PGD"):
        clean = clean.to(device, non_blocking=True)
        targets = {key: value.to(device) for key, value in targets.items()}
        output_clean = [root / "clean" / "images" / f"{name}.png" for name in batch_names]
        output_adv = [root / "adv" / "images" / f"{name}.png" for name in batch_names]
        if not all(path.is_file() for path in output_clean + output_adv):
            adversarial = pgd_attack(detector, criterion, clean, targets, batch_names, cfg)
            for index, name in enumerate(batch_names):
                save_png(clean[index], output_clean[index])
                save_png(adversarial[index], output_adv[index])
        for name, label_path in zip(batch_names, label_paths):
            source = Path(label_path)
            for split in ("clean", "adv"):
                destination = root / split / "labels" / f"{name}.txt"
                if source.exists():
                    shutil.copy2(source, destination)
                elif destination.exists():
                    destination.unlink()
    attack_signature["manifest_created_utc"] = now_utc()
    json_dump(manifest_path, attack_signature)
    return attack_signature


class SharedPairDataset(Dataset):
    def __init__(self, root: Path, names: Sequence[str]):
        self.root = root
        self.names = list(names)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        clean_path = self.root / "clean" / "images" / f"{name}.png"
        adv_path = self.root / "adv" / "images" / f"{name}.png"
        clean = self.to_tensor(Image.open(clean_path).convert("RGB"))
        adv = self.to_tensor(Image.open(adv_path).convert("RGB"))
        return clean, adv, name


def pair_collate(batch):
    clean, adversarial, names = zip(*batch)
    return torch.stack(clean), torch.stack(adversarial), list(names)


def make_pair_loader(cfg: AuditConfig, root: Path, names: Sequence[str]):
    return DataLoader(
        SharedPairDataset(root, names),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=pair_collate,
    )


def resize_for_purifier(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def tensor_stats(probability: torch.Tensor, hard: torch.Tensor) -> List[Dict]:
    flat_probability = probability.flatten(1)
    flat_hard = hard.flatten(1)
    margin = (flat_probability - 0.5).abs()
    result = []
    for index in range(probability.shape[0]):
        values = margin[index]
        result.append({
            "bit_count": int(flat_hard.shape[1]),
            "one_rate": float(flat_hard[index].mean()),
            "margin_mean": float(values.mean()),
            "margin_min": float(values.min()),
            "margin_p10": float(torch.quantile(values, 0.10)),
            "margin_p50": float(torch.quantile(values, 0.50)),
        })
    return result


def paired_stream_rows(names, clean_repr, adv_repr) -> List[Dict]:
    streams = {
        "main": (
            clean_repr["main_probability"], clean_repr["main_hard"],
            adv_repr["main_probability"], adv_repr["main_hard"],
        )
    }
    for name in sorted(clean_repr["skip_hard"]):
        streams[name] = (
            clean_repr["skip_probability"][name], clean_repr["skip_hard"][name],
            adv_repr["skip_probability"][name], adv_repr["skip_hard"][name],
        )
    per_stream = {}
    for stream, (clean_probability, clean_hard, adv_probability, adv_hard) in streams.items():
        clean_stats = tensor_stats(clean_probability, clean_hard)
        adv_stats = tensor_stats(adv_probability, adv_hard)
        flip = (clean_hard != adv_hard).flatten(1).float().mean(1)
        probability_shift = (
            clean_probability - adv_probability
        ).abs().flatten(1).mean(1)
        per_stream[stream] = [
            {
                "clean": clean_stats[index],
                "adv": adv_stats[index],
                "flip_rate": float(flip[index]),
                "mean_probability_shift": float(probability_shift[index]),
            }
            for index in range(len(names))
        ]
    return [
        {
            "name": name,
            "streams": {stream: values[index] for stream, values in per_stream.items()},
        }
        for index, name in enumerate(names)
    ]


def save_code_tensor_shard(path: Path, names, clean_repr, adv_repr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"names": np.asarray(names)}
    for prefix, representation in (("clean", clean_repr), ("adv", adv_repr)):
        arrays[f"{prefix}_main_probability"] = representation["main_probability"].cpu().numpy()
        arrays[f"{prefix}_main_hard"] = representation["main_hard"].to(torch.uint8).cpu().numpy()
        for name in sorted(representation["skip_hard"]):
            safe = name.replace("x", "_")
            arrays[f"{prefix}_{safe}_probability"] = representation["skip_probability"][name].cpu().numpy()
            arrays[f"{prefix}_{safe}_hard"] = representation["skip_hard"][name].to(torch.uint8).cpu().numpy()
    np.savez_compressed(path, **arrays)


def condition_directories(output: Path, condition_names: Sequence[str]) -> Dict[str, Path]:
    result = {}
    for name in condition_names:
        root = output / "conditions" / name
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "excluded_images").mkdir(parents=True, exist_ok=True)
        result[name] = root
    return result


def save_condition_batch(
    images: torch.Tensor,
    names: Sequence[str],
    directory: Path,
    valid_set: set,
    shared_label_dir: Path,
) -> None:
    for image, name in zip(images, names):
        if name in valid_set:
            path = directory / "images" / f"{name}.png"
            label_source = shared_label_dir / f"{name}.txt"
            label_destination = directory / "labels" / f"{name}.txt"
            if label_source.exists() and not label_destination.exists():
                shutil.copy2(label_source, label_destination)
        else:
            path = directory / "excluded_images" / f"{name}.png"
        if not path.exists():
            save_png(image, path)


def verify_condition_counts(
    directories: Dict[str, Path], valid_count: int, excluded_count: int
) -> None:
    failures = []
    for name, directory in directories.items():
        valid_actual = len(list((directory / "images").glob("*.png")))
        excluded_actual = len(list((directory / "excluded_images").glob("*.png")))
        if (valid_actual, excluded_actual) != (valid_count, excluded_count):
            failures.append({
                "condition": name, "valid": valid_actual, "excluded": excluded_actual,
                "expected_valid": valid_count, "expected_excluded": excluded_count,
            })
    if failures:
        raise RuntimeError(f"Incomplete reconstruction outputs: {failures}")


def old_slice_code_equal(old_repr, expanded_repr) -> Dict[str, torch.Tensor]:
    result = {
        "main": (old_repr["main_hard"] == expanded_repr["main_hard"]).flatten(1).all(1)
    }
    for name, old_code in old_repr["skip_hard"].items():
        count = old_code.shape[1]
        candidate = expanded_repr["skip_hard"][name][:, :count]
        result[name] = (old_code == candidate).flatten(1).all(1)
    return result


def evaluate_map(
    cfg: AuditConfig, detector, directory: Path, output: Path, condition: str
) -> Dict:
    dataset_yaml = output / "map_inputs" / f"{condition}.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        yaml.safe_dump({
            "path": str(directory), "train": "images", "val": "images",
            "names": {0: "fire", 1: "smoke"},
        }), encoding="utf-8"
    )
    result = detector.val(
        data=str(dataset_yaml), imgsz=cfg.detector_size,
        batch=cfg.map_batch_size, conf=cfg.map_conf, iou=cfg.map_iou,
        max_det=cfg.map_max_det, verbose=False, plots=False,
        project=str(output / "ultralytics_runs"), name=condition, exist_ok=True,
    )
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "parameters": {
            "imgsz": cfg.detector_size, "batch": cfg.map_batch_size,
            "conf": cfg.map_conf, "iou": cfg.map_iou,
            "max_det": cfg.map_max_det,
        },
    }


def copy_valid_reference(
    shared_root: Path, split: str, names: Sequence[str], output: Path
) -> Path:
    root = output / "references" / f"original_{split}"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for name in tqdm(names, desc=f"Prepare original_{split}"):
        for folder, suffix in (("images", ".png"), ("labels", ".txt")):
            source = shared_root / split / folder / f"{name}{suffix}"
            destination = root / folder / f"{name}{suffix}"
            if source.exists() and not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
    return root


def compute_recovery(results: Dict[str, Dict], condition: str) -> Dict[str, float]:
    clean = results["original_clean"]
    adversarial = results["original_adv"]
    value = results[condition]
    derived = {}
    for suffix in ("map50", "map50_95"):
        denominator = clean[suffix] - adversarial[suffix]
        derived[f"recovery_{suffix}"] = (
            (value[suffix] - adversarial[suffix]) / denominator
            if abs(denominator) > 1e-12 else float("nan")
        )
    return derived


def xywhn_to_xyxy(box, size: int) -> List[float]:
    x, y, width, height = box
    return [
        (x - width / 2) * size, (y - height / 2) * size,
        (x + width / 2) * size, (y + height / 2) * size,
    ]


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def ground_truth_from_label(path: Path, size: int) -> List[Dict]:
    result = []
    if not path.exists():
        return result
    for object_index, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        class_value, x, y, width, height = map(float, raw.split())
        area = width * height * size * size
        size_group = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
        result.append({
            "object_index": object_index,
            "class_id": int(class_value),
            "class_name": "fire" if int(class_value) == 0 else "smoke",
            "xyxy": xywhn_to_xyxy((x, y, width, height), size),
            "xywh_normalized": [x, y, width, height],
            "area_px2": area,
            "size_group": size_group,
        })
    return result


def match_image_detections(
    ground_truth: List[Dict], predictions: List[Dict], conf: float, iou_threshold: float
) -> Dict:
    candidates = [item for item in predictions if item["confidence"] >= conf]
    candidates.sort(key=lambda item: (-item["confidence"], item["prediction_index"]))
    unmatched_gt = set(range(len(ground_truth)))
    matches = []
    unmatched_predictions = []
    for prediction in candidates:
        possible = []
        for gt_index in unmatched_gt:
            target = ground_truth[gt_index]
            if target["class_id"] != prediction["class_id"]:
                continue
            iou = box_iou(target["xyxy"], prediction["xyxy"])
            if iou >= iou_threshold:
                possible.append((iou, gt_index))
        if possible:
            iou, gt_index = max(possible, key=lambda item: (item[0], -item[1]))
            unmatched_gt.remove(gt_index)
            matches.append({
                "ground_truth_index": gt_index,
                "prediction_index": prediction["prediction_index"],
                "iou": iou,
            })
        else:
            unmatched_predictions.append(prediction["prediction_index"])
    if not ground_truth:
        outcome = "true_negative" if not candidates else "background_false_positive"
    elif not unmatched_gt and not unmatched_predictions:
        outcome = "all_objects_matched_no_false_positive"
    elif not unmatched_gt:
        outcome = "all_objects_matched_with_false_positive"
    elif matches:
        outcome = "partial_match"
    else:
        outcome = "no_object_matched"
    return {
        "rule": {
            "confidence_threshold": conf,
            "iou_threshold": iou_threshold,
            "class_aware": True,
            "assignment": "prediction confidence descending, best-IoU unmatched GT",
            "background_rule": "no GT: true_negative iff no prediction passes confidence",
        },
        "outcome": outcome,
        "matches": matches,
        "unmatched_ground_truth_indices": sorted(unmatched_gt),
        "unmatched_prediction_indices": unmatched_predictions,
    }


def save_detection_details(
    cfg: AuditConfig,
    detector,
    directory: Path,
    names: Sequence[str],
    output_path: Path,
) -> Dict:
    image_paths = [directory / "images" / f"{name}.png" for name in names]
    label_paths = [directory / "labels" / f"{name}.txt" for name in names]
    if output_path.exists():
        output_path.unlink()
    outcome_counts: Dict[str, int] = {}
    class_gt = {"fire": 0, "smoke": 0}
    size_gt = {"small": 0, "medium": 0, "large": 0}
    for start in tqdm(range(0, len(names), cfg.predict_batch_size), desc=output_path.stem):
        batch_paths = image_paths[start : start + cfg.predict_batch_size]
        results = detector.predict(
            source=[str(path) for path in batch_paths], imgsz=cfg.detector_size,
            conf=cfg.map_conf, iou=cfg.map_iou, max_det=cfg.map_max_det,
            verbose=False,
        )
        rows = []
        for offset, result in enumerate(results):
            index = start + offset
            predictions = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidence = result.boxes.conf.detach().cpu().tolist()
                for prediction_index, (box, class_id, score) in enumerate(
                    zip(xyxy, classes, confidence)
                ):
                    predictions.append({
                        "prediction_index": prediction_index,
                        "class_id": int(class_id),
                        "class_name": "fire" if int(class_id) == 0 else "smoke",
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box],
                    })
            ground_truth = ground_truth_from_label(label_paths[index], cfg.detector_size)
            for item in ground_truth:
                class_gt[item["class_name"]] += 1
                size_gt[item["size_group"]] += 1
            matching = match_image_detections(
                ground_truth, predictions, cfg.detail_conf, cfg.detail_iou
            )
            outcome_counts[matching["outcome"]] = outcome_counts.get(matching["outcome"], 0) + 1
            rows.append({
                "name": names[index], "ground_truth": ground_truth,
                "predictions": predictions, "matching": matching,
            })
        append_jsonl(output_path, rows)
    summary = {
        "images": len(names), "outcome_counts": outcome_counts,
        "ground_truth_by_class": class_gt, "ground_truth_by_size": size_gt,
        "details": str(output_path),
    }
    json_dump(output_path.with_suffix(".summary.json"), summary)
    return summary


def load_valid_names(
    cfg: AuditConfig, shared_root: Path, output: Path,
    enforce_expected_counts: bool,
) -> Tuple[Dict, List[str], set]:
    audit = audit_shared_dataset(
        cfg, shared_root, output / "dataset_audit", enforce_expected_counts
    )
    names = audit["manifest"]["names"]
    valid = audit["valid_names"]
    return audit["manifest"], names, set(valid)


def architecture_signature(model) -> Dict:
    return {
        "purifier_input": [3, 128, 128],
        "detector_input": [3, 384, 384],
        "main": {"shape": [8, 16, 16], "bits": 2048},
        "skips": {
            name: {
                "shape": [quantizer.code_channels, quantizer.code_spatial,
                          quantizer.code_spatial],
                "bits": quantizer.n_bits,
                "source_spatial": quantizer.source_spatial,
                "hidden_channels": quantizer.hidden_channels,
                "mode": quantizer.mode,
                "decoder_insert_after_index": next(
                    index for index, value in model.decoder._insert_after.items()
                    if value == name
                ),
            }
            for name, quantizer in model.decoder.skip_quantizers.items()
        },
        "total_bits": model.total_bit_budget(),
        "hard_threshold": "probability > 0.5 (exactly 0.5 maps to zero)",
        "base_normalization": "GroupNorm(min(32, channels), channels)",
        "skip_normalization": "GroupNorm(largest <=16 divisor of hidden channels)",
        "base_upsample": "nearest x2 then 3x3 convolution",
        "skip_upsample": "bilinear align_corners=False then 3x3 convolution",
        "384_to_128_and_128_to_384": "bilinear align_corners=False",
        "image_serialization": "round(clamp(x,0,1)*255) uint8 RGB PNG",
    }


@torch.inference_mode()
def run_p0(cfg: AuditConfig) -> Dict:
    output = Path(cfg.output_root) / "p0"
    output.mkdir(parents=True, exist_ok=True)
    shared_root = Path(cfg.p0_shared_root)
    manifest, names, valid_set = load_valid_names(
        cfg, shared_root, output, enforce_expected_counts=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3 = models.load_model(cfg.robust_v3, expanded=False, device=device)
    expanded = models.load_model(cfg.expanded_epoch8, expanded=True, device=device)
    inherited = models.compare_inherited_state(v3, expanded)
    json_dump(output / "inherited_state_equivalence.json", inherited)
    if not inherited["equal"]:
        raise RuntimeError("Expanded checkpoint changed inherited parameters; see report")

    condition_names = [
        "v3_clean_hard", "v3_adv_hard",
        "expanded_old_only_clean", "expanded_old_only_adv",
        "expanded_full_clean", "expanded_full_adv",
    ]
    directories = condition_directories(output, condition_names)
    loader = make_pair_loader(cfg, shared_root, names)
    equivalence_path = output / "per_image_equivalence.jsonl"
    if equivalence_path.exists():
        equivalence_path.unlink()
    aggregate = {"images": 0, "main_mismatch": 0, "skip64_mismatch": 0,
                 "skip32_mismatch": 0, "max_old_only_reconstruction_error": 0.0}
    with torch.inference_mode():
        for clean384, adv384, batch_names in tqdm(loader, desc="P0 inference"):
            clean128 = resize_for_purifier(clean384.to(device), cfg.purifier_size)
            adv128 = resize_for_purifier(adv384.to(device), cfg.purifier_size)
            rows = []
            for source_name, images128 in (("clean", clean128), ("adv", adv128)):
                old_repr = v3.encode_representations(images128)
                new_repr = expanded.encode_representations(images128)
                equal = old_slice_code_equal(old_repr, new_repr)
                old_rec = v3.decode_from_codes(old_repr["main_hard"], old_repr["skip_hard"])
                new_old_codes = models.old_only_skip_codes(new_repr["skip_hard"])
                new_old_rec = expanded.decode_from_codes(new_repr["main_hard"], new_old_codes)
                new_full_rec = expanded.decode_from_codes(
                    new_repr["main_hard"], new_repr["skip_hard"]
                )
                errors = (old_rec - new_old_rec).abs().flatten(1).max(1).values
                for index, name in enumerate(batch_names):
                    row = next((item for item in rows if item["name"] == name), None)
                    if row is None:
                        row = {"name": name}
                        rows.append(row)
                    row[source_name] = {
                        "main_equal": bool(equal["main"][index]),
                        "skip64_old_slice_equal": bool(equal["64x64"][index]),
                        "skip32_old_slice_equal": bool(equal["32x32"][index]),
                        "old_only_max_abs_reconstruction_error": float(errors[index]),
                    }
                    aggregate["images"] += 1
                    aggregate["main_mismatch"] += int(not equal["main"][index])
                    aggregate["skip64_mismatch"] += int(not equal["64x64"][index])
                    aggregate["skip32_mismatch"] += int(not equal["32x32"][index])
                    aggregate["max_old_only_reconstruction_error"] = max(
                        aggregate["max_old_only_reconstruction_error"], float(errors[index])
                    )
                old_rec_384 = F.interpolate(
                    old_rec, size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear", align_corners=False,
                )
                new_old_rec_384 = F.interpolate(
                    new_old_rec, size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear", align_corners=False,
                )
                new_full_rec_384 = F.interpolate(
                    new_full_rec, size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear", align_corners=False,
                )
                save_condition_batch(
                    old_rec_384, batch_names, directories[f"v3_{source_name}_hard"],
                    valid_set, shared_root / "clean" / "labels",
                )
                save_condition_batch(
                    new_old_rec_384, batch_names, directories[f"expanded_old_only_{source_name}"],
                    valid_set, shared_root / "clean" / "labels",
                )
                save_condition_batch(
                    new_full_rec_384, batch_names, directories[f"expanded_full_{source_name}"],
                    valid_set, shared_root / "clean" / "labels",
                )
            append_jsonl(equivalence_path, rows)
    json_dump(output / "equivalence_summary.json", aggregate)
    if any(aggregate[key] for key in ("main_mismatch", "skip64_mismatch", "skip32_mismatch")):
        raise RuntimeError("Old code slices are not equivalent; see equivalence_summary.json")
    if aggregate["max_old_only_reconstruction_error"] > cfg.equivalence_atol:
        raise RuntimeError(
            "Expanded old_only reconstruction differs from v3 beyond tolerance: "
            f"{aggregate['max_old_only_reconstruction_error']} > {cfg.equivalence_atol}"
        )
    verify_condition_counts(directories, len(valid_set), len(names) - len(valid_set))

    from ultralytics import YOLO
    detector = YOLO(cfg.yolo_ckpt)
    valid_names = [name for name in names if name in valid_set]
    reference_clean = copy_valid_reference(shared_root, "clean", valid_names, output)
    reference_adv = copy_valid_reference(shared_root, "adv", valid_names, output)
    map_results = {
        "original_clean": evaluate_map(cfg, detector, reference_clean, output, "original_clean"),
        "original_adv": evaluate_map(cfg, detector, reference_adv, output, "original_adv"),
    }
    for condition, directory in directories.items():
        map_results[condition] = evaluate_map(cfg, detector, directory, output, condition)
    derived = {
        condition: compute_recovery(map_results, condition)
        for condition in condition_names if condition.endswith("adv") or "adv_hard" in condition
    }
    report = {
        "schema": SCHEMA, "stage": "p0", "created_utc": now_utc(),
        "checkpoint_sha256": {
            "robust_v3": sha256_file(Path(cfg.robust_v3)),
            "expanded_epoch8": sha256_file(Path(cfg.expanded_epoch8)),
            "yolo": sha256_file(Path(cfg.yolo_ckpt)),
        },
        "architecture_v3": architecture_signature(v3),
        "architecture_expanded": architecture_signature(expanded),
        "shared_manifest": manifest,
        "valid_images": len(valid_names), "excluded_images": len(names) - len(valid_names),
        "old_only_definition": (
            "Use expanded encoder and all old skip channels; set skip64 channels [2:] "
            "and skip32 channels [8:] to zero immediately before decode_from_codes."
        ),
        "equivalence": aggregate,
        "equivalence_atol": cfg.equivalence_atol,
        "maps": map_results, "derived": derived,
    }
    json_dump(output / "metrics_p0.json", report)
    if cfg.save_detections:
        details = {}
        for condition, directory in {
            "original_clean": reference_clean, "original_adv": reference_adv,
            **directories,
        }.items():
            details[condition] = save_detection_details(
                cfg, detector, directory, valid_names,
                output / "detections" / f"{condition}.jsonl",
            )
        json_dump(output / "detection_detail_index.json", details)
    return report


def combine_codes(main, skips):
    return main, {name: value for name, value in skips.items()}


@torch.inference_mode()
def run_p1(cfg: AuditConfig) -> Dict:
    if not cfg.p1_shared_root:
        raise ValueError("--p1-shared-root is required for P1")
    p0_root = Path(cfg.p0_shared_root).resolve()
    p1_root = Path(cfg.p1_shared_root).resolve()
    if p0_root == p1_root and not cfg.allow_p1_on_p0_data:
        raise RuntimeError(
            "P1 root equals the historically used P0 test set. Generate a fixed validation "
            "PGD root or explicitly pass --allow-p1-on-p0-data."
        )
    output = Path(cfg.output_root) / "p1"
    output.mkdir(parents=True, exist_ok=True)
    manifest, names, valid_set = load_valid_names(
        cfg, p1_root, output, enforce_expected_counts=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.load_model(cfg.robust_v3, expanded=False, device=device)

    condition_names = [
        "clean_hard_all",
        "adv_hard_all",
        "oracle_clean_main_adv_skips_hard",
        "oracle_adv_main_clean_skips_hard",
        "diagnostic_clean_soft_all",
        "diagnostic_adv_soft_all",
        "diagnostic_clean_soft_main_hard_skips",
        "diagnostic_adv_soft_main_hard_skips",
        "diagnostic_clean_hard_main_soft_skips",
        "diagnostic_adv_hard_main_soft_skips",
    ]
    directories = condition_directories(output, condition_names)
    loader = make_pair_loader(cfg, p1_root, names)
    bit_stats_path = output / "per_image_bit_statistics.jsonl"
    if bit_stats_path.exists():
        bit_stats_path.unlink()
    shard_index = 0

    with torch.inference_mode():
        for clean384, adv384, batch_names in tqdm(loader, desc="P1 code diagnostics"):
            clean128 = resize_for_purifier(clean384.to(device), cfg.purifier_size)
            adv128 = resize_for_purifier(adv384.to(device), cfg.purifier_size)
            clean = model.encode_representations(clean128)
            adversarial = model.encode_representations(adv128)
            append_jsonl(bit_stats_path, paired_stream_rows(batch_names, clean, adversarial))
            if cfg.save_code_tensors:
                save_code_tensor_shard(
                    output / "code_tensors" / f"batch_{shard_index:06d}.npz",
                    batch_names, clean, adversarial,
                )
                shard_index += 1

            conditions = {
                "clean_hard_all": combine_codes(clean["main_hard"], clean["skip_hard"]),
                "adv_hard_all": combine_codes(adversarial["main_hard"], adversarial["skip_hard"]),
                "oracle_clean_main_adv_skips_hard": combine_codes(
                    clean["main_hard"], adversarial["skip_hard"]
                ),
                "oracle_adv_main_clean_skips_hard": combine_codes(
                    adversarial["main_hard"], clean["skip_hard"]
                ),
                "diagnostic_clean_soft_all": combine_codes(
                    clean["main_probability"], clean["skip_probability"]
                ),
                "diagnostic_adv_soft_all": combine_codes(
                    adversarial["main_probability"], adversarial["skip_probability"]
                ),
                "diagnostic_clean_soft_main_hard_skips": combine_codes(
                    clean["main_probability"], clean["skip_hard"]
                ),
                "diagnostic_adv_soft_main_hard_skips": combine_codes(
                    adversarial["main_probability"], adversarial["skip_hard"]
                ),
                "diagnostic_clean_hard_main_soft_skips": combine_codes(
                    clean["main_hard"], clean["skip_probability"]
                ),
                "diagnostic_adv_hard_main_soft_skips": combine_codes(
                    adversarial["main_hard"], adversarial["skip_probability"]
                ),
            }
            for condition, (main_code, skip_codes) in conditions.items():
                reconstruction = model.decode_from_codes(main_code, skip_codes)
                reconstruction = F.interpolate(
                    reconstruction, size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear", align_corners=False,
                )
                save_condition_batch(
                    reconstruction, batch_names, directories[condition], valid_set,
                    p1_root / "clean" / "labels",
                )

    verify_condition_counts(directories, len(valid_set), len(names) - len(valid_set))

    from ultralytics import YOLO
    detector = YOLO(cfg.yolo_ckpt)
    valid_names = [name for name in names if name in valid_set]
    reference_clean = copy_valid_reference(p1_root, "clean", valid_names, output)
    reference_adv = copy_valid_reference(p1_root, "adv", valid_names, output)
    maps = {
        "original_clean": evaluate_map(cfg, detector, reference_clean, output, "original_clean"),
        "original_adv": evaluate_map(cfg, detector, reference_adv, output, "original_adv"),
    }
    for condition, directory in directories.items():
        maps[condition] = evaluate_map(cfg, detector, directory, output, condition)
    derived = {
        condition: compute_recovery(maps, condition)
        for condition in condition_names
    }
    report = {
        "schema": SCHEMA, "stage": "p1", "created_utc": now_utc(),
        "checkpoint_sha256": {
            "robust_v3": sha256_file(Path(cfg.robust_v3)),
            "yolo": sha256_file(Path(cfg.yolo_ckpt)),
        },
        "architecture_v3": architecture_signature(model),
        "shared_manifest": manifest,
        "valid_images": len(valid_names), "excluded_images": len(names) - len(valid_names),
        "condition_semantics": {
            "oracle_*": (
                "Uses clean codes unavailable at deployment; diagnostic only and must not be "
                "reported as a deployable purification result."
            ),
            "diagnostic_*soft*": (
                "Feeds probabilities to a decoder trained on hard codes. This is an OOD "
                "diagnostic and cannot identify an independent causal module loss."
            ),
            "interactions": (
                "Differences between conditions include module interactions and must not be "
                "summed into an exact decomposition of the 0.1215 mAP gap."
            ),
        },
        "bit_statistics": str(bit_stats_path),
        "raw_code_tensor_shards_saved": cfg.save_code_tensors,
        "maps": maps, "derived": derived,
    }
    json_dump(output / "metrics_p1.json", report)
    if cfg.save_detections:
        details = {}
        for condition, directory in {
            "original_clean": reference_clean, "original_adv": reference_adv,
            **directories,
        }.items():
            details[condition] = save_detection_details(
                cfg, detector, directory, valid_names,
                output / "detections" / f"{condition}.jsonl",
            )
        json_dump(output / "detection_detail_index.json", details)
    return report


def parse_args() -> argparse.Namespace:
    defaults = AuditConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("archive", "make-splits", "generate-val-pgd", "p0", "p1", "all"),
    )
    parser.add_argument("--robust-v3", default=defaults.robust_v3)
    parser.add_argument("--expanded-epoch8", default=defaults.expanded_epoch8)
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--p0-shared-root", default=defaults.p0_shared_root)
    parser.add_argument("--p1-shared-root")
    parser.add_argument("--dfire-root", default=defaults.dfire_root)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--repo-root", default=defaults.repo_root)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--map-batch-size", type=int, default=defaults.map_batch_size)
    parser.add_argument("--predict-batch-size", type=int, default=defaults.predict_batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--expected-total", type=int, default=defaults.expected_total)
    parser.add_argument("--expected-valid", type=int, default=defaults.expected_valid)
    parser.add_argument("--expected-excluded", type=int, default=defaults.expected_excluded)
    parser.add_argument("--validation-count", type=int, default=defaults.validation_count)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--attack-eps-px", type=int, default=defaults.attack_eps_px)
    parser.add_argument("--attack-alpha-px", type=int, default=defaults.attack_alpha_px)
    parser.add_argument("--attack-steps", type=int, default=defaults.attack_steps)
    parser.add_argument("--attack-seed", type=int, default=defaults.attack_seed)
    parser.add_argument("--save-code-tensors", action="store_true")
    parser.add_argument("--no-save-detections", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--allow-p1-on-p0-data", action="store_true")
    parser.add_argument("--equivalence-atol", type=float, default=defaults.equivalence_atol)
    parser.add_argument(
        "--source-file", action="append", default=[],
        help="Additional exact Colab definition/evaluation source to archive; repeatable",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        robust_v3=args.robust_v3,
        expanded_epoch8=args.expanded_epoch8,
        yolo_ckpt=args.yolo_ckpt,
        p0_shared_root=args.p0_shared_root,
        p1_shared_root=args.p1_shared_root,
        dfire_root=args.dfire_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
        batch_size=args.batch_size,
        map_batch_size=args.map_batch_size,
        predict_batch_size=args.predict_batch_size,
        num_workers=args.num_workers,
        expected_total=args.expected_total,
        expected_valid=args.expected_valid,
        expected_excluded=args.expected_excluded,
        validation_count=args.validation_count,
        split_seed=args.split_seed,
        attack_eps_px=args.attack_eps_px,
        attack_alpha_px=args.attack_alpha_px,
        attack_steps=args.attack_steps,
        attack_seed=args.attack_seed,
        save_code_tensors=args.save_code_tensors,
        save_detections=not args.no_save_detections,
        allow_count_mismatch=args.allow_count_mismatch,
        allow_p1_on_p0_data=args.allow_p1_on_p0_data,
        equivalence_atol=args.equivalence_atol,
        source_files=args.source_file,
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    json_dump(Path(cfg.output_root) / "last_invocation.json", {
        "schema": SCHEMA, "started_utc": now_utc(), "stage": args.stage,
        "config": asdict(cfg),
    })
    if args.stage in ("archive", "all"):
        run_archive(cfg)
    if args.stage in ("make-splits", "all"):
        run_make_splits(cfg)
    if args.stage == "generate-val-pgd":
        run_generate_validation_pgd(cfg)
    if args.stage in ("p0", "all"):
        run_p0(cfg)
    if args.stage in ("p1", "all"):
        run_p1(cfg)
    print(f"Completed stage={args.stage}; outputs={cfg.output_root}", flush=True)


if __name__ == "__main__":
    main()
