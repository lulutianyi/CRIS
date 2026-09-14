"""Lightweight ZeroPur evaluation on D-Fire shared-PGD images.

The two optimization stages follow ZeroPur (arXiv:2406.03143): Guided Shift
and Adaptive Projection.  The paper's classifier features are replaced by
frozen YOLOv8 detector features for the D-Fire detection adaptation.

Reused from existing local modules:
  * shared-PGD manifest/dataset, image saving, YOLO loading
  * image-list based Ultralytics mAP evaluation

This script performs evaluation only.  It never trains or overwrites weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
for module_dir in (HERE, HERE.parent / "core"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import freqpure_dfire as map_utils  # noqa: E402
import shared_pgd_gcn_flowpure_compare as shared  # noqa: E402


@dataclass
class Config:
    shared_root: str = "/content/drive/MyDrive/dfire_shared_pgd_png_eps8_seed20260822"
    output_root: str = "/content/drive/MyDrive/zeropur_dfire_eval"
    yolo_ckpt: str = "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    max_images: int = 32
    subset_seed: int = 20260912
    batch_size: int = 1
    detector_batch_size: int = 8
    num_workers: int = 0
    feature_layers: Tuple[int, ...] = (4, 6, 9)
    blur_sigma: float = 1.8
    purification_eps: float = 10 / 255
    guided_steps: int = 10
    guided_step_size: float = 1 / 255
    projection_steps: int = 50
    projection_weight: float = 1e-4
    include_clean: bool = False


def read_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def select_names(cfg: Config) -> Tuple[Dict, List[str]]:
    manifest = read_json(Path(cfg.shared_root) / "manifest.json")
    names = [str(name) for name in manifest.get("names", [])]
    if not names:
        raise RuntimeError("Shared-PGD manifest has no image names")
    count = len(names) if cfg.max_images == 0 else cfg.max_images
    if count > len(names):
        raise ValueError(f"Requested {count} images, manifest contains {len(names)}")
    selected = random.Random(cfg.subset_seed).sample(names, count)
    for split in ("clean", "adv"):
        directory = Path(cfg.shared_root) / split / "images"
        missing = [name for name in selected if not (directory / f"{name}.png").is_file()]
        if missing:
            raise RuntimeError(f"Missing {split} image: {missing[0]}")
    return manifest, selected


def gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinate.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel[:, None] * kernel[None, :]
    return kernel_2d[None, None]


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, images.device, images.dtype)
    kernel = kernel.expand(images.shape[1], 1, -1, -1)
    padding = kernel.shape[-1] // 2
    padded = F.pad(images, (padding,) * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=images.shape[1])


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    return None


class YOLOFeatures:
    """Capture selected frozen YOLO feature maps while preserving input gradients."""

    def __init__(self, model: nn.Module, layer_indices: Sequence[int]):
        self.model = model
        modules = getattr(model, "model", None)
        if modules is None:
            raise RuntimeError("YOLO model has no indexable .model layers")
        self.indices = tuple(layer_indices)
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= len(modules):
            raise ValueError(f"Invalid feature layers {self.indices}; YOLO has {len(modules)} layers")
        self.values: Dict[int, torch.Tensor] = {}
        self.handles = [
            modules[index].register_forward_hook(self._hook(index))
            for index in self.indices
        ]

    def _hook(self, index: int):
        def save(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise RuntimeError(f"YOLO layer {index} did not return a tensor")
            self.values[index] = tensor
        return save

    def __call__(self, images: torch.Tensor) -> List[torch.Tensor]:
        self.values.clear()
        _ = self.model(images)
        missing = [index for index in self.indices if index not in self.values]
        if missing:
            raise RuntimeError(f"YOLO hooks did not fire: {missing}")
        return [self.values[index] for index in self.indices]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def cosine_embedding(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.adaptive_avg_pool2d(left, 1).flatten(1)
    right = F.adaptive_avg_pool2d(right, 1).flatten(1)
    return F.cosine_similarity(left, right, dim=1).mean()


def normalized_feature_distance(
    current: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]
) -> torch.Tensor:
    per_layer = []
    for current_layer, reference_layer in zip(current, reference):
        height, width = current_layer.shape[-2:]
        current_norm = current_layer / (
            current_layer.square().sum(1, keepdim=True).sqrt() + 1e-10
        ) / math.sqrt(height * width)
        reference_norm = reference_layer / (
            reference_layer.square().sum(1, keepdim=True).sqrt() + 1e-10
        ) / math.sqrt(height * width)
        per_layer.append((current_norm - reference_norm).flatten(1).square().sum(1))
    return torch.stack(per_layer).sum(0).clamp_min(1e-12).sqrt().mean()


class ZeroPurYOLO:
    def __init__(self, detector, cfg: Config):
        self.cfg = cfg
        self.model = detector.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.features = YOLOFeatures(self.model, cfg.feature_layers)

    def guided_shift(self, inputs: torch.Tensor) -> torch.Tensor:
        lower = (inputs - self.cfg.purification_eps).clamp(0, 1)
        upper = (inputs + self.cfg.purification_eps).clamp(0, 1)
        value = inputs.detach().clone()
        momentum = torch.zeros_like(value)
        for _ in range(self.cfg.guided_steps):
            value.requires_grad_(True)
            blurred = gaussian_blur(value.detach(), self.cfg.blur_sigma)
            with torch.no_grad():
                blurred_feature = self.features(blurred)[-1].detach()
            current_feature = self.features(value)[-1]
            loss = -cosine_embedding(current_feature, blurred_feature)
            gradient = torch.autograd.grad(loss, value, only_inputs=True)[0]
            gradient = gradient / gradient.abs().sum().clamp_min(1e-12)
            momentum = gradient.detach() + momentum
            value = value.detach() - self.cfg.guided_step_size * momentum.sign()
            value = torch.maximum(torch.minimum(value, upper), lower).clamp(0, 1)
        return value.detach()

    def adaptive_projection(
        self, original: torch.Tensor, guided: torch.Tensor
    ) -> torch.Tensor:
        lower = (original - self.cfg.purification_eps).clamp(0, 1)
        upper = (original + self.cfg.purification_eps).clamp(0, 1)
        with torch.no_grad():
            original_features = [value.detach() for value in self.features(original)]
            guided_features = [value.detach() for value in self.features(guided)]
        value = original.detach().clone()
        step_size = self.cfg.purification_eps / max(self.cfg.projection_steps, 1)
        for _ in range(self.cfg.projection_steps):
            value.requires_grad_(True)
            current_features = self.features(value)
            projection = value.new_zeros(())
            for coarse, current, initial in zip(
                guided_features, current_features, original_features
            ):
                direction = (coarse - initial).flatten(1)
                displacement = (current - initial).flatten(1)
                projection = projection - (direction * displacement).sum(1).mean()
            projection = projection / len(current_features)
            perceptual = normalized_feature_distance(current_features, original_features)
            loss = self.cfg.projection_weight * projection + perceptual
            gradient = torch.autograd.grad(loss, value, only_inputs=True)[0]
            value = value.detach() - step_size * gradient.sign()
            value = torch.maximum(torch.minimum(value, upper), lower).clamp(0, 1)
        return value.detach()

    def __call__(self, images: torch.Tensor):
        guided = self.guided_shift(images)
        projected = self.adaptive_projection(images, guided)
        return guided, projected

    def close(self):
        self.features.close()


def output_splits(cfg: Config) -> Tuple[str, ...]:
    splits = ("adv_blur", "adv_guided_shift", "adv_zeropur")
    return splits + (("clean_zeropur",) if cfg.include_clean else ())


def output_complete(root: Path, split: str, names: Sequence[str]) -> bool:
    image_dir = root / split / "images"
    return all((image_dir / f"{name}.png").is_file() for name in names)


def run_root(cfg: Config, names: Sequence[str]) -> Path:
    crc = zlib.crc32("|".join(names).encode("utf-8"))
    mode = "withclean" if cfg.include_clean else "advonly"
    tag = (
        f"zeropur_yolo_{mode}_n{len(names)}_crc{crc:08x}_"
        f"eps{cfg.purification_eps * 255:g}_g{cfg.guided_steps}_"
        f"p{cfg.projection_steps}_seed{cfg.subset_seed}"
    )
    root = Path(cfg.output_root) / tag
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "zeropur_dfire_yolo_adaptation_v1",
        "config": asdict(cfg),
        "selected_names": list(names),
        "adaptation_note": (
            "ZeroPur GS/AP equations with frozen YOLOv8 intermediate features; "
            "the paper's original classifier-specific feature wrapper is replaced."
        ),
    }
    path = root / "run_config.json"
    if path.exists() and read_json(path) != identity:
        raise RuntimeError(f"Existing run has different configuration: {path}")
    write_json(path, identity)
    return root


def generate(cfg: Config, names: Sequence[str], root: Path, device) -> None:
    detector = shared.YOLO(cfg.yolo_ckpt)
    purifier = ZeroPurYOLO(detector, cfg)
    pair_cfg = shared.CompareConfig(
        shared_root=cfg.shared_root,
        purify_batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    loader = shared.make_shared_loader(
        pair_cfg, {"count": len(names), "names": list(names)}, device
    )
    labels = str(Path(cfg.shared_root) / "clean" / "labels")
    started = time.perf_counter()
    processed = 0
    try:
        for clean, adv, batch_names in tqdm(loader, desc="ZeroPur D-Fire"):
            required = output_splits(cfg)
            if all(output_complete(root, split, batch_names) for split in required):
                processed += len(batch_names)
                continue
            clean = clean.to(device, non_blocking=True)
            adv = adv.to(device, non_blocking=True)
            blurred = gaussian_blur(adv, cfg.blur_sigma)
            guided, purified = purifier(adv)
            generated = {
                "adv_blur": blurred,
                "adv_guided_shift": guided,
                "adv_zeropur": purified,
            }
            if cfg.include_clean:
                _clean_guided, generated["clean_zeropur"] = purifier(clean)
            for split, images in generated.items():
                if not output_complete(root, split, batch_names):
                    shared.save_output_batch(images, batch_names, str(root / split), labels)
            processed += len(batch_names)
            elapsed = time.perf_counter() - started
            if processed == len(batch_names) or processed % 20 == 0:
                rate = processed / max(elapsed, 1e-9)
                eta = (len(names) - processed) / max(rate, 1e-9)
                print(f"processed={processed}/{len(names)}, {rate:.3f} img/s, ETA={eta/60:.1f} min")
    finally:
        purifier.close()
        del purifier, detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify(cfg: Config, names: Sequence[str], root: Path) -> None:
    expected = set(names)
    for split in output_splits(cfg):
        actual = {path.stem for path in (root / split / "images").glob("*.png")}
        print(f"{split:24s} images={len(actual & expected)}/{len(expected)}")
        if actual != expected:
            raise RuntimeError(
                f"{split} mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}"
            )


def evaluate(cfg: Config, manifest: Dict, names: Sequence[str], root: Path, device):
    verify(cfg, names, root)
    detector = shared.YOLO(cfg.yolo_ckpt)
    eval_cfg = map_utils.FreqPureDFireConfig(
        shared_root=cfg.shared_root,
        yolo_ckpt=cfg.yolo_ckpt,
        detector_size=384,
        detector_batch_size=cfg.detector_batch_size,
    )
    shared_root = Path(cfg.shared_root)
    directories = {
        "original_clean": [shared_root / "clean" / "images" / f"{name}.png" for name in names],
        "adv_original": [shared_root / "adv" / "images" / f"{name}.png" for name in names],
    }
    for split in output_splits(cfg):
        directories[split] = [root / split / "images" / f"{name}.png" for name in names]

    results = {}
    for split, paths in directories.items():
        image_list = map_utils.write_image_list(root / "image_lists" / f"{split}.txt", paths)
        map50, map50_95 = map_utils.eval_map(detector, image_list, eval_cfg, root, split, device)
        results[split] = {"map50": map50, "map50_95": map50_95}
        print(f"{split:24s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")

    clean, adv = results["original_clean"], results["adv_original"]
    def recovery(name: str, metric: str) -> float:
        return (results[name][metric] - adv[metric]) / max(
            clean[metric] - adv[metric], 1e-8
        )
    derived = {}
    for split in ("adv_blur", "adv_guided_shift", "adv_zeropur"):
        derived[f"{split}_recovery_map50"] = recovery(split, "map50")
        derived[f"{split}_recovery_map50_95"] = recovery(split, "map50_95")
    if cfg.include_clean:
        derived["clean_zeropur_drop_map50"] = (
            clean["map50"] - results["clean_zeropur"]["map50"]
        )
    print("\n=== ZeroPur D-Fire shared-PGD evaluation ===")
    print(
        f"Guided Shift recovery: mAP50={derived['adv_guided_shift_recovery_map50']:.1%}, "
        f"mAP50-95={derived['adv_guided_shift_recovery_map50_95']:.1%}"
    )
    print(
        f"Full ZeroPur recovery: mAP50={derived['adv_zeropur_recovery_map50']:.1%}, "
        f"mAP50-95={derived['adv_zeropur_recovery_map50_95']:.1%}"
    )
    report = {
        "schema": "zeropur_dfire_yolo_adaptation_v1",
        "config": asdict(cfg),
        "shared_attack_manifest": manifest,
        "selected_names": list(names),
        "results": results,
        "derived": derived,
        "method_note": (
            "Training-free ZeroPur GS/AP adapted from classification to object "
            "detection using frozen YOLOv8 feature layers."
        ),
    }
    path = root / "metrics_zeropur_dfire.json"
    write_json(path, report)
    print("Metrics saved:", path)


def parse_args():
    defaults = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("purify", "map", "all"), default="all")
    parser.add_argument("--shared-root", default=defaults.shared_root)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--max-images", type=int, default=defaults.max_images)
    parser.add_argument("--subset-seed", type=int, default=defaults.subset_seed)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--detector-batch-size", type=int, default=defaults.detector_batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--feature-layers", default=",".join(map(str, defaults.feature_layers)))
    parser.add_argument("--blur-sigma", type=float, default=defaults.blur_sigma)
    parser.add_argument("--purification-eps", type=float, default=defaults.purification_eps)
    parser.add_argument("--guided-steps", type=int, default=defaults.guided_steps)
    parser.add_argument("--guided-step-size", type=float, default=defaults.guided_step_size)
    parser.add_argument("--projection-steps", type=int, default=defaults.projection_steps)
    parser.add_argument("--projection-weight", type=float, default=defaults.projection_weight)
    parser.add_argument("--include-clean", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)
    cfg = Config(
        shared_root=args.shared_root,
        output_root=args.output_root,
        yolo_ckpt=args.yolo_ckpt,
        max_images=args.max_images,
        subset_seed=args.subset_seed,
        batch_size=args.batch_size,
        detector_batch_size=args.detector_batch_size,
        num_workers=args.num_workers,
        feature_layers=tuple(int(value) for value in args.feature_layers.split(",")),
        blur_sigma=args.blur_sigma,
        purification_eps=args.purification_eps,
        guided_steps=args.guided_steps,
        guided_step_size=args.guided_step_size,
        projection_steps=args.projection_steps,
        projection_weight=args.projection_weight,
        include_clean=args.include_clean,
    )
    if cfg.max_images < 0 or cfg.batch_size < 1:
        parser.error("--max-images must be >= 0 and --batch-size must be >= 1")
    return args.stage, cfg


def main():
    stage, cfg = parse_args()
    if not Path(cfg.yolo_ckpt).is_file():
        raise FileNotFoundError(cfg.yolo_ckpt)
    manifest, names = select_names(cfg)
    root = run_root(cfg, names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Selected images:", len(names))
    print("Output:", root)
    if stage in ("purify", "all"):
        generate(cfg, names, root, device)
        verify(cfg, names, root)
    if stage in ("map", "all"):
        evaluate(cfg, manifest, names, root, device)


if __name__ == "__main__":
    main()
