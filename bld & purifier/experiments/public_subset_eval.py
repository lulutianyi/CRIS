"""Evaluate the notebook-defined GCN/HF-AE on CIFAR-10 or COCO subsets.

This file deliberately reuses the model definitions already executed in
``gcn.ipynb``.  In Colab run it with ``%run -i`` so that names such as ``bld``,
``BinaryAutoEncoderAttn`` and ``BinaryAutoEncoderGCN`` are visible here.

Examples
--------
CIFAR-10::

    %run -i experiments/public_subset_eval.py \
        --dataset cifar10 --subset-size 1000 --download \
        --checkpoint /content/drive/MyDrive/.../gcn_skip_learned_v2_hf_taskaware_best.pt \
        --model-name hf_ae --output-dir /content/public_subset_eval

COCO reconstruction and detection::

    %run -i experiments/public_subset_eval.py \
        --dataset coco --subset-size 500 \
        --coco-images /content/coco/val2017 \
        --coco-annotations /content/coco/annotations/instances_val2017.json \
        --coco-detection --coco-yolo yolov8n.pt \
        --checkpoint /content/drive/MyDrive/.../gcn_skip_learned_v2_hf_taskaware_best.pt \
        --model-name hf_ae --output-dir /content/public_subset_eval
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


NOTEBOOK_NAMES = (
    "bld",
    "BinaryAutoEncoderAttn",
    "BinaryAutoEncoderGCN",
    "GCNSkipAEConfig",
    "GCNBlockConfig",
    "auto_detect_skip_levels",
)


def require_notebook_context() -> None:
    missing = [name for name in NOTEBOOK_NAMES if name not in globals()]
    if missing:
        raise RuntimeError(
            "缺少gcn.ipynb模型定义: "
            + ", ".join(missing)
            + "。请先运行notebook的模型定义单元，再用 `%run -i` 执行本脚本。"
        )


def unwrap_state_dict(raw):
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def build_notebook_gcn_ae(checkpoint: str):
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


def highpass(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = kernel_size // 2
    blur = F.avg_pool2d(
        F.pad(x, (padding,) * 4, mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )
    return x - blur


def ssim_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dependency-free global SSIM, used only for like-for-like comparisons."""
    dims = (1, 2, 3)
    mu_x = x.mean(dim=dims)
    mu_y = y.mean(dim=dims)
    var_x = ((x - mu_x[:, None, None, None]) ** 2).mean(dim=dims)
    var_y = ((y - mu_y[:, None, None, None]) ** 2).mean(dim=dims)
    covariance = (
        (x - mu_x[:, None, None, None])
        * (y - mu_y[:, None, None, None])
    ).mean(dim=dims)
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1)
        * (var_x + var_y + c2)
    )


class MetricAccumulator:
    def __init__(self):
        self.count = 0
        self.psnr = 0.0
        self.ssim = 0.0
        self.mae = 0.0
        self.highpass_psnr = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)
        mse = (prediction - target).square().mean(dim=(1, 2, 3))
        hp_mse = (
            highpass(prediction) - highpass(target)
        ).square().mean(dim=(1, 2, 3))
        batch = prediction.shape[0]
        self.count += batch
        self.psnr += (-10 * torch.log10(mse + 1e-12)).sum().item()
        self.highpass_psnr += (
            -10 * torch.log10(hp_mse + 1e-12)
        ).sum().item()
        self.ssim += ssim_per_image(prediction, target).sum().item()
        self.mae += (
            (prediction - target).abs().mean(dim=(1, 2, 3)).sum().item()
        )

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "psnr": self.psnr / self.count,
            "ssim": self.ssim / self.count,
            "mae": self.mae / self.count,
            "highpass_psnr": self.highpass_psnr / self.count,
        }


class CifarSubset(Dataset):
    def __init__(self, root: str, indices: Sequence[int], download: bool):
        from torchvision.datasets import CIFAR10

        self.dataset = CIFAR10(
            root=root, train=False, download=download, transform=T.ToTensor()
        )
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return image, label, self.indices[index]


def stratified_cifar_indices(root: str, size: int, seed: int, download: bool):
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=root, train=False, download=download)
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {index: [] for index in range(10)}
    for index, target in enumerate(dataset.targets):
        by_class[int(target)].append(index)
    for values in by_class.values():
        rng.shuffle(values)
    selected = []
    base, remainder = divmod(min(size, len(dataset)), 10)
    for class_id in range(10):
        take = base + (1 if class_id < remainder else 0)
        selected.extend(by_class[class_id][:take])
    rng.shuffle(selected)
    return selected


def save_comparison_grid(
    target: torch.Tensor,
    reconstructed: torch.Tensor,
    path: Path,
    maximum: int = 32,
) -> None:
    from torchvision.utils import make_grid

    count = min(maximum, target.shape[0])
    paired = torch.stack(
        [item for i in range(count) for item in (target[i], reconstructed[i])]
    )
    grid = make_grid(paired, nrow=8, padding=2)
    T.ToPILImage()(grid.cpu()).save(path)


@torch.inference_mode()
def evaluate_cifar(args, model) -> Dict[str, float]:
    indices = stratified_cifar_indices(
        args.cifar_root, args.subset_size, args.seed, args.download
    )
    loader = DataLoader(
        CifarSubset(args.cifar_root, indices, args.download),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    metrics_128 = MetricAccumulator()
    metrics_native = MetricAccumulator()
    first_pair = None
    for native, _labels, _indices in tqdm(loader, desc="CIFAR-10 subset"):
        native = native.to(bld.DEVICE, non_blocking=True)
        target128 = F.interpolate(
            native, size=(128, 128), mode="bilinear", align_corners=False
        )
        reconstructed128, _ = model(target128)
        reconstructed_native = F.interpolate(
            reconstructed128,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        metrics_128.update(reconstructed128, target128)
        metrics_native.update(reconstructed_native, native)
        if first_pair is None:
            first_pair = (target128.cpu(), reconstructed128.cpu())
    if first_pair is not None:
        save_comparison_grid(
            first_pair[0],
            first_pair[1],
            args.run_dir / "cifar10_original_reconstructed.png",
        )
    return {
        "dataset": "cifar10",
        "subset_size": len(indices),
        "seed": args.seed,
        "metrics_128": metrics_128.summary(),
        "metrics_native_32": metrics_native.summary(),
    }


class CocoSubset(Dataset):
    def __init__(self, coco, image_root: Path, image_ids: Sequence[int]):
        self.coco = coco
        self.image_root = image_root
        self.image_ids = list(image_ids)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        path = self.image_root / info["file_name"]
        image = self.to_tensor(Image.open(path).convert("RGB"))
        target128 = F.interpolate(
            image.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return target128, image_id, info["file_name"], info["height"], info["width"]


def coco_collate(batch):
    target128, image_ids, names, heights, widths = zip(*batch)
    return (
        torch.stack(target128),
        list(image_ids),
        list(names),
        list(heights),
        list(widths),
    )


def select_coco_ids(coco, size: int, seed: int) -> List[int]:
    image_ids = [
        image_id
        for image_id in coco.getImgIds()
        if len(coco.getAnnIds(imgIds=[image_id], iscrowd=None)) > 0
    ]
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return image_ids[: min(size, len(image_ids))]


@torch.inference_mode()
def reconstruct_coco(args, model):
    from pycocotools.coco import COCO

    coco = COCO(str(args.coco_annotations))
    image_ids = select_coco_ids(coco, args.subset_size, args.seed)
    dataset = CocoSubset(coco, args.coco_images, image_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=coco_collate,
    )
    reconstruction_dir = args.run_dir / "coco_reconstructed"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricAccumulator()
    reconstructed_paths: List[str] = []
    original_paths: List[str] = []
    ordered_image_ids: List[int] = []
    for target128, batch_ids, names, heights, widths in tqdm(
        loader, desc="COCO subset"
    ):
        target128 = target128.to(bld.DEVICE, non_blocking=True)
        reconstructed128, _ = model(target128)
        metrics.update(reconstructed128, target128)
        for index, (image_id, name, height, width) in enumerate(
            zip(batch_ids, names, heights, widths)
        ):
            reconstructed_native = F.interpolate(
                reconstructed128[index : index + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            output_path = reconstruction_dir / f"{Path(name).stem}.png"
            T.ToPILImage()(reconstructed_native.cpu().clamp(0, 1)).save(
                output_path, format="PNG", compress_level=3
            )
            reconstructed_paths.append(str(output_path))
            original_paths.append(str(args.coco_images / name))
            ordered_image_ids.append(int(image_id))
    summary = {
        "dataset": "coco",
        "subset_size": len(image_ids),
        "seed": args.seed,
        "metrics_128": metrics.summary(),
    }
    return coco, ordered_image_ids, original_paths, reconstructed_paths, summary


def predict_coco(
    detector,
    paths: Sequence[str],
    image_ids: Sequence[int],
    coco,
    batch_size: int,
):
    categories = coco.loadCats(coco.getCatIds())
    category_by_name = {category["name"]: category["id"] for category in categories}
    model_names = detector.names
    if isinstance(model_names, list):
        model_names = dict(enumerate(model_names))
    predictions = []
    results = detector.predict(
        source=list(paths),
        imgsz=640,
        batch=batch_size,
        conf=0.001,
        iou=0.7,
        verbose=False,
        stream=True,
    )
    for image_id, result in zip(image_ids, results):
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.detach().cpu()
        confidence = result.boxes.conf.detach().cpu()
        classes = result.boxes.cls.detach().cpu().long()
        for box, score, class_index in zip(xyxy, confidence, classes):
            class_name = model_names[int(class_index)]
            category_id = category_by_name.get(class_name)
            if category_id is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return predictions


def coco_map(coco, image_ids: Sequence[int], predictions) -> Dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
    }


def evaluate_coco(args, model) -> Dict:
    coco, image_ids, originals, reconstructed, summary = reconstruct_coco(
        args, model
    )
    if args.coco_detection:
        from ultralytics import YOLO

        # COCO reconstruction and 640px detector inference do not need to share
        # GPU memory.  Moving the AE away first avoids OOM on a 15 GB T4.
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        detector = YOLO(args.coco_yolo)
        original_predictions = predict_coco(
            detector,
            originals,
            image_ids,
            coco,
            batch_size=args.coco_det_batch,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reconstructed_predictions = predict_coco(
            detector,
            reconstructed,
            image_ids,
            coco,
            batch_size=args.coco_det_batch,
        )
        summary["coco_detection"] = {
            "detector": args.coco_yolo,
            "original": coco_map(coco, image_ids, original_predictions),
            "reconstructed": coco_map(
                coco, image_ids, reconstructed_predictions
            ),
        }
        original_map = summary["coco_detection"]["original"]
        reconstructed_map = summary["coco_detection"]["reconstructed"]
        summary["coco_detection"]["retention"] = {
            "map50": reconstructed_map["map50"] / max(original_map["map50"], 1e-12),
            "map50_95": reconstructed_map["map50_95"]
            / max(original_map["map50_95"], 1e-12),
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "coco"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="hf_ae")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/public_subset_eval"))
    parser.add_argument("--cifar-root", default="/content/cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--coco-images", type=Path)
    parser.add_argument("--coco-annotations", type=Path)
    parser.add_argument("--coco-detection", action="store_true")
    parser.add_argument("--coco-yolo", default="yolov8n.pt")
    parser.add_argument("--coco-det-batch", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_dir = args.output_dir / f"{args.model_name}_{args.dataset}_n{args.subset_size}_seed{args.seed}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "coco" and (
        args.coco_images is None or args.coco_annotations is None
    ):
        raise ValueError("COCO评测必须提供--coco-images和--coco-annotations")
    model = build_notebook_gcn_ae(args.checkpoint)
    if args.dataset == "cifar10":
        summary = evaluate_cifar(args, model)
    else:
        summary = evaluate_coco(args, model)
    summary["model_name"] = args.model_name
    summary["checkpoint"] = args.checkpoint
    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== 公共数据集子集评测 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果：", summary_path)


if __name__ == "__main__":
    main()
