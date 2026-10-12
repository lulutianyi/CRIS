"""COCO resize-only control for the 128x128 HF-AE pipeline.

The script evaluates the same seeded COCO subset before and after
``native -> 128x128 -> native`` bilinear resizing.  No AE checkpoint or
notebook model definition is required.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm


def highpass(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = kernel_size // 2
    blur = F.avg_pool2d(
        F.pad(x, (padding,) * 4, mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )
    return x - blur


def global_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
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
    value = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1)
        * (var_x + var_y + c2)
    )
    return float(value.mean())


class NativeMetricAccumulator:
    def __init__(self):
        self.count = 0
        self.psnr = 0.0
        self.ssim = 0.0
        self.mae = 0.0
        self.highpass_psnr = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        mse = (prediction - target).square().mean()
        hp_mse = (highpass(prediction) - highpass(target)).square().mean()
        self.count += 1
        self.psnr += float(-10 * torch.log10(mse + 1e-12))
        self.highpass_psnr += float(-10 * torch.log10(hp_mse + 1e-12))
        self.ssim += global_ssim(prediction, target)
        self.mae += float((prediction - target).abs().mean())

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "psnr": self.psnr / self.count,
            "ssim": self.ssim / self.count,
            "mae": self.mae / self.count,
            "highpass_psnr": self.highpass_psnr / self.count,
        }


def select_coco_ids(coco, size: int, seed: int) -> List[int]:
    image_ids = [
        image_id
        for image_id in coco.getImgIds()
        if len(coco.getAnnIds(imgIds=[image_id], iscrowd=None)) > 0
    ]
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return image_ids[: min(size, len(image_ids))]


def create_resize_only_images(args, coco, image_ids: Sequence[int]):
    output_dir = args.run_dir / "resize_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = T.ToTensor()
    metrics = NativeMetricAccumulator()
    original_paths: List[str] = []
    resized_paths: List[str] = []

    for image_id in tqdm(image_ids, desc="COCO resize-only"):
        info = coco.loadImgs([image_id])[0]
        original_path = args.coco_images / info["file_name"]
        original = to_tensor(Image.open(original_path).convert("RGB")).unsqueeze(0)
        reduced = F.interpolate(
            original,
            size=(args.ae_size, args.ae_size),
            mode="bilinear",
            align_corners=False,
        )
        restored = F.interpolate(
            reduced,
            size=original.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        metrics.update(restored, original)
        output_path = output_dir / f"{Path(info['file_name']).stem}.png"
        T.ToPILImage()(restored.squeeze(0)).save(
            output_path, format="PNG", compress_level=3
        )
        original_paths.append(str(original_path))
        resized_paths.append(str(output_path))

    return original_paths, resized_paths, metrics.summary()


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
            category_id = category_by_name.get(model_names[int(class_index)])
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
        "map_small": float(evaluator.stats[3]),
        "map_medium": float(evaluator.stats[4]),
        "map_large": float(evaluator.stats[5]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-images", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--ae-size", type=int, default=128)
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--det-batch", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/content/coco_resize_only_eval")
    )
    return parser.parse_args()


def main():
    from pycocotools.coco import COCO
    from ultralytics import YOLO

    args = parse_args()
    args.run_dir = (
        args.output_dir
        / f"coco_resize{args.ae_size}_n{args.subset_size}_seed{args.seed}"
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(args.coco_annotations))
    image_ids = select_coco_ids(coco, args.subset_size, args.seed)
    originals, resized, reconstruction_metrics = create_resize_only_images(
        args, coco, image_ids
    )
    detector = YOLO(args.detector)
    original_predictions = predict_coco(
        detector, originals, image_ids, coco, args.det_batch
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resized_predictions = predict_coco(
        detector, resized, image_ids, coco, args.det_batch
    )
    original_map = coco_map(coco, image_ids, original_predictions)
    resize_map = coco_map(coco, image_ids, resized_predictions)
    summary = {
        "dataset": "coco",
        "method": f"resize_only_{args.ae_size}",
        "subset_size": len(image_ids),
        "seed": args.seed,
        "native_reconstruction_metrics": reconstruction_metrics,
        "detection": {
            "detector": args.detector,
            "original": original_map,
            "resize_only": resize_map,
            "retention": {
                "map50": resize_map["map50"] / max(original_map["map50"], 1e-12),
                "map50_95": resize_map["map50_95"]
                / max(original_map["map50_95"], 1e-12),
            },
        },
    }
    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== COCO resize-only对照 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果：", summary_path)


if __name__ == "__main__":
    main()
