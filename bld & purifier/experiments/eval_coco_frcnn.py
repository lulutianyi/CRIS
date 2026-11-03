"""Evaluate official Faster R-CNN v2 on clean or manifest-backed COCO images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from pycocotools.coco import COCO
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import (  # noqa: E402
    CocoFixedSubset,
    atomic_write_json,
    evaluate_coco_predictions,
    load_manifest,
    load_victim,
    predictions_to_coco,
    select_image_ids,
)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(str(args.annotations))

    replacement_paths = None
    if args.manifest:
        manifest = load_manifest(args.manifest)
        samples = manifest["samples"]
        image_ids = [int(sample["image_id"]) for sample in samples]
        if args.source == "adversarial":
            replacement_paths = {
                int(sample["image_id"]): sample["adversarial_path"]
                for sample in samples
            }
        source_name = args.source
    else:
        if args.source == "adversarial":
            raise ValueError("--source adversarial 必须同时提供 --manifest")
        image_ids = select_image_ids(
            coco, args.subset_size, args.seed
        )
        source_name = "clean"

    dataset = CocoFixedSubset(
        args.images,
        args.annotations,
        image_ids,
        replacement_paths=replacement_paths,
    )
    model, weights = load_victim(device)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for image, target, _path, _name in tqdm(
            dataset, desc=f"Faster R-CNN {source_name}"
        ):
            image = image.to(device)
            prediction = model([image])[0]
            predictions.extend(
                predictions_to_coco(
                    int(target["image_id"].item()),
                    prediction,
                    score_threshold=args.score_threshold,
                )
            )

    metrics = evaluate_coco_predictions(coco, image_ids, predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{source_name}_predictions.json"
    atomic_write_json(prediction_path, predictions)
    summary = {
        "source": source_name,
        "subset_size": len(image_ids),
        "image_ids": image_ids,
        "seed": args.seed,
        "victim": "fasterrcnn_resnet50_fpn_v2",
        "weights": str(weights),
        "manifest": str(args.manifest) if args.manifest else None,
        "score_threshold": args.score_threshold,
        "metrics": metrics,
        "predictions": str(prediction_path),
    }
    summary_path = output_dir / f"{source_name}_summary.json"
    atomic_write_json(summary_path, summary)
    print("\n========== Faster R-CNN COCO评测 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果:", summary_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--source", choices=("clean", "adversarial"), default="clean"
    )
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
