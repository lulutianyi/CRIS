"""Shared COCO/Faster R-CNN utilities for adversarial-purification experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


def load_victim(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights


def valid_image_ids(coco: COCO):
    valid = {
        int(annotation["image_id"])
        for annotation in coco.dataset["annotations"]
        if annotation.get("iscrowd", 0) == 0
        and annotation.get("bbox", [0, 0, 0, 0])[2] > 1
        and annotation.get("bbox", [0, 0, 0, 0])[3] > 1
    }
    return [int(image_id) for image_id in coco.getImgIds() if image_id in valid]


def select_image_ids(coco: COCO, subset_size: int, seed: int):
    image_ids = valid_image_ids(coco)
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return image_ids[: min(subset_size, len(image_ids))]


def target_from_coco(coco: COCO, image_id: int, width: int, height: int):
    boxes = []
    labels = []
    areas = []
    crowds = []
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        x, y, box_width, box_height = annotation["bbox"]
        x1 = max(0.0, min(float(x), float(width)))
        y1 = max(0.0, min(float(y), float(height)))
        x2 = max(0.0, min(float(x + box_width), float(width)))
        y2 = max(0.0, min(float(y + box_height), float(height)))
        if x2 <= x1 + 1e-6 or y2 <= y1 + 1e-6:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(int(annotation["category_id"]))
        areas.append(float((x2 - x1) * (y2 - y1)))
        crowds.append(0)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.tensor(crowds, dtype=torch.int64),
    }


class CocoFixedSubset(Dataset):
    def __init__(
        self,
        image_root,
        annotation_path,
        image_ids,
        replacement_paths=None,
    ):
        self.image_root = Path(image_root)
        self.coco = COCO(str(annotation_path))
        self.image_ids = [int(image_id) for image_id in image_ids]
        self.replacement_paths = {
            int(key): Path(value)
            for key, value in (replacement_paths or {}).items()
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        path = self.replacement_paths.get(
            image_id, self.image_root / info["file_name"]
        )
        image = TF.to_tensor(Image.open(path).convert("RGB"))
        target = target_from_coco(
            self.coco, image_id, info["width"], info["height"]
        )
        return image, target, str(path), info["file_name"]


def target_to_device(target, device, attack_only=False):
    if attack_only:
        keys = ("boxes", "labels")
    else:
        keys = target.keys()
    return {
        key: target[key].to(device)
        for key in keys
        if torch.is_tensor(target[key])
    }


def predictions_to_coco(image_id, prediction, score_threshold=0.001):
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    rows = []
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append({
            "image_id": int(image_id),
            "category_id": int(label),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return rows


def evaluate_coco_predictions(coco, image_ids, predictions):
    if not predictions:
        return {
            "map50_95": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "map_small": 0.0,
            "map_medium": 0.0,
            "map_large": 0.0,
        }
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = [int(image_id) for image_id in image_ids]
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


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
