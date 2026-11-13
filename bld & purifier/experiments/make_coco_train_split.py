"""Expand an existing COCO train split without leaking old samples into validation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import atomic_write_json, valid_image_ids  # noqa: E402


def main():
    args = parse_args()
    old_split = json.loads(args.old_split.read_text(encoding="utf-8"))
    old_train = [int(value) for value in old_split["train_image_ids"]]
    old_val = [int(value) for value in old_split["val_image_ids"]]

    if len(old_train) > args.train_size:
        raise ValueError("新训练集不能小于旧训练集")

    coco = COCO(str(args.annotations))
    valid_ids = valid_image_ids(coco)
    old_ids = set(old_train) | set(old_val)
    candidates = [image_id for image_id in valid_ids if image_id not in old_ids]
    random.Random(args.seed).shuffle(candidates)

    extra_train_count = args.train_size - len(old_train)
    required = extra_train_count + args.val_size
    if len(candidates) < required:
        raise RuntimeError(
            f"可用新图片只有 {len(candidates)} 张，需要至少 {required} 张"
        )

    extra_train = candidates[:extra_train_count]
    new_val = candidates[extra_train_count:required]
    train_ids = old_train + extra_train

    split = {
        "dataset": "coco_train2017",
        "seed": args.seed,
        "source_split": str(args.old_split),
        "old_train_count": len(old_train),
        "excluded_old_val_count": len(old_val),
        "new_train_count": len(extra_train),
        "train_image_ids": train_ids,
        "val_image_ids": new_val,
    }
    atomic_write_json(args.output, split)

    print("旧训练图片：", len(old_train))
    print("排除旧验证图片：", len(old_val))
    print("新增训练图片：", len(extra_train))
    print("最终训练/验证：", len(train_ids), len(new_val))
    print("新分割文件：", args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--old-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261110)
    return parser.parse_args()


if __name__ == "__main__":
    main()
