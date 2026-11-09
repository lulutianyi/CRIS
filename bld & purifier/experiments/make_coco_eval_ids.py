"""Create a fixed COCO evaluation subset while excluding prior development IDs."""

import argparse
import json
import random
from pathlib import Path

from pycocotools.coco import COCO

from coco_frcnn_common import atomic_write_json, valid_image_ids


def read_image_ids(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    values = value["image_ids"] if isinstance(value, dict) else value
    return {int(image_id) for image_id in values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    coco = COCO(str(args.annotations))
    excluded = set().union(*(read_image_ids(path) for path in args.exclude))
    candidates = []
    for image_id in valid_image_ids(coco):
        file_name = coco.loadImgs([image_id])[0]["file_name"]
        if image_id not in excluded and (args.images / file_name).is_file():
            candidates.append(image_id)
    random.Random(args.seed).shuffle(candidates)
    image_ids = candidates[: args.subset_size]

    if len(image_ids) < args.subset_size:
        raise RuntimeError(
            f"可用且未被排除的图片只有 {len(image_ids)} 张，少于 {args.subset_size} 张"
        )

    atomic_write_json(
        args.output,
        {
            "dataset": "coco_val2017",
            "seed": args.seed,
            "subset_size": len(image_ids),
            "excluded_count": len(excluded),
            "image_ids": image_ids,
        },
    )
    print("最终评测图片数：", len(image_ids))
    print("已排除旧图片数：", len(excluded))
    print("编号文件：", args.output)


if __name__ == "__main__":
    main()
