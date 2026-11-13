"""Download a deterministic COCO train2017 subset without the full image zip."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3
from PIL import Image
from pycocotools.coco import COCO
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_frcnn_common import atomic_write_json, select_image_ids  # noqa: E402


def valid_image(path, width, height):
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            size_matches = image.size == (width, height)
            image.verify()
            return size_matches
    except Exception:
        return False


def download_one(info, output_dir, retries, insecure):
    destination = output_dir / info["file_name"]
    if valid_image(destination, info["width"], info["height"]):
        return int(info["id"]), str(destination), "existing"

    url = info.get("coco_url") or (
        "https://images.cocodataset.org/train2017/" + info["file_name"]
    )
    url = url.replace("http://", "https://", 1)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=(20, 120),
                verify=not insecure,
            )
            response.raise_for_status()
            temporary.write_bytes(response.content)
            if not valid_image(temporary, info["width"], info["height"]):
                raise ValueError("下载文件不是完整的目标图片")
            temporary.replace(destination)
            return int(info["id"]), str(destination), "downloaded"
        except Exception as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            time.sleep(2 ** attempt)
    raise RuntimeError(f'{info["file_name"]}: {last_error}')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    coco = COCO(str(args.annotations))
    if args.split:
        split = json.loads(args.split.read_text(encoding="utf-8"))
        image_ids = [
            int(value)
            for value in split["train_image_ids"] + split["val_image_ids"]
        ]
    else:
        image_ids = select_image_ids(coco, args.subset_size, args.seed)
    infos = coco.loadImgs(image_ids)
    info_by_id = {int(info["id"]): info for info in infos}
    infos = [info_by_id[image_id] for image_id in image_ids]

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                info,
                output_dir,
                args.retries,
                args.insecure,
            ): info
            for info in infos
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="COCO train subset"
        ):
            info = futures[future]
            try:
                image_id, path, status = future.result()
                records.append({
                    "image_id": image_id,
                    "file_name": info["file_name"],
                    "path": path,
                    "status": status,
                })
            except Exception as error:
                failures.append({
                    "image_id": int(info["id"]),
                    "file_name": info["file_name"],
                    "error": str(error),
                })

    record_by_id = {record["image_id"]: record for record in records}
    records = [record_by_id[image_id] for image_id in image_ids if image_id in record_by_id]
    manifest = {
        "dataset": "coco_train2017",
        "annotations": str(args.annotations),
        "output_dir": str(output_dir),
        "requested_subset_size": len(image_ids),
        "seed": args.seed,
        "selected_image_ids": image_ids,
        "completed": len(records),
        "failures": failures,
        "samples": records,
    }
    manifest_path = output_dir / "clean_subset_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print("\nclean subset manifest:", manifest_path)
    print("完成:", len(records), "失败:", len(failures))
    if failures:
        raise RuntimeError("部分图片下载失败；保持相同参数重新运行即可续传")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="仅在当前环境出现 images.cocodataset.org 证书错误时使用",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
