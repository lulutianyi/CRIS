"""Prepare an isolated Colab workspace for the official ZeroPur implementation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision


REPOSITORY = "https://github.com/erhul/ZeroPur.git"


def run(command, cwd=None, check=True):
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content/zeropur_workspace"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    official = args.root / "official"

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "timm",
        "torchattacks",
        "opencv-python-headless",
        "scikit-learn",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/fra31/auto-attack.git",
    ])

    if not (official / ".git").exists():
        run(["git", "clone", REPOSITORY, str(official)])

    run(["git", "lfs", "pull"], cwd=official, check=False)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official, text=True
    ).strip()

    metadata = {
        "repository": REPOSITORY,
        "commit": commit,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    metadata_path = args.root / "environment.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("ZeroPur官方代码：", official)
    print("固定commit：", commit)
    print("环境记录：", metadata_path)
    print("GPU：", metadata["cuda_device"])


if __name__ == "__main__":
    main()
