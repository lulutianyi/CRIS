"""Compare GCN-Skip and an existing FreqPure run on exactly the same images.

This is intentionally a thin orchestration script. It imports the existing
GCN implementation from shared_pgd_gcn_flowpure_compare.py and the image-list
mAP evaluator from freqpure_dfire.py. It does not train anything and does not
rerun FreqPure.

Required under /content:
  - core/shared_pgd_gcn_flowpure_compare.py
  - methods/freqpure_dfire.py
  - comparisons/compare_gcn_freqpure_subset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
for module_dir in (SCRIPT_DIR, SCRIPT_DIR.parent / "methods", SCRIPT_DIR.parent / "core"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import freqpure_dfire as freq  # noqa: E402
import shared_pgd_gcn_flowpure_compare as shared  # noqa: E402


DEFAULT_FREQ_METRICS = (
    "/content/drive/MyDrive/freqpure_adv400/"
    "imagenet256_a10_p10_d0p3_t50_k5_c8_ddpm_official_"
    "adv_freqpure_only_subset20260909_seed20260904_n400/"
    "metrics_freqpure_dfire.json"
)
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/gcn_freqpure_same_subset"


def read_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def selected_names(report: Dict) -> List[str]:
    names = report.get("selected_names")
    if not isinstance(names, list) or not names:
        raise RuntimeError("FreqPure metrics has no non-empty selected_names")
    names = [str(name) for name in names]
    if len(set(names)) != len(names):
        raise RuntimeError("FreqPure selected_names contains duplicates")
    return names


def require_freq_results(report: Dict) -> Dict[str, Dict[str, float]]:
    results = report.get("results", {})
    required = (
        "original_clean",
        "adv_original",
        "adv_resize_baseline",
        "adv_freqpure",
    )
    missing = [name for name in required if name not in results]
    if missing:
        raise RuntimeError(f"FreqPure metrics missing results: {missing}")
    return results


def resolve_shared_root(args, report: Dict) -> str:
    root = args.shared_root
    if root is None:
        root = report.get("config", {}).get("shared_root")
    if not root:
        raise RuntimeError("Cannot determine shared_root; pass --shared-root")
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return str(Path(root))


def validate_shared_images(root: str, names: List[str]) -> None:
    for split in ("clean", "adv"):
        image_dir = Path(root) / split / "images"
        missing = [name for name in names if not (image_dir / f"{name}.png").is_file()]
        if missing:
            raise RuntimeError(
                f"Shared {split} is missing {len(missing)} selected images; "
                f"first={missing[0]}"
            )


def make_gcn_config(args, shared_root: str) -> shared.CompareConfig:
    return shared.CompareConfig(
        shared_root=shared_root,
        output_root=args.output_root,
        yolo_ckpt=args.yolo_ckpt,
        gcn_ckpt=args.gcn_ckpt,
        binary_diffusion_ckpt=args.binary_diffusion_ckpt,
        t_star=args.t_star,
        purification_seed=args.purification_seed,
        purify_batch_size=args.gcn_batch_size,
        num_workers=args.num_workers,
        detector_size=384,
        purifier_size=128,
    )


def comparison_root(args, cfg: shared.CompareConfig, names: List[str]) -> Path:
    subset_crc = zlib.crc32("|".join(names).encode("utf-8"))
    gcn_hash = shared.checkpoint_fingerprint(cfg.gcn_ckpt)
    root = Path(args.output_root) / (
        f"same{len(names)}_crc{subset_crc:08x}_gcn{gcn_hash}_t{cfg.t_star}_"
        f"pseed{cfg.purification_seed}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def check_outputs(root: Path, names: List[str], t_star: int) -> None:
    expected = set(names)
    for split in (
        "gcn_clean_reconstructed",
        "gcn_adv_reconstructed",
        f"gcn_adv_diffusion_t{t_star}",
    ):
        image_dir = root / split / "images"
        actual = {path.stem for path in image_dir.glob("*.png")}
        missing = expected - actual
        extra = actual - expected
        print(f"{split:30s} selected={len(actual & expected)}/{len(expected)}")
        if missing or extra:
            raise RuntimeError(
                f"Output mismatch for {split}: missing={len(missing)}, extra={len(extra)}"
            )


def evaluate_gcn_maps(
    args,
    cfg: shared.CompareConfig,
    root: Path,
    names: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    detector = shared.YOLO(cfg.yolo_ckpt)
    eval_cfg = freq.FreqPureDFireConfig(
        shared_root=cfg.shared_root,
        yolo_ckpt=cfg.yolo_ckpt,
        detector_size=cfg.detector_size,
        detector_batch_size=args.detector_batch_size,
    )
    split_names = (
        "gcn_clean_reconstructed",
        "gcn_adv_reconstructed",
        f"gcn_adv_diffusion_t{cfg.t_star}",
    )
    results: Dict[str, Dict[str, float]] = {}
    for split in split_names:
        paths = [root / split / "images" / f"{name}.png" for name in names]
        image_list = freq.write_image_list(root / "image_lists" / f"{split}.txt", paths)
        map50, map50_95 = freq.eval_map(
            detector, image_list, eval_cfg, root, split, device
        )
        results[split] = {"map50": map50, "map50_95": map50_95}
        print(f"{split:30s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")
    return results


def combine_report(
    args,
    cfg: shared.CompareConfig,
    root: Path,
    names: List[str],
    freq_report: Dict,
    gcn_results: Dict[str, Dict[str, float]],
) -> Path:
    results = dict(require_freq_results(freq_report))
    results.update(gcn_results)
    clean = results["original_clean"]
    adv = results["adv_original"]

    def recovery(name: str, metric: str) -> float:
        denominator = max(clean[metric] - adv[metric], 1e-8)
        return (results[name][metric] - adv[metric]) / denominator

    direct = "gcn_adv_reconstructed"
    diffusion = f"gcn_adv_diffusion_t{cfg.t_star}"
    derived = {
        "freqpure_recovery_map50": recovery("adv_freqpure", "map50"),
        "freqpure_recovery_map50_95": recovery("adv_freqpure", "map50_95"),
        "gcn_direct_recovery_map50": recovery(direct, "map50"),
        "gcn_direct_recovery_map50_95": recovery(direct, "map50_95"),
        "gcn_diffusion_recovery_map50": recovery(diffusion, "map50"),
        "gcn_diffusion_recovery_map50_95": recovery(diffusion, "map50_95"),
        "gcn_clean_drop_map50": (
            clean["map50"] - results["gcn_clean_reconstructed"]["map50"]
        ),
        "freqpure_minus_gcn_direct_map50": (
            results["adv_freqpure"]["map50"] - results[direct]["map50"]
        ),
    }

    print("\n=== Same-subset GCN vs FreqPure ===")
    print(
        f"FreqPure:   mAP50 recovery={derived['freqpure_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['freqpure_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN direct: mAP50 recovery={derived['gcn_direct_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['gcn_direct_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN T*={cfg.t_star}: mAP50 recovery="
        f"{derived['gcn_diffusion_recovery_map50']:.1%}, mAP50-95 recovery="
        f"{derived['gcn_diffusion_recovery_map50_95']:.1%}"
    )
    print(
        "FreqPure - GCN direct raw mAP50: "
        f"{derived['freqpure_minus_gcn_direct_map50']:+.4f}"
    )

    report = {
        "schema": "gcn_freqpure_same_selected_names_v1",
        "selected_count": len(names),
        "selected_names": names,
        "freqpure_metrics_source": args.freq_metrics,
        "gcn_config": asdict(cfg),
        "results": results,
        "derived": derived,
    }
    path = root / "metrics_gcn_vs_freqpure_same_subset.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Metrics saved:", path)
    return path


def parse_args():
    defaults = shared.CompareConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "gcn", "map"), default="all")
    parser.add_argument("--freq-metrics", default=DEFAULT_FREQ_METRICS)
    parser.add_argument("--shared-root", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--gcn-ckpt", default=defaults.gcn_ckpt)
    parser.add_argument(
        "--binary-diffusion-ckpt", default=defaults.binary_diffusion_ckpt
    )
    parser.add_argument("--t-star", type=int, default=3)
    parser.add_argument("--purification-seed", type=int, default=20260904)
    parser.add_argument("--gcn-batch-size", type=int, default=8)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)
    return args


def main() -> None:
    args = parse_args()
    freq_report = read_json(Path(args.freq_metrics))
    names = selected_names(freq_report)
    require_freq_results(freq_report)
    shared_root = resolve_shared_root(args, freq_report)
    validate_shared_images(shared_root, names)
    cfg = make_gcn_config(args, shared_root)
    for path in (cfg.yolo_ckpt, cfg.gcn_ckpt, cfg.binary_diffusion_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = comparison_root(args, cfg, names)
    subset_manifest = {"count": len(names), "names": names}
    print("Device:", device)
    print("Selected names from:", args.freq_metrics)
    print("Selected images:", len(names))
    print("Output:", root)

    if args.stage in ("all", "gcn"):
        shared.evaluate_gcn_from_shared(cfg, subset_manifest, device, str(root))
    if args.stage in ("all", "map"):
        check_outputs(root, names, cfg.t_star)
        gcn_results = evaluate_gcn_maps(args, cfg, root, names, device)
        combine_report(args, cfg, root, names, freq_report, gcn_results)


if __name__ == "__main__":
    main()
