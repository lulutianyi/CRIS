"""Run GCN-Skip and FreqPure on three fixed, disjoint shared-PGD subsets.

This file is deliberately only an orchestrator.  Network definitions, shared
PGD loading, purification, image writing, and YOLO mAP evaluation are reused
from ``shared_pgd_gcn_flowpure_compare.py`` and ``freqpure_dfire.py``.

Designed for Colab.  It performs no training and never changes checkpoints.
All selected names and metrics are persisted below ``--output-root`` so an
interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import torch


HERE = Path(__file__).resolve().parent
for module_dir in (HERE, HERE.parent / "methods", HERE.parent / "core"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import freqpure_dfire as freq  # noqa: E402
import shared_pgd_gcn_flowpure_compare as shared  # noqa: E402


SCHEMA = "gcn_freqpure_three_disjoint_subsets_v1"


def read_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fingerprint_names(names: Sequence[str]) -> str:
    return f"{zlib.crc32('|'.join(names).encode('utf-8')):08x}"


def configure_freqpure_adv_only_compatibility() -> None:
    """Support both the original and the later output-mode FreqPure files."""

    def active_output_splits(_cfg):
        return ("adv_resize_baseline", "adv_freqpure")

    # The local/Colab copies seen during development did not all contain this
    # helper.  Assigning it here does not edit freqpure_dfire.py.
    freq.active_output_splits = active_output_splits


def experiment_identity(args, shared_manifest: Dict, subsets: List[List[str]]) -> Dict:
    return {
        "schema": SCHEMA,
        "subset_size": args.subset_size,
        "num_subsets": args.num_subsets,
        "subset_seed": args.subset_seed,
        "purification_seed": args.purification_seed,
        "t_star": args.t_star,
        "shared_root": str(Path(args.shared_root)),
        "shared_manifest_count": shared_manifest.get("count"),
        "gcn_checkpoint": args.gcn_ckpt,
        "binary_diffusion_checkpoint": args.binary_diffusion_ckpt,
        "freqpure_diffusion_checkpoint": args.freqpure_diffusion_ckpt,
        "first_freq_metrics": args.first_freq_metrics,
        "subsets": [
            {
                "index": index + 1,
                "count": len(names),
                "names_crc32": fingerprint_names(names),
                "names": names,
            }
            for index, names in enumerate(subsets)
        ],
    }


def prepare_subsets(args) -> tuple[Dict, List[List[str]]]:
    manifest_path = Path(args.shared_root) / "manifest.json"
    manifest = read_json(manifest_path)
    names = [str(name) for name in manifest.get("names", [])]
    if not names:
        raise RuntimeError(f"No names in {manifest_path}")
    if len(names) != len(set(names)):
        raise RuntimeError("Shared PGD manifest contains duplicate image names")
    required = args.subset_size * args.num_subsets
    if required > len(names):
        raise ValueError(f"Need {required} images, but manifest has only {len(names)}")

    subsets = []
    remaining = list(names)
    if args.first_freq_metrics:
        first_report = read_json(Path(args.first_freq_metrics))
        first_names = [str(name) for name in first_report.get("selected_names", [])]
        if len(first_names) != args.subset_size:
            raise RuntimeError(
                f"First FreqPure metrics contains {len(first_names)} names, "
                f"expected {args.subset_size}"
            )
        unknown = set(first_names) - set(names)
        if unknown:
            raise RuntimeError(
                f"First FreqPure subset is not from this manifest; first={next(iter(unknown))}"
            )
        subsets.append(first_names)
        first_set = set(first_names)
        remaining = [name for name in remaining if name not in first_set]

    random.Random(args.subset_seed).shuffle(remaining)
    needed = args.num_subsets - len(subsets)
    subsets.extend(
        remaining[index * args.subset_size : (index + 1) * args.subset_size]
        for index in range(needed)
    )
    flat = [name for subset in subsets for name in subset]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Generated subsets are not disjoint")

    for split in ("clean", "adv"):
        image_dir = Path(args.shared_root) / split / "images"
        missing = [name for name in flat if not (image_dir / f"{name}.png").is_file()]
        if missing:
            raise RuntimeError(
                f"Shared PGD {split} is missing {len(missing)} selected images; "
                f"first={missing[0]}"
            )

    output_root = Path(args.output_root)
    identity_path = output_root / "experiment_manifest.json"
    identity = experiment_identity(args, manifest, subsets)
    if identity_path.exists():
        existing = read_json(identity_path)
        if existing != identity:
            raise RuntimeError(
                "Existing experiment manifest has different settings. "
                "Use a new --output-root instead of mixing experiments."
            )
    else:
        write_json(identity_path, identity)
    return manifest, subsets


def selected_indices(args) -> List[int]:
    if args.subset_index == 0:
        return list(range(args.num_subsets))
    if not 1 <= args.subset_index <= args.num_subsets:
        raise ValueError(
            f"--subset-index must be 0..{args.num_subsets}; got {args.subset_index}"
        )
    return [args.subset_index - 1]


def subset_root(args, index: int, names: Sequence[str]) -> Path:
    root = Path(args.output_root) / (
        f"subset_{index + 1:02d}_n{len(names)}_crc{fingerprint_names(names)}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_freq_config(args) -> freq.FreqPureDFireConfig:
    # Only constructor fields common to every saved version are passed here.
    cfg = freq.FreqPureDFireConfig(
        shared_root=args.shared_root,
        output_root=args.output_root,
        yolo_ckpt=args.yolo_ckpt,
        freqpure_repo=args.freqpure_repo,
        diffusion_ckpt=args.freqpure_diffusion_ckpt,
        detector_size=384,
        purifier_size=256,
        max_images=None,
        purify_batch_size=args.freqpure_batch_size,
        detector_batch_size=args.detector_batch_size,
        num_workers=args.num_workers,
        purification_seed=args.purification_seed,
        use_amp=not args.no_amp,
    )
    cfg.output_mode = "adv_freqpure_only"
    cfg.subset_seed = args.subset_seed
    return cfg


def make_gcn_config(args, output_root: Path) -> shared.CompareConfig:
    return shared.CompareConfig(
        shared_root=args.shared_root,
        output_root=str(output_root),
        yolo_ckpt=args.yolo_ckpt,
        gcn_ckpt=args.gcn_ckpt,
        binary_diffusion_ckpt=args.binary_diffusion_ckpt,
        detector_size=384,
        purifier_size=128,
        t_star=args.t_star,
        purification_seed=args.purification_seed,
        purify_batch_size=args.gcn_batch_size,
        num_workers=args.num_workers,
    )


def verify_gcn_outputs(root: Path, names: Sequence[str], t_star: int) -> None:
    expected = set(names)
    for split in (
        "gcn_clean_reconstructed",
        "gcn_adv_reconstructed",
        f"gcn_adv_diffusion_t{t_star}",
    ):
        actual = {path.stem for path in (root / split / "images").glob("*.png")}
        missing = expected - actual
        extra = actual - expected
        print(f"{split:30s} images={len(actual & expected)}/{len(expected)}")
        if missing or extra:
            raise RuntimeError(
                f"Incomplete {split}: missing={len(missing)}, extra={len(extra)}"
            )


def evaluate_gcn_maps(
    args,
    cfg: shared.CompareConfig,
    root: Path,
    names: Sequence[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    detector = shared.YOLO(cfg.yolo_ckpt)
    eval_cfg = freq.FreqPureDFireConfig(
        shared_root=cfg.shared_root,
        yolo_ckpt=cfg.yolo_ckpt,
        detector_size=384,
        detector_batch_size=args.detector_batch_size,
    )
    results = {}
    for split in (
        "gcn_clean_reconstructed",
        "gcn_adv_reconstructed",
        f"gcn_adv_diffusion_t{cfg.t_star}",
    ):
        paths = [root / split / "images" / f"{name}.png" for name in names]
        image_list = freq.write_image_list(
            root / "image_lists" / f"{split}.txt", paths
        )
        map50, map50_95 = freq.eval_map(
            detector, image_list, eval_cfg, root, split, device
        )
        results[split] = {"map50": map50, "map50_95": map50_95}
        print(f"{split:30s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def combine_subset_report(
    args,
    index: int,
    names: Sequence[str],
    root: Path,
    freq_report: Dict,
    gcn_results: Dict[str, Dict[str, float]],
) -> Dict:
    results = dict(freq_report["results"])
    required = ("original_clean", "adv_original", "adv_resize_baseline", "adv_freqpure")
    missing = [key for key in required if key not in results]
    if missing:
        raise RuntimeError(f"FreqPure report is missing: {missing}")
    results.update(gcn_results)
    clean = results["original_clean"]
    adv = results["adv_original"]

    def recovery(name: str, metric: str) -> float:
        denominator = max(clean[metric] - adv[metric], 1e-8)
        return (results[name][metric] - adv[metric]) / denominator

    direct = "gcn_adv_reconstructed"
    diffusion = f"gcn_adv_diffusion_t{args.t_star}"
    derived = {
        "freqpure_recovery_map50": recovery("adv_freqpure", "map50"),
        "freqpure_recovery_map50_95": recovery("adv_freqpure", "map50_95"),
        "gcn_direct_recovery_map50": recovery(direct, "map50"),
        "gcn_direct_recovery_map50_95": recovery(direct, "map50_95"),
        "gcn_diffusion_recovery_map50": recovery(diffusion, "map50"),
        "gcn_diffusion_recovery_map50_95": recovery(diffusion, "map50_95"),
        "freqpure_minus_gcn_direct_raw_map50": (
            results["adv_freqpure"]["map50"] - results[direct]["map50"]
        ),
    }
    report = {
        "schema": SCHEMA,
        "subset_index": index + 1,
        "selected_count": len(names),
        "selected_names_crc32": fingerprint_names(names),
        "selected_names": list(names),
        "t_star": args.t_star,
        "purification_seed": args.purification_seed,
        "results": results,
        "derived": derived,
    }
    path = root / "metrics_gcn_freqpure_subset.json"
    write_json(path, report)
    print(f"\n=== Subset {index + 1}: same-image GCN vs FreqPure ===")
    print(
        "FreqPure:  "
        f"mAP50 recovery={derived['freqpure_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['freqpure_recovery_map50_95']:.1%}"
    )
    print(
        "GCN direct: "
        f"mAP50 recovery={derived['gcn_direct_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['gcn_direct_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN T*={args.t_star}: "
        f"mAP50 recovery={derived['gcn_diffusion_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['gcn_diffusion_recovery_map50_95']:.1%}"
    )
    print(
        "FreqPure - GCN direct raw mAP50: "
        f"{derived['freqpure_minus_gcn_direct_raw_map50']:+.4f}"
    )
    print("Subset metrics saved:", path)
    return report


def summarize(args, subsets: List[List[str]]) -> Path:
    reports = []
    missing = []
    for index, names in enumerate(subsets):
        path = subset_root(args, index, names) / "metrics_gcn_freqpure_subset.json"
        if path.is_file():
            reports.append(read_json(path))
        else:
            missing.append(index + 1)
    if missing:
        raise RuntimeError(f"Cannot summarize; missing subset metrics: {missing}")

    keys = (
        "freqpure_recovery_map50",
        "freqpure_recovery_map50_95",
        "gcn_direct_recovery_map50",
        "gcn_direct_recovery_map50_95",
        "gcn_diffusion_recovery_map50",
        "gcn_diffusion_recovery_map50_95",
        "freqpure_minus_gcn_direct_raw_map50",
    )
    aggregate = {}
    for key in keys:
        values = [float(report["derived"][key]) for report in reports]
        aggregate[key] = {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    summary = {
        "schema": SCHEMA,
        "completed_subsets": len(reports),
        "subset_size": args.subset_size,
        "aggregate": aggregate,
    }
    path = Path(args.output_root) / "summary_three_subsets.json"
    write_json(path, summary)

    def show(label: str, key50: str, key95: str) -> None:
        value50 = aggregate[key50]
        value95 = aggregate[key95]
        print(
            f"{label:12s} mAP50={value50['mean']:.1%} ± "
            f"{value50['sample_std']:.1%}; mAP50-95={value95['mean']:.1%} ± "
            f"{value95['sample_std']:.1%}"
        )

    print("\n=== Three disjoint subsets: mean ± sample std ===")
    show("FreqPure", "freqpure_recovery_map50", "freqpure_recovery_map50_95")
    show("GCN direct", "gcn_direct_recovery_map50", "gcn_direct_recovery_map50_95")
    show(
        f"GCN T*={args.t_star}",
        "gcn_diffusion_recovery_map50",
        "gcn_diffusion_recovery_map50_95",
    )
    difference = aggregate["freqpure_minus_gcn_direct_raw_map50"]
    print(
        "FreqPure - GCN direct raw mAP50: "
        f"{difference['mean']:+.4f} ± {difference['sample_std']:.4f}"
    )
    print("Summary saved:", path)
    return path


def run_subset(args, manifest: Dict, names: List[str], index: int, device) -> None:
    root = subset_root(args, index, names)
    freq_root = root / "freqpure"
    gcn_root = root / "gcn"
    freq_root.mkdir(parents=True, exist_ok=True)
    gcn_root.mkdir(parents=True, exist_ok=True)
    subset_manifest = {"count": len(names), "names": names}
    write_json(root / "selected_names.json", subset_manifest)

    print("\n" + "=" * 76)
    print(
        f"Subset {index + 1}/{args.num_subsets}: n={len(names)}, "
        f"crc32={fingerprint_names(names)}"
    )
    print("=" * 76)

    freq_cfg = make_freq_config(args)
    gcn_cfg = make_gcn_config(args, gcn_root)

    reused_freq_metrics = (
        Path(args.first_freq_metrics)
        if index == 0 and args.first_freq_metrics
        else None
    )
    if args.stage in ("all", "purify") and reused_freq_metrics is None:
        freq.seed_everything(args.purification_seed)
        freq.generate_outputs(freq_cfg, names, freq_root, device)
        freq.verify_outputs(freq_cfg, freq_root, names)
    elif args.stage in ("all", "purify"):
        print("Reusing completed FreqPure metrics:", reused_freq_metrics)
    if args.stage in ("all", "gcn"):
        shared.evaluate_gcn_from_shared(
            gcn_cfg, subset_manifest, device, str(gcn_root)
        )
        verify_gcn_outputs(gcn_root, names, args.t_star)
    if args.stage in ("all", "map"):
        if reused_freq_metrics is None:
            freq_report = freq.compute_maps(
                freq_cfg, manifest, names, freq_root, device
            )
        else:
            freq_report = read_json(reused_freq_metrics)
            recorded_names = [str(name) for name in freq_report.get("selected_names", [])]
            if recorded_names != names:
                raise RuntimeError("Reused FreqPure metrics names/order do not match subset 1")
        verify_gcn_outputs(gcn_root, names, args.t_star)
        gcn_results = evaluate_gcn_maps(
            args, gcn_cfg, gcn_root, names, device
        )
        combine_subset_report(
            args, index, names, root, freq_report, gcn_results
        )


def parse_args():
    defaults = shared.CompareConfig()
    freq_defaults = freq.FreqPureDFireConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "purify", "gcn", "map", "summary"),
        default="all",
    )
    parser.add_argument("--shared-root", default=defaults.shared_root)
    parser.add_argument(
        "--output-root",
        default="/content/drive/MyDrive/gcn_freqpure_three400",
    )
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--gcn-ckpt", default=defaults.gcn_ckpt)
    parser.add_argument(
        "--binary-diffusion-ckpt", default=defaults.binary_diffusion_ckpt
    )
    parser.add_argument("--freqpure-repo", default=freq_defaults.freqpure_repo)
    parser.add_argument(
        "--freqpure-diffusion-ckpt", default=freq_defaults.diffusion_ckpt
    )
    parser.add_argument(
        "--first-freq-metrics",
        default=None,
        help="Reuse one completed FreqPure metrics JSON as subset 1",
    )
    parser.add_argument("--subset-size", type=int, default=400)
    parser.add_argument("--num-subsets", type=int, default=3)
    parser.add_argument("--subset-seed", type=int, default=20260910)
    parser.add_argument(
        "--subset-index",
        type=int,
        default=0,
        help="0 runs every subset; 1..N runs only that subset",
    )
    parser.add_argument("--purification-seed", type=int, default=20260904)
    parser.add_argument("--t-star", type=int, default=3)
    parser.add_argument("--freqpure-batch-size", type=int, default=1)
    parser.add_argument("--gcn-batch-size", type=int, default=8)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)
    if args.subset_size < 1 or args.num_subsets < 1:
        parser.error("--subset-size and --num-subsets must be positive")
    if args.t_star < 1:
        parser.error("--t-star must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    configure_freqpure_adv_only_compatibility()
    manifest, subsets = prepare_subsets(args)
    print("Three subsets are fixed and mutually disjoint.")
    print("Experiment manifest:", Path(args.output_root) / "experiment_manifest.json")
    if args.stage == "summary":
        summarize(args, subsets)
        return
    for path in (
        args.yolo_ckpt,
        args.gcn_ckpt,
        args.binary_diffusion_ckpt,
        args.freqpure_diffusion_ckpt,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    for index in selected_indices(args):
        run_subset(args, manifest, subsets[index], index, device)

    if args.subset_index == 0 and args.stage in ("all", "map"):
        summarize(args, subsets)


if __name__ == "__main__":
    main()
