"""Same-subset comparison of GCN Joint-3072 and an existing FreqPure run.

Thin wrapper only: existing architecture, dataset, image saving, and mAP helpers
are imported instead of copied. No training, PGD generation, or FreqPure rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
for module_dir in (HERE, HERE.parent / "methods", HERE.parent / "core"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import compare_gcn_freqpure_subset as common  # noqa: E402
import freqpure_dfire as freq  # noqa: E402
import shared_pgd_gcn_flowpure_compare as shared  # noqa: E402


DEFAULT_FREQ_METRICS = common.DEFAULT_FREQ_METRICS
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/gcn_joint_freqpure_same400"
DEFAULT_JOINT_CKPT = (
    "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2/"
    "gcn_joint_3072_diffusion_v1.pt"
)


class JointReverseModel(nn.Module):
    """Exact 12x16x16 joint wrapper used by the notebook checkpoint."""

    def __init__(self):
        super().__init__()
        self.channels = 12
        self.height = 16
        self.width = 16
        self.unet = shared.DiffUNet(in_ch=12, latent_size=16)

    def forward(self, z_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch = z_t.shape[0]
        image = z_t.float().view(batch, self.channels, self.height, self.width)
        return self.unet(image, timestep).view(batch, -1)


def encode_all_codes(model, images: torch.Tensor):
    """2048-bit main plus the two learned-v2 512-bit skip streams."""
    main, features = model.encoder(images, hard=True)
    skip_codes = {
        name: quantizer.encode(features[name])
        for name, quantizer in model.decoder.skip_quantizers.items()
    }
    return main, skip_codes


def decode_from_codes(model, main: torch.Tensor, skip_codes: Dict[str, torch.Tensor]):
    """Decode only explicit bit streams; never reuse adversarial float skips."""
    expected = set(model.decoder.gcn_blocks.keys())
    missing = expected - set(skip_codes)
    if missing:
        raise KeyError(f"Missing skip codes: {sorted(missing)}")
    hidden = main.float()
    for index, module in enumerate(model.decoder.base_decoder.conv_layers):
        hidden = module(hidden)
        if index in model.decoder._insert_after:
            name = model.decoder._insert_after[index]
            quantizer = model.decoder.skip_quantizers[name]
            feature = quantizer.decode(
                skip_codes[name].float(), quantizer.source_spatial
            )
            hidden = model.decoder.gcn_blocks[name](hidden, feature)
    return hidden


def pack_joint_codes(main, skip_codes) -> torch.Tensor:
    skip64 = skip_codes["64x64"].float()
    skip32 = skip_codes["32x32"].float()
    if tuple(main.shape[1:]) != (8, 16, 16):
        raise ValueError(f"Unexpected main shape: {tuple(main.shape)}")
    if tuple(skip64.shape[1:]) != (2, 16, 16):
        raise ValueError(f"Unexpected skip64 shape: {tuple(skip64.shape)}")
    if tuple(skip32.shape[1:]) != (8, 8, 8):
        raise ValueError(f"Unexpected skip32 shape: {tuple(skip32.shape)}")
    skip32_packed = F.pixel_shuffle(skip32, upscale_factor=2)
    return torch.cat([main.float(), skip64, skip32_packed], dim=1)


def unpack_joint_codes(joint: torch.Tensor):
    if tuple(joint.shape[1:]) != (12, 16, 16):
        raise ValueError(f"Unexpected joint shape: {tuple(joint.shape)}")
    main = joint[:, :8]
    skip64 = joint[:, 8:10]
    skip32 = F.pixel_unshuffle(joint[:, 10:12], downscale_factor=2)
    return main, {"64x64": skip64, "32x32": skip32}


def load_joint_model(path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    raw = shared.torch_load(path, device)
    metadata = raw if isinstance(raw, dict) else {}
    model = JointReverseModel().to(device)
    model.load_state_dict(shared.unwrap_state_dict(raw), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print("Strictly loaded Joint-3072:", path)
    if metadata:
        print(
            "Joint checkpoint format/step:",
            metadata.get("format", "unknown"),
            metadata.get("optimizer_step", "unknown"),
        )
    return model, metadata


def run_root(args, names: List[str]) -> Path:
    subset_crc = zlib.crc32("|".join(names).encode("utf-8"))
    gcn_hash = shared.checkpoint_fingerprint(args.gcn_ckpt)
    joint_hash = shared.checkpoint_fingerprint(args.joint_diffusion_ckpt)
    root = Path(args.output_root) / (
        f"same{len(names)}_crc{subset_crc:08x}_gcn{gcn_hash}_"
        f"joint{joint_hash}_t{args.t_star}_pseed{args.purification_seed}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_names(t_star: int) -> Tuple[str, str, str]:
    return (
        "gcn_clean_reconstructed",
        "gcn_adv_direct",
        f"gcn_joint_diffusion_t{t_star}",
    )


@torch.inference_mode()
def generate_gcn_outputs(args, cfg, names: List[str], root: Path, device):
    ae = shared.load_gcn_model(cfg, device)
    joint_model, metadata = load_joint_model(args.joint_diffusion_ckpt, device)
    loader = shared.make_shared_loader(
        cfg, {"count": len(names), "names": names}, device
    )
    labels = str(Path(cfg.shared_root) / "clean" / "labels")
    splits = output_names(args.t_star)
    directories = {name: str(root / name) for name in splits}
    betas, flip_probability = shared.get_schedule(device)

    probe = torch.zeros(1, 12, 16, 16, device=device)
    probe[:, :, ::2, 1::2] = 1
    probe_main, probe_skips = unpack_joint_codes(probe)
    if not torch.equal(probe, pack_joint_codes(probe_main, probe_skips)):
        raise RuntimeError("Joint pack/unpack is not exactly invertible")

    for clean384, adv384, batch_names in tqdm(loader, desc="GCN Joint-3072 same subset"):
        if all(shared.output_complete(path, batch_names) for path in directories.values()):
            continue
        clean128 = F.interpolate(
            clean384.to(device, non_blocking=True),
            size=(128, 128), mode="bilinear", align_corners=False,
        )
        adv128 = F.interpolate(
            adv384.to(device, non_blocking=True),
            size=(128, 128), mode="bilinear", align_corners=False,
        )
        main_clean, skip_clean = encode_all_codes(ae, clean128)
        main_adv, skip_adv = encode_all_codes(ae, adv128)
        clean_reconstructed = decode_from_codes(ae, main_clean, skip_clean)
        adv_direct = decode_from_codes(ae, main_adv, skip_adv)

        batch_seed = args.purification_seed + zlib.crc32(
            "|".join(batch_names).encode("utf-8")
        )
        shared.seed_everything(batch_seed)
        joint_adv = pack_joint_codes(main_adv, skip_adv)
        flat = joint_adv.long().view(joint_adv.shape[0], -1)
        timestep = torch.full(
            (flat.shape[0],), args.t_star - 1,
            device=device, dtype=torch.long,
        )
        noised = shared.q_sample(flat, timestep, flip_probability)
        recovered = shared.reverse_binary(
            joint_model, noised, args.t_star, betas, flip_probability
        ).float().view_as(joint_adv)
        recovered_main, recovered_skips = unpack_joint_codes(recovered)
        joint_reconstructed = decode_from_codes(
            ae, recovered_main, recovered_skips
        )

        generated = {
            splits[0]: clean_reconstructed,
            splits[1]: adv_direct,
            splits[2]: joint_reconstructed,
        }
        for split, images128 in generated.items():
            images384 = F.interpolate(
                images128, size=(384, 384), mode="bilinear", align_corners=False
            )
            shared.save_output_batch(
                images384, batch_names, directories[split], labels
            )

    metadata_path = root / "joint_checkpoint_metadata.json"
    safe_metadata = {
        "checkpoint": args.joint_diffusion_ckpt,
        "sha256_prefix": shared.checkpoint_fingerprint(args.joint_diffusion_ckpt),
        "format": metadata.get("format") if metadata else None,
        "optimizer_step": metadata.get("optimizer_step") if metadata else None,
        "gcn_checkpoint_recorded": metadata.get("gcn_checkpoint") if metadata else None,
    }
    metadata_path.write_text(
        json.dumps(safe_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def verify_outputs(root: Path, names: List[str], t_star: int) -> None:
    expected = set(names)
    for split in output_names(t_star):
        actual = {path.stem for path in (root / split / "images").glob("*.png")}
        missing, extra = expected - actual, actual - expected
        print(f"{split:32s} selected={len(actual & expected)}/{len(expected)}")
        if missing or extra:
            raise RuntimeError(
                f"{split} mismatch: missing={len(missing)}, extra={len(extra)}"
            )


def evaluate_maps(args, cfg, names: List[str], root: Path, device) -> Dict:
    detector = shared.YOLO(cfg.yolo_ckpt)
    eval_cfg = freq.FreqPureDFireConfig(
        shared_root=cfg.shared_root,
        yolo_ckpt=cfg.yolo_ckpt,
        detector_size=384,
        detector_batch_size=args.detector_batch_size,
    )
    results = {}
    for split in output_names(args.t_star):
        paths = [root / split / "images" / f"{name}.png" for name in names]
        image_list = freq.write_image_list(
            root / "image_lists" / f"{split}.txt", paths
        )
        map50, map50_95 = freq.eval_map(
            detector, image_list, eval_cfg, root, split, device
        )
        results[split] = {"map50": map50, "map50_95": map50_95}
        print(f"{split:32s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")
    return results


def save_comparison(args, names, root, freq_report, gcn_results) -> Path:
    results = dict(common.require_freq_results(freq_report))
    results.update(gcn_results)
    clean, adv = results["original_clean"], results["adv_original"]

    def recovery(name: str, metric: str) -> float:
        denominator = max(clean[metric] - adv[metric], 1e-8)
        return (results[name][metric] - adv[metric]) / denominator

    direct = "gcn_adv_direct"
    joint = f"gcn_joint_diffusion_t{args.t_star}"
    derived = {
        "freqpure_recovery_map50": recovery("adv_freqpure", "map50"),
        "freqpure_recovery_map50_95": recovery("adv_freqpure", "map50_95"),
        "gcn_direct_recovery_map50": recovery(direct, "map50"),
        "gcn_direct_recovery_map50_95": recovery(direct, "map50_95"),
        "gcn_joint_recovery_map50": recovery(joint, "map50"),
        "gcn_joint_recovery_map50_95": recovery(joint, "map50_95"),
        "gcn_clean_drop_map50": (
            clean["map50"] - results["gcn_clean_reconstructed"]["map50"]
        ),
        "joint_gain_over_direct_map50": (
            results[joint]["map50"] - results[direct]["map50"]
        ),
        "freqpure_minus_joint_map50": (
            results["adv_freqpure"]["map50"] - results[joint]["map50"]
        ),
    }

    print("\n=== Same-subset latest Joint-3072 vs FreqPure ===")
    print(
        f"FreqPure:  mAP50 recovery={derived['freqpure_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['freqpure_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN direct: mAP50 recovery={derived['gcn_direct_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['gcn_direct_recovery_map50_95']:.1%}"
    )
    print(
        f"GCN Joint T*={args.t_star}: "
        f"mAP50 recovery={derived['gcn_joint_recovery_map50']:.1%}, "
        f"mAP50-95 recovery={derived['gcn_joint_recovery_map50_95']:.1%}"
    )
    print(
        "Joint gain over direct raw mAP50: "
        f"{derived['joint_gain_over_direct_map50']:+.4f}"
    )
    print(
        "FreqPure - Joint raw mAP50: "
        f"{derived['freqpure_minus_joint_map50']:+.4f}"
    )

    report = {
        "schema": "gcn_joint3072_freqpure_same_subset_v1",
        "selected_count": len(names),
        "selected_names": names,
        "freqpure_metrics_source": args.freq_metrics,
        "gcn_checkpoint": args.gcn_ckpt,
        "joint_diffusion_checkpoint": args.joint_diffusion_ckpt,
        "joint_checkpoint_sha256_prefix": shared.checkpoint_fingerprint(
            args.joint_diffusion_ckpt
        ),
        "t_star": args.t_star,
        "purification_seed": args.purification_seed,
        "results": results,
        "derived": derived,
    }
    path = root / "metrics_joint3072_vs_freqpure_same_subset.json"
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
    parser.add_argument("--joint-diffusion-ckpt", default=DEFAULT_JOINT_CKPT)
    parser.add_argument("--t-star", type=int, default=3)
    parser.add_argument("--purification-seed", type=int, default=20260904)
    parser.add_argument("--gcn-batch-size", type=int, default=8)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)
    if args.t_star < 1:
        parser.error("--t-star must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    freq_report = common.read_json(Path(args.freq_metrics))
    names = common.selected_names(freq_report)
    common.require_freq_results(freq_report)
    shared_root = common.resolve_shared_root(args, freq_report)
    common.validate_shared_images(shared_root, names)
    cfg = shared.CompareConfig(
        shared_root=shared_root,
        yolo_ckpt=args.yolo_ckpt,
        gcn_ckpt=args.gcn_ckpt,
        t_star=args.t_star,
        purification_seed=args.purification_seed,
        purify_batch_size=args.gcn_batch_size,
        num_workers=args.num_workers,
        detector_size=384,
        purifier_size=128,
    )
    for path in (args.yolo_ckpt, args.gcn_ckpt, args.joint_diffusion_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = run_root(args, names)
    print("Device:", device)
    print("Selected names from:", args.freq_metrics)
    print("Selected images:", len(names))
    print("Output:", root)

    if args.stage in ("all", "gcn"):
        generate_gcn_outputs(args, cfg, names, root, device)
    if args.stage in ("all", "map"):
        verify_outputs(root, names, args.t_star)
        gcn_results = evaluate_maps(args, cfg, names, root, device)
        save_comparison(args, names, root, freq_report, gcn_results)


if __name__ == "__main__":
    main()
