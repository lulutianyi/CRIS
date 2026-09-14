"""
FreqPure on D-Fire with the existing shared-PGD evaluation protocol.

This is a detection-task adaptation of the official ICML 2025 FreqPure code:
https://github.com/GaozhengPei/FreqPure

The frequency-domain algorithm and ImageNet 256x256 guided-diffusion backbone
are retained.  The official classification wrapper is deliberately removed;
purified RGB images are saved and evaluated by the fixed D-Fire YOLO detector.

Colab setup (run once):

    from google.colab import drive
    drive.mount('/content/drive')
    %pip install -q ultralytics==8.4.126 pyyaml blobfile
    !git clone https://github.com/GaozhengPei/FreqPure.git /content/FreqPure
    !mkdir -p /content/drive/MyDrive/freqpure_checkpoints
    !wget -c \
      https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
      -O /content/drive/MyDrive/freqpure_checkpoints/256x256_diffusion_uncond.pt

Very small interface smoke run (only checks that the model and paths work;
these reduced settings are NOT reportable as FreqPure results):

    %run /content/methods/freqpure_dfire.py --stage all --max-images 2 \
      --purification-cycles 1 --denoising-steps 1

Official-parameter 32-image smoke evaluation:

    %run /content/methods/freqpure_dfire.py --stage all --max-images 32

Full 4,306-image evaluation after the smoke run succeeds:

    %run /content/methods/freqpure_dfire.py --stage all --max-images 0

The full official ImageNet schedule is expensive: eight purification cycles,
five reverse steps per cycle, plus one reference-denoiser call at every reverse
step.  Outputs are resumable and are never deleted automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is missing. In Colab run: "
        "%pip install -q ultralytics==8.4.126"
    ) from exc


@dataclass
class FreqPureDFireConfig:
    shared_root: str = (
        "/content/drive/MyDrive/dfire_shared_pgd_png_eps8_seed20260822"
    )
    output_root: str = "/content/drive/MyDrive/freqpure_dfire_eval"
    yolo_ckpt: str = (
        "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    )
    freqpure_repo: str = "/content/FreqPure"
    diffusion_ckpt: str = (
        "/content/drive/MyDrive/freqpure_checkpoints/"
        "256x256_diffusion_uncond.pt"
    )

    detector_size: int = 384
    purifier_size: int = 256
    max_images: Optional[int] = 32
    purify_batch_size: int = 1
    detector_batch_size: int = 8
    num_workers: int = 0

    # Official ImageNet FreqPure parameters.
    amplitude_cut_range: int = 10
    phase_cut_range: int = 10
    phase_delta: float = 0.3
    forward_noise_steps: int = 50
    max_timestep: int = 50
    denoising_steps: int = 5
    purification_cycles: int = 8
    sampling_method: str = "ddpm"
    phase_mode: str = "official_code"

    purification_seed: int = 20260904
    use_amp: bool = True


OUTPUT_SPLITS = (
    "clean_resize_baseline",
    "adv_resize_baseline",
    "clean_diffpure",
    "adv_diffpure",
    "clean_freqpure",
    "adv_freqpure",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: str, map_location):
    # Work around Colab sessions where torch._utils was not attached lazily.
    importlib.import_module("torch._utils")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def list_pngs(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.png"))


def stable_batch_seed(base_seed: int, names: Sequence[str], stream: int = 0) -> int:
    token = "|".join(names).encode("utf-8")
    value = zlib.crc32(token)
    return int((base_seed + value + stream * 1_000_003) % (2**63 - 1))


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    return images.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()


def save_png_batch(
    images: torch.Tensor,
    names: Sequence[str],
    output_dir: Path,
    label_dir: Path,
) -> None:
    image_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    images_u8 = tensor_to_uint8(images)
    for image, name in zip(images_u8, names):
        array = image.permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(array, mode="RGB").save(
            image_dir / f"{name}.png", format="PNG", compress_level=3
        )
        source_label = label_dir / f"{name}.txt"
        destination_label = output_label_dir / f"{name}.txt"
        if source_label.exists() and not destination_label.exists():
            shutil.copy2(source_label, destination_label)


class SharedPGDDataset(Dataset):
    def __init__(self, shared_root: str, names: Sequence[str]):
        self.root = Path(shared_root)
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _read(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0

    def __getitem__(self, index: int):
        name = self.names[index]
        clean = self._read(self.root / "clean" / "images" / f"{name}.png")
        adv = self._read(self.root / "adv" / "images" / f"{name}.png")
        return clean, adv, name


def pair_collate(batch):
    clean, adv, names = zip(*batch)
    return torch.stack(clean), torch.stack(adv), list(names)


def load_shared_manifest(cfg: FreqPureDFireConfig) -> Tuple[Dict, List[str]]:
    manifest_path = Path(cfg.shared_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared PGD manifest: {manifest_path}. "
            "Generate the shared PGD set first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in (None, "complete"):
        raise RuntimeError(
            f"Shared PGD manifest is not complete: status={manifest.get('status')!r}"
        )
    names = list(manifest.get("names", []))
    if not names:
        raise RuntimeError("Shared PGD manifest contains no image names")
    if manifest.get("count") != len(names):
        raise RuntimeError(
            f"Manifest count={manifest.get('count')} but names={len(names)}"
        )
    if cfg.max_images is not None:
        if cfg.max_images > len(names):
            raise ValueError(
                f"max_images={cfg.max_images} exceeds manifest size {len(names)}"
            )
        if cfg.subset_seed is None:
            names = names[: cfg.max_images]
        else:
            # A fixed random subset is less sensitive to manifest ordering and
            # can be reused verbatim by another purifier through selected_names.
            names = random.Random(cfg.subset_seed).sample(names, cfg.max_images)
    for split in ("clean", "adv"):
        image_dir = Path(cfg.shared_root) / split / "images"
        missing = [name for name in names if not (image_dir / f"{name}.png").exists()]
        if missing:
            raise RuntimeError(
                f"Shared PGD {split} is incomplete; first missing image: {missing[0]}"
            )
    return manifest, names


def run_tag(cfg: FreqPureDFireConfig, count: int) -> str:
    delta = str(cfg.phase_delta).replace(".", "p")
    phase = "official" if cfg.phase_mode == "official_code" else "projection"
    subset = "ordered" if cfg.subset_seed is None else f"subset{cfg.subset_seed}"
    return (
        f"imagenet256_a{cfg.amplitude_cut_range}_p{cfg.phase_cut_range}_d{delta}_"
        f"t{cfg.max_timestep}_k{cfg.denoising_steps}_c{cfg.purification_cycles}_"
        f"{cfg.sampling_method}_{phase}_{cfg.output_mode}_{subset}_"
        f"seed{cfg.purification_seed}_n{count}"
    )


def prepare_run_root(
    cfg: FreqPureDFireConfig, manifest: Dict, names: Sequence[str]
) -> Path:
    root = Path(cfg.output_root) / run_tag(cfg, len(names))
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "freqpure_dfire_shared_pgd_v1",
        "config": asdict(cfg),
        "selected_count": len(names),
        "selected_names_crc32": zlib.crc32("|".join(names).encode("utf-8")),
        "shared_manifest": manifest,
    }
    identity_path = root / "run_config.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                f"Existing output has different configuration: {identity_path}. "
                "Use another --output-root."
            )
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    for split in active_output_splits(cfg):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    return root


def load_official_imagenet_diffusion(
    cfg: FreqPureDFireConfig, device: torch.device
) -> nn.Module:
    repo = Path(cfg.freqpure_repo)
    config_path = repo / "diffusion_configs" / "imagenet.yml"
    if not repo.is_dir() or not config_path.exists():
        raise FileNotFoundError(
            f"Official FreqPure repository not found at {repo}. In Colab run:\n"
            "!git clone https://github.com/GaozhengPei/FreqPure.git /content/FreqPure"
        )
    if not Path(cfg.diffusion_ckpt).is_file():
        raise FileNotFoundError(
            f"ImageNet diffusion checkpoint not found: {cfg.diffusion_ckpt}\n"
            "Download 256x256_diffusion_uncond.pt as shown in this script's header."
        )

    sys.path.insert(0, str(repo))
    from guided_diffusion.script_util import (  # type: ignore
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    official_model_config = raw_config.get("model", raw_config)
    defaults = model_and_diffusion_defaults()
    unknown = sorted(set(official_model_config) - set(defaults))
    model_config = {
        **defaults,
        **{key: value for key, value in official_model_config.items() if key in defaults},
    }
    if unknown:
        print("Ignoring non-model keys from official config:", unknown)

    model, _ = create_model_and_diffusion(**model_config)
    state = torch_load(cfg.diffusion_ckpt, "cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(
        str(key).startswith("module.") for key in state
    ):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print("Strictly loaded official ImageNet diffusion:", cfg.diffusion_ckpt)
    return model


class FreqPurePurifier(nn.Module):
    """Vectorized detection-task wrapper around official FreqPure purification."""

    def __init__(
        self,
        diffusion: nn.Module,
        cfg: FreqPureDFireConfig,
        device: torch.device,
    ):
        super().__init__()
        self.diffusion = diffusion
        self.cfg = cfg
        self.device = device
        self.eta = 1.0 if cfg.sampling_method == "ddpm" else 0.0
        betas = torch.linspace(1e-4, 2e-2, 1000, device=device)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))
        self._mask_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

        if cfg.purifier_size != 256:
            raise ValueError("Official ImageNet FreqPure backbone requires 256x256")
        if cfg.sampling_method not in ("ddpm", "ddim"):
            raise ValueError("sampling_method must be ddpm or ddim")
        if cfg.phase_mode not in ("official_code", "paper_projection"):
            raise ValueError(
                "phase_mode must be official_code or paper_projection"
            )
        if not 1 <= cfg.max_timestep <= 1000:
            raise ValueError("max_timestep must be in [1, 1000]")
        if not 0 <= cfg.forward_noise_steps <= 999:
            raise ValueError("forward_noise_steps must be in [0, 999]")
        if not 1 <= cfg.denoising_steps <= cfg.max_timestep:
            raise ValueError("invalid denoising_steps")

    @staticmethod
    def to_diffusion(x_01: torch.Tensor) -> torch.Tensor:
        return ((x_01 - 0.5) * 2.0).clamp(-1, 1)

    @staticmethod
    def to_image(x_diff: torch.Tensor) -> torch.Tensor:
        return (x_diff / 2.0 + 0.5).clamp(0, 1)

    def _alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        # timestep=-1 denotes alpha_bar=1 at the terminal x0 state.
        flat = timestep.long().flatten()
        values = torch.ones_like(flat, dtype=self.alpha_bar.dtype)
        nonterminal = flat >= 0
        if nonterminal.any():
            values[nonterminal] = self.alpha_bar.index_select(0, flat[nonterminal])
        return values.view(-1, 1, 1, 1)

    @staticmethod
    def _randn_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )

    def _noise(
        self,
        x: torch.Tensor,
        timestep_index: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        t = torch.full(
            (x.shape[0],), timestep_index, device=x.device, dtype=torch.long
        )
        alpha = self._alpha(t)
        noise = self._randn_like(x, generator)
        return x * alpha.sqrt() + noise * (1.0 - alpha).sqrt()

    def _predict_epsilon(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.use_amp and x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_context:
            output = self.diffusion(x, timestep.float())
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.shape[1] < 3:
            raise RuntimeError(f"Diffusion output shape is invalid: {output.shape}")
        # The official ImageNet model learns variance and returns six channels;
        # FreqPure uses the first three epsilon channels.
        return output[:, :3].float()

    def _frequency_mask(self, height: int, width: int, radius: int) -> torch.Tensor:
        key = (height, width, radius)
        if key not in self._mask_cache:
            y = torch.arange(-height // 2, height // 2, device=self.device)
            x = torch.arange(-width // 2, width // 2, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            mask = (xx.float().square() + yy.float().square()).sqrt() <= radius
            self._mask_cache[key] = mask[None, None]
        return self._mask_cache[key]

    def _frequency_exchange(
        self,
        reference_diff: torch.Tensor,
        estimated_x0_diff: torch.Tensor,
        reference_generator: torch.Generator,
    ) -> torch.Tensor:
        # Official code denoises a separately noised copy of the current input
        # to construct the frequency reference at every reverse step.
        # Unlike def_max_timesteps (which is converted from 50 to index 49 by
        # get_diffusion_params), the released FreqPure code uses U=50 directly
        # in amplitude_phase_exchange_torch. Keep that asymmetry here.
        reference_t = self.cfg.forward_noise_steps
        reference_noised = self._noise(
            reference_diff, reference_t, reference_generator
        )
        t = torch.full(
            (reference_diff.shape[0],),
            reference_t,
            device=reference_diff.device,
            dtype=torch.long,
        )
        alpha = self._alpha(t)
        epsilon = self._predict_epsilon(reference_noised, t)
        reference_x0 = (
            reference_noised.float() - epsilon * (1.0 - alpha).sqrt()
        ) / alpha.sqrt()

        # Official implementation performs FFT on [0,255] RGB values.
        reference_rgb = self.to_image(reference_x0).float() * 255.0
        estimate_rgb = self.to_image(estimated_x0_diff).float() * 255.0
        reference_fft = torch.fft.fftshift(
            torch.fft.fft2(reference_rgb, dim=(-2, -1)), dim=(-2, -1)
        )
        estimate_fft = torch.fft.fftshift(
            torch.fft.fft2(estimate_rgb, dim=(-2, -1)), dim=(-2, -1)
        )
        reference_amplitude = reference_fft.abs()
        estimate_amplitude = estimate_fft.abs()
        # The +pi representation matches the official phase code.
        reference_phase = torch.angle(reference_fft) + math.pi
        estimate_phase = torch.angle(estimate_fft) + math.pi

        height, width = estimate_rgb.shape[-2:]
        amplitude_mask = self._frequency_mask(
            height, width, self.cfg.amplitude_cut_range
        )
        phase_mask = self._frequency_mask(height, width, self.cfg.phase_cut_range)
        mixed_amplitude = torch.where(
            amplitude_mask, reference_amplitude, estimate_amplitude
        )

        if self.cfg.phase_mode == "official_code":
            # This exactly follows the released implementation: low-frequency
            # phase is replaced first; the subsequent clip is therefore a no-op.
            mixed_phase = torch.where(phase_mask, reference_phase, estimate_phase)
            clipped = torch.maximum(
                torch.minimum(
                    mixed_phase, reference_phase + self.cfg.phase_delta
                ),
                reference_phase - self.cfg.phase_delta,
            )
            mixed_phase = torch.where(phase_mask, clipped, mixed_phase)
        else:
            # The paper describes projection, not hard replacement.  This mode
            # is provided as a separately named ablation and is not the default.
            projected = torch.maximum(
                torch.minimum(
                    estimate_phase, reference_phase + self.cfg.phase_delta
                ),
                reference_phase - self.cfg.phase_delta,
            )
            mixed_phase = torch.where(phase_mask, projected, estimate_phase)

        mixed_fft = mixed_amplitude * torch.exp(
            1j * (mixed_phase - math.pi)
        )
        reconstructed = torch.fft.ifft2(
            torch.fft.ifftshift(mixed_fft, dim=(-2, -1)), dim=(-2, -1)
        ).abs()
        reconstructed_01 = reconstructed.clamp(0, 255) / 255.0
        return self.to_diffusion(reconstructed_01)

    def _reverse_sequence(self) -> List[int]:
        stride = max(self.cfg.max_timestep // self.cfg.denoising_steps, 1)
        sequence = [
            value - 1
            for value in range(stride, self.cfg.max_timestep + 1, stride)
        ]
        if sequence[-1] != self.cfg.max_timestep - 1:
            sequence.append(self.cfg.max_timestep - 1)
        return sequence

    def _denoise_cycle(
        self,
        reference_diff: torch.Tensor,
        noised: torch.Tensor,
        use_frequency: bool,
        main_generator: torch.Generator,
        reference_generator: torch.Generator,
    ) -> torch.Tensor:
        sequence = self._reverse_sequence()
        next_sequence = [-1] + sequence[:-1]
        state = noised
        for current, next_value in zip(
            reversed(sequence), reversed(next_sequence)
        ):
            t = torch.full(
                (state.shape[0],), current, device=state.device, dtype=torch.long
            )
            next_t = torch.full_like(t, next_value)
            alpha = self._alpha(t)
            alpha_next = self._alpha(next_t)
            epsilon = self._predict_epsilon(state, t)
            estimated_x0 = (
                state.float() - epsilon * (1.0 - alpha).sqrt()
            ) / alpha.sqrt()
            if use_frequency:
                estimated_x0 = self._frequency_exchange(
                    reference_diff, estimated_x0, reference_generator
                )

            c1 = self.eta * torch.sqrt(
                ((1.0 - alpha / alpha_next) * (1.0 - alpha_next) /
                 (1.0 - alpha).clamp_min(1e-12)).clamp_min(0.0)
            )
            c2 = torch.sqrt((1.0 - alpha_next - c1.square()).clamp_min(0.0))
            reverse_noise = self._randn_like(state, main_generator)
            state = (
                alpha_next.sqrt() * estimated_x0
                + c1 * reverse_noise
                + c2 * epsilon
            )
        return state

    @torch.inference_mode()
    def purify(
        self,
        images_01: torch.Tensor,
        use_frequency: bool,
        main_seed: int,
        reference_seed: int,
    ) -> torch.Tensor:
        if tuple(images_01.shape[-2:]) != (256, 256):
            images_01 = F.interpolate(
                images_01,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
        state = self.to_diffusion(images_01)
        main_generator = torch.Generator(device=state.device).manual_seed(main_seed)
        reference_generator = torch.Generator(device=state.device).manual_seed(
            reference_seed
        )
        for _ in range(self.cfg.purification_cycles):
            reference = state
            noised = self._noise(
                state, self.cfg.max_timestep - 1, main_generator
            )
            state = self._denoise_cycle(
                reference,
                noised,
                use_frequency,
                main_generator,
                reference_generator,
            )
        return self.to_image(state)


def roundtrip_resize(
    images: torch.Tensor, purifier_size: int, detector_size: int
) -> torch.Tensor:
    reduced = F.interpolate(
        images, size=(purifier_size, purifier_size), mode="bilinear", align_corners=False
    )
    return F.interpolate(
        reduced, size=(detector_size, detector_size), mode="bilinear", align_corners=False
    )


def output_batch_exists(root: Path, split: str, names: Sequence[str]) -> bool:
    image_dir = root / split / "images"
    return all((image_dir / f"{name}.png").exists() for name in names)


def generate_outputs(
    cfg: FreqPureDFireConfig,
    names: Sequence[str],
    run_root: Path,
    device: torch.device,
) -> None:
    diffusion = load_official_imagenet_diffusion(cfg, device)
    purifier = FreqPurePurifier(diffusion, cfg, device).eval()
    dataset = SharedPGDDataset(cfg.shared_root, names)
    loader = DataLoader(
        dataset,
        batch_size=cfg.purify_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pair_collate,
    )
    labels = Path(cfg.shared_root) / "clean" / "labels"
    started = time.perf_counter()
    processed = 0
    enabled = set(active_output_splits(cfg))

    for clean, adv, batch_names in tqdm(loader, desc="FreqPure D-Fire"):
        clean = clean.to(device, non_blocking=True)
        adv = adv.to(device, non_blocking=True)

        if (
            "clean_resize_baseline" in enabled
            and not output_batch_exists(
                run_root, "clean_resize_baseline", batch_names
            )
        ):
            save_png_batch(
                roundtrip_resize(clean, cfg.purifier_size, cfg.detector_size),
                batch_names,
                run_root / "clean_resize_baseline",
                labels,
            )
        if (
            "adv_resize_baseline" in enabled
            and not output_batch_exists(
                run_root, "adv_resize_baseline", batch_names
            )
        ):
            save_png_batch(
                roundtrip_resize(adv, cfg.purifier_size, cfg.detector_size),
                batch_names,
                run_root / "adv_resize_baseline",
                labels,
            )

        main_seed = stable_batch_seed(cfg.purification_seed, batch_names, stream=0)
        reference_seed = stable_batch_seed(
            cfg.purification_seed, batch_names, stream=1
        )
        for prefix, images in (("clean", clean), ("adv", adv)):
            diffpure_name = f"{prefix}_diffpure"
            if (
                diffpure_name in enabled
                and not output_batch_exists(run_root, diffpure_name, batch_names)
            ):
                purified = purifier.purify(
                    images,
                    use_frequency=False,
                    main_seed=main_seed,
                    reference_seed=reference_seed,
                )
                purified384 = F.interpolate(
                    purified,
                    size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear",
                    align_corners=False,
                )
                save_png_batch(
                    purified384,
                    batch_names,
                    run_root / diffpure_name,
                    labels,
                )

            freqpure_name = f"{prefix}_freqpure"
            if (
                freqpure_name in enabled
                and not output_batch_exists(run_root, freqpure_name, batch_names)
            ):
                purified = purifier.purify(
                    images,
                    use_frequency=True,
                    main_seed=main_seed,
                    reference_seed=reference_seed,
                )
                purified384 = F.interpolate(
                    purified,
                    size=(cfg.detector_size, cfg.detector_size),
                    mode="bilinear",
                    align_corners=False,
                )
                save_png_batch(
                    purified384,
                    batch_names,
                    run_root / freqpure_name,
                    labels,
                )
        processed += len(batch_names)
        if processed == len(batch_names) or processed % 32 == 0:
            elapsed = time.perf_counter() - started
            rate = processed / max(elapsed, 1e-9)
            eta = (len(names) - processed) / max(rate, 1e-9)
            print(
                f"processed={processed}/{len(names)}, "
                f"{rate:.3f} images/s, ETA={eta / 60:.1f} min"
            )

    del purifier, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_outputs(
    cfg: FreqPureDFireConfig, run_root: Path, names: Sequence[str]
) -> None:
    expected = set(names)
    for split in active_output_splits(cfg):
        actual = {path.stem for path in list_pngs(run_root / split / "images")}
        missing = expected - actual
        unexpected = actual - expected
        print(f"{split:24s} images={len(actual)}/{len(expected)}")
        if missing or unexpected:
            raise RuntimeError(
                f"Incomplete {split}: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )


def write_image_list(path: Path, image_paths: Iterable[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = [str(item.resolve()) for item in image_paths]
    path.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return path


def eval_map(
    detector,
    image_list: Path,
    cfg: FreqPureDFireConfig,
    run_root: Path,
    name: str,
    device: torch.device,
) -> Tuple[float, float]:
    yaml_path = run_root / "map_data" / f"{name}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": "/",
                "train": str(image_list.resolve()),
                "val": str(image_list.resolve()),
                "names": {0: "fire", 1: "smoke"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = detector.val(
        data=str(yaml_path),
        imgsz=cfg.detector_size,
        batch=cfg.detector_batch_size,
        verbose=False,
        plots=False,
        project=str(run_root / "ultralytics_val"),
        name=name,
        exist_ok=True,
        device=0 if device.type == "cuda" else "cpu",
    )
    if hasattr(result, "box"):
        return float(result.box.map50), float(result.box.map)
    values = getattr(result, "results_dict", {})
    return (
        float(values["metrics/mAP50(B)"]),
        float(values["metrics/mAP50-95(B)"]),
    )


def compute_maps(
    cfg: FreqPureDFireConfig,
    manifest: Dict,
    names: Sequence[str],
    run_root: Path,
    device: torch.device,
) -> Dict:
    verify_outputs(cfg, run_root, names)
    if not Path(cfg.yolo_ckpt).is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {cfg.yolo_ckpt}")
    detector = YOLO(cfg.yolo_ckpt)

    lists_dir = run_root / "image_lists"
    shared_root = Path(cfg.shared_root)
    directories: Dict[str, List[Path]] = {
        "original_clean": [
            shared_root / "clean" / "images" / f"{name}.png" for name in names
        ],
        "adv_original": [
            shared_root / "adv" / "images" / f"{name}.png" for name in names
        ],
    }
    for split in active_output_splits(cfg):
        directories[split] = [
            run_root / split / "images" / f"{name}.png" for name in names
        ]

    results: Dict[str, Dict[str, float]] = {}
    for name, paths in directories.items():
        image_list = write_image_list(lists_dir / f"{name}.txt", paths)
        map50, map50_95 = eval_map(
            detector, image_list, cfg, run_root, name, device
        )
        results[name] = {"map50": map50, "map50_95": map50_95}
        print(
            f"{name:24s} mAP50={map50:.4f}  mAP50-95={map50_95:.4f}"
        )

    clean = results["original_clean"]
    adv = results["adv_original"]

    def recovery(name: str, metric: str) -> float:
        denominator = max(clean[metric] - adv[metric], 1e-8)
        return (results[name][metric] - adv[metric]) / denominator

    derived: Dict[str, float] = {
        "resize_recovery_map50": recovery("adv_resize_baseline", "map50"),
        "resize_recovery_map50_95": recovery(
            "adv_resize_baseline", "map50_95"
        ),
        "freqpure_recovery_map50": recovery("adv_freqpure", "map50"),
        "freqpure_recovery_map50_95": recovery("adv_freqpure", "map50_95"),
    }
    if "clean_freqpure" in results:
        derived["freqpure_clean_drop_map50"] = (
            clean["map50"] - results["clean_freqpure"]["map50"]
        )
    if "adv_diffpure" in results:
        derived["diffpure_recovery_map50"] = recovery(
            "adv_diffpure", "map50"
        )
        derived["diffpure_recovery_map50_95"] = recovery(
            "adv_diffpure", "map50_95"
        )
        derived["diffpure_gain_over_resize_map50"] = (
            results["adv_diffpure"]["map50"]
            - results["adv_resize_baseline"]["map50"]
        )
        derived["freqpure_gain_over_diffpure_map50"] = (
            results["adv_freqpure"]["map50"]
            - results["adv_diffpure"]["map50"]
        )
    if "clean_diffpure" in results:
        derived["diffpure_clean_drop_map50"] = (
            clean["map50"] - results["clean_diffpure"]["map50"]
        )

    print("\n=== FreqPure D-Fire shared-PGD comparison ===")
    print(
        f"FreqPure recovery: mAP50={derived['freqpure_recovery_map50']:.1%}, "
        f"mAP50-95={derived['freqpure_recovery_map50_95']:.1%}"
    )
    if "diffpure_recovery_map50" in derived:
        print(
            f"DiffPure control recovery: "
            f"mAP50={derived['diffpure_recovery_map50']:.1%}, "
            f"mAP50-95={derived['diffpure_recovery_map50_95']:.1%}"
        )
        print(
            "FreqPure gain over matched DiffPure mAP50: "
            f"{derived['freqpure_gain_over_diffpure_map50']:+.4f}"
        )
    if "freqpure_clean_drop_map50" in derived:
        print(
            "FreqPure clean mAP50 loss: "
            f"{derived['freqpure_clean_drop_map50']:+.4f}"
        )

    report = {
        "schema": "freqpure_dfire_shared_pgd_metrics_v1",
        "config": asdict(cfg),
        "shared_attack_manifest": manifest,
        "selected_names": list(names),
        "results": results,
        "derived": derived,
        "method_note": (
            "Official ImageNet-256 FreqPure frequency algorithm adapted from "
            "classification to D-Fire YOLO detection. This is an off-the-shelf "
            "transfer evaluation, not a D-Fire-trained diffusion model."
        ),
    }
    metrics_path = run_root / "metrics_freqpure_dfire.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Metrics saved:", metrics_path)
    return report


def parse_args() -> Tuple[str, FreqPureDFireConfig]:
    defaults = FreqPureDFireConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("purify", "map", "all"), default="all")
    parser.add_argument("--shared-root", default=defaults.shared_root)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--yolo-ckpt", default=defaults.yolo_ckpt)
    parser.add_argument("--freqpure-repo", default=defaults.freqpure_repo)
    parser.add_argument("--diffusion-ckpt", default=defaults.diffusion_ckpt)
    parser.add_argument(
        "--max-images",
        type=int,
        default=defaults.max_images,
        help="0 means all manifest images; default 32 is a smoke run",
    )
    parser.add_argument("--purify-batch-size", type=int, default=defaults.purify_batch_size)
    parser.add_argument("--detector-batch-size", type=int, default=defaults.detector_batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument(
        "--output-mode",
        choices=("all", "freqpure_only", "adv_freqpure_only"),
        default=defaults.output_mode,
        help="Select generated controls; adv_freqpure_only is the fastest recovery test",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=defaults.subset_seed,
        help="Deterministically sample max-images instead of taking manifest prefix",
    )
    parser.add_argument("--amplitude-cut-range", type=int, default=defaults.amplitude_cut_range)
    parser.add_argument("--phase-cut-range", type=int, default=defaults.phase_cut_range)
    parser.add_argument("--phase-delta", type=float, default=defaults.phase_delta)
    parser.add_argument("--forward-noise-steps", type=int, default=defaults.forward_noise_steps)
    parser.add_argument("--max-timestep", type=int, default=defaults.max_timestep)
    parser.add_argument("--denoising-steps", type=int, default=defaults.denoising_steps)
    parser.add_argument("--purification-cycles", type=int, default=defaults.purification_cycles)
    parser.add_argument("--sampling-method", choices=("ddpm", "ddim"), default=defaults.sampling_method)
    parser.add_argument(
        "--phase-mode",
        choices=("official_code", "paper_projection"),
        default=defaults.phase_mode,
    )
    parser.add_argument("--purification-seed", type=int, default=defaults.purification_seed)
    parser.add_argument("--no-amp", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook arguments:", unknown)

    if args.max_images < 0:
        parser.error("--max-images must be >= 0")
    max_images = None if args.max_images == 0 else args.max_images
    cfg = FreqPureDFireConfig(
        shared_root=args.shared_root,
        output_root=args.output_root,
        yolo_ckpt=args.yolo_ckpt,
        freqpure_repo=args.freqpure_repo,
        diffusion_ckpt=args.diffusion_ckpt,
        max_images=max_images,
        purify_batch_size=args.purify_batch_size,
        detector_batch_size=args.detector_batch_size,
        num_workers=args.num_workers,
        output_mode=args.output_mode,
        subset_seed=args.subset_seed,
        amplitude_cut_range=args.amplitude_cut_range,
        phase_cut_range=args.phase_cut_range,
        phase_delta=args.phase_delta,
        forward_noise_steps=args.forward_noise_steps,
        max_timestep=args.max_timestep,
        denoising_steps=args.denoising_steps,
        purification_cycles=args.purification_cycles,
        sampling_method=args.sampling_method,
        phase_mode=args.phase_mode,
        purification_seed=args.purification_seed,
        use_amp=not args.no_amp,
    )
    return args.stage, cfg


def main() -> None:
    stage, cfg = parse_args()
    seed_everything(cfg.purification_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    manifest, names = load_shared_manifest(cfg)
    run_root = prepare_run_root(cfg, manifest, names)
    print("Shared PGD:", cfg.shared_root)
    print("Selected images:", len(names))
    print("Output:", run_root)

    if stage in ("purify", "all"):
        generate_outputs(cfg, names, run_root, device)
        verify_outputs(cfg, run_root, names)
    if stage in ("map", "all"):
        compute_maps(cfg, manifest, names, run_root, device)


if __name__ == "__main__":
    main()
