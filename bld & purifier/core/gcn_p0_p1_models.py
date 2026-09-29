"""Inference-only GCN model definitions used by the P0/P1 audit.

This module deliberately contains no optimizer or training entry point.  It
reuses the Attn/base blocks from ``shared_pgd_gcn_flowpure_compare.py`` and
adds the fixed-hidden-width skip quantizer used by the preserved 3072->6144
expansion.  Checkpoints are always loaded strictly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import shared_pgd_gcn_flowpure_compare as shared
except ImportError:  # Allows ``%run /content/core/...`` and direct execution.
    import shared_pgd_gcn_flowpure_compare as shared


OLD_SKIP_SHAPES: Dict[str, Tuple[int, int]] = {
    "64x64": (2, 16),
    "32x32": (8, 8),
}
EXPANDED_SKIP_SHAPES: Dict[str, Tuple[int, int]] = {
    "64x64": (8, 16),
    "32x32": (32, 8),
}
FIXED_SKIP_HIDDEN: Dict[str, int] = {"64x64": 32, "32x32": 64}


@dataclass
class AuditGCNConfig:
    active_levels: Optional[List[str]] = None
    block_cfg: shared.GCNBlockConfig = field(default_factory=shared.GCNBlockConfig)
    skip_quantizer_mode: str = "learned_v2"
    skip_level_code_shapes: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: OLD_SKIP_SHAPES.copy()
    )
    skip_hidden_channels: Dict[str, int] = field(
        default_factory=lambda: FIXED_SKIP_HIDDEN.copy()
    )

    def code_shape_for(self, level_name: str) -> Tuple[int, int]:
        channels, spatial = self.skip_level_code_shapes[level_name]
        return int(channels), int(spatial)


class FixedWidthSkipQuantizer(nn.Module):
    """Learned-v2 quantizer with an explicit, checkpoint-stable hidden width."""

    def __init__(
        self,
        channels: int,
        code_channels: int,
        code_spatial: int,
        source_spatial: int,
        hidden_channels: int,
        mode: str = "learned_v2",
    ):
        super().__init__()
        if mode != "learned_v2":
            raise ValueError("P0/P1 expects learned_v2 skip quantizers")
        self.channels = int(channels)
        self.code_channels = int(code_channels)
        self.code_spatial = int(code_spatial)
        self.source_spatial = int(source_spatial)
        self.hidden_channels = int(hidden_channels)
        self.mode = mode
        self.n_bits = self.code_channels * self.code_spatial * self.code_spatial

        ratio = self.source_spatial // self.code_spatial
        if (
            self.source_spatial % self.code_spatial
            or ratio < 1
            or ratio & (ratio - 1)
        ):
            raise ValueError("source/code spatial ratio must be a positive power of two")

        hidden = self.hidden_channels
        encoder_layers: List[nn.Module] = [
            nn.Conv2d(self.channels, hidden, 3, padding=1),
            nn.GroupNorm(shared.skip_group_count(hidden), hidden),
            nn.SiLU(),
        ]
        current = self.source_spatial
        while current > self.code_spatial:
            encoder_layers.extend(
                [
                    nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                    nn.GroupNorm(shared.skip_group_count(hidden), hidden),
                    nn.SiLU(),
                ]
            )
            current //= 2
        encoder_layers.append(nn.Conv2d(hidden, self.code_channels, 1))
        self.encoder_net = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = [
            nn.Conv2d(self.code_channels, hidden, 3, padding=1),
            nn.GroupNorm(shared.skip_group_count(hidden), hidden),
            nn.SiLU(),
        ]
        current = self.code_spatial
        while current < self.source_spatial:
            decoder_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.GroupNorm(shared.skip_group_count(hidden), hidden),
                    nn.SiLU(),
                ]
            )
            current *= 2
        decoder_layers.append(nn.Conv2d(hidden, self.channels, 3, padding=1))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def logits(self, encoder_feature: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(encoder_feature)

    def probability(self, encoder_feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(encoder_feature))

    @staticmethod
    def hard_from_probability(probability: torch.Tensor) -> torch.Tensor:
        return (probability > 0.5).to(probability.dtype)

    def encode(self, encoder_feature: torch.Tensor, hard: bool = True) -> torch.Tensor:
        probability = self.probability(encoder_feature)
        return self.hard_from_probability(probability) if hard else probability

    def decode(self, code: torch.Tensor, spatial_size: int) -> torch.Tensor:
        # Intentionally does not threshold: P1 soft-code inputs must reach the
        # decoder unchanged.  Soft-code results are diagnostic/OOD conditions.
        output = self.decoder_net(code.float())
        if output.shape[-2:] != (spatial_size, spatial_size):
            output = F.interpolate(
                output,
                size=(spatial_size, spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return output

    def forward(self, encoder_feature: torch.Tensor):
        code = self.encode(encoder_feature, hard=True)
        return self.decode(code, encoder_feature.shape[-1]), code


class AuditDecoderGCN(nn.Module):
    def __init__(self, base_decoder, levels, config: AuditGCNConfig):
        super().__init__()
        self.base_decoder = base_decoder
        active = set(config.active_levels) if config.active_levels is not None else None
        self.gcn_blocks = nn.ModuleDict()
        self.skip_quantizers = nn.ModuleDict()
        self._insert_after: Dict[int, str] = {}
        for level in levels:
            if active is not None and level.name not in active:
                continue
            self.gcn_blocks[level.name] = shared.GraphConvSkipBlock(
                level.channels, config.block_cfg
            )
            code_channels, code_spatial = config.code_shape_for(level.name)
            self.skip_quantizers[level.name] = FixedWidthSkipQuantizer(
                channels=level.channels,
                code_channels=code_channels,
                code_spatial=code_spatial,
                source_spatial=level.spatial_size,
                hidden_channels=config.skip_hidden_channels[level.name],
                mode=config.skip_quantizer_mode,
            )
            self._insert_after[level.dec_insert_after] = level.name

    def forward_from_codes(
        self, main_code: torch.Tensor, skip_codes: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        expected = set(self.gcn_blocks)
        if set(skip_codes) != expected:
            raise KeyError(
                f"skip stream mismatch: expected={sorted(expected)}, "
                f"actual={sorted(skip_codes)}"
            )
        hidden = main_code.float()
        for index, module in enumerate(self.base_decoder.conv_layers):
            hidden = module(hidden)
            if index in self._insert_after:
                name = self._insert_after[index]
                quantizer = self.skip_quantizers[name]
                feature = quantizer.decode(skip_codes[name], quantizer.source_spatial)
                hidden = self.gcn_blocks[name](hidden, feature)
        return hidden

    def forward(self, main_code, encoder_features):
        skip_codes = {
            name: quantizer.encode(encoder_features[name], hard=True)
            for name, quantizer in self.skip_quantizers.items()
        }
        return self.forward_from_codes(main_code, skip_codes)

    def skip_bit_budget(self) -> int:
        return sum(item.n_bits for item in self.skip_quantizers.values())


class AuditGCNAutoEncoder(nn.Module):
    def __init__(self, skip_shapes: Dict[str, Tuple[int, int]]):
        super().__init__()
        base = shared.BinaryAutoEncoderAttn()
        levels = shared.auto_detect_skip_levels(base, shared.IMG_SIZE)
        config = AuditGCNConfig(skip_level_code_shapes=skip_shapes.copy())
        self.encoder = shared.EncoderGCN(base.encoder, levels)
        self.decoder = AuditDecoderGCN(base.decoder, levels, config)
        self.latent_h = base.latent_h
        self.latent_w = base.latent_w
        self.D = base.D
        self.skip_shapes = skip_shapes.copy()

    def total_bit_budget(self) -> int:
        return self.D + self.decoder.skip_bit_budget()

    def _encoder_logits_and_features(self, images: torch.Tensor):
        features = {}
        hidden = images
        for index, module in enumerate(self.encoder.base_encoder.conv_layers):
            hidden = module(hidden)
            if index in self.encoder._hook_at:
                features[self.encoder._hook_at[index]] = hidden
        return hidden, features

    def encode_representations(self, images: torch.Tensor) -> Dict[str, object]:
        """Return logits, probabilities and thresholded hard codes for all streams."""
        main_logits, features = self._encoder_logits_and_features(images)
        main_probability = torch.sigmoid(main_logits)
        skip_logits = {}
        skip_probability = {}
        skip_hard = {}
        for name, quantizer in self.decoder.skip_quantizers.items():
            logits = quantizer.logits(features[name])
            probability = torch.sigmoid(logits)
            skip_logits[name] = logits
            skip_probability[name] = probability
            skip_hard[name] = (probability > 0.5).to(probability.dtype)
        return {
            "main_logits": main_logits,
            "main_probability": main_probability,
            "main_hard": (main_probability > 0.5).to(main_probability.dtype),
            "skip_logits": skip_logits,
            "skip_probability": skip_probability,
            "skip_hard": skip_hard,
        }

    def encode_all_codes(self, images: torch.Tensor, hard: bool = True):
        values = self.encode_representations(images)
        if hard:
            return values["main_hard"], values["skip_hard"]
        return values["main_probability"], values["skip_probability"]

    def decode_from_codes(self, main_code, skip_codes):
        return self.decoder.forward_from_codes(main_code, skip_codes)

    def forward(self, images: torch.Tensor):
        main, skips = self.encode_all_codes(images, hard=True)
        return self.decode_from_codes(main, skips), main


def unwrap_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
    return raw


def load_model(
    checkpoint: str,
    expanded: bool,
    device: torch.device,
) -> AuditGCNAutoEncoder:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    shapes = EXPANDED_SKIP_SHAPES if expanded else OLD_SKIP_SHAPES
    model = AuditGCNAutoEncoder(shapes).to(device)
    try:
        raw = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=device)
    model.load_state_dict(unwrap_state_dict(raw), strict=True)
    model.eval()
    expected_bits = 6144 if expanded else 3072
    if model.total_bit_budget() != expected_bits:
        raise RuntimeError(
            f"bit budget mismatch: {model.total_bit_budget()} != {expected_bits}"
        )
    return model


def old_only_skip_codes(
    expanded_skip_codes: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Disable every added skip channel while preserving expanded-model encoding."""
    result = {}
    for name, code in expanded_skip_codes.items():
        old_channels = OLD_SKIP_SHAPES[name][0]
        if code.shape[1] < old_channels:
            raise ValueError(f"{name} has fewer than {old_channels} channels")
        masked = torch.zeros_like(code)
        masked[:, :old_channels].copy_(code[:, :old_channels])
        result[name] = masked
    return result


def compare_inherited_state(
    old_model: AuditGCNAutoEncoder,
    expanded_model: AuditGCNAutoEncoder,
) -> Dict[str, object]:
    """Check unchanged tensors and old slices of the three widened tensors."""
    old_state = old_model.state_dict()
    new_state = expanded_model.state_dict()
    if set(old_state) != set(new_state):
        raise RuntimeError("3072 and 6144 state_dict keys differ")

    widened = {}
    for level, quantizer in expanded_model.decoder.skip_quantizers.items():
        prefix = f"decoder.skip_quantizers.{level}."
        last = len(quantizer.encoder_net) - 1
        widened[prefix + f"encoder_net.{last}.weight"] = (0, OLD_SKIP_SHAPES[level][0])
        widened[prefix + f"encoder_net.{last}.bias"] = (0, OLD_SKIP_SHAPES[level][0])
        widened[prefix + "decoder_net.0.weight"] = (1, OLD_SKIP_SHAPES[level][0])

    mismatches = []
    compared = 0
    for key, old_value in old_state.items():
        new_value = new_state[key]
        if key in widened:
            axis, count = widened[key]
            index = [slice(None)] * new_value.ndim
            index[axis] = slice(0, count)
            candidate = new_value[tuple(index)]
        else:
            candidate = new_value
        compared += old_value.numel()
        if old_value.shape != candidate.shape or not torch.equal(old_value.cpu(), candidate.cpu()):
            delta = None
            if old_value.shape == candidate.shape:
                delta = float((old_value.cpu() - candidate.cpu()).abs().max())
            mismatches.append(
                {"name": key, "old_shape": list(old_value.shape),
                 "new_slice_shape": list(candidate.shape), "max_abs": delta}
            )
    return {
        "equal": not mismatches,
        "compared_scalar_count": compared,
        "mismatches": mismatches,
        "widened_tensor_rules": {
            key: {"axis": axis, "old_count": count}
            for key, (axis, count) in widened.items()
        },
    }
