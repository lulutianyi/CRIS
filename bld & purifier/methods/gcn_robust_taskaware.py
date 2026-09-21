"""Adversarial task-aware Robust-v4 fine-tuning cell for gcn.ipynb.

Run from the notebook with ``%run -i /content/methods/gcn_robust_taskaware.py`` after
the Learned-v2 GCN cell.  The serialized architecture and 3072-bit budget do
not change, so the resulting state_dict remains compatible with the shared
evaluators.  Robust-v4 changes the trainable path: it lightly unfreezes the
encoder tail and explicitly regularizes main/skip codes, swapped streams, and
random bit corruption.
"""

from dataclasses import asdict, dataclass
import copy
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


ROBUST_DIR = "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust_v4"
ROBUST_FINAL = ROBUST_DIR + "/gcn_skip_learned_v2_robust_v4_final.pt"
ROBUST_V3_FINAL = (
    "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust/"
    "gcn_skip_learned_v2_robust_taskaware_final.pt"
)


@dataclass
class RobustTaskAwareConfig:
    epochs: int = 6
    batch_size: int = 2
    lr: float = 2e-5
    encoder_lr_multiplier: float = 0.2
    new_module_lr_multiplier: float = 2.0
    weight_decay: float = 1e-4
    encoder_unfreeze_tail_modules: int = 2
    clean_l1_weight: float = 0.7
    clean_mse_weight: float = 0.3
    adv_reconstruction_weight: float = 0.5
    consistency_weight: float = 0.25
    code_consistency_weight: float = 0.10
    code_margin_weight: float = 0.02
    swap_reconstruction_weight: float = 0.25
    bit_corruption_weight: float = 0.15
    bit_flip_probability: float = 0.01
    clean_detection_weight: float = 0.005
    adv_detection_weight: float = 0.01
    detection_warmup_epochs: int = 2
    pgd_eps: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps_start: int = 3
    pgd_steps_end: int = 10
    grad_clip: float = 1.0
    num_workers: int = 2
    max_train_batches: int | None = None
    map_eval_samples: int | None = 512
    map_eval_batch_size: int = 16
    map_eval_dir: str = "/content/gcn_robust_clean_map_eval"
    detector_ckpt: str = (
        "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    )
    initial_checkpoint: str = ROBUST_V3_FINAL
    local_dir: str = "/content/gcn_skip_learned_v2_robust_v4"
    drive_dir: str = ROBUST_DIR
    final_alias: str = ROBUST_FINAL


def _require_notebook_context():
    required = (
        "ae_gcn", "bld", "DFireDetDataset", "det_collate",
        "_sanitize_det_targets", "_LossAttrDict", "evaluate_gcn_ae_map",
        "unwrap_state_dict", "LEARNED_V2_GCN_CKPT",
    )
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Run the notebook through the Learned-v2 definition cell first; "
            f"missing globals: {missing}"
        )


def make_robust_v4_initial_model(cfg):
    """Prefer the previous robust checkpoint so v4 does not discard its gain."""
    model = copy.deepcopy(ae_gcn).to(bld.DEVICE)
    candidates = [cfg.initial_checkpoint, LEARNED_V2_GCN_CKPT]
    for checkpoint in candidates:
        if checkpoint and os.path.isfile(checkpoint):
            state = torch.load(checkpoint, map_location=bld.DEVICE)
            model.load_state_dict(unwrap_state_dict(state), strict=True)
            cfg._resolved_initial_checkpoint = checkpoint
            print("Robust-v4 strictly initialized from:", checkpoint)
            return model
    raise FileNotFoundError(
        "Neither the previous Robust-v3 checkpoint nor the Learned-v2 "
        f"checkpoint exists: {candidates}"
    )


def _prepare_detector(cfg):
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    detector = YOLO(cfg.detector_ckpt)
    detector.model.to(bld.DEVICE).train()
    for parameter in detector.model.parameters():
        parameter.requires_grad_(False)
    for module in detector.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = v8DetectionLoss(detector.model)
    defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    if isinstance(criterion.hyp, dict):
        wrapped = _LossAttrDict(criterion.hyp)
        for key, value in defaults.items():
            wrapped.setdefault(key, value)
        criterion.hyp = wrapped
    else:
        for key, value in defaults.items():
            if not hasattr(criterion.hyp, key):
                setattr(criterion.hyp, key, value)
    return detector, criterion


def _pgd(detector, criterion, clean, targets, cfg, steps):
    lower = (clean - cfg.pgd_eps).clamp(0, 1)
    upper = (clean + cfg.pgd_eps).clamp(0, 1)
    adv = (clean + torch.empty_like(clean).uniform_(
        -cfg.pgd_eps, cfg.pgd_eps
    )).clamp(0, 1)
    with torch.enable_grad():
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            loss_vector, _ = criterion(detector(adv), targets)
            gradient = torch.autograd.grad(
                loss_vector.sum(), adv, only_inputs=True
            )[0]
            adv = adv.detach() + cfg.pgd_alpha * gradient.sign()
            adv = torch.maximum(torch.minimum(adv, upper), lower).clamp(0, 1)
    return adv.detach()


def _save_epoch(model, cfg, epoch, clean_map50, clean_map5095):
    os.makedirs(cfg.local_dir, exist_ok=True)
    os.makedirs(cfg.drive_dir, exist_ok=True)
    filename = f"gcn_skip_learned_v2_robust_epoch{epoch}.pt"
    local_path = os.path.join(cfg.local_dir, filename)
    drive_path = os.path.join(cfg.drive_dir, filename)
    torch.save(model.state_dict(), local_path)
    shutil.copy2(local_path, drive_path)
    metadata = {
        "format": "gcn_skip_learned_v2_robust_taskaware_v4",
        "epoch": epoch,
        "clean_map50_diagnostic": float(clean_map50),
        "clean_map50_95_diagnostic": float(clean_map5095),
        "base_checkpoint": getattr(
            cfg, "_resolved_initial_checkpoint", cfg.initial_checkpoint
        ),
        "config": asdict(cfg),
    }
    with open(drive_path + ".json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print("Saved epoch checkpoint:", drive_path)
    return drive_path


def _skip_probability(quantizer, feature):
    """Return the pre-threshold skip probability without changing the model API."""
    if quantizer.mode == "legacy":
        logits = quantizer.encode_conv(quantizer.pool(feature))
    else:
        logits = quantizer.encoder_net(feature)
    return torch.sigmoid(logits)


def _straight_through_binary(probability):
    hard = (probability > 0.5).to(probability.dtype)
    return probability + (hard - probability).detach()


def _encode_all_soft_and_hard(model, images):
    """Encode once and expose differentiable probabilities plus hard STE codes."""
    main_probability, features = model.encoder(images, hard=False)
    main_code = _straight_through_binary(main_probability)
    skip_probability = {
        name: _skip_probability(quantizer, features[name])
        for name, quantizer in model.decoder.skip_quantizers.items()
    }
    skip_codes = {
        name: _straight_through_binary(probability)
        for name, probability in skip_probability.items()
    }
    return main_probability, main_code, skip_probability, skip_codes


def _balanced_code_bce(adv_main, clean_main, adv_skip, clean_skip):
    """Give main, skip64, and skip32 equal influence regardless of bit count."""
    losses = [
        F.binary_cross_entropy(
            adv_main.clamp(1e-5, 1 - 1e-5),
            (clean_main.detach() > 0.5).to(adv_main.dtype),
        )
    ]
    for name in sorted(clean_skip):
        losses.append(
            F.binary_cross_entropy(
                adv_skip[name].clamp(1e-5, 1 - 1e-5),
                (clean_skip[name].detach() > 0.5).to(adv_skip[name].dtype),
            )
        )
    return torch.stack(losses).mean()


def _code_margin(main_probability, skip_probability):
    """Penalize uncertain probabilities near 0.5; zero is attained at 0 or 1."""
    penalties = [(main_probability * (1 - main_probability)).mean()]
    penalties.extend(
        (probability * (1 - probability)).mean()
        for probability in skip_probability.values()
    )
    return torch.stack(penalties).mean()


def _flip_binary_code(code, probability):
    if probability <= 0:
        return code
    mask = (torch.rand_like(code) < probability).to(code.dtype)
    return code.detach() * (1 - mask) + (1 - code.detach()) * mask


def _corrupt_all_codes(main_code, skip_codes, probability):
    return (
        _flip_binary_code(main_code, probability),
        {
            name: _flip_binary_code(code, probability)
            for name, code in skip_codes.items()
        },
    )


def _pgd_steps_for_epoch(cfg, epoch):
    if cfg.epochs <= 1:
        return int(cfg.pgd_steps_end)
    fraction = epoch / (cfg.epochs - 1)
    steps = cfg.pgd_steps_start + fraction * (
        cfg.pgd_steps_end - cfg.pgd_steps_start
    )
    return max(1, int(round(steps)))


def train_gcn_robust_taskaware(model, cfg):
    from torch.utils.data import DataLoader
    from ultralytics import YOLO

    if not hasattr(model, "decode_from_codes"):
        raise TypeError(
            "Robust-v4 requires the Learned-v2 model with decode_from_codes()."
        )
    if not getattr(model.decoder, "skip_quantizers", None):
        raise TypeError("Robust-v4 requires quantized skip64/skip32 streams.")
    if not 0 <= cfg.bit_flip_probability <= 0.5:
        raise ValueError("bit_flip_probability must be in [0, 0.5].")
    if cfg.pgd_steps_start < 1 or cfg.pgd_steps_end < cfg.pgd_steps_start:
        raise ValueError("Require 1 <= pgd_steps_start <= pgd_steps_end.")

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    encoder_layers = list(model.encoder.base_encoder.conv_layers)
    unfreeze_count = min(
        max(int(cfg.encoder_unfreeze_tail_modules), 0), len(encoder_layers)
    )
    encoder_tail = encoder_layers[-unfreeze_count:] if unfreeze_count else []
    for module in encoder_tail:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    named = [
        (name, value) for name, value in model.named_parameters()
        if value.requires_grad
    ]
    encoder = [value for name, value in named if name.startswith("encoder.")]
    encoder_ids = {id(value) for value in encoder}
    special = [
        value for name, value in named
        if id(value) not in encoder_ids and (
            "decoder.gcn_blocks" in name or "decoder.skip_quantizers" in name
        )
    ]
    special_ids = {id(value) for value in special}
    base = [
        value for _name, value in named
        if id(value) not in special_ids and id(value) not in encoder_ids
    ]
    trainable = [value for _name, value in named]

    dataset = DFireDetDataset(bld.DFIRE_ROOT, split="train", size=384)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=det_collate, drop_last=True,
    )
    map_dataset = bld.DFireDataset(bld.DFIRE_ROOT, split="test", size=bld.IMG_SIZE)
    loss_yolo, criterion = _prepare_detector(cfg)
    eval_yolo = YOLO(cfg.detector_ckpt)
    groups = []
    if encoder:
        groups.append({
            "params": encoder,
            "lr": cfg.lr * cfg.encoder_lr_multiplier,
        })
    if base:
        groups.append({"params": base, "lr": cfg.lr})
    if special:
        groups.append({"params": special, "lr": cfg.lr * cfg.new_module_lr_multiplier})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    print(
        f"Robust-v4 fine-tune: train={len(dataset)}, YOLO=frozen, "
        f"encoder_tail_modules={unfreeze_count}, "
        f"encoder_tail={sum(p.numel() for p in encoder):,}, "
        f"base_decoder={sum(p.numel() for p in base):,}, "
        f"GCN+skip={sum(p.numel() for p in special):,}"
    )

    for epoch in range(cfg.epochs):
        model.train()
        model.encoder.eval()
        for module in encoder_tail:
            module.train()
        warmup = min(1.0, (epoch + 1) / max(cfg.detection_warmup_epochs, 1))
        pgd_steps = _pgd_steps_for_epoch(cfg, epoch)
        totals = {key: 0.0 for key in (
            "loss", "clean_rec", "adv_rec", "consistency", "code_consistency",
            "code_margin", "swap_rec", "bit_rec", "clean_det", "adv_det"
        )}
        seen = dropped = 0
        iterator = tqdm(loader, desc=f"robust epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch_index, (clean384, targets, _paths) in enumerate(iterator):
            if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
                break
            clean384 = clean384.to(bld.DEVICE, non_blocking=True)
            clean_targets, invalid = _sanitize_det_targets(targets, bld.DEVICE)
            dropped += invalid
            adv384 = _pgd(
                loss_yolo.model, criterion, clean384, clean_targets, cfg, pgd_steps
            )
            clean128 = F.interpolate(clean384, size=(128, 128), mode="bilinear", align_corners=False)
            adv128 = F.interpolate(adv384, size=(128, 128), mode="bilinear", align_corners=False)

            clean_main_p, clean_z, clean_skip_p, clean_skip = (
                _encode_all_soft_and_hard(model, clean128)
            )
            adv_main_p, adv_z, adv_skip_p, adv_skip = (
                _encode_all_soft_and_hard(model, adv128)
            )
            clean_rec = model.decode_from_codes(clean_z, clean_skip)
            adv_rec = model.decode_from_codes(adv_z, adv_skip)
            clean_l1 = F.l1_loss(clean_rec, clean128)
            clean_mse = F.mse_loss(clean_rec, clean128)
            clean_rec_loss = cfg.clean_l1_weight * clean_l1 + cfg.clean_mse_weight * clean_mse
            adv_rec_loss = F.l1_loss(adv_rec, clean128)
            consistency = F.l1_loss(adv_rec, clean_rec.detach())

            code_consistency = _balanced_code_bce(
                adv_main_p, clean_main_p, adv_skip_p, clean_skip_p
            )
            code_margin = 0.5 * (
                _code_margin(clean_main_p, clean_skip_p)
                + _code_margin(adv_main_p, adv_skip_p)
            )

            clean_main_adv_skip = model.decode_from_codes(
                clean_z, adv_skip
            )
            adv_main_clean_skip = model.decode_from_codes(
                adv_z, clean_skip
            )
            swap_rec = 0.5 * (
                F.l1_loss(clean_main_adv_skip, clean128)
                + F.l1_loss(adv_main_clean_skip, clean128)
            )

            noisy_main, noisy_skip = _corrupt_all_codes(
                clean_z, clean_skip, cfg.bit_flip_probability
            )
            bit_rec = F.l1_loss(
                model.decode_from_codes(noisy_main, noisy_skip), clean128
            )
            clean384_out = F.interpolate(clean_rec, size=(384, 384), mode="bilinear", align_corners=False)
            adv384_out = F.interpolate(adv_rec, size=(384, 384), mode="bilinear", align_corners=False)
            clean_det_vec, _ = criterion(loss_yolo.model(clean384_out), clean_targets)
            adv_det_vec, _ = criterion(loss_yolo.model(adv384_out), clean_targets)
            clean_det = clean_det_vec.sum() / clean384.shape[0]
            adv_det = adv_det_vec.sum() / clean384.shape[0]
            loss = (
                clean_rec_loss
                + cfg.adv_reconstruction_weight * adv_rec_loss
                + cfg.consistency_weight * consistency
                + cfg.code_consistency_weight * code_consistency
                + cfg.code_margin_weight * code_margin
                + cfg.swap_reconstruction_weight * swap_rec
                + cfg.bit_corruption_weight * bit_rec
                + cfg.clean_detection_weight * warmup * clean_det
                + cfg.adv_detection_weight * warmup * adv_det
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            batch = clean384.shape[0]
            seen += batch
            values = {
                "loss": loss, "clean_rec": clean_rec_loss,
                "adv_rec": adv_rec_loss, "consistency": consistency,
                "code_consistency": code_consistency,
                "code_margin": code_margin, "swap_rec": swap_rec,
                "bit_rec": bit_rec,
                "clean_det": clean_det, "adv_det": adv_det,
            }
            for key, value in values.items():
                totals[key] += float(value.detach()) * batch
            iterator.set_postfix(
                loss=f"{loss.item():.4f}", adv_det=f"{adv_det.item():.3f}",
                code=f"{code_consistency.item():.3f}", pgd=pgd_steps,
            )

        if seen == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step()
        averages = {key: value / seen for key, value in totals.items()}
        print(
            "epoch", epoch + 1, averages,
            "pgd_steps=", pgd_steps,
            "dropped_invalid_boxes=", dropped,
        )
        map50, map5095 = evaluate_gcn_ae_map(
            model, map_dataset, eval_yolo,
            save_dir=cfg.map_eval_dir,
            batch_size=cfg.map_eval_batch_size,
            n_samples=cfg.map_eval_samples,
            img_size=384,
        )
        drive_path = _save_epoch(model, cfg, epoch + 1, map50, map5095)
        if epoch + 1 == cfg.epochs:
            os.makedirs(os.path.dirname(cfg.final_alias), exist_ok=True)
            shutil.copy2(drive_path, cfg.final_alias)
            shutil.copy2(drive_path + ".json", cfg.final_alias + ".json")
            print("Updated final-epoch alias:", cfg.final_alias)
    print("Training complete. Select any alternative epoch only on a separate validation split.")
    return model


_require_notebook_context()
robust_task_cfg = RobustTaskAwareConfig()
RUN_GCN_ROBUST_TASKAWARE_TRAINING = False

if RUN_GCN_ROBUST_TASKAWARE_TRAINING:
    ae_gcn_robust_v4 = train_gcn_robust_taskaware(
        make_robust_v4_initial_model(robust_task_cfg), robust_task_cfg
    )
elif os.path.exists(ROBUST_FINAL):
    ae_gcn_robust_v4 = copy.deepcopy(ae_gcn).to(bld.DEVICE)
    state = torch.load(ROBUST_FINAL, map_location=bld.DEVICE)
    ae_gcn_robust_v4.load_state_dict(unwrap_state_dict(state), strict=True)
    ae_gcn_robust_v4.eval()
    print("Strictly loaded Robust-v4 task-aware GCN:", ROBUST_FINAL)
else:
    ae_gcn_robust_v4 = None
    print(
        "Robust-v4 definitions loaded. First set max_train_batches=2 "
        "for a smoke run; then use None for formal training."
    )
