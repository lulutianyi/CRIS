"""Adversarial task-aware fine-tuning cell for gcn.ipynb.

Run from the notebook with ``%run -i /content/methods/gcn_robust_taskaware.py`` after
the Learned-v2 GCN cell.  The architecture and 3072-bit budget do not change,
so the resulting state_dict remains compatible with the shared evaluators.
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


ROBUST_DIR = "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust"
ROBUST_FINAL = ROBUST_DIR + "/gcn_skip_learned_v2_robust_taskaware_final.pt"


@dataclass
class RobustTaskAwareConfig:
    epochs: int = 4
    batch_size: int = 4
    lr: float = 2e-5
    new_module_lr_multiplier: float = 2.0
    weight_decay: float = 1e-4
    clean_l1_weight: float = 0.7
    clean_mse_weight: float = 0.3
    adv_reconstruction_weight: float = 0.5
    consistency_weight: float = 0.25
    clean_detection_weight: float = 0.005
    adv_detection_weight: float = 0.01
    detection_warmup_epochs: int = 2
    pgd_eps: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 3
    grad_clip: float = 1.0
    num_workers: int = 2
    max_train_batches: int | None = None
    map_eval_samples: int | None = 512
    map_eval_batch_size: int = 16
    map_eval_dir: str = "/content/gcn_robust_clean_map_eval"
    detector_ckpt: str = (
        "/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt"
    )
    local_dir: str = "/content/gcn_skip_learned_v2_robust"
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


def _pgd(detector, criterion, clean, targets, cfg):
    lower = (clean - cfg.pgd_eps).clamp(0, 1)
    upper = (clean + cfg.pgd_eps).clamp(0, 1)
    adv = (clean + torch.empty_like(clean).uniform_(
        -cfg.pgd_eps, cfg.pgd_eps
    )).clamp(0, 1)
    with torch.enable_grad():
        for _ in range(cfg.pgd_steps):
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
        "format": "gcn_skip_learned_v2_robust_taskaware_v3",
        "epoch": epoch,
        "clean_map50_selection_subset": float(clean_map50),
        "clean_map50_95_selection_subset": float(clean_map5095),
        "base_checkpoint": LEARNED_V2_GCN_CKPT,
        "config": asdict(cfg),
    }
    with open(drive_path + ".json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print("Saved epoch checkpoint:", drive_path)
    return drive_path


def train_gcn_robust_taskaware(model, cfg):
    from torch.utils.data import DataLoader
    from ultralytics import YOLO

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    special = [
        value for name, value in named
        if "decoder.gcn_blocks" in name or "decoder.skip_quantizers" in name
    ]
    special_ids = {id(value) for value in special}
    base = [value for _name, value in named if id(value) not in special_ids]
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
    if base:
        groups.append({"params": base, "lr": cfg.lr})
    if special:
        groups.append({"params": special, "lr": cfg.lr * cfg.new_module_lr_multiplier})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    print(
        f"Robust fine-tune: train={len(dataset)}, encoder=frozen, YOLO=frozen, "
        f"base_decoder={sum(p.numel() for p in base):,}, "
        f"GCN+skip={sum(p.numel() for p in special):,}"
    )

    for epoch in range(cfg.epochs):
        model.train()
        model.encoder.eval()
        warmup = min(1.0, (epoch + 1) / max(cfg.detection_warmup_epochs, 1))
        totals = {key: 0.0 for key in (
            "loss", "clean_rec", "adv_rec", "consistency", "clean_det", "adv_det"
        )}
        seen = dropped = 0
        iterator = tqdm(loader, desc=f"robust epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch_index, (clean384, targets, _paths) in enumerate(iterator):
            if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
                break
            clean384 = clean384.to(bld.DEVICE, non_blocking=True)
            clean_targets, invalid = _sanitize_det_targets(targets, bld.DEVICE)
            dropped += invalid
            adv384 = _pgd(loss_yolo.model, criterion, clean384, clean_targets, cfg)
            clean128 = F.interpolate(clean384, size=(128, 128), mode="bilinear", align_corners=False)
            adv128 = F.interpolate(adv384, size=(128, 128), mode="bilinear", align_corners=False)

            with torch.no_grad():
                clean_z, clean_features = model.encoder(clean128, hard=True)
                adv_z, adv_features = model.encoder(adv128, hard=True)
                clean_features = {key: value.detach() for key, value in clean_features.items()}
                adv_features = {key: value.detach() for key, value in adv_features.items()}
            clean_rec = model.decoder(clean_z.detach(), clean_features)
            adv_rec = model.decoder(adv_z.detach(), adv_features)
            clean_l1 = F.l1_loss(clean_rec, clean128)
            clean_mse = F.mse_loss(clean_rec, clean128)
            clean_rec_loss = cfg.clean_l1_weight * clean_l1 + cfg.clean_mse_weight * clean_mse
            adv_rec_loss = F.l1_loss(adv_rec, clean128)
            consistency = F.l1_loss(adv_rec, clean_rec.detach())
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
                "clean_det": clean_det, "adv_det": adv_det,
            }
            for key, value in values.items():
                totals[key] += float(value.detach()) * batch
            iterator.set_postfix(loss=f"{loss.item():.4f}", adv_det=f"{adv_det.item():.3f}")

        if seen == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step()
        averages = {key: value / seen for key, value in totals.items()}
        print("epoch", epoch + 1, averages, "dropped_invalid_boxes=", dropped)
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
    ae_gcn_robust_v3 = train_gcn_robust_taskaware(
        copy.deepcopy(ae_gcn).to(bld.DEVICE), robust_task_cfg
    )
elif os.path.exists(ROBUST_FINAL):
    ae_gcn_robust_v3 = copy.deepcopy(ae_gcn).to(bld.DEVICE)
    state = torch.load(ROBUST_FINAL, map_location=bld.DEVICE)
    ae_gcn_robust_v3.load_state_dict(unwrap_state_dict(state), strict=True)
    ae_gcn_robust_v3.eval()
    print("Strictly loaded robust task-aware GCN:", ROBUST_FINAL)
else:
    ae_gcn_robust_v3 = None
    print(
        "Robust task-aware definitions loaded. First set max_train_batches=2 "
        "for a smoke run; then use None for formal training."
    )
