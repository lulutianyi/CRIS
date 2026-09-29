"""Notebook entry: %run -i /content/methods/gcn_decoder_only.py.

Requires the notebook's dataset and Learned-v2 definitions.
Loading this file does not train. No test-set evaluation or automatic best alias.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import islice
import hashlib
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


@dataclass
class DecoderOnlyConfig:
    source: str = ('/content/drive/MyDrive/gcn_ae_checkpoints/'
                   'skip_learned_v2_robust/gcn_skip_learned_v2_robust_taskaware_final.pt')
    detector_ckpt: str = '/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt'
    output_dir: str = '/content/gcn_decoder_only_smoke'
    epochs: int = 1
    batch_size: int = 2
    max_train_batches: int | None = 2
    lr: float = 5e-6
    seed: int = 20260926
    holdout_size: int = 512
    split_seed: int = 20260926
    pgd_steps: int = 3
    pgd_eps: float = 8 / 255
    pgd_alpha: float = 2 / 255


def _decoder_helpers():
    # Do not execute a notebook training script to import two helpers: its
    # top-level checks/entry points require an unrelated notebook namespace.
    return {'_prepare_detector': _decoder_prepare_detector, '_pgd': _decoder_pgd}


def _decoder_prepare_detector(cfg):
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    detector = YOLO(cfg.detector_ckpt)
    detector.model.to(bld.DEVICE).train()
    detector.model.requires_grad_(False)
    for module in detector.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    criterion = v8DetectionLoss(detector.model)
    if isinstance(criterion.hyp, dict):
        criterion.hyp = _LossAttrDict(dict(criterion.hyp))
    for key, value in {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}.items():
        if not hasattr(criterion.hyp, key):
            setattr(criterion.hyp, key, value)
    return detector, criterion


def _decoder_pgd(detector, criterion, clean, targets, cfg, steps):
    lower, upper = (clean-cfg.pgd_eps).clamp(0, 1), (clean+cfg.pgd_eps).clamp(0, 1)
    adv = (clean + torch.empty_like(clean).uniform_(-cfg.pgd_eps, cfg.pgd_eps)).clamp(0, 1)
    with torch.enable_grad():
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            loss, _ = criterion(detector(adv), targets)
            gradient = torch.autograd.grad(loss.sum(), adv, only_inputs=True)[0]
            adv = adv.detach() + cfg.pgd_alpha * gradient.sign()
            adv = torch.maximum(torch.minimum(adv, upper), lower).clamp(0, 1)
    return adv.detach()


def _decoder_initial_model(cfg):
    # Explicit original shapes: works even inside the 6144 notebook.
    config = GCNSkipAEConfig(
        active_levels=['64x64', '32x32'], skip_quantizer_mode='learned_v2',
        skip_level_code_shapes={'64x64': (2, 16), '32x32': (8, 8)},
    )
    model = build_gcn_ae(BinaryAutoEncoderAttn().to(bld.DEVICE), config)
    raw = torch.load(cfg.source, map_location='cpu', weights_only=True)
    state = raw
    for key in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(raw, dict) and isinstance(raw.get(key), dict):
            state = raw[key]
            break
    model.load_state_dict(state, strict=True)
    if model.total_bit_budget() != 3072:
        raise RuntimeError('Expected original 3072-bit architecture')
    model.requires_grad_(False)
    model.decoder.base_decoder.requires_grad_(True)
    # Eval mode freezes running buffers/dropout; gradients still work.
    return model.eval()


def train_gcn_decoder_only(cfg=None):
    cfg = cfg or DecoderOnlyConfig()
    required = ('bld', 'GCNSkipAEConfig', 'build_gcn_ae', 'BinaryAutoEncoderAttn',
                'DFireDetDataset', 'det_collate', '_sanitize_det_targets', '_LossAttrDict')
    missing = [key for key in required if key not in globals()]
    if missing:
        raise RuntimeError(f'Run notebook definitions first: {missing}')
    if min(cfg.epochs, cfg.batch_size, cfg.pgd_steps) < 1 or cfg.lr <= 0:
        raise ValueError('Invalid training configuration')
    if cfg.max_train_batches is not None and cfg.max_train_batches < 1:
        raise ValueError('max_train_batches must be positive or None')
    for path in (cfg.source, cfg.detector_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output = Path(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new empty output directory: {output}')
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    helpers = _decoder_helpers()
    model = _decoder_initial_model(cfg)
    prefix = 'decoder.base_decoder.'
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith(prefix)}
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(n.startswith(prefix) for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=1e-4)
    dataset = DFireDetDataset(bld.DFIRE_ROOT, split='train', size=384)
    if not 0 < cfg.holdout_size < len(dataset):
        raise ValueError('holdout_size must be between 1 and train size - 1')
    # Sort by path so split identity does not depend on filesystem enumeration.
    paths = [str(p) for p in dataset.img_paths]
    indices = sorted(range(len(dataset)), key=lambda i: paths[i])
    random.Random(cfg.split_seed).shuffle(indices)
    heldout, train_indices = indices[:cfg.holdout_size], indices[cfg.holdout_size:]
    loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg.batch_size,
        shuffle=True, num_workers=0, collate_fn=det_collate,
        generator=torch.Generator().manual_seed(cfg.seed))
    detector, criterion = helpers['_prepare_detector'](cfg)
    digest = hashlib.sha256()
    with open(cfg.source, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(format='gcn_v3_decoder_only_v1', config=asdict(cfg),
        source_sha256=digest.hexdigest(), bit_budget=3072,
        train_paths=[paths[i] for i in train_indices],
        holdout_paths=[paths[i] for i in heldout],
        holdout_note='Excluded from this fine-tune, but may have been seen by source model.',
        trainable_parameters=sum(p.numel() for p in parameters))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Decoder-only: train={len(train_indices)}, holdout={len(heldout)}, '
          f'bits=3072, trainable={metadata["trainable_parameters"]:,}', flush=True)
    print('Strict source:', cfg.source, '\nNo test evaluation; no best checkpoint selection.', flush=True)
    for epoch in range(1, cfg.epochs + 1):
        totals, seen = {}, 0
        count = len(loader) if cfg.max_train_batches is None else min(len(loader), cfg.max_train_batches)
        for clean, targets, _ in tqdm(islice(loader, count), total=count,
                                     desc=f'Decoder-only {epoch}/{cfg.epochs}'):
            clean = clean.to(bld.DEVICE)
            targets, _ = _sanitize_det_targets(targets, bld.DEVICE)
            adv = helpers['_pgd'](detector.model, criterion, clean, targets, cfg, cfg.pgd_steps)
            x, a = [F.interpolate(t, (128, 128), mode='bilinear', align_corners=False)
                    for t in (clean, adv)]
            with torch.no_grad():
                zc, sc = model.encode_all_codes(x)
                za, sa = model.encode_all_codes(a)
            # Never no_grad here: frozen GCN must transmit decoder gradients.
            rc, ra = model.decode_from_codes(zc, sc), model.decode_from_codes(za, sa)
            l1, mse = F.l1_loss(rc, x), F.mse_loss(rc, x)
            adv_rec, cons = F.l1_loss(ra, x), F.l1_loss(ra, rc.detach())
            dc, da = [criterion(detector.model(F.interpolate(
                t, (384, 384), mode='bilinear', align_corners=False)), targets)[0].sum() / len(clean)
                for t in (rc, ra)]
            loss = .7*l1 + .3*mse + .5*adv_rec + .25*cons + .005*dc + .01*da
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if not any(p.grad is not None for p in parameters):
                raise RuntimeError('No base-decoder gradients')
            optimizer.step()
            values = dict(loss=loss, clean_l1=l1, clean_mse=mse, adv_rec=adv_rec,
                          consistency=cons, clean_det=dc, adv_det=da, grad_norm=norm)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + value.item()*len(clean)
            seen += len(clean)
        for key, expected in frozen.items():
            if not torch.equal(model.state_dict()[key].detach().cpu(), expected):
                raise RuntimeError(f'Frozen state changed: {key}')
        losses = {k: v/seen for k, v in totals.items()}
        path = output / f'epoch{epoch}.pt'
        torch.save({'model_state_dict': model.state_dict(), 'config': asdict(cfg),
                    'source_sha256': digest.hexdigest(), 'bit_budget': 3072,
                    'epoch': epoch, 'losses': losses, 'samples_seen_epoch': seen,
                    'frozen_state_unchanged': True}, path)
        (output / f'epoch{epoch}.json').write_text(json.dumps(
            dict(epoch=epoch, losses=losses, samples_seen=seen, checkpoint=str(path)),
            indent=2), encoding='utf-8')
        print('Frozen encoder/GCN/skip unchanged. Losses:', losses, flush=True)
        print('Saved:', path, flush=True)
    return model.eval()


print('Decoder-only definitions loaded. No training started.')
