"""Notebook extension: %run -i after preserved-expansion GCN definitions.

Freeze every inherited parameter, including the old slices of widened layers.
Train only added encoder outputs and decoder inputs. Default is smoke budget.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import islice
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NewChannelsConfig:
    source: str = (
        '/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust/'
        'gcn_skip_learned_v2_robust_taskaware_final.pt'
    )
    output_dir: str = '/content/gcn6144_new_channels_smoke'
    detector_ckpt: str = '/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt'
    epochs: int = 1
    max_train_batches: int = 2
    batch_size: int = 2
    lr: float = 2e-5
    seed: int = 20260920
    pgd_steps: int = 3
    eps: float = 8/255
    alpha: float = 2/255


def _lock_inherited_parameters(model, old_channels):
    """Masks plus explicit projection prevent Adam/decay changing old slices."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    masks, handles = {}, []
    for level, quantizer in model.decoder.skip_quantizers.items():
        if level != "64x64":
            continue
        count = old_channels[level]
        last = len(quantizer.encoder_net)-1
        prefix = f'decoder.skip_quantizers.{level}.'
        for suffix in ('weight', 'bias'):
            key = prefix + f'encoder_net.{last}.{suffix}'
            parameter = dict(model.named_parameters())[key]
            mask = torch.zeros_like(parameter)
            mask[count:] = 1
            masks[key] = mask
        key = prefix + 'decoder_net.0.weight'
        parameter = dict(model.named_parameters())[key]
        mask = torch.zeros_like(parameter)
        mask[:, count:] = 1
        masks[key] = mask
    parameters = dict(model.named_parameters())
    for key, mask in masks.items():
        parameters[key].requires_grad_(True)
        handles.append(parameters[key].register_hook(lambda grad, m=mask: grad*m))
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # The six widened tensors are small enough to retain device-side copies.
    locked = {key: parameters[key].detach().clone() for key in masks}
    return masks, locked, initial, handles


@torch.no_grad()
def _restore_old_slices(model, masks, locked):
    for key, parameter in model.named_parameters():
        if key in masks:
            mask = masks[key].bool()
            parameter.copy_(torch.where(mask, parameter, locked[key]))


def _check_old_parameters(model, initial, masks):
    for key, value in model.state_dict().items():
        actual = value.detach().cpu()
        if key in masks:
            protected = ~masks[key].detach().cpu().bool()
            actual, expected = actual[protected], initial[key][protected]
        else:
            expected = initial[key]
        if not torch.equal(actual, expected):
            raise RuntimeError(f'Inherited parameter/buffer changed: {key}')


def train_new_channels(cfg):
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from tqdm.auto import tqdm
    required = ('make_gcn6144', 'DFireDetDataset', 'det_collate',
                '_sanitize_det_targets', '_LossAttrDict', 'bld')
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(f'Run notebook definitions first: {missing}')
    if min(cfg.epochs, cfg.max_train_batches, cfg.batch_size, cfg.pgd_steps) < 1:
        raise ValueError('Training counts must be positive')
    output = Path(cfg.output_dir)
    if (output / 'experiment.json').exists() or any(
            (output / f'epoch{i}.pt').exists() for i in range(1, cfg.epochs+1)):
        raise FileExistsError(f'Use a new experiment directory: {output}')
    for path in (cfg.source, cfg.detector_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = make_gcn6144(cfg.source, resume6144=False).eval()
    # 保留32层旧通道，关闭新增通道的解码贡献
    with torch.no_grad():
        model.decoder.skip_quantizers["32x32"].decoder_net[0].weight[
            :, 8:, :, :
        ].zero_()
    masks, locked, initial, handles = _lock_inherited_parameters(
        model, {'64x64': 2, '32x32': 8})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    dataset = DFireDetDataset(bld.DFIRE_ROOT, split='train', size=384)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=det_collate,
        generator=torch.Generator().manual_seed(cfg.seed))
    detector = YOLO(cfg.detector_ckpt).model.to(bld.DEVICE).train()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    for module in detector.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = v8DetectionLoss(detector)
    if isinstance(criterion.hyp, dict):
        criterion.hyp = _LossAttrDict(dict(criterion.hyp))
    for key, value in {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}.items():
        if not hasattr(criterion.hyp, key):
            setattr(criterion.hyp, key, value)
    metadata = {
        'format': 'gcn6144_v3_added_channels_v1', 'config': asdict(cfg),
        'source_sha256': model._initial_checkpoint_sha256,
        'expansion_check': model._expansion_check,
        'bit_budget': 6144,
        'skip_level_code_shapes': {'64x64': [8,16], '32x32': [32,8]},
        'skip_hidden_channels': {'64x64': 32, '32x32': 64},
        'trainable_scalar_count': int(sum(m.sum().item() for m in masks.values())),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Only added-channel scalars train:', metadata['trainable_scalar_count'], flush=True)
    try:
        for epoch in range(1, cfg.epochs+1):
            # eval mode prevents all inherited running buffers from changing;
            # it does not disable autograd on the added-channel weights.
            model.eval()
            totals = dict(loss=0., clean_rec=0., adv_rec=0., clean_det=0., adv_det=0.)
            seen = 0
            count = min(len(loader), cfg.max_train_batches)
            for clean, targets, _ in tqdm(islice(loader,count), total=count,
                                         desc=f'New channels {epoch}/{cfg.epochs}'):
                clean = clean.to(bld.DEVICE)
                targets, _ = _sanitize_det_targets(targets,bld.DEVICE)
                adv = (clean+torch.empty_like(clean).uniform_(-cfg.eps,cfg.eps)).clamp(0,1)
                for _ in range(cfg.pgd_steps):
                    adv = adv.detach().requires_grad_(True)
                    attack_loss, _ = criterion(detector(adv),targets)
                    gradient = torch.autograd.grad(attack_loss.sum(),adv)[0]
                    adv = adv.detach()+cfg.alpha*gradient.sign()
                    adv = torch.maximum(torch.minimum(adv,clean+cfg.eps),clean-cfg.eps).clamp(0,1)
                x = F.interpolate(clean,(128,128),mode='bilinear',align_corners=False)
                a = F.interpolate(adv.detach(),(128,128),mode='bilinear',align_corners=False)
                with torch.no_grad():
                    zc,fc = model.encoder(x,hard=True)
                    za,fa = model.encoder(a,hard=True)
                rc,ra = model.decoder(zc,fc),model.decoder(za,fa)
                clean_rec = .7*F.l1_loss(rc,x)+.3*F.mse_loss(rc,x)
                adv_rec = F.l1_loss(ra,x)
                dc,_ = criterion(detector(F.interpolate(rc,(384,384),mode='bilinear',align_corners=False)),targets)
                da,_ = criterion(detector(F.interpolate(ra,(384,384),mode='bilinear',align_corners=False)),targets)
                dc,da = dc.sum()/len(clean),da.sum()/len(clean)
                # Fixed v3 loss weights; no curriculum or repeated detection warmup.
                loss = clean_rec+.5*adv_rec+.25*F.l1_loss(ra,rc.detach())+.005*dc+.01*da
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step()
                _restore_old_slices(model,masks,locked)
                values = dict(loss=loss,clean_rec=clean_rec,adv_rec=adv_rec,clean_det=dc,adv_det=da)
                for key,value in values.items():
                    totals[key] += value.item()*len(clean)
                seen += len(clean)
            if not seen:
                raise RuntimeError('No training samples')
            _check_old_parameters(model,initial,masks)
            history = {key:value/seen for key,value in totals.items()}
            path = output/f'epoch{epoch}.pt'
            torch.save({**metadata,'epoch':epoch,'losses':history,
                        'inherited_parameters_unchanged':True,
                        'model_state_dict':model.state_dict()},path)
            print('Inherited parameters unchanged; losses:',history,flush=True)
            print('Saved:',path,flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return model.eval()


print('Added-channel training definitions loaded. No training started.')
