"""Run with %run -i after the NEW cells 6/7 of gcn_6144bit.ipynb.

Evaluation only: original/resize controls and four model states on the same
saved image list. No FreqPure inference, training, or checkpoint writes.
"""


def run_gcn6144_diagnostics():
    import hashlib
    import importlib
    import json
    import sys
    import time
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from ultralytics import YOLO

    required = ("_new_skip_model", "make_gcn6144", "_unwrap_gcn_state_dict",
                "SOURCE3072_SHAPES", "GCN6144_HIDDEN", "bld")
    missing = [key for key in required if key not in globals()]
    if missing:
        raise RuntimeError(f"Run NEW definition cells 6/7 first: missing {missing}")
    if GCN6144_HIDDEN != {"64x64": 32, "32x32": 64}:
        raise RuntimeError("Expected preserved-expansion architecture: hidden=32/64")

    for folder in (Path(__file__).resolve().parent, Path('/content/methods'), Path('/content')):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    ev = importlib.import_module("freqpure_dfire")
    source = Path('/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2/'
                  'gcn_skip_learned_v2_taskaware_best.pt')
    experiment = Path('/content/drive/MyDrive/gcn_ae_checkpoints/skip6144_preserve_learnedv2')
    adapt = experiment / 'adapt_pilot500/adapt_epoch1.pt'
    joint = experiment / 'joint_pilot500_2/joint_epoch8.pt'
    shared = Path('/content/drive/MyDrive/dfire_shared_pgd_png_eps8_seed20260822')
    freq_metrics = Path('/content/drive/MyDrive/freqpure_adv400/'
        'imagenet256_a10_p10_d0p3_t50_k5_c8_ddpm_official_'
        'adv_freqpure_only_subset20260909_seed20260904_n400/metrics_freqpure_dfire.json')
    yolo_path = Path('/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt')
    for path in (source, adapt, joint, shared / 'manifest.json', freq_metrics, yolo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    previous = json.loads(freq_metrics.read_text(encoding='utf-8'))
    names = previous['selected_names']
    if len(names) != 400 or len(set(names)) != 400:
        raise ValueError('Expected the fixed list of 400 unique image names')
    manifest = json.loads((shared / 'manifest.json').read_text(encoding='utf-8'))
    if not set(names).issubset(set(manifest['names'])):
        raise ValueError('Image list is not contained in the shared PGD manifest')
    for split in ('clean', 'adv'):
        for name in names:
            path = shared / split / 'images' / f'{name}.png'
            if not path.is_file():
                raise FileNotFoundError(path)

    def sha256(path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                result.update(chunk)
        return result.hexdigest()

    tag = str(time.time_ns())
    root = Path('/content/gcn6144_diagnostics') / tag
    root.mkdir(parents=True)
    saved = experiment / 'diagnostics' / f'metrics_{tag}.json'
    saved.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'running', 'selected_names': names, 'local_output': str(root),
        'shared_root': str(shared), 'freqpure_metrics_source': str(freq_metrics),
        'checkpoints': {key: {'path': str(path), 'sha256': sha256(path)}
                        for key, path in {'source3072': source, 'adapt6144': adapt,
                                          'joint6144': joint}.items()},
        'shared_manifest_sha256': sha256(shared / 'manifest.json'),
        'detector_sha256': sha256(yolo_path),
        'expansion': {'max_abs_output_error': 0.0, 'main_bit_mismatches': 0,
                      'old_skip_bit_mismatches': 0},
        'results': {},
    }

    def save_report():
        content = json.dumps(report, ensure_ascii=False, indent=2)
        (root / 'metrics.json').write_text(content, encoding='utf-8')
        # Save partial metrics after each split, as well as the completed report.
        temporary = saved.with_suffix('.json.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(saved)

    save_report()  # Also checks Drive can save before spending GPU time.
    device = torch.device(bld.DEVICE)
    raw = torch.load(str(source), map_location='cpu')
    state = raw['model'] if isinstance(raw, dict) and 'model' in raw else _unwrap_gcn_state_dict(raw)
    old = _new_skip_model(SOURCE3072_SHAPES)
    old.load_state_dict(state, strict=True)
    models = {
        'source3072': old.eval(),
        'expanded6144': make_gcn6144(str(source)).eval(),
        'adapt6144': make_gcn6144(str(adapt), resume6144=True).eval(),
        'joint6144': make_gcn6144(str(joint), resume6144=True).eval(),
    }
    loader = DataLoader(ev.SharedPGDDataset(str(shared), names), batch_size=2,
                        shuffle=False, num_workers=0, collate_fn=ev.pair_collate)
    folders = {'original_clean': shared / 'clean', 'adv_original': shared / 'adv'}
    for prefix in ('clean', 'adv'):
        folders[f'{prefix}_resize128'] = root / f'{prefix}_resize128'
        for stage in models:
            folders[f'{prefix}_{stage}'] = root / f'{prefix}_{stage}'
    print('Images: 400; model states: source / expanded / adapted / joint', flush=True)
    print('Metrics will be saved to:', saved, flush=True)
    with torch.inference_mode():
        for clean, adv, batch_names in tqdm(loader, desc='Four-state direct diagnostics'):
            for prefix, images in (('clean', clean), ('adv', adv)):
                small = F.interpolate(images.to(device), (128, 128), mode='bilinear', align_corners=False)

                def save_images(key, reconstructed):
                    large = F.interpolate(reconstructed, (384, 384), mode='bilinear', align_corners=False)
                    ev.save_png_batch(large, batch_names, folders[key], shared / 'clean' / 'labels')

                save_images(f'{prefix}_resize128', small)
                reference = None
                for stage, model in models.items():
                    main, skip = model.encode_all_codes(small)
                    reconstruction = model.decode_from_codes(main, skip)
                    if not torch.isfinite(reconstruction).all():
                        raise FloatingPointError(f'Nonfinite reconstruction: {stage}')
                    if stage == 'source3072':
                        reference = (main, skip, reconstruction)
                    elif stage == 'expanded6144':
                        check = report['expansion']
                        check['main_bit_mismatches'] += int((main != reference[0]).sum().item())
                        for key, code in reference[1].items():
                            check['old_skip_bit_mismatches'] += int(
                                (skip[key][:, :code.shape[1]] != code).sum().item())
                        error = (reconstruction-reference[2]).abs().max().item()
                        check['max_abs_output_error'] = max(check['max_abs_output_error'], error)
                    save_images(f'{prefix}_{stage}', reconstruction)
    models.clear()
    del model, old, raw, state, reference, reconstruction, main, skip
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('Actual-image expansion check:', report['expansion'], flush=True)
    save_report()

    detector = YOLO(str(yolo_path))
    cfg = ev.FreqPureDFireConfig(shared_root=str(shared), yolo_ckpt=str(yolo_path),
                               detector_size=384, detector_batch_size=8)
    for key, folder in folders.items():
        image_list = ev.write_image_list(root / 'image_lists' / f'{key}.txt',
                    [folder / 'images' / f'{name}.png' for name in names])
        m50, m95 = ev.eval_map(detector, image_list, cfg, root, key, device)
        report['results'][key] = {'map50': m50, 'map50_95': m95}
        save_report()
        print(f'{key}: mAP50={m50:.4f}, mAP50-95={m95:.4f}', flush=True)

    result = report['results']
    report['recovery'] = {}
    print('\nStage                  clean mAP50   adv mAP50   recovery50/95')
    for stage in ('resize128', 'source3072', 'expanded6144', 'adapt6144', 'joint6144'):
        rates = {}
        for metric in ('map50', 'map50_95'):
            denominator = result['original_clean'][metric]-result['adv_original'][metric]
            if denominator <= 0:
                raise ValueError('Nonpositive recovery denominator')
            rates[metric] = (result[f'adv_{stage}'][metric]-result['adv_original'][metric])/denominator
        report['recovery'][stage] = rates
        print(f"{stage:22s} {result[f'clean_{stage}']['map50']:.4f}        "
              f"{result[f'adv_{stage}']['map50']:.4f}      "
              f"{rates['map50']:.1%}/{rates['map50_95']:.1%}")
    report['status'] = 'complete'
    save_report()
    print('Saved to Drive:', saved)
    return report


if __name__ == '__main__':
    gcn6144_diagnostic_report = run_gcn6144_diagnostics()
