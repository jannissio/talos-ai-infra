"""Preserve a stopped mug-pose fit, every numeric example and both candidates."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from simulation_lab.mug_visual_observer import MugShapePoseNet
from scripts.mug_observer_experiment import load_protocol, read, score, sha, space, write_json


def package(args):
    p = load_protocol(args.protocol)
    raw = ROOT/p['raw_root']
    roots = {key: ROOT/p[key+'_package'] for key in ('model', 'training', 'evidence')}
    fit, selection = read(raw/'fit/training.json'), read(raw/'fit/selection.json')
    if fit['protocol_sha256'] != sha(args.protocol) or selection['training_report_sha256'] != sha(raw/'fit/training.json'):
        raise ValueError('A preserved training result changed.')
    if fit['development_gate_passed'] or selection['development_gate_passed']:
        raise ValueError('This stop package requires both development candidates to fail; a passing protocol needs its further gates.')
    if fit['updates'] != p['fit']['updates'] or [r['update'] for r in fit['checkpoints']] != p['fit']['checkpoints']:
        raise ValueError('The complete declared fit and both candidates are required.')
    if (raw/'evaluation').exists() or (raw/'openvino').exists():
        raise ValueError('A stopped development fit must not expose fresh data or export.')
    if not args.verify_only and any(root.exists() for root in roots.values()):
        raise FileExistsError('Preserve earlier packages.')
    for name, digest in fit['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A frozen fit dependency changed: '+name)
    source_hashes = {}
    for split in ('training', 'development'):
        manifest, audit = read(raw/split/'manifest.json'), read(raw/split/'audit.json')
        if manifest['data_sha256'] != sha(raw/split/'data.npz') or not audit['passed'] or audit['data_sha256'] != manifest['data_sha256']:
            raise ValueError('Every compact example must pass independent image and label reconstruction.')
        for name, digest in manifest['source_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('A dataset source changed: '+name)
            source_hashes[name] = digest
    with np.load(raw/'development/data.npz', allow_pickle=False) as archive:
        development = {name: archive[name] for name in ('features', 'labels_m', 'present', 'usable_views')}
    torch.set_num_threads(2)
    verified = []
    for candidate in fit['checkpoints']:
        checkpoint = ROOT/candidate['checkpoint']
        if sha(checkpoint) != candidate['checkpoint_sha256']:
            raise ValueError('A failed checkpoint changed.')
        model = MugShapePoseNet().eval()
        model.load_state_dict(load_file(str(checkpoint)))
        with torch.inference_mode():
            values = model(torch.from_numpy(development['features'])).numpy().astype(float)
        result = score(p, values*p['observer']['target_scale_m']+np.asarray(p['observer']['target_origin_m']),
                       development['labels_m'], development['present'], development['usable_views'])
        original = read(raw/f'fit/step-{candidate["update"]:06d}-development.json')
        if result['summary'] != candidate['development'] or result['summary'] != original['summary'] or result['rows'] != original['rows']:
            raise ValueError('An independent checkpoint score disagrees with a preserved outcome.')
        verified.append({'update': candidate['update'], 'checkpoint_sha256': sha(checkpoint), 'exact_score_match': True})
    audit = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'source_sha256': sha(Path(__file__)),
        'training_states': p['data']['training_states'], 'development_states': p['data']['development_states'],
        'independently_reconstructed_rgb_views': sum(read(raw/split/'audit.json')['rgb_views'] for split in ('training', 'development')),
        'checkpoint_scores': verified, 'passed': True, 'fresh_evaluation_not_generated': True,
        'export_not_performed': True, 'new_physical_trials': 0, 'selected_model_unchanged': True}
    if args.verify_only:
        total = 0
        for root in roots.values():
            manifest = read(root/'manifest.json')
            for name, digest in manifest['files'].items():
                if sha(root/name) != digest:
                    raise ValueError('A packaged artifact changed: '+name)
            total += len(manifest['files'])
        print({'verified_files': total, **audit})
        return
    payloads = {key: [] for key in roots}
    for split in ('training', 'development'):
        payloads['training'].extend((path, Path(split)/path.name) for path in sorted((raw/split).iterdir()) if path.is_file())
    for path in sorted((raw/'fit').iterdir()):
        if path.is_file():
            payloads['model' if path.suffix == '.safetensors' else 'evidence'].append((path, Path(path.name)))
    sources = [ROOT/name for name in source_hashes if name.endswith('.py')]
    sources += [ROOT/name for name in fit['source_sha256'] if name.endswith('.py')]
    sources += [Path(__file__), ROOT/'scripts/audit_mug_visual_observer.py', ROOT/'scripts/export_mug_visual_observer.py']
    for source in sorted(set(sources)):
        payloads['training'].append((source, Path('source')/source.relative_to(ROOT)))
    for key in roots:
        payloads[key].append((args.protocol, Path('protocol.json')))
    expected = sum(source.stat().st_size for values in payloads.values() for source, _ in values)+8*1024**2
    space(p, roots['evidence'], expected)
    for root in roots.values():
        root.mkdir(parents=True)
    for key, values in payloads.items():
        for source, name in values:
            destination = roots[key]/name
            space(p, destination, source.stat().st_size+1024)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
    write_json(roots['evidence']/'audit.json', audit)
    write_json(roots['training']/'reproduction-inputs.json', {'source_sha256': source_hashes,
        'requires_existing_repository_inputs': [p['training_input'], p['initial_scene_root'], p['visibility_protocol']],
        'source_restoration': 'Packaged source/ contains the exact frozen Python files. Reconstruct each image from qpos, episode_index, exposure_scale and fixed calibrations in data.npz/manifest.json; audit every image hash before use.',
        'scope': 'Synthetic geometry-to-pose inputs only, no physically valid grasp or unseen physical generalization claim.'})
    counts = {}
    for key, root in roots.items():
        files = {path.relative_to(root).as_posix(): sha(path) for path in root.rglob('*') if path.is_file()}
        size = sum((root/name).stat().st_size for name in files)
        write_json(root/'manifest.json', {'schema': p['schema'], 'files': files, 'bytes': size})
        counts[key] = {'files': len(files), 'bytes': size}
    space(p, roots['evidence'], 0)
    print({'packages': counts, **audit})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-observer-v1.json')
    parser.add_argument('--verify-only', action='store_true')
    package(parser.parse_args())
