"""Preserve both bottle refiners, exact compact fit inputs and the failed gate."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, space, write, arrays
from simulation_lab.bottle_refinement_runtime import RefinedBottleObserver


def array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def verify(p):
    folders = [ROOT/p[key] for key in ('model_package', 'training_package', 'evidence_package')]
    checked = 0
    for folder in folders:
        manifest = read(folder/'manifest.json')
        for name, digest in manifest['files'].items():
            if sha(folder/name) != digest:
                raise ValueError('Packaged file changed: '+name)
            checked += 1
    RefinedBottleObserver(folders[0]/'openvino')
    if read(folders[2]/'fresh-perception.json')['passed']:
        raise ValueError('This preserved experiment stopped at its failed fresh gate.')
    print({'packaged_hashes_checked': checked, 'experimental_cpu_runtime_loads': True, 'promoted': False}, flush=True)


def run(args):
    args.protocol = args.protocol.resolve()
    p = checked_protocol(args.protocol)
    if args.verify_only:
        verify(p)
        return
    raw = ROOT/p['raw_root']
    model, training, evidence = [ROOT/p[key] for key in ('model_package', 'training_package', 'evidence_package')]
    if any(folder.exists() for folder in (model, training, evidence)):
        raise FileExistsError('Preserve each existing model, input and evidence package.')
    selected = read(raw/'fit/selection.json')
    fresh = read(raw/'fresh-perception.json')
    export = read(raw/'openvino/parity-development.json')
    assert selected['development_gate_passed'] and export['passed'] and not fresh['passed']
    assert fresh['protocol_sha256'] == sha(args.protocol)
    for split in ('training', 'development', 'evaluation'):
        audit = read(raw/split/'audit.json')
        assert audit['passed'] and audit['data_manifest_sha256'] == sha(raw/split/'manifest.json')
    preflight = space(p, training, 384*1024**2)
    for folder in (model, training, evidence):
        folder.mkdir(parents=True)
    copies = {}
    def copy(source, destination):
        source = source.resolve()
        space(p, destination, source.stat().st_size+1024**2)
        write(destination, source.read_bytes())
        assert sha(destination) == sha(source)
        copies[destination.relative_to(ROOT).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for candidate in selected['candidates']:
        source = ROOT/candidate['checkpoint']
        copy(source, model/'candidates'/source.parent.name/source.name)
        copy(source.parent/'development.json', evidence/'candidates'/source.parent.name/'development.json')
    chosen = ROOT/selected['selected']['checkpoint']
    copy(chosen, model/'refiner.safetensors')
    copy(ROOT/p['coarse_checkpoint'], model/'coarse.safetensors')
    for source in (raw/'openvino').iterdir():
        if source.is_file():
            copy(source, (evidence/'export' if source.name == 'parity-development.json' else model/'openvino')/source.name)
    for name in ('fresh-perception.json', 'fresh-perception-freeze.json', 'evaluation-collect.log', 'evaluation-audit.log'):
        copy(raw/name, evidence/name)
    copy(raw/'fit/selection.json', evidence/'selection.json')
    copy(raw/'fit/fit-freeze.json', evidence/'fit-freeze.json')
    copy(args.protocol, evidence/'protocol.json')
    copy(args.protocol, training/'protocol.json')
    for source in (raw/'audit-attempt-1').rglob('*'):
        if source.is_file():
            copy(source, evidence/'audit-attempt-1'/source.relative_to(raw/'audit-attempt-1'))
    for name in ('source.py', 'incident.json'):
        if (raw/'package-attempt-1'/name).exists():
            copy(raw/'package-attempt-1'/name, evidence/'package-attempt-1'/name)
    names = [name for name in p['source_sha256'] if name.endswith('.py')]
    names += ['scripts/train_bottle_refinement.py', 'scripts/export_bottle_refinement.py',
        'scripts/audit_bottle_refinement_data.py', 'scripts/evaluate_bottle_refinement.py',
        'scripts/package_bottle_refinement.py', 'simulation_lab/bottle_refinement_runtime.py']
    for name in sorted(set(names)):
        copy(ROOT/name, evidence/'frozen-source'/name)
    logs = ['training-collect', 'development-collect', 'fit', 'export', 'fresh-perception',
            'training-audit', 'training-audit-attempt2', 'development-audit-attempt2']
    for name in logs:
        source = ROOT/'.run/final-goal'/('bottle-refinement-v1-'+name+'.log')
        if source.exists():
            copy(source, evidence/'console'/source.name)
    compact = {}
    for split in ('training', 'development', 'evaluation'):
        folder, target = raw/split, training/split
        manifest = read(folder/'manifest.json')
        for name in ('recipes.npz', 'manifest.json', 'audit.json'):
            copy(folder/name, target/name)
        copy(folder/'scene.xml', target/'source-scene.xml')
        scene = ET.parse(folder/'scene.xml')
        scene.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', target).replace('\\', '/'))
        write(target/'scene.xml', ET.tostring(scene.getroot(), encoding='utf-8'))
        for source in folder.glob('example-*.png'):
            copy(source, target/source.name)
        labels = []
        for shard in manifest['shards']:
            source = folder/shard['file']
            assert sha(source) == shard['sha256']
            with np.load(source, allow_pickle=False) as z:
                values = {key: z[key] for key in z.files if key != 'rgb'}
            destination = target/'labels'/shard['file']
            destination.parent.mkdir(parents=True, exist_ok=True)
            space(p, destination, 8*1024**2)
            arrays(destination, **values)
            with np.load(destination, allow_pickle=False) as check:
                assert all(np.array_equal(values[key], check[key]) for key in values)
            labels.append({'file': destination.relative_to(target).as_posix(), 'sha256': sha(destination),
                'original_rgb_shard_sha256': shard['sha256'], 'states': shard['states']})
        compact[split] = {'states': manifest['states'], 'labels': labels,
            'recipes_sha256': sha(target/'recipes.npz'), 'original_manifest_sha256': sha(target/'manifest.json'),
            'rgb': 'Exact scene/joint/light recipes and all original pixel hashes retained. Full RGB shards remain in the local archive; regenerate images from these recipes.'}
    prepared = {}
    for split in ('training', 'development'):
        folder, target = raw/('prepared-'+split), training/('prepared-'+split)
        original = read(folder/'manifest.json')
        assert sha(folder/'crops.npz') == original['prepared_arrays_sha256']
        copy(folder/'manifest.json', target/'source-manifest.json')
        with np.load(folder/'crops.npz', allow_pickle=False) as z:
            values = {key: z[key] for key in z.files}
        descriptions = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'array_sha256': array_digest(value)} for key, value in values.items()}
        rows = []
        for first in range(0, original['states'], 64):
            last = min(original['states'], first+64)
            parts = {key: value[first:last] if key in ('world_points', 'present') else value[first*6:last*6]
                for key, value in values.items()}
            destination = target/f'inputs-{first//64:03d}.npz'
            space(p, destination, 16*1024**2)
            arrays(destination, **parts)
            with np.load(destination, allow_pickle=False) as z:
                assert all(np.array_equal(z[key], part) for key, part in parts.items())
            rows.append({'file': destination.name, 'first_state': first, 'states': last-first, 'sha256': sha(destination)})
        prepared[split] = {'states': original['states'], 'crop_count': original['crop_count'], 'original_npz_sha256': sha(folder/'crops.npz'),
            'arrays': descriptions, 'shards': rows, 'all_arrays_verified_identical': True,
            'reconstruction': 'Concatenate each named array across the ordered shards on axis zero. All dtypes, shapes and logical-array SHA-256 digests are declared. The inputs include exact FP16 crop pixels, centers, labels, visibility and privileged scoring-only truth.'}
        write(target/'manifest.json', prepared[split])
        del values
    write(training/'inputs.json', {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'compact_rgb': compact,
        'exact_prepared_inputs': prepared, 'training_code': '../../scripts/train_bottle_refinement.py',
        'scope': 'Exact crop inputs from the original run plus compact RGB regeneration recipes. Reproduction does not create a new evaluation or permit fitting on the exposed final split.'})
    write(model/'model.json', {'schema': p['schema'], 'status': 'experimental; fresh perception gate failed; not promoted',
        'license': 'MIT', 'selected_checkpoint_sha256': sha(chosen), 'coarse_checkpoint_sha256': sha(ROOT/p['coarse_checkpoint']),
        'training_states': 4096, 'development_states': 512, 'fresh_perception': fresh['summary'],
        'openvino': 'openvino', 'physical_v1_trials': 0, 'inference': p['inference'],
        'evidence': '../../docs/robotics/evidence/bottle-refinement-v1'})
    write(evidence/'audit.json', {'schema': p['schema'], 'preflight': preflight, 'exact_copies': copies,
        'prepared_inputs_verified': prepared, 'original_physical_reserved_stages_started': False,
        'development': export['summary'], 'fresh_perception': fresh['summary'], 'promoted': False,
        'scope': 'All 5,120 observation recipes/labels and pixel hashes retained; all exact fit/development crop arrays preserved and compared. Original raw RGB remains local. This packaging audit does not repeat physical trials.'})
    text = '''# Bottle image refinement V1: stopped at fresh perception

The RTX 4070 fits a new 51,986-parameter local RGB refinement network after the frozen coarse bottle detector. The original coarse model is preserved. Training uses 4,096 wider and physical-context observation states, with 512 development states. The single 8,000-update fit takes 82.18 seconds and reserves 1.084 GiB of Torch GPU memory. Both checkpoints and every score are retained.

The selected step-8,000 CPU export passes development: **433/449 present**, **0/63 absent false accepts**, **1.322 mm p95 / 3.971 mm maximum**. CPU/export decisions match the training runtime on every development state.

The frozen fresh test **fails**: **433/456 present (94.956%)**, **0/56 absent false accepts**, **1.494 mm p95 / 4.427 mm maximum**. The unchanged limits require at least 95% acceptance, no absent false accepts, at most 2 mm p95 and 4 mm maximum. All 23 refusals and the largest error remain in the original result. This model is **not promoted**. The original six development and 24 final physical seeds remain unexposed under this protocol.

All 5,120 observation states pass the geometry/image-hash audits, with seven fixed states (21 views) independently re-rendered per split. A first read-only audit was stopped for repeated decompression; its source and logs are retained. Caching one shard at a time changes no verification assertion.

The separate exposed-grid physical diagnostic asks where this observer/controller actually works, using already exposed contexts that contributed training views. It does not convert this failed gate into a pass or establish fresh-scene generalization.

The model package contains both checkpoints, selected weights, the frozen coarse checkpoint and actual FP32 CPU exports. The training package retains exact fit/development crop inputs in small ordered shards, all scene/joint/light recipes, every label and original RGB hash. Full raw RGB shards remain locally preserved. Privileged geometry and masks are labels/scoring only; the observer input is three RGB images and fixed calibration.
'''
    write(evidence/'README.md', text.encode())
    write(model/'README.md', ('''# Experimental bottle RGB refinement

This model **failed its fresh perception gate and is not deployed**. It uses current RGB, a frozen coarse CNN, a new semantic-point crop CNN and fixed stereo geometry. No simulator object pose, depth, segmentation or teacher enters inference. Physical generalization is unverified.

See [all results and limitations](../../docs/robotics/evidence/bottle-refinement-v1/README.md) and [exact training input](../../training/bottle_refinement_v1/README.md). Both candidates and the selected CPU export are preserved. License: MIT.
''').encode())
    write(training/'README.md', ('''# Exact compact bottle-refinement input

`prepared-training` contains the exact 24,576 FP16 crops used to fit the network, grouped by 64 scene states. `prepared-development` contains all 3,072 development crops. Concatenate each named array along axis zero in manifest order to recover the original arrays; verify their dtypes, shapes and logical SHA-256 digests. Every shard was compared element-for-element with the original run. No single shard exceeds the repository's file-size limit.

`training`, `development` and `evaluation` retain all 5,120 scene/joint/light recipes, projection/visibility/geometry labels and every original RGB image hash, plus representative images and portable scenes. Regeneration uses the repository assets, the fixed camera protocol and the frozen collector's rendering settings. Labels and masks are privileged offline supervision/scoring only. Full RGB archives remain local and are not duplicated here.

Training source and original fit settings are preserved in [the evidence package](../../docs/robotics/evidence/bottle-refinement-v1/README.md). The model failed its fresh perception gate; its final split is exposed and must not be described as a new test when reproduced. Original physical V1 seeds remain unexposed. License: MIT.
''').encode())
    for folder in (model, training, evidence):
        records = {file.relative_to(folder).as_posix(): sha(file) for file in folder.rglob('*') if file.is_file()}
        write(folder/'manifest.json', {'schema': p['schema'], 'files': records})
    verify(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args())
