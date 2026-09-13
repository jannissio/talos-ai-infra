"""Preserve a full-batch motor fit and rescore its checkpoints without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.package_rgb_servo_motor_refinement import check_score
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    require_space(path, 1024**2)
    path.write_bytes((json.dumps(value, indent=2)+'\n').encode('utf-8'))


def rescore(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = ROOT / row['file']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Packaged file changed: ' + row['file'])
    for name, expected in manifest['reproduction_inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reproduction input changed: ' + name)
    training = json.loads((output / 'training.json').read_text(encoding='utf-8'))
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    settings = protocol['motor_training']
    with np.load(ROOT / settings['data'] / 'development.npz', allow_pickle=False) as arrays:
        development = {name: arrays[name] for name in arrays.files}
    scores = [{'update': 0, **check_score(ROOT / settings['warm_start'], training['baseline'], development, 'cpu')}]
    for row in training['candidates']:
        path = output / 'candidates' / ('update-%06d.safetensors' % row['update'])
        if sha(path) != row['sha256']:
            raise ValueError('Checkpoint does not match the original training report.')
        scores.append({'update': row['update'], **check_score(path, row['development'], development, 'cpu')})
    if training['training_completed']:
        if [c['update'] for c in training['candidates']] != settings['candidate_updates']:
            raise ValueError('The declared candidate set changed.')
        selected = min(training['candidates'], key=lambda c: (-c['development']['accepted'], c['development']['position_error_mm']['p95']))
        if selected['update'] != training['selected_update']:
            raise ValueError('The predeclared selection rule changed.')
        candidate = next(row for row in scores if row['update'] == selected['update'])
        gate = protocol['kinematic_gate']
        passed = (candidate['accepted'] >= gate['minimum_accepted'] and candidate['total'] == gate['total']
                  and candidate['position_error_mm']['p95'] <= gate['maximum_p95_position_error_mm'])
    else:
        passed = False
    if passed != training['kinematic_gate_passed']:
        raise ValueError('CPU rescoring changes the declared gate outcome.')
    return {'verified_files': len(manifest['files']), 'device': 'cpu', 'scores': scores,
            'kinematic_gate_passed': passed,
            'scope': 'Checkpoint reload and independent MuJoCo forward-geometry checks only; no new physical test.'}


def package(source, output):
    if output.exists():
        raise FileExistsError('Preserve prior packages; use a new output directory.')
    training = json.loads((source / 'training.json').read_text(encoding='utf-8'))
    protocol_path = ROOT / training['protocol']
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if sha(protocol_path) != training['protocol_sha256']:
        raise ValueError('The declared training protocol changed.')
    for name, expected in training['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Training source changed before packaging: ' + name)
    raw_bytes = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if raw_bytes*2 + 16*1024**2 > protocol['budget']['max_new_data_mib']*1024**2:
        raise ValueError('The raw plus offline-package estimate exceeds the declared data budget.')
    require_space(output, raw_bytes + 16*1024**2)
    output.mkdir(parents=True)
    files = []

    def copy(path, relative):
        target = output / relative
        require_space(target, path.stat().st_size + 1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files.append({'source': path.relative_to(ROOT).as_posix(), 'file': target.relative_to(ROOT).as_posix(),
                      'sha256': sha(target), 'bytes': target.stat().st_size, 'transformation': 'byte-identical copy'})

    copy(source / 'training.json', Path('training.json'))
    copy(protocol_path, Path('protocol.json'))
    for name in training['source_sha256']:
        copy(ROOT / name, Path('evaluated-source') / name)
    for row in training['candidates']:
        path = ROOT / row['checkpoint']
        if path.parent != source or sha(path) != row['sha256']:
            raise ValueError('Unexpected checkpoint or hash.')
        copy(path, Path('candidates') / ('update-%06d.safetensors' % row['update']))
    # Preserve the existing dependency inventory; no shared runtime changed.
    inherited = json.loads((ROOT / 'docs/robotics/evidence/rgb-servo-v3/manifest.json').read_text(encoding='utf-8'))
    inputs = dict(inherited['reproduction_inputs_sha256'])
    settings = protocol['motor_training']
    inputs[settings['warm_start']] = training['warm_start_sha256']
    inputs[training['protocol']] = training['protocol_sha256']
    for split, expected in training['data_sha256'].items():
        inputs[(Path(settings['data']) / (split + '.npz')).as_posix()] = expected
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Inherited reproduction input changed: ' + name)
    summary = {key: training[key] for key in ('training_completed', 'stop_reason', 'completed_updates',
               'gradient_evaluations', 'optimizer_iterations', 'selected_update', 'kinematic_gate_passed', 'cuda_device',
               'cuda_geometry_maximum_absolute_error')}
    summary.update(baseline={k: v for k, v in training['baseline'].items() if k != 'rows'},
                   candidates=[{'update': row['update'], 'sha256': row['sha256'],
                                **{k: v for k, v in row['development'].items() if k != 'rows'}} for row in training['candidates']],
                   wall_seconds_at_last_log=training['curve'][-1]['wall_seconds'] if training['curve'] else None,
                   physical_trials_at_packaging=0, physical_development_seeds_exposed=[], physical_evaluation_seeds_exposed=[],
                   status='offline gate passed; physical validation still required' if training['kinematic_gate_passed'] else 'stopped at the declared offline gate; not promoted')
    write_json(output / 'summary.json', summary)
    files.append({'file': (output / 'summary.json').relative_to(ROOT).as_posix(), 'sha256': sha(output / 'summary.json'),
                  'bytes': (output / 'summary.json').stat().st_size, 'transformation': 'Summary derived from the preserved original report.'})
    write_json(output / 'manifest.json', {'schema': 'talos.full-batch-motor-evidence.v1', 'files': files,
                                        'reproduction_inputs_sha256': inputs, 'original_raw_bytes': raw_bytes,
                                        'scope': 'Offline training package. Inherited dependency hashes are checked again at packaging; they were not newly captured by the trainer.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep the evidence package inside this repository.')
    if not args.verify_only:
        if not args.source:
            parser.error('--source is required when creating a package')
        package((ROOT / args.source).resolve(), output)
    result = rescore(output)
    if not args.verify_only:
        write_json(output / 'audit.json', result)
    print(json.dumps(result, indent=2))
