"""Audit and preserve physical development after a full-batch motor refinement."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space

CONDITIONS = ('nominal', 'nominal-frozen', 'push-live', 'push-frozen')
ARTIFACTS = ('observer.safetensors', 'motor.safetensors', 'openvino/observer.xml',
             'openvino/observer.bin', 'openvino/motor.xml', 'openvino/motor.bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(batch, model, offline, sources=None):
    protocol, training = read(offline / 'protocol.json'), read(offline / 'training.json')
    summary, frozen = read(batch / 'summary.json'), read(batch / 'frozen-inputs.json')
    if summary['frozen_inputs'] != frozen or summary['split'] != 'development':
        raise ValueError('Incorrect development inputs.')
    if sha(offline / 'protocol.json') != frozen['protocol_sha256'] or training['protocol_sha256'] != frozen['protocol_sha256']:
        raise ValueError('Training and physical protocols differ.')
    selected = next(row for row in training['candidates'] if row['update'] == training['selected_update'])
    if not training['training_completed'] or not training['kinematic_gate_passed'] or selected['sha256'] != sha(model / 'motor.safetensors'):
        raise ValueError('Physical trials did not use the preselected, offline-qualified motor.')
    manifest = read(model / 'experiment.json')
    if sha(model / 'experiment.json') != frozen['model_manifest_sha256']:
        raise ValueError('Model manifest differs from the physical freeze.')
    for name in ARTIFACTS:
        if sha(model / name) != frozen['model_artifact_sha256'][name]:
            raise ValueError('Model differs from freeze: ' + name)
    for name in ('observer.safetensors', 'openvino/observer.xml', 'openvino/observer.bin'):
        if sha(model / name) != sha(ROOT / 'models/bottle_servo_v1' / name):
            raise ValueError('The original observer was changed.')
    for name, digest in frozen['source_sha256'].items():
        source = sources / (digest + '.py') if sources else ROOT / name
        if sha(source) != digest:
            raise ValueError('Frozen source is missing or changed: ' + name)
    seeds = protocol['development_seeds']
    expected = {(seed, condition) for seed in seeds for condition in CONDITIONS}
    rows = summary['rows']
    if (summary['attempted'] != len(expected) or summary['planned'] != len(expected)
            or len(rows) != len(expected) or {(r['seed'], r['case']) for r in rows} != expected):
        raise ValueError('Missing, duplicate or unexpected physical trial.')
    reports, frames, legs = {}, 0, 0
    reasons = {condition: Counter() for condition in CONDITIONS}
    for row in rows:
        path = batch / row['report']
        if sha(path) != row['report_sha256']:
            raise ValueError('A retained report changed.')
        report = read(path)
        reports[row['seed'], row['case']] = report
        mode = 'frozen' if row['case'].endswith('frozen') else 'live'
        if report['seed'] != row['seed'] or report['split'] != 'development' or report['mode'] != mode:
            raise ValueError('Physical trial identity differs from its declared condition.')
        for name in ('protocol_sha256', 'routing_sha256'):
            if report[name] != frozen[name]:
                raise ValueError('Physical input hash mismatch: ' + name)
        for name in ('observer_sha256', 'motor_sha256', 'openvino_artifact_sha256'):
            if report[name] != manifest[name]:
                raise ValueError('Physical model mismatch: ' + name)
        for name, digest in report['source_sha256'].items():
            source = sources / (digest + '.py') if sources else path.parent / 'evaluated-source' / name
            if frozen['source_sha256'][name] != digest or sha(source) != digest:
                raise ValueError('Physical source mismatch: ' + name)
        for name in ('physics_state_writes_during_control', 'hidden_forces', 'inverse_solver_calls_during_control'):
            if report[name] != 0:
                raise ValueError('Forbidden runtime assistance: ' + name)
        physical = report.get('physical') or {}
        passed = report['status'] == 'succeeded' and bool(physical.get('passed'))
        if row['passed'] != passed or row['status'] != report['status']:
            raise ValueError('Summary contradicts the physical outcome.')
        if passed and (len(report['completed_legs']) != len(report['route']['legs'])
                       or report['cumulative_non_target_displacement_m'] > .004):
            raise ValueError('A successful route failed completeness or displacement checks.')
        for leg in report.get('completed_legs', []):
            result = leg['physical']
            metrics = result['metrics']
            if (not result['passed'] or metrics['placement_error_mm'] > 8
                    or not metrics['both_arms_parked'] or metrics['unexpected_collisions'] != 0
                    or metrics['stable_release_and_park_s'] < .5 - 1e-9):
                raise ValueError('A completed leg violates the existing physical limits.')
            legs += 1
        if not passed:
            reasons[row['case']][report.get('message', report['status'])] += 1
        states_path = path.parent / 'states.npz'
        if states_path.exists():
            with np.load(states_path, allow_pickle=False) as states:
                length = len(states['time'])
                if (any(len(states[name]) != length for name in states.files)
                        or (not length and report.get('simulation_seconds') != 0)
                        or np.any(np.diff(states['time']) <= 0)):
                    raise ValueError('Incomplete or unordered synchronized state arrays.')
                for name in ('qpos', 'qvel', 'time', 'targets'):
                    if not np.isfinite(states[name]).all():
                        raise ValueError('Nonfinite physical state.')
                frames += length
        elif report['status'] != 'invalid_start':
            raise ValueError('Missing state recording for a valid starting scene.')
    counts = {case: sum(r['passed'] for r in rows if r['case'] == case) for case in CONDITIONS}
    if counts != summary['passed_by_case']:
        raise ValueError('Summary counts disagree with original reports.')
    lookup = {(r['seed'], r['case']): r['passed'] for r in rows}
    paired = [s for s in seeds if lookup[s, 'nominal'] and lookup[s, 'nominal-frozen']]
    deltas = []
    for seed in seeds:
        baseline = reports[seed, 'nominal']
        for case in CONDITIONS:
            if any(reports[seed, case][key] != baseline[key] for key in ('initial_pose', 'destination_m')):
                raise ValueError('Paired conditions used different scenes or goals.')
        for case in ('nominal', 'nominal-frozen'):
            if reports[seed, case].get('disclosed_push_ticks', 0):
                raise ValueError('An undisturbed condition received a push.')
        a, b = reports[seed, 'push-live'], reports[seed, 'push-frozen']
        if a.get('disclosed_push_ticks', 0) != b.get('disclosed_push_ticks', 0):
            raise ValueError('Paired force durations differ.')
        left, right = a.get('push_translation_before_grasp_mm'), b.get('push_translation_before_grasp_mm')
        if left is not None and right is not None:
            deltas.append(float(np.max(np.abs(np.asarray(left) - right))))
    gate = protocol['development_gate']
    passed = counts['nominal'] >= gate['minimum_nominal_live_passes'] and counts['push-live'] >= gate['minimum_pushed_live_passes']
    return {'schema': 'talos.rgb-servo-full-batch-physical-audit.v1',
            'experiment': protocol['schema'], 'all_trials_verified': len(rows), 'seeds_per_condition': len(seeds),
            'passed_by_case': counts, 'synchronized_state_frames': frames, 'verified_completed_legs': legs,
            'paired_nominal_pass_seeds': paired,
            'pushed_passes_in_paired_subset': {case: sum(lookup[s, case] for s in paired) for case in ('push-live', 'push-frozen')},
            'maximum_paired_push_translation_difference_mm': max(deltas, default=None),
            'failures_by_case': {case: dict(reasons[case]) for case in CONDITIONS},
            'development_gate_passed': passed, 'physical_final_trials': 0,
            'final_seeds_exposed': [], 'promoted': False,
            'status': 'development gate passed; final evaluation still required' if passed else 'stopped at physical development gate',
            'scope': 'All four development conditions retained. Per-leg frozen-image ablation; live RGB during approach/descent, proprioceptive carry. No final-test, Intel or arbitrary-workspace claim.'}


def package(args):
    if args.output.exists():
        raise FileExistsError('Preserve the existing package; use a fresh output directory.')
    result = audit(args.batch, args.model, args.offline)
    # This package records a stopped development experiment; never infer a final freeze.
    if result['development_gate_passed']:
        raise ValueError('This stopped-experiment packager requires a failed development gate.')
    frozen, protocol = read(args.batch / 'frozen-inputs.json'), read(args.offline / 'protocol.json')
    inputs = [(args.batch / name, args.output / 'development' / name, False)
              for name in ('summary.json', 'frozen-inputs.json')]
    inputs += [(args.model / name, args.output / 'model' / name, False)
               for name in (*ARTIFACTS, 'experiment.json', 'motor-export-parity.json')]
    inputs += [(ROOT / name, args.output / 'source-blobs' / (digest + '.py'), False)
               for name, digest in frozen['source_sha256'].items()]
    for name in ('routing', 'push_protocol'):
        inputs.append((ROOT / protocol[name], args.output / (name + '.json'), False))
    for row in read(args.batch / 'summary.json')['rows']:
        folder = (args.batch / row['report']).parent
        for name in ('report.json', 'states.npz', 'observations.json', 'image-counterfactuals.json', 'scene.xml'):
            if (folder / name).exists():
                inputs.append((folder / name, args.output / 'development' / folder.name / name, name == 'scene.xml'))
    expected = sum(source.stat().st_size for source, _, _ in inputs) + 4*1024**2
    version = protocol['schema'].rsplit('.', 1)[-1]
    folders = list((ROOT / '.run').glob('rgb-servo-' + version + '*'))
    folders += [args.offline, ROOT / 'models' / ('bottle_servo_' + version)]
    existing = sum(p.stat().st_size for folder in folders if folder.is_dir() for p in folder.rglob('*') if p.is_file())
    existing += sum(p.stat().st_size for p in (ROOT / '.run/final-goal').glob('rgb-servo-' + version + '*') if p.is_file())
    if existing + expected > protocol['budget']['max_new_data_mib']*1024**2:
        raise ValueError('Package exceeds the declared data budget.')
    require_space(args.output, expected)
    args.output.mkdir(parents=True)
    files = []
    for source, destination, scene in inputs:
        require_space(destination, source.stat().st_size + 1024**2)
        payload, transformation = source.read_bytes(), 'byte-identical copy'
        if scene:
            tree = ET.fromstring(payload)
            compiler = tree.find('compiler')
            meshes = (source.parent / compiler.get('meshdir')).resolve()
            if not meshes.is_relative_to(ROOT / 'simulation_lab/assets') or not meshes.is_dir():
                raise ValueError('Scene mesh directory is outside repository assets.')
            compiler.set('meshdir', os.path.relpath(meshes, destination.parent).replace('\\', '/'))
            payload = ET.tostring(tree, encoding='utf-8')
            transformation = 'Only compiler meshdir remapped to the same repository assets.'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError('Conflicting content-addressed source.')
            continue
        destination.write_bytes(payload)
        files.append({'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
                      'file': destination.relative_to(args.output).as_posix(), 'sha256': sha(destination),
                      'bytes': len(payload), 'transformation': transformation})
    packaged = audit(args.output / 'development', args.output / 'model', args.offline, args.output / 'source-blobs')
    if packaged != result:
        raise ValueError('Portable package does not reproduce the original audit.')
    for name, value in [('audit.json', result), ('manifest.json', {
            'schema': 'talos.rgb-servo-full-batch-physical-package.v1', 'files': files,
            'originals_modified': False, 'offline_package': args.offline.relative_to(ROOT).as_posix(),
            'omissions': 'Raw camera PNGs and console logs remain local. Every trial report, synchronized state array, compact RGB observation and image-action comparison is preserved.',
            'observer_export': 'Actual observer weights and IR are identical to V1. The export parity report also probes a newly exported observer; only its motor export was used in these trials.'})]:
        require_space(args.output / name, 1024**2)
        (args.output / name).write_bytes((json.dumps(value, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'files': len(files), 'bytes': sum(r['bytes'] for r in files), 'audit': result}))


def verify(args):
    manifest = read(args.output / 'manifest.json')
    for row in manifest['files']:
        if sha(args.output / row['file']) != row['sha256']:
            raise ValueError('Packaged file changed: ' + row['file'])
    if sha(args.output / 'routing.json') != read(args.output / 'development/frozen-inputs.json')['routing_sha256']:
        raise ValueError('Packaged routing configuration changed.')
    result = audit(args.output / 'development', args.output / 'model', args.offline, args.output / 'source-blobs')
    if result != read(args.output / 'audit.json'):
        raise ValueError('Physical audit no longer matches the saved outcome.')
    print(json.dumps({'manifest_files_verified': len(manifest['files']), 'audit': result}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and (not args.batch or not args.model):
        parser.error('--batch and --model are required for a new package')
    for name in ('offline', 'output', 'batch', 'model'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    verify(args) if args.verify_only else package(args)
