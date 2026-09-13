"""Evaluate a gated motor refinement on all declared RGB-control conditions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('nominal', 'nominal-frozen', 'push-live', 'push-frozen')
ARTIFACTS = ('observer.safetensors', 'motor.safetensors', 'openvino/observer.xml',
             'openvino/observer.bin', 'openvino/motor.xml', 'openvino/motor.bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    require_space(path, 1024**2)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_bytes((json.dumps(value, indent=2)+'\n').encode('utf-8'))
    temporary.replace(path)


def frozen_inputs(protocol_path, routing_path, model):
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    original = json.loads((ROOT / 'docs/robotics/experiments/rgb-servo-final-evaluation-v1.json').read_text(encoding='utf-8'))
    names = list(original['source_sha256']) + ['simulation_lab/rgb_servo_routing.py',
            'scripts/evaluate_rgb_servo_routing.py', 'scripts/evaluate_rgb_servo_full_batch_suite.py']
    return {'protocol_sha256': sha(protocol_path), 'routing_sha256': sha(routing_path),
            'source_sha256': {name: sha(ROOT / name) for name in names},
            'model_manifest_sha256': sha(model / 'experiment.json'),
            'model_artifact_sha256': {name: sha(model / name) for name in ARTIFACTS},
            'conditions': list(CONDITIONS), 'evaluation_seeds': protocol['evaluation_seeds'],
            'maximum_trial_wall_seconds': 300,
            'ablation': 'Initial image frozen separately per leg; a later leg acquires fresh RGB after release and parking.'}


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve previous trials; choose a fresh output directory.')
    if not 1 <= args.workers <= 3:
        raise ValueError('Use one to three isolated physical workers.')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    training = json.loads(args.training_report.read_text(encoding='utf-8'))
    if not training['training_completed'] or not training['kinematic_gate_passed']:
        raise ValueError('The completed training fit must pass its declared offline gate.')
    if training['protocol_sha256'] != sha(args.protocol):
        raise ValueError('Training used a different protocol.')
    selected = next(row for row in training['candidates'] if row['update'] == training['selected_update'])
    if sha(args.model / 'motor.safetensors') != selected['sha256']:
        raise ValueError('Only the preselected motor checkpoint may be evaluated.')
    baseline = ROOT / 'models/bottle_servo_v1'
    for name in ('observer.safetensors', 'openvino/observer.xml', 'openvino/observer.bin'):
        if sha(args.model / name) != sha(baseline / name):
            raise ValueError('The observer or its deployed export changed.')
    frozen = frozen_inputs(args.protocol, args.routing, args.model)
    seeds = protocol[args.split + '_seeds']
    if args.split == 'evaluation':
        if not args.freeze:
            raise ValueError('A completed development gate and frozen selection are required.')
        selection = json.loads(args.freeze.read_text(encoding='utf-8'))
        for name, expected in frozen.items():
            if selection[name] != expected:
                raise ValueError('The final candidate differs from its freeze: ' + name)
        if not selection['development_gate_passed']:
            raise ValueError('The development gate did not pass.')
    version = protocol['schema'].rsplit('.', 1)[-1]
    if not version.startswith('v') or not version[1:].isdigit():
        raise ValueError('Unexpected experiment version.')
    resolved_output = args.output.resolve()
    if resolved_output.parent != ROOT / '.run' or not resolved_output.name.startswith('rgb-servo-' + version + '-'):
        raise ValueError('Use an unused .run/rgb-servo-' + version + '-... directory so all generated data is counted.')
    byte_limit = protocol['budget']['max_new_data_mib']*1024**2

    def used_bytes():
        folders = list((ROOT / '.run').glob('rgb-servo-' + version + '*'))
        folders += [ROOT / 'docs/robotics/evidence' / ('rgb-servo-' + version), ROOT / 'models' / ('bottle_servo_' + version)]
        paths = [p for folder in folders if folder.is_dir() for p in folder.rglob('*')]
        paths += list((ROOT / '.run/final-goal').glob('rgb-servo-' + version + '*'))
        total = 0
        for path in paths:
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # A live summary may have atomically replaced its small .pending file.
                continue
        return total

    expected = len(seeds)*len(CONDITIONS)*3*1024**2
    if used_bytes() + expected > byte_limit:
        raise ValueError('The full planned batch would exceed its declared data budget.')
    require_space(args.output, expected)
    args.output.mkdir(parents=True)
    write_json(args.output / 'frozen-inputs.json', frozen)
    began, rows, lock, reserved = time.perf_counter(), [], Lock(), 0

    def one(seed, condition):
        nonlocal reserved
        name, reservation = f'{seed}-{condition}', 16*1024**2
        folder = args.output / name
        with lock:
            if used_bytes() + reserved + reservation > byte_limit:
                return {'seed': seed, 'case': condition, 'passed': False, 'status': 'resource_limit',
                        'message': 'Data budget reached; existing evidence preserved.'}
            require_space(folder, reservation)
            reserved += reservation
        try:
            command = [sys.executable, 'scripts/evaluate_rgb_servo_routing.py', '--protocol', str(args.protocol),
                       '--routing', str(args.routing), '--model', str(args.model), '--seed', str(seed),
                       '--split', args.split, '--device', 'CPU', '--output', str(folder)]
            if condition in ('nominal-frozen', 'push-frozen'):
                command += ['--mode', 'frozen']
            if condition.startswith('push-'):
                command += ['--push', protocol['push_protocol']]
            with (args.output / (name + '.log')).open('w', encoding='utf-8') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, cwd=ROOT)
                except subprocess.TimeoutExpired:
                    return {'seed': seed, 'case': condition, 'passed': False, 'status': 'execution_timeout',
                            'message': 'Worker exceeded the fixed 300-second wall limit.'}
            path = folder / 'report.json'
            if process.returncode or not path.exists():
                return {'seed': seed, 'case': condition, 'passed': False, 'status': 'execution_error', 'exit_code': process.returncode}
            report = json.loads(path.read_text(encoding='utf-8'))
            for name, expected_hash in report['source_sha256'].items():
                if frozen['source_sha256'][name] != expected_hash:
                    raise ValueError('A worker used a different source: ' + name)
            for field in ('protocol_sha256', 'routing_sha256'):
                if report[field] != frozen[field]:
                    raise ValueError('A worker used a different protocol or routing configuration.')
            physical = report.get('physical') or {}
            metrics = physical.get('metrics') or {}
            return {'seed': seed, 'case': condition,
                    'passed': report['status'] == 'succeeded' and bool(physical.get('passed')),
                    'status': report['status'], 'message': report.get('message'),
                    'route': (report.get('route') or {}).get('kind'), 'completed_legs': len(report.get('completed_legs', [])),
                    'simulation_seconds': report.get('simulation_seconds'), 'wall_seconds': report.get('wall_seconds'),
                    'placement_error_mm': metrics.get('placement_error_mm'),
                    'report': path.relative_to(args.output).as_posix(), 'report_sha256': sha(path)}
        finally:
            with lock:
                reserved -= reservation

    def summary():
        return {'frozen_inputs': frozen, 'split': args.split, 'attempted': len(rows), 'planned': len(seeds)*len(CONDITIONS),
                'passed_by_case': {case: sum(row['passed'] for row in rows if row['case'] == case) for case in CONDITIONS},
                'rows': sorted(rows, key=lambda row: (row['seed'], row['case'])), 'wall_seconds': time.perf_counter()-began,
                'generated_bytes_in_experiment': used_bytes(),
                'scope': 'All declared seeds and conditions retained, including invalid starts, refusals and execution errors. AMD PC/OpenVINO CPU; no Intel or arbitrary-workspace claim.'}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, seed, case): (seed, case) for seed in seeds for case in CONDITIONS}
        for job in as_completed(jobs):
            try:
                row = job.result()
            except Exception as exc:
                seed, case = jobs[job]
                row = {'seed': seed, 'case': case, 'passed': False, 'status': 'audit_error', 'error_type': type(exc).__name__, 'message': str(exc)}
            rows.append(row)
            write_json(args.output / 'summary.json', summary())
            print(json.dumps(row), flush=True)
    print(json.dumps({k: v for k, v in summary().items() if k not in ('rows', 'frozen_inputs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--routing', type=Path, default=Path('docs/robotics/experiments/rgb-servo-routing-v2.json'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
