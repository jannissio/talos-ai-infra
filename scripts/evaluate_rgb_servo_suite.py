"""Run every declared physical case, retaining failures and paired ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text()); seeds = protocol[args.split+'_seeds']
    if not 1 <= args.workers <= 3:
        raise ValueError('Use one to three independent workers.')
    modes = ('nominal', 'nominal-frozen', 'push-live', 'push-frozen') if args.include_frozen_nominal else ('nominal', 'push-live', 'push-frozen')
    cases = [(seed, mode) for seed in seeds for mode in modes]
    require_space(args.output, len(cases)*16*1024**2); args.output.mkdir(parents=True)
    frozen = {'protocol': args.protocol.as_posix(), 'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'split': args.split, 'cases': [{'seed': seed, 'mode': mode} for seed, mode in cases],
              'observer': args.observer.as_posix(), 'observer_sha256': hashlib.sha256(args.observer.read_bytes()).hexdigest(),
              'motor': args.motor.as_posix(), 'motor_sha256': hashlib.sha256(args.motor.read_bytes()).hexdigest(),
              'minimum_views': args.minimum_views, 'push_protocol': args.push.as_posix(),
              'rigid_geometry': args.rigid_geometry,
              'openvino': args.openvino.as_posix() if args.openvino else None,
              'include_frozen_nominal': args.include_frozen_nominal,
              'scope': 'Original finite workspace and all three requested destinations, including known training-stage reach failures. No case is excluded.'}
    (args.output/'frozen-inputs.json').write_text(json.dumps(frozen, indent=2)+'\n')
    began = time.perf_counter(); rows = []

    def one(seed, mode):
        name = f'{seed}-{mode}'; folder = args.output/name
        require_space(folder, 16*1024**2)
        command = [sys.executable, 'scripts/evaluate_rgb_servo_bottle.py', '--protocol', str(args.protocol),
                   '--split', args.split, '--seed', str(seed), '--observer', str(args.observer), '--motor', str(args.motor),
                   '--minimum-views', str(args.minimum_views), '--output', str(folder)]
        if mode.startswith('push-'):
            command += ['--push', str(args.push)]
        if args.rigid_geometry:
            command += ['--rigid-geometry']
        if args.openvino:
            command += ['--openvino', str(args.openvino)]
        if mode in ('push-frozen', 'nominal-frozen'):
            command += ['--mode', 'frozen']
        with (args.output/(name+'.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = folder/'report.json'
        if path.exists():
            data = json.loads(path.read_text()); physical = data.get('physical') or {}
            metrics = physical.get('metrics') or {}
            return {'seed': seed, 'case': mode, 'status': data['status'], 'passed': bool(physical.get('passed')),
                    'message': data.get('message'), 'arm': data.get('arm'), 'stage': data.get('stage'),
                    'inference_runtime': data.get('inference_runtime'),
                    'placement_error_mm': metrics.get('placement_error_mm'), 'simulation_seconds': data.get('simulation_seconds'),
                    'wall_seconds': data.get('wall_seconds'), 'push_translation_mm': data.get('push_translation_before_grasp_mm'),
                    'counterfactual_max_action_delta_rad': data.get('image_counterfactual_max_action_delta_rad'),
                    'report': name+'/report.json', 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return {'seed': seed, 'case': mode, 'status': 'execution_error', 'passed': False, 'exit_code': process.returncode,
                'message': 'No physical report; inspect the preserved local log.'}

    def summary():
        paired = []
        for seed in seeds:
            outcomes = {r['case']: r for r in rows if r['seed'] == seed}
            if len(outcomes) == len(modes):
                pair = {'seed': seed, 'nominal': outcomes['nominal']['passed'],
                        'live': outcomes['push-live']['passed'], 'frozen': outcomes['push-frozen']['passed']}
                if args.include_frozen_nominal:
                    pair['nominal_frozen'] = outcomes['nominal-frozen']['passed']
                paired.append(pair)
        return {'frozen_inputs': frozen, 'attempted': len(rows), 'planned': len(cases),
                'passed_by_case': {mode: sum(r['passed'] for r in rows if r['case'] == mode) for mode in modes},
                'paired': paired, 'rows': sorted(rows, key=lambda r: (r['seed'], r['case'])),
                'wall_seconds': time.perf_counter()-began}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, seed, mode) for seed, mode in cases]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); require_space(args.output, 1024**2)
            (args.output/'summary.json').write_text(json.dumps(summary(), indent=2)+'\n')
            print(json.dumps(row), flush=True)
    print(json.dumps({k: v for k, v in summary().items() if k not in ('rows', 'frozen_inputs')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-bottle-v1.json'))
    p.add_argument('--split', choices=['development', 'evaluation'], required=True)
    p.add_argument('--observer', type=Path, required=True)
    p.add_argument('--motor', type=Path, default=Path('.run/rgb-servo-motor-fit-v1/model.safetensors'))
    p.add_argument('--minimum-views', type=int, choices=[2, 3], default=2)
    p.add_argument('--rigid-geometry', action='store_true')
    p.add_argument('--openvino', type=Path)
    p.add_argument('--include-frozen-nominal', action='store_true')
    p.add_argument('--push', type=Path, default=Path('docs/robotics/experiments/rgb-servo-push-v1.json'))
    p.add_argument('--workers', type=int, default=2); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
