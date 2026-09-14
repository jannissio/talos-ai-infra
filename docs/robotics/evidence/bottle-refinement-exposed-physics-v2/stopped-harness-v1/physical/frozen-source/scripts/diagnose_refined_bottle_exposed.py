"""Physical diagnosis on the already exposed bottle grid; never a promotion test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, space, write
from scripts.evaluate_refined_bottle_physics import execute_trial, integration_path, checked_integration
from scripts.measure_manipulation_coverage import setup as coverage_setup


def declare(path):
    parent_path = ROOT/'docs/robotics/experiments/bottle-refinement-v1.json'
    parent = checked_protocol(parent_path)
    raw = ROOT/parent['raw_root']
    perception = read(raw/'fresh-perception.json')
    coverage_path = ROOT/'docs/robotics/experiments/manipulation-coverage-v1.json'
    coverage = read(coverage_path)
    package = ROOT/'docs/robotics/evidence/manipulation-coverage-v1'
    cases = [case for case in coverage['cases'] if case['skill'] == 'bottle']
    if path.exists():
        raise FileExistsError('Preserve each independent protocol.')
    names = ['scripts/diagnose_refined_bottle_exposed.py', 'scripts/evaluate_refined_bottle_physics.py',
        'scripts/measure_manipulation_coverage.py', 'scripts/bottle_refinement_experiment.py']
    names += [file.relative_to(ROOT).as_posix() for file in (ROOT/'simulation_lab').glob('*.py')]
    motor = ROOT/parent['physical']['motor']
    inputs = [parent_path, coverage_path, raw/'fresh-perception.json', raw/'openvino/artifacts.json',
        ROOT/'docs/robotics/experiments/rgb-servo-routing-v2.json', motor/'motor.safetensors',
        motor/'openvino/motor.xml', motor/'openvino/motor.bin', package/'audit.json', package/'manifest.json']
    inputs += [file for file in (raw/'openvino').glob('*') if file.is_file()]
    inputs += [ROOT/coverage['context']/name for name in coverage['context_sha256']]
    for case in cases:
        for mode in ('learned', 'teacher'):
            inputs += [package/(case['id']+'-'+mode)/name for name in ('states.npz', 'report.json')]
    frozen = {name: sha(ROOT/name) for name in names}
    frozen.update({file.relative_to(ROOT).as_posix(): sha(file) for file in inputs})
    p = {'schema': 'talos.bottle-refinement-exposed-physics.v1', 'declared_on': '2026-09-14',
        'raw_root': '.run/bottle-refinement-exposed-physics-v1', 'model_package': parent['model_package'],
        'training_package': parent['training_package'], 'evidence_package': 'docs/robotics/evidence/bottle-refinement-exposed-physics-v1',
        'camera_protocol': parent['camera_protocol'], 'camera_configuration': parent['camera_configuration'],
        'coverage_protocol': coverage_path.relative_to(ROOT).as_posix(), 'coverage_evidence': package.relative_to(ROOT).as_posix(),
        'cases': cases, 'modes': ['live', 'frozen'], 'frozen_sha256': frozen,
        'budget': {'maximum_raw_and_packaged_gib': 4, 'maximum_physical_trials': 58, 'reserve_gib': 10},
        'decision': 'A separate exposed-grid diagnostic follows a FAILED fresh synthetic perception gate. Do not run or tune on bottle-refinement-v1 physical seeds. No promotion, new-scene generalization or qualifying Intel claim can follow from this diagnostic.',
        'reason': 'Locate physical failure mechanisms and usable execution regions before another development change. Vision error alone does not identify motor/contact failures. These contexts contributed training images, and the older grid outcomes were already exposed.',
        'parent_perception_result': perception['summary'],
        'selection': 'Every original bottle case: one anchor, all 24 XY grid points, both yaw variations and both additional destinations. Keep invalid cases. No success-based case selection.',
        'comparison': 'Use exact original reset recipes. Reuse the preserved preset/teacher results only after verifying identical starting qpos/qvel. Live and per-leg frozen RGB use identical observer, motor, routes and physical thresholds.',
        'stop': 'Stop for a harness error; physical failures do not remove planned cases. Finish the declared diagnostic, preserve all outcomes and choose any change under a new protocol.'}
    folder = ROOT/p['raw_root']
    if folder.exists():
        raise FileExistsError('Preserve every diagnostic folder.')
    preflight = space(p, folder, 512*1024**2)
    write(path, p)
    spec = {'schema': p['schema'], 'protocol_sha256': sha(path), 'frozen_sha256': frozen,
        'observer': (raw/'openvino').relative_to(ROOT).as_posix(), 'motor': parent['physical']['motor'],
        'routing': 'docs/robotics/experiments/rgb-servo-routing-v2.json', 'baseline': 'models/bottle_visual',
        'settle_steps': coverage['settle_steps'], 'maximum_simulated_seconds': 100., 'maximum_wall_seconds': 180.,
        'maximum_workers': 2, 'preflight': preflight,
        'scope': 'Training-exposed physical diagnostic across the entire previous bottle grid. Failed perception V1 stays failed; no candidate promotion or new holdout claim.'}
    write(integration_path(p), spec)
    for name in names:
        write(folder/'physical/frozen-source'/name, (ROOT/name).read_bytes())
    print({'cases': len(cases), 'physical_trials': len(cases)*2, 'protocol_sha256': sha(path)}, flush=True)


def trial(args):
    p = read(args.protocol)
    spec = checked_integration(p, args.protocol)
    case = next(case for case in p['cases'] if case['id'] == args.case)
    if args.mode not in p['modes']:
        raise ValueError('Undeclared diagnostic mode.')
    coverage = read(ROOT/p['coverage_protocol'])
    def setup(_p, _spec, _seed, _split, folder):
        model, data, layout, state = coverage_setup(coverage, case, folder)
        old_folder = ROOT/p['coverage_evidence']/(case['id']+'-learned')
        with np.load(old_folder/'states.npz', allow_pickle=False) as archive:
            if not np.array_equal(data.qpos, archive['qpos'][0]) or not np.array_equal(data.qvel, archive['qvel'][0]):
                raise ValueError('Exposed initial state is not identical to the preserved comparison.')
        state['identical_to_original_qpos_qvel'] = True
        destination = next(t['position_m'][:2] for t in layout['targets'] if t['object_id'] == 'bottle')
        return model, data, layout, destination, state
    args.seed, args.split = coverage['source_seed'], 'exposed'
    folder = ROOT/p['raw_root']/'physical/exposed'/(args.case+'-'+args.mode)
    return execute_trial(p, spec, args, folder, setup_function=setup)


def batch(args):
    p = read(args.protocol)
    spec = checked_integration(p, args.protocol)
    root = ROOT/p['raw_root']/'physical/exposed'
    if root.exists():
        raise FileExistsError('Preserve the previous exposed physical batch.')
    space(p, root, 512*1024**2)
    root.mkdir()
    def launch(item):
        case, mode = item
        folder = root/(case['id']+'-'+mode)
        log = root/(case['id']+'-'+mode+'.log')
        if folder.exists() or log.exists():
            raise FileExistsError('Preserve each physical output and log.')
        space(p, log, 24*1024**2)
        with log.open('xb') as stream:
            completed = subprocess.run([sys.executable, '-I', str(Path(__file__)), '--protocol', str(args.protocol.resolve()),
                '--case', case['id'], '--mode', mode], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                timeout=spec['maximum_wall_seconds']+90)
        report = read(folder/'report.json') if (folder/'report.json').exists() else {'status': 'harness_error', 'message': 'No report.'}
        row = {'case': case['id'], 'kind': case['kind'], 'mode': mode, 'status': report['status'],
            'message': report['message'], 'exit_code': completed.returncode,
            'report_sha256': sha(folder/'report.json') if (folder/'report.json').exists() else None}
        print(row, flush=True)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=spec['maximum_workers']) as workers:
        for case in p['cases']:
            new = list(workers.map(launch, [(case, mode) for mode in p['modes']]))
            rows += new
            if any(row['status'] == 'harness_error' or row['exit_code'] for row in new):
                write(root/'stopped.json', {'reason': 'Harness error; no later case exposed.', 'rows': rows})
                return 1
    write(root/'summary.json', {'schema': p['schema'], 'protocol_sha256': sha(args.protocol),
        'rows': rows, 'successes': {mode: sum(row['mode'] == mode and row['status'] == 'succeeded' for row in rows) for mode in p['modes']},
        'planned_cases_per_mode': len(p['cases']), 'all_outcomes_retained': len(rows) == len(p['cases'])*2,
        'scope': spec['scope'], 'promotion': False})
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--mode', choices=['live', 'frozen'])
    args = parser.parse_args()
    if args.declare:
        declare(args.declare)
    else:
        raise SystemExit(trial(args) if args.case else batch(args))
