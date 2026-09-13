"""Preserve and independently rescore the stopped V3 motor-only experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from safetensors.torch import load_file
from simulation_lab.rgb_servo_motor import CartesianMotorNet, RobotGeometry
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('docs/robotics/evidence/rgb-servo-v3')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    payload = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    require_space(path, len(payload) + 1024**2)
    path.write_bytes(payload)


def check_score(checkpoint, recorded, development, device):
    """Reload weights and use MuJoCo, not the differentiable training geometry."""
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    network = CartesianMotorNet().to(device)
    network.load_state_dict(load_file(str(checkpoint), device=device))
    points = torch.tensor(development['points'], device=device)
    sides = torch.tensor(development['sides'], device=device)
    with torch.inference_mode():
        predicted = network(points, sides).cpu().numpy()
    model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])
    reference = RobotGeometry(model)
    rows = []
    for point, sign, joints in zip(development['points'], development['sides'], predicted):
        side, offset = ('left', 0) if sign < 0 else ('right', 6)
        actual, rotation = reference.pose(side, joints)
        error = float(np.linalg.norm(actual - point))
        axis = float(np.linalg.norm(rotation[:, 1] - [0., 0., 1.]))
        limits = model.actuator_ctrlrange[offset:offset+5]
        accepted = bool(error <= .0015 and axis <= .025 and
                        np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
        rows.append({'side': side, 'position_error_mm': error*1000,
                     'axis_error': axis, 'accepted': accepted})
    if len(rows) != recorded['total'] or len(rows) != len(recorded['rows']):
        raise ValueError('Development denominator changed.')
    position_deltas, axis_deltas = [], []
    for actual, expected in zip(rows, recorded['rows']):
        if actual['accepted'] != expected['accepted'] or actual['side'] != expected['side']:
            raise ValueError('A recorded point changed acceptance or arm.')
        position_deltas.append(abs(actual['position_error_mm'] - expected['position_error_mm']))
        axis_deltas.append(abs(actual['axis_error'] - expected['axis_error']))
    # Allow small CPU/CUDA arithmetic differences; never change the physical guard.
    if max(position_deltas) > .001 or max(axis_deltas) > 1e-5:
        raise ValueError('Checkpoint predictions do not reproduce the recorded scores.')
    errors = [r['position_error_mm'] for r in rows]
    result = {'accepted': sum(r['accepted'] for r in rows), 'total': len(rows),
              'position_error_mm': {'p95': float(np.quantile(errors, .95)),
                                    'median': float(np.median(errors)), 'max': max(errors)},
              'maximum_rescore_position_delta_mm': max(position_deltas),
              'maximum_rescore_axis_delta': max(axis_deltas)}
    if result['accepted'] != recorded['accepted']:
        raise ValueError('Recorded acceptance count differs.')
    return result


def verify(output, device):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['file']
        if digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('Packaged file changed: ' + item['file'])
    for name, expected in manifest['reproduction_inputs_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Reproduction input changed: ' + name)
    training = json.loads((output / 'training.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'training/bottle_servo_v1/motor/development.npz', allow_pickle=False) as data:
        development = {name: data[name] for name in data.files}
    results = [{'step': 0, **check_score(ROOT / 'models/bottle_servo_v1/motor.safetensors',
                                        training['baseline'], development, device)}]
    for candidate in training['candidates']:
        checkpoint = output / 'candidates' / ('step-%06d.safetensors' % candidate['step'])
        if digest(checkpoint) != candidate['sha256']:
            raise ValueError('Saved candidate differs from the trained checkpoint.')
        results.append({'step': candidate['step'],
                        **check_score(checkpoint, candidate['development'], development, device)})
    selected = min(training['candidates'], key=lambda c: (-c['development']['accepted'],
                   c['development']['position_error_mm']['p95']))
    if selected['step'] != training['selected_step'] or training['kinematic_gate_passed']:
        raise ValueError('This package must preserve the failed declared gate.')
    chosen = next(row for row in results if row['step'] == selected['step'])
    if chosen['accepted'] >= 360 and chosen['position_error_mm']['p95'] <= .5:
        raise ValueError('Rescoring unexpectedly changes the declared gate.')
    return {'verified_files': len(manifest['files']), 'device': device, 'scores': results,
            'kinematic_gate_passed': False, 'physical_trials': 0,
            'status': 'stopped at the unchanged kinematic gate; not promoted'}


def package(source, output):
    if output.exists():
        raise FileExistsError(output)
    training = json.loads((source / 'training.json').read_text(encoding='utf-8'))
    protocol_path = ROOT / training['protocol']
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if digest(protocol_path) != training['protocol_sha256']:
        raise ValueError('The pre-training protocol changed.')
    if training['steps'] != 6000 or [c['step'] for c in training['candidates']] != [3000, 6000]:
        raise ValueError('Unexpected training budget or candidate set.')
    if training['kinematic_gate_passed']:
        raise ValueError('This packager is for the stopped V3 experiment.')
    total_raw = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if total_raw*2 + 16*1024**2 > protocol['budget']['max_new_data_mib']*1024**2:
        raise ValueError('The raw plus packaged estimate exceeds the declared budget.')
    require_space(output, total_raw + 16*1024**2)
    output.mkdir(parents=True)
    files = []

    def copy(path, relative):
        target = output / relative
        require_space(target, path.stat().st_size + 1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files.append({'source': path.relative_to(ROOT).as_posix(), 'source_sha256': digest(path),
                      'file': target.relative_to(ROOT).as_posix(), 'sha256': digest(target),
                      'bytes': target.stat().st_size, 'transformation': 'byte-identical copy'})

    copy(source / 'training.json', Path('training.json'))
    copy(protocol_path, Path('protocol.json'))
    for name, expected in training['source_sha256'].items():
        path = ROOT / name.replace('\\', '/')
        if digest(path) != expected:
            raise ValueError('Training source changed before packaging.')
        copy(path, Path('evaluated-source') / path.relative_to(ROOT))
    for candidate in training['candidates']:
        path = ROOT / candidate['checkpoint']
        if digest(path) != candidate['sha256'] or path.parent.parent != source:
            raise ValueError('Unexpected candidate input.')
        copy(path, Path('candidates') / ('step-%06d.safetensors' % candidate['step']))
    inputs = [protocol_path, ROOT / 'models/bottle_servo_v1/motor.safetensors']
    data = ROOT / 'training/bottle_servo_v1/motor'
    inputs += [data / name for name in ('training.npz', 'development.npz', 'manifest.json')]
    for split in ('training', 'development'):
        if digest(data / (split + '.npz')) != training['data_sha256'][split]:
            raise ValueError('The original training input changed.')
    if digest(inputs[1]) != training['warm_start_sha256']:
        raise ValueError('The original motor changed.')
    # These dependencies were unchanged throughout this training-only change.
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', 'simulation_lab'], cwd=ROOT).decode().split('\0')
    dependencies = [ROOT / name for name in tracked if name and
                    (name.endswith('.py') or name.startswith('simulation_lab/assets/'))]
    for path in dependencies:
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest(path):
            raise ValueError('Shared dependency differs from the unchanged parent commit: ' + relative)
    inputs += dependencies
    summary = {
        'schema': 'talos.rgb-servo-motor-refinement-result.v3',
        'status': 'stopped at the unchanged kinematic gate; not promoted',
        'selected_step': training['selected_step'], 'kinematic_gate_passed': False,
        'gate': {'minimum_accepted': 360, 'total': 362, 'maximum_p95_position_error_mm': .5},
        'baseline': {k: v for k, v in training['baseline'].items() if k != 'rows'},
        'candidates': [{'step': c['step'], 'sha256': c['sha256'],
                        **{k: v for k, v in c['development'].items() if k != 'rows'}} for c in training['candidates']],
        'training_steps': training['steps'], 'wall_seconds_at_last_training_step': training['curve'][-1]['wall_seconds'],
        'maximum_torch_allocated_vram_gib': max(row['peak_vram_gib'] for row in training['curve']),
        'cuda_device': training['cuda_device'],
        'cuda_geometry_maximum_absolute_error': training['cuda_geometry_maximum_absolute_error'],
        'physical_trials': 0, 'physical_development_seeds_exposed': [], 'physical_evaluation_seeds_exposed': [],
        'scope': 'Original exposed kinematic development labels only. No physical success, OpenVINO parity or wider coverage is established.',
        'training_log': 'Structured curve preserved in training.json. The raw local console log is not packaged because its warning includes a personal absolute path.'}
    write_json(output / 'summary.json', summary)
    files.append({'file': (output / 'summary.json').relative_to(ROOT).as_posix(),
                  'sha256': digest(output / 'summary.json'), 'bytes': (output / 'summary.json').stat().st_size,
                  'transformation': 'Summary derived from the preserved training report; no scores changed.'})
    write_json(output / 'manifest.json', {
        'schema': 'talos.rgb-servo-motor-refinement-package.v3', 'files': files,
        'reproduction_inputs_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs},
        'dependency_check_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dependency_scope': 'Shared dependencies checked at packaging against the unchanged parent revision; the original training report hashes the two new training sources.',
        'original_raw_bytes': total_raw, 'expected_raw_plus_package_bytes_upper_bound': total_raw*2 + 16*1024**2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.run/rgb-servo-v3-motor-refinement'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep the evidence package within this repository.')
    if not args.verify_only:
        package((ROOT / args.source).resolve(), output)
    result = verify(output, args.device)
    if not args.verify_only:
        write_json(output / 'audit.json', result)
    print(json.dumps(result, indent=2))
