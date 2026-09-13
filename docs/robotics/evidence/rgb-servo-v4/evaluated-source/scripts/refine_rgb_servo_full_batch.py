"""Bounded full-data motor refinement; no physical evaluation or deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scripts.torch_robot_geometry import TorchRobotGeometry
from simulation_lab.rgb_servo_motor import CartesianMotorNet, RobotGeometry
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingBudgetReached(RuntimeError):
    pass


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior outputs; use an unused directory.')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    settings, budget = protocol['motor_training'], protocol['budget']
    if not torch.cuda.is_available():
        raise RuntimeError('The declared experiment requires CUDA.')
    warm_start = ROOT / settings['warm_start']
    if sha(warm_start) != settings['warm_start_sha256']:
        raise ValueError('The frozen warm-start checkpoint changed.')
    dataset = ROOT / settings['data']
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    tensors, data_hashes = {}, {}
    for split in ('training', 'development'):
        path = dataset / (split + '.npz')
        data_hashes[split] = sha(path)
        if data_hashes[split] != manifest['splits'][split]['sha256']:
            raise ValueError('The original kinematic input changed.')
        with np.load(path, allow_pickle=False) as arrays:
            tensors[split] = {key: torch.tensor(arrays[key], device='cuda') for key in arrays.files}
    if len(tensors['training']['points']) != 4056 or len(tensors['development']['points']) != 362:
        raise ValueError('Unexpected original split sizes.')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(settings['rng_seed'])
    model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])
    reference = RobotGeometry(model)
    geometry = {side: TorchRobotGeometry(model, side, dtype=torch.float32).cuda() for side in ('left', 'right')}
    network = CartesianMotorNet().cuda()
    network.load_state_dict(load_file(str(warm_start), device='cuda'))
    development = tensors['development']

    def score():
        with torch.inference_mode():
            predicted = network(development['points'], development['sides']).cpu().numpy()
        rows = []
        for point, sign, joints in zip(development['points'].cpu().numpy(), development['sides'].cpu().numpy(), predicted):
            side, offset = ('left', 0) if sign < 0 else ('right', 6)
            actual, rotation = reference.pose(side, joints)
            error = float(np.linalg.norm(actual-point))
            axis_error = float(np.linalg.norm(rotation[:, 1]-[0., 0., 1.]))
            limits = model.actuator_ctrlrange[offset:offset+5]
            accepted = bool(error <= .0015 and axis_error <= .025 and
                            np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
            rows.append({'side': side, 'position_error_mm': error*1000, 'axis_error': axis_error, 'accepted': accepted})
        errors = [row['position_error_mm'] for row in rows]
        return {'accepted': sum(row['accepted'] for row in rows), 'total': len(rows),
                'position_error_mm': {'p95': float(np.quantile(errors, .95)),
                                      'median': float(np.median(errors)), 'max': max(errors)}, 'rows': rows}

    baseline = score()
    maximum_geometry_error = 0.
    with torch.inference_mode():
        for side, sign in [('left', -1), ('right', 1)]:
            joints = development['joints'][development['sides'] == sign]
            points, rotations = geometry[side](joints)
            for joint, point, rotation in zip(joints.cpu().numpy(), points.cpu().numpy(), rotations.cpu().numpy()):
                expected_point, expected_rotation = reference.pose(side, joint)
                maximum_geometry_error = max(maximum_geometry_error, float(np.max(np.abs(point-expected_point))),
                                             float(np.max(np.abs(rotation-expected_rotation))))
    if maximum_geometry_error > 2e-6:
        raise ValueError('Training geometry failed its independent MuJoCo check.')
    require_space(args.output, 32*1024**2)
    args.output.mkdir(parents=True)
    source_paths = [Path(__file__).resolve(), ROOT / 'scripts/torch_robot_geometry.py',
                    ROOT / 'simulation_lab/rgb_servo_motor.py', ROOT / 'simulation_lab/scene.py']
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    optimizer = torch.optim.LBFGS(network.parameters(), lr=settings['learning_rate'], max_iter=1,
                                 max_eval=25, tolerance_grad=1e-9, tolerance_change=1e-12,
                                 history_size=settings['history_size'], line_search_fn=settings['line_search'])
    points, sides, labels = (tensors['training'][key] for key in ('points', 'sides', 'joints'))
    masks = {side: sides == sign for side, sign in [('left', -1), ('right', 1)]}
    up = torch.tensor([0., 0., 1.], device='cuda')
    torch.cuda.reset_peak_memory_stats()
    began, evaluations, completed_updates = time.perf_counter(), 0, 0
    curve, candidates, stop_reason = [], [], None
    last_loss = None

    def closure():
        nonlocal evaluations, last_loss
        if evaluations >= settings['maximum_gradient_evaluations']:
            raise TrainingBudgetReached('Declared gradient-evaluation limit reached.')
        if time.perf_counter()-began > budget['max_training_minutes']*60:
            raise TrainingBudgetReached('Declared training-time limit reached.')
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        predicted = network(points, sides)
        loss = .01 * (((predicted-labels)/network.joint_scale)**2).mean()
        geometry_loss = torch.zeros((), device='cuda')
        for side in ('left', 'right'):
            mask = masks[side]
            actual, rotation = geometry[side](predicted[mask])
            geometry_loss = geometry_loss + (((actual-points[mask])/.005)**2).sum()
            geometry_loss = geometry_loss + (((rotation[:, :, 1]-up)/.05)**2).sum()
        loss = loss + geometry_loss/(len(points)*3)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective.')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise RuntimeError('Nonfinite training gradient.')
        last_loss = float(loss.detach())
        if torch.cuda.max_memory_allocated()/1024**3 > budget['max_peak_vram_gib']:
            raise TrainingBudgetReached('Declared peak Torch-memory limit reached.')
        return loss

    try:
        for update in range(1, settings['updates']+1):
            optimizer.step(closure)
            completed_updates = update
            if time.perf_counter()-began > budget['max_training_minutes']*60:
                raise TrainingBudgetReached('Declared training-time limit reached after an update.')
            if torch.cuda.max_memory_allocated()/1024**3 > budget['max_peak_vram_gib']:
                raise TrainingBudgetReached('Declared peak Torch-memory limit reached after an update.')
            if update % 25 == 0:
                torch.cuda.synchronize()
                row = {'update': update, 'gradient_evaluations': evaluations, 'last_closure_loss': last_loss,
                       'wall_seconds': time.perf_counter()-began,
                       'peak_vram_gib': torch.cuda.max_memory_allocated()/1024**3}
                curve.append(row)
                print(json.dumps(row), flush=True)
                require_space(args.output, 16*1024**2)
            if update in settings['candidate_updates']:
                result = score()
                checkpoint = args.output / ('update-%06d.safetensors' % update)
                require_space(checkpoint, 16*1024**2)
                save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(checkpoint))
                candidates.append({'update': update, 'gradient_evaluations': evaluations,
                                   'checkpoint': checkpoint.as_posix(), 'sha256': sha(checkpoint), 'development': result})
                print(json.dumps({'candidate_update': update, 'accepted': result['accepted'],
                                  'position_error_mm': result['position_error_mm']}), flush=True)
    except TrainingBudgetReached as exc:
        stop_reason = str(exc)
    complete = completed_updates == settings['updates'] and stop_reason is None
    selected = min(candidates, key=lambda c: (-c['development']['accepted'], c['development']['position_error_mm']['p95'])) if complete else None
    gate = protocol['kinematic_gate']
    passed = bool(selected and selected['development']['accepted'] >= gate['minimum_accepted']
                  and selected['development']['total'] == gate['total']
                  and selected['development']['position_error_mm']['p95'] <= gate['maximum_p95_position_error_mm'])
    result = {'protocol': args.protocol.as_posix(), 'protocol_sha256': sha(args.protocol),
              'warm_start_sha256': sha(warm_start), 'data_sha256': data_hashes, 'source_sha256': source_hashes,
              'baseline': baseline, 'candidates': candidates, 'completed_updates': completed_updates,
              'gradient_evaluations': evaluations, 'training_completed': complete, 'stop_reason': stop_reason,
              'optimizer_iterations': int(optimizer.state[next(network.parameters())].get('n_iter', 0)),
              'selected_update': selected['update'] if selected else None,
              'selected_checkpoint': selected['checkpoint'] if selected else None,
              'kinematic_gate_passed': passed, 'cuda_geometry_maximum_absolute_error': maximum_geometry_error,
              'curve': curve, 'cuda_device': torch.cuda.get_device_name(),
              'scope': 'Offline refinement on original exposed kinematic development points; no physical success established.'}
    require_space(args.output, 1024**2)
    (args.output / 'training.json').write_bytes((json.dumps(result, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({key: result[key] for key in ('completed_updates', 'gradient_evaluations', 'training_completed',
                                                'stop_reason', 'selected_update', 'kinematic_gate_passed')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-bottle-v4.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
