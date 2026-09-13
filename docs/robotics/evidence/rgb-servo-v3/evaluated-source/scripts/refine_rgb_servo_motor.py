"""Refine an existing motor network using offline differentiable gripper geometry."""
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
from simulation_lab.rgb_servo_motor import CartesianMotorNet, RobotGeometry
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from scripts.torch_robot_geometry import TorchRobotGeometry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    protocol = json.loads(args.protocol.read_text())
    settings = protocol['motor_training']
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('The declared refinement requires CUDA.')
    require_space(args.output, 32*1024**2)
    args.output.mkdir(parents=True)
    dataset = Path(settings['data'])
    manifest = json.loads((dataset / 'manifest.json').read_text())
    tensors, input_hashes = {}, {}
    for split in ('training', 'development'):
        path = dataset / (split + '.npz')
        input_hashes[split] = digest(path)
        if input_hashes[split] != manifest['splits'][split]['sha256']:
            raise ValueError('Original kinematic input changed.')
        with np.load(path, allow_pickle=False) as arrays:
            tensors[split] = {k: torch.tensor(arrays[k], device='cuda') for k in arrays.files}
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(settings['rng_seed'])
    model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])
    geometry = RobotGeometry(model)
    differentiable = {side: TorchRobotGeometry(model, side, dtype=torch.float32).cuda() for side in ('left', 'right')}
    network = CartesianMotorNet().cuda()
    warm_start = Path(settings['warm_start'])
    network.load_state_dict(load_file(str(warm_start), device='cuda'))
    development = tensors['development']

    def score():
        with torch.inference_mode():
            outputs = network(development['points'], development['sides']).cpu().numpy()
        rows = []
        for point, side_value, joints in zip(development['points'].cpu().numpy(), development['sides'].cpu().numpy(), outputs):
            side, offset = ('left', 0) if side_value < 0 else ('right', 6)
            actual, rotation = geometry.pose(side, joints)
            position_error = float(np.linalg.norm(actual-point))
            axis_error = float(np.linalg.norm(rotation[:, 1]-[0., 0., 1.]))
            limits = model.actuator_ctrlrange[offset:offset+5]
            accepted = (position_error <= .0015 and axis_error <= .025
                        and np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
            rows.append({'side': side, 'position_error_mm': position_error*1000,
                         'axis_error': axis_error, 'accepted': bool(accepted)})
        errors = [r['position_error_mm'] for r in rows]
        return {'accepted': sum(r['accepted'] for r in rows), 'total': len(rows),
                'position_error_mm': {'p95': float(np.quantile(errors, .95)), 'median': float(np.median(errors)), 'max': max(errors)},
                'rows': rows}

    baseline = score()
    # Float32/CUDA geometry must also match the separate MuJoCo implementation.
    maximum_geometry_error = 0.
    with torch.inference_mode():
        for side, sign in [('left', -1), ('right', 1)]:
            joints = development['joints'][development['sides'] == sign]
            points, rotations = differentiable[side](joints)
            for row, point, rotation in zip(joints.cpu().numpy(), points.cpu().numpy(), rotations.cpu().numpy()):
                expected_point, expected_rotation = geometry.pose(side, row)
                maximum_geometry_error = max(maximum_geometry_error, float(np.max(np.abs(point-expected_point))),
                                             float(np.max(np.abs(rotation-expected_rotation))))
    if maximum_geometry_error > 2e-6:
        raise ValueError('CUDA training geometry differs from MuJoCo.')
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['steps'], eta_min=settings['final_learning_rate'])
    training = tensors['training']
    target_axis = torch.tensor([0., 0., 1.], device='cuda')
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    curve, candidates = [], []
    for step in range(1, settings['steps']+1):
        indices = torch.randint(len(training['points']), (settings['batch_size'],), device='cuda')
        points, sides, labels = (training[k][indices] for k in ('points', 'sides', 'joints'))
        predicted = network(points, sides)
        loss = .01 * (((predicted-labels) / network.joint_scale)**2).mean()
        geometric_loss = torch.zeros((), device='cuda')
        for side, sign in [('left', -1), ('right', 1)]:
            mask = sides == sign
            actual, rotation = differentiable[side](predicted[mask])
            geometric_loss = geometric_loss + (((actual-points[mask])/.005)**2).sum()
            geometric_loss = geometric_loss + (((rotation[:, :, 1]-target_axis)/.05)**2).sum()
        loss = loss + geometric_loss / (len(points)*3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 500 == 0:
            torch.cuda.synchronize()
            wall = time.perf_counter()-began
            peak = torch.cuda.max_memory_allocated()/1024**3
            row = {'step': step, 'loss': float(loss), 'wall_seconds': wall, 'peak_vram_gib': peak}
            curve.append(row)
            print(json.dumps(row), flush=True)
            require_space(args.output, 16*1024**2)
            if wall > protocol['budget']['max_training_minutes']*60 or peak > protocol['budget']['max_peak_vram_gib']:
                raise RuntimeError('Declared training resource limit exceeded.')
        if step in settings['candidate_steps']:
            result = score()
            folder = args.output / f'step-{step:06d}'
            require_space(folder, 16*1024**2)
            folder.mkdir()
            checkpoint = folder / 'model.safetensors'
            save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(checkpoint))
            candidates.append({'step': step, 'checkpoint': checkpoint.as_posix(), 'sha256': digest(checkpoint), 'development': result})
            print(json.dumps({'candidate_step': step, 'accepted': result['accepted'], 'position_error_mm': result['position_error_mm']}), flush=True)
    selected = min(candidates, key=lambda c: (-c['development']['accepted'], c['development']['position_error_mm']['p95']))
    gate = selected['development']['accepted'] >= 360 and selected['development']['position_error_mm']['p95'] <= .5
    result = {'protocol': args.protocol.as_posix(), 'protocol_sha256': digest(args.protocol),
              'warm_start_sha256': digest(warm_start), 'data_sha256': input_hashes,
              'source_sha256': {str(p): digest(p) for p in (Path(__file__).relative_to(Path.cwd()), Path('scripts/torch_robot_geometry.py'))},
              'baseline': baseline, 'candidates': candidates, 'selected_step': selected['step'],
              'selected_checkpoint': selected['checkpoint'], 'kinematic_gate_passed': gate,
              'cuda_geometry_maximum_absolute_error': maximum_geometry_error,
              'steps': settings['steps'], 'curve': curve, 'cuda_device': torch.cuda.get_device_name(),
              'scope': 'Offline kinematic refinement only. Original development points are exposed; no physical success is established.'}
    (args.output / 'training.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'selected_step': selected['step'], 'kinematic_gate_passed': gate,
                      'accepted': selected['development']['accepted'], 'total': selected['development']['total'],
                      'position_error_mm': selected['development']['position_error_mm']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-bottle-v3.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
