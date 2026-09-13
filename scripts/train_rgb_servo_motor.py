"""Fit and independently score the bounded neural Cartesian motor map."""
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
from safetensors.torch import save_file
from simulation_lab.rgb_servo_motor import CartesianMotorNet, RobotGeometry
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded training run requires the declared CUDA GPU.')
    protocol = json.loads(args.protocol.read_text()); manifest = json.loads((args.dataset/'manifest.json').read_text())
    require_space(args.output, 32*1024**2); args.output.mkdir(parents=True)
    batches = {}
    for split in ('training', 'development'):
        path = args.dataset/(split+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['splits'][split]['sha256']:
            raise ValueError('Kinematic dataset checksum mismatch.')
        with np.load(path, allow_pickle=False) as data:
            batches[split] = {k: torch.tensor(data[k], device='cuda') for k in data.files}
    torch.set_num_threads(2); torch.manual_seed(2026095503); torch.set_float32_matmul_precision('high')
    network = CartesianMotorNet().cuda(); train = batches['training']; dev = batches['development']
    with torch.no_grad():
        network.joint_center.copy_(train['joints'].mean(0)); network.joint_scale.copy_(train['joints'].std(0).clamp_min(.1))
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['steps'], eta_min=.00002)
    began = time.perf_counter(); curve = []; torch.cuda.reset_peak_memory_stats()
    for step in range(1, protocol['steps']+1):
        ix = torch.randint(len(train['points']), (protocol['batch_size'],), device='cuda')
        prediction = network(train['points'][ix], train['sides'][ix])
        loss = (((prediction-train['joints'][ix])/network.joint_scale)**2).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
        if step % 1000 == 0:
            with torch.inference_mode():
                error = network(dev['points'], dev['sides'])-dev['joints']
                row = {'step': step, 'training_loss': float(loss), 'development_joint_mae_rad': float(error.abs().mean()),
                       'wall_seconds': time.perf_counter()-began}
            curve.append(row); print(json.dumps(row), flush=True)
            if row['wall_seconds'] > protocol['budget']['max_training_minutes']*60:
                raise RuntimeError('Declared training time budget exceeded.')
            require_space(args.output, 16*1024**2)
    torch.cuda.synchronize(); wall = time.perf_counter()-began
    if torch.cuda.max_memory_allocated()/1024**3 > protocol['budget']['max_peak_vram_gib']:
        raise RuntimeError('Declared GPU memory budget exceeded.')
    network.eval()
    with torch.inference_mode():
        predicted = network(dev['points'], dev['sides']).cpu().numpy()
    xml, _ = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml); geometry = RobotGeometry(model)
    rows = []
    for i, (point, side_number, joints) in enumerate(zip(dev['points'].cpu().numpy(), dev['sides'].cpu().numpy(), predicted)):
        side = 'left' if side_number < 0 else 'right'; offset = 0 if side == 'left' else 6
        actual, rotation = geometry.pose(side, joints)
        error = float(np.linalg.norm(actual-point)); axis = float(np.linalg.norm(rotation[:, 1]-[0., 0., 1.]))
        limits = model.actuator_ctrlrange[offset:offset+5]
        accepted = bool(error <= .0015 and axis <= .025 and np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
        rows.append({'index': i, 'side': side, 'position_error_mm': error*1000, 'axis_error': axis, 'accepted': accepted})
    require_space(args.output, 16*1024**2)
    save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}, str(args.output/'model.safetensors'))
    errors = [r['position_error_mm'] for r in rows]
    result = {'protocol': args.protocol.as_posix(), 'dataset': args.dataset.as_posix(), 'curve': curve,
              'training_seconds': wall, 'peak_vram_gib': torch.cuda.max_memory_allocated()/1024**3,
              'cuda_device': torch.cuda.get_device_name(), 'steps': protocol['steps'],
              'checkpoint_sha256': hashlib.sha256((args.output/'model.safetensors').read_bytes()).hexdigest(),
              'development': {'attempted_solver_points': manifest['splits']['development']['attempted'],
                              'solver_accepted': len(rows), 'neural_guard_accepted': sum(r['accepted'] for r in rows),
                              'position_error_mm': {'median': float(np.median(errors)), 'p95': float(np.quantile(errors, .95)), 'max': max(errors)},
                              'rows': rows},
              'scope': 'Kinematic development accuracy only; physical control and corrective benefit are not established.'}
    (args.output/'training.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**{k: v for k, v in result.items() if k not in ('curve', 'development')},
                      'development': {k: v for k, v in result['development'].items() if k != 'rows'}}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-motor-v1.json'))
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
