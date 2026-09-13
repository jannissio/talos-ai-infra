"""Collect offline robot kinematic labels, retaining every solver refusal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.autonomy import ArmIK, PlanningError
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    require_space(args.output, 32*1024**2); args.output.mkdir(parents=True)
    xml, _ = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2; mujoco.mj_forward(model, data)
    solvers = {side: ArmIK(model, data, side) for side in ('left', 'right')}
    bounds = np.asarray(protocol['position_bounds_m'])
    began = time.perf_counter(); summaries = {}
    for split in ('training', 'development'):
        rng = np.random.default_rng(protocol[split+'_rng_seed'])
        records, points, sides, joints = [], [], [], []
        for index in range(protocol[split+'_points']):
            side = 'left' if index % 2 == 0 else 'right'
            point = rng.uniform(bounds[:, 0], bounds[:, 1])
            row = {'index': index, 'side': side, 'point_m': point.tolist(), 'accepted': False}
            try:
                q = solvers[side].solve(point, np.asarray(HOME[:5]))
                points.append(point); sides.append(-1. if side == 'left' else 1.); joints.append(q)
                row['accepted'] = True
            except PlanningError as exc:
                row['reason'] = str(exc)
            records.append(row)
            if index % 256 == 0:
                require_space(args.output, 16*1024**2)
                print(json.dumps({'split': split, 'attempted': index+1, 'accepted': len(points),
                                  'wall_seconds': time.perf_counter()-began}), flush=True)
        require_space(args.output, 16*1024**2)
        path = args.output/(split+'.npz')
        np.savez_compressed(path, points=np.asarray(points, dtype='float32'), sides=np.asarray(sides, dtype='float32'),
                            joints=np.asarray(joints, dtype='float32'))
        (args.output/(split+'-attempts.json')).write_text(json.dumps(records, indent=2)+'\n')
        summaries[split] = {'attempted': len(records), 'accepted': len(points), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'protocol': args.protocol.as_posix(), 'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'splits': summaries, 'wall_seconds': time.perf_counter()-began,
              'scope': 'Kinematic labels only. No physical episode or learned-control success is asserted.'}
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-motor-v1.json'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
