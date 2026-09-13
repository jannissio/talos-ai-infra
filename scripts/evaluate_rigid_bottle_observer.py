"""Evaluate a declared rigid refinement on already saved RGB predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulation_lab.rgb_servo_geometry import rigid_bottle_keypoints
from simulation_lab.rgb_servo_cameras import project, VIEWS
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.observations.read_text()); manifest = json.loads((args.dataset/'manifest.json').read_text())
    if manifest['split'] != 'development':
        raise ValueError('This selection diagnostic accepts original development data only.')
    require_space(args.output, 4*1024**2); args.output.mkdir(parents=True)
    truth = []
    for shard in manifest['shards']:
        with np.load(args.dataset/shard['file'], allow_pickle=False) as data:
            truth.extend(data['world_points'])
    rows = []
    for record, actual in zip(source['rows'], truth):
        row = {key: record.get(key) for key in ('index', 'status', 'scoring_only_present', 'scoring_only_error_mm')}
        if record['status'] == 'observed':
            selected = [name for name in VIEWS if record['views'][name]['accepted']]
            pixels = np.asarray([record['views'][name]['keypoints_px'] for name in selected])
            matrices = [manifest['calibrations'][name]['projection'] for name in selected]
            points, fit = rigid_bottle_keypoints(record['keypoints_m'], pixels, matrices)
            residual = max(float(np.linalg.norm(project(points, matrix)-xy, axis=1).max()) for matrix, xy in zip(matrices, pixels))
            row.update(fit=fit, rigid_points_m=points.tolist(), residual_px=residual)
            if residual > 1.5 or points[1, 2]-points[0, 2] <= .075:
                row['status'] = 'refused'
            elif row['scoring_only_present']:
                row['rigid_error_mm'] = float(np.linalg.norm(points-actual, axis=1).max()*1000)
        rows.append(row)
    errors = [r['rigid_error_mm'] for r in rows if 'rigid_error_mm' in r]
    result = {'protocol': 'docs/robotics/experiments/rgb-servo-rigid-observer-v1.json',
              'prediction_source': args.observations.as_posix(), 'prediction_source_sha256': hashlib.sha256(args.observations.read_bytes()).hexdigest(),
              'frames': len(rows), 'accepted_present': len(errors), 'false_accepted_absent': sum(r['status'] == 'observed' and not r['scoring_only_present'] for r in rows),
              'refused': sum(r['status'] == 'refused' for r in rows),
              'error_mm': {'median': float(np.median(errors)), 'p95': float(np.quantile(errors, .95)), 'max': max(errors)}, 'rows': rows}
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
