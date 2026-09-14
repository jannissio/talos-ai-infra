"""Fit one declared RGB crop refiner; retain both candidates and every score."""
import argparse
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, write, arrays, space
from simulation_lab.rgb_servo_network import BottleKeypointNet, decode_heatmaps
from simulation_lab.bottle_refinement import KeypointCropRefiner, image_crops, decode, reconstruct, refinement_loss


def prepare(p, protocol_path, split):
    root = ROOT/p['raw_root']; folder = root/split; output = root/('prepared-'+split)
    if output.exists():
        raise FileExistsError('Preserve earlier prepared image crops.')
    manifest = read(folder/'manifest.json')
    assert manifest['protocol_sha256'] == sha(protocol_path)
    assert manifest['completed_states'] == p['data'][split+'_states']
    assert sha(folder/'recipes.npz') == manifest['recipes_sha256']
    space(p, output, 1024**3 if split == 'training' else 192*1024**2)
    output.mkdir()
    coarse = BottleKeypointNet().cuda().eval()
    coarse.load_state_dict(load_file(str(ROOT/p['coarse_checkpoint']), device='cuda'))
    rng = np.random.default_rng(p['training']['seed']+1)
    parts = {k: [] for k in ('crops', 'centers', 'labels', 'visible', 'world_points', 'present')}
    began = time.perf_counter()
    for shard in manifest['shards']:
        path = folder/shard['file']; assert sha(path) == shard['sha256']
        with np.load(path, allow_pickle=False) as z:
            images = z['rgb'].reshape(-1, 240, 320, 3)
            truth = z['keypoints_px'].reshape(-1, 2, 2)
            visible = np.repeat(z['visible'].reshape(-1, 1), 2, axis=1)
            parts['world_points'].append(z['world_points'].copy()); parts['present'].append(z['present'].copy())
        for start in range(0, len(images), 12):
            rgb = torch.from_numpy(images[start:start+12].transpose(0, 3, 1, 2).copy()).cuda().float()/255
            with torch.inference_mode():
                output_values = coarse(rgb)
                points = decode_heatmaps(output_values[0])[0]
                if split == 'training':
                    jitter = rng.uniform(-p['data']['training_crop_jitter_px'], p['data']['training_crop_jitter_px'], tuple(points.shape))
                    points = points+torch.as_tensor(jitter, device='cuda', dtype=points.dtype)
                crops = image_crops(rgb, points).half().cpu().numpy()
            centers = points.cpu().numpy()
            labels = truth[start:start+12]-centers+31.5
            valid = visible[start:start+12] & (labels >= 2).all(-1) & (labels < 62).all(-1)
            parts['crops'].append(crops)
            parts['centers'].append(centers.reshape(-1, 2))
            parts['labels'].append(labels.reshape(-1, 2))
            parts['visible'].append(valid.reshape(-1))
        print(f'{split}: prepared {shard["file"]}', flush=True)
    values = {key: np.concatenate(rows) for key, rows in parts.items()}
    space(p, output, sum(v.nbytes for v in values.values())+8*1024**2)
    arrays(output/'crops.npz', **values)
    write(output/'manifest.json', {'protocol_sha256': sha(protocol_path), 'data_manifest_sha256': sha(folder/'manifest.json'),
        'coarse_checkpoint_sha256': sha(ROOT/p['coarse_checkpoint']), 'prepared_arrays_sha256': sha(output/'crops.npz'),
        'trainer_sha256': sha(Path(__file__)), 'states': manifest['states'], 'crop_count': len(values['crops']),
        'calibrations': manifest['calibrations'], 'visible_crops': int(values['visible'].sum()),
        'input_precision': 'Saved FP16 crops promoted to FP32 for the network; coarse detections use FP32 CUDA.',
        'wall_seconds': time.perf_counter()-began})
    del coarse
    torch.cuda.empty_cache()
    return values, manifest['calibrations']


def score(network, data, calibrations, p, device='cuda'):
    predictions = {k: [] for k in ('points', 'peaks', 'variance', 'visibility')}
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(data['crops']), p['training']['batch_size']):
            inputs = torch.from_numpy(data['crops'][start:start+p['training']['batch_size']]).to(device).float()
            outputs = decode(network(inputs))
            for key, value in zip(predictions, outputs):
                predictions[key].append(value.cpu().numpy())
    values = {key: np.concatenate(items) for key, items in predictions.items()}
    values['points'] += data['centers']-31.5
    count = len(data['present'])
    values = {key: value.reshape(count, 3, 2, *value.shape[1:]) for key, value in values.items()}
    rows = []
    for index in range(count):
        observed = reconstruct(values['points'][index], values['peaks'][index], values['variance'][index],
            values['visibility'][index], calibrations, p['inference'])
        present = bool(data['present'][index])
        error = float(np.linalg.norm(np.asarray(observed['grasp_point_m'])-data['world_points'][index, 1])*1000) if present and observed['status'] == 'observed' else None
        rows.append({'index': index, 'scoring_only_present': present, 'observation': observed,
                     'error_mm': error, 'scoring_only_grasp_m': data['world_points'][index, 1].tolist() if present else None})
    present_count = int(data['present'].sum())
    errors = [row['error_mm'] for row in rows if row['error_mm'] is not None]
    false = sum(not row['scoring_only_present'] and row['observation']['status'] == 'observed' for row in rows)
    summary = {'states': count, 'present': present_count, 'accepted_present': len(errors),
        'absent': count-present_count, 'false_accepted_absent': false,
        'accepted_fraction': len(errors)/max(1, present_count),
        'p95_error_mm': float(np.quantile(errors, .95)) if errors else None, 'maximum_error_mm': max(errors) if errors else None}
    gate = p['perception_gate']
    summary['gate_passed'] = bool(errors and summary['accepted_fraction'] >= gate['minimum_present_acceptance_fraction']
        and false <= gate['maximum_false_absent_accepts'] and summary['p95_error_mm'] <= gate['maximum_p95_error_mm']
        and summary['maximum_error_mm'] <= gate['maximum_error_mm'])
    return {'summary': summary, 'rows': rows}


def train(args):
    p = checked_protocol(args.protocol); raw = ROOT/p['raw_root']; output = raw/'fit'
    if output.exists() or (raw/'evaluation').exists():
        raise FileExistsError('Preserve prior fits and keep evaluation unexposed until selection.')
    if not torch.cuda.is_available():
        raise ValueError('The declared training requires CUDA.')
    torch.set_num_threads(4); torch.manual_seed(p['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    space(p, output, 64*1024**2); output.mkdir(parents=True)
    write(output/'fit-freeze.json', {'protocol_sha256': sha(args.protocol), 'trainer_sha256': sha(Path(__file__)),
        'network_sha256': sha(ROOT/'simulation_lab/bottle_refinement.py'),
        'checkpoint_selection': p['training']['selection'], 'data_splits': ['training', 'development']})
    training, _ = prepare(p, args.protocol, 'training')
    development, calibrations = prepare(p, args.protocol, 'development')
    tensors = {key: torch.from_numpy(training[key]).cuda() for key in ('crops', 'labels', 'visible')}
    del training
    network = KeypointCropRefiner().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=p['training']['learning_rate'], weight_decay=p['training']['weight_decay'])
    began = time.perf_counter(); curve, candidates = [], []
    for step in range(1, p['training']['updates']+1):
        if time.perf_counter()-began > p['budget']['maximum_fit_minutes']*60:
            raise TimeoutError('Declared fitting wall budget exceeded.')
        if torch.cuda.max_memory_reserved()/1024**3 > p['budget']['maximum_torch_reserved_gib']:
            raise MemoryError('Declared GPU-memory cap exceeded.')
        fraction = (step-1)/(p['training']['updates']-1)
        lr = p['training']['final_learning_rate']+(p['training']['learning_rate']-p['training']['final_learning_rate'])*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:
            group['lr'] = lr
        index = torch.randint(len(tensors['crops']), (p['training']['batch_size'],), device='cuda')
        network.train(); optimizer.zero_grad(set_to_none=True)
        loss = refinement_loss(network(tensors['crops'][index].float()), tensors['labels'][index], tensors['visible'][index])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite refinement loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5.); optimizer.step()
        if step == 1 or step % 250 == 0:
            torch.cuda.synchronize()
            row = {'step': step, 'loss': float(loss.detach()), 'learning_rate': lr,
                'wall_seconds': time.perf_counter()-began, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3}
            curve.append(row); print(row, flush=True)
        if step in p['training']['checkpoints']:
            folder = output/f'step-{step:06d}'; space(p, folder, 8*1024**2); folder.mkdir()
            checkpoint = folder/'refiner.safetensors'
            assert not checkpoint.exists()
            save_file({key: value.detach().cpu().contiguous() for key, value in network.state_dict().items()}, str(checkpoint))
            result = score(network, development, calibrations, p)
            result.update(protocol_sha256=sha(args.protocol), checkpoint_sha256=sha(checkpoint), step=step, inference_precision='CUDA FP32')
            write(folder/'development.json', result)
            candidates.append({'step': step, 'checkpoint': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint), 'development': result['summary']})
            print({'step': step, 'development': result['summary']}, flush=True)
    qualified = [c for c in candidates if c['development']['gate_passed']]
    selected = min(qualified, key=lambda c: (-c['development']['accepted_present'], c['development']['p95_error_mm'], c['step'])) if qualified else None
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'trainer_sha256': sha(Path(__file__)),
        'curve': curve, 'candidates': candidates, 'selected': selected, 'development_gate_passed': selected is not None,
        'wall_seconds': time.perf_counter()-began, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
        'parameter_count': sum(v.numel() for v in network.parameters()), 'scope': 'Perception prerequisite only; wider physical success still requires its separately frozen integration and held-out trial gates.'}
    write(output/'selection.json', result)
    print({'development_gate_passed': result['development_gate_passed'], 'wall_seconds': result['wall_seconds']}, flush=True)
    return int(not result['development_gate_passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    raise SystemExit(train(parser.parse_args()))
