"""Run exactly the preregistered compact mug-pose fit and development selection."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from simulation_lab.mug_visual_observer import MugShapePoseNet
from scripts.mug_observer_experiment import load_protocol, read, score, sha, space, write_json


def dataset(folder, p, protocol_path):
    manifest = read(folder/'manifest.json')
    if manifest['protocol_sha256'] != sha(protocol_path) or sha(folder/'data.npz') != manifest['data_sha256']:
        raise ValueError('A declared dataset changed.')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A dataset dependency changed: '+name)
    with np.load(folder/'data.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in ('features', 'labels_m', 'present', 'usable_views')}
    if len(arrays['features']) != p['data'][manifest['split']+'_states']:
        raise ValueError('An incomplete split cannot be fitted or scored.')
    return arrays, manifest


def train(args):
    p = load_protocol(args.protocol)
    raw = ROOT/p['raw_root']
    output = raw/'fit'
    if output.exists() or (raw/'evaluation').exists():
        raise FileExistsError('No overwrite, repeat fit or evaluation-before-selection is allowed.')
    if not torch.cuda.is_available():
        raise RuntimeError('The declared fit requires CUDA.')
    torch.set_num_threads(2)
    torch.manual_seed(p['fit']['rng_seed'])
    np.random.seed(p['fit']['rng_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    training, train_manifest = dataset(raw/'training', p, args.protocol)
    development, dev_manifest = dataset(raw/'development', p, args.protocol)
    preflight = space(p, output, 16*1024**2)
    output.mkdir(parents=True)
    model = MugShapePoseNet().cuda()
    valid = training['present'] & (training['usable_views'] >= 2)
    if not valid.any():
        raise ValueError('No image-accepted present training examples.')
    origin, scale = np.asarray(p['observer']['target_origin_m']), p['observer']['target_scale_m']
    features = torch.tensor(training['features'][valid], device='cuda')
    labels = torch.tensor((training['labels_m'][valid]-origin)/scale, dtype=torch.float32, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['fit']['learning_rate'], weight_decay=p['fit']['weight_decay'])
    started, curve, checkpoints = time.perf_counter(), [], []
    for update in range(1, p['fit']['updates']+1):
        if time.perf_counter()-started > p['budget']['maximum_fit_minutes']*60:
            raise TimeoutError('The one declared fit exhausted its time budget; preserve partial artifacts.')
        fraction = (update-1)/(p['fit']['updates']-1)
        lr = p['fit']['final_learning_rate']+(p['fit']['learning_rate']-p['fit']['final_learning_rate'])*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:
            group['lr'] = lr
        ids = torch.randint(len(features), (p['fit']['batch_size'],), device='cuda')
        prediction = model(features[ids])
        loss = torch.mean((prediction-labels[ids])**2)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; preserve the failed fit.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update == 1 or update % 1000 == 0:
            torch.cuda.synchronize()
            reserved = torch.cuda.max_memory_reserved()/1024**3
            if reserved > p['budget']['maximum_torch_reserved_gib']:
                raise MemoryError('Declared GPU memory limit exceeded.')
            space(p, output, 8*1024**2)
            entry = {'update': update, 'batch_mse': float(loss.detach()), 'learning_rate': lr,
                     'wall_seconds': time.perf_counter()-started, 'peak_torch_reserved_gib': reserved}
            curve.append(entry)
            print(entry, flush=True)
        if update in p['fit']['checkpoints']:
            path = output/f'step-{update:06d}.safetensors'
            space(p, path, 4*1024**2)
            save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(path))
            cpu = MugShapePoseNet().eval()
            cpu.load_state_dict(load_file(str(path)))
            with torch.inference_mode():
                prediction_m = cpu(torch.from_numpy(development['features'])).numpy().astype(float)*scale+origin
            result = score(p, prediction_m, development['labels_m'], development['present'], development['usable_views'])
            result.update({'update': update, 'checkpoint_sha256': sha(path), 'development_data_sha256': dev_manifest['data_sha256'],
                           'protocol_sha256': sha(args.protocol), 'scoring_device': 'PyTorch CPU FP32'})
            write_json(output/f'step-{update:06d}-development.json', result)
            checkpoints.append({'update': update, 'checkpoint': path.relative_to(ROOT).as_posix(),
                                'checkpoint_sha256': sha(path), 'development': result['summary']})
            print({'checkpoint': update, 'development': result['summary']}, flush=True)
    torch.cuda.synchronize()
    passing = [r for r in checkpoints if r['development']['gate_passed']]
    selected = min(passing, key=lambda r: (r['development']['p95_3d_error_mm'], r['development']['maximum_3d_error_mm'], r['update'])) if passing else None
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in [Path(__file__), ROOT/'simulation_lab/mug_visual_observer.py', ROOT/'scripts/mug_observer_experiment.py']}
    report = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'source_sha256': source_hashes,
        'parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'training_data_sha256': train_manifest['data_sha256'], 'development_data_sha256': dev_manifest['data_sha256'],
        'updates': p['fit']['updates'], 'training_examples_used': int(valid.sum()), 'training_examples_retained': len(valid),
        'gpu_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'peak_torch_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
        'wall_seconds': time.perf_counter()-started, 'preflight': preflight, 'curve': curve, 'checkpoints': checkpoints,
        'development_gate_passed': bool(selected), 'scope': 'One synthetic RGB geometry-to-pose fit; no learned physical trial or promotion.'}
    write_json(output/'training.json', report)
    selection = {'protocol_sha256': sha(args.protocol), 'training_report_sha256': sha(output/'training.json'),
        'development_gate_passed': bool(selected), 'checkpoint': selected['checkpoint'] if selected else None,
        'checkpoint_sha256': selected['checkpoint_sha256'] if selected else None,
        'selected_update': selected['update'] if selected else None,
        'next_step': 'Export parity then once-only fresh synthetic perception' if selected else 'Stop: no export, fresh data or motor integration under this protocol.'}
    write_json(output/'selection.json', selection)
    space(p, output, 0)
    print({'selection': selection, 'wall_seconds': report['wall_seconds']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-observer-v1.json')
    train(parser.parse_args())
