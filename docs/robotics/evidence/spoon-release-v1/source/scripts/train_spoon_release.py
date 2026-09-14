"""Warm-fit one spoon candidate with frozen v6 normalization after the data gate."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from simulation_lab.primitive_policy import PrimitiveNet
from simulation_lab.storage import GIB, require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    require_space(path,4*1024**2)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_bytes((json.dumps(value,indent=2)+'\n').encode())
    pending.replace(path)


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve the existing fit.')
    protocol, info = read(args.protocol), read(args.source/'retrieval.json')
    gate, audit = read(args.source/'input-replay.json'), read(args.source/'audit.json')
    if (not gate['passed'] or not gate['complete'] or not audit['gate_passed'] or len(gate['episodes']) != 39
            or gate['source_sha256'] != sha(args.source/'retrieval.npz')
            or info['release_protocol_sha256'] != sha(args.protocol)):
        raise ValueError('Every modified physical input must pass before training.')
    if not torch.cuda.is_available():
        raise RuntimeError('This declared fit requires the RTX training environment.')
    baseline = ROOT/protocol['baseline_spoon']
    if sha(baseline/'primitive.safetensors') != protocol['baseline_spoon_sha256']:
        raise ValueError('Warm-start checkpoint changed.')
    for entry in read(args.source/'manifest.json')['files']:
        if sha(args.source/entry['file']) != entry['sha256']:
            raise ValueError('Packaged training input changed: '+entry['file'])
    settings = protocol['training']
    torch.set_num_threads(2)
    torch.manual_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    require_space(args.output,32*1024**2)
    args.output.mkdir(parents=True)
    with np.load(args.source/'retrieval.npz',allow_pickle=False) as source:
        data = {name:source[name] for name in ('visual','actions','seconds','bounds')}
    metadata = read(baseline/'primitive.json')
    visual = torch.tensor((data['visual']-metadata['visual_mean'])/metadata['visual_std'],dtype=torch.float32,device='cuda')
    seconds = torch.tensor(data['seconds'],dtype=torch.float32,device='cuda')
    targets = torch.tensor((data['actions']-metadata['action_mean'])/metadata['action_std'],dtype=torch.float32,device='cuda')
    network = PrimitiveNet().cuda().eval()
    network.load_state_dict(load_file(str(baseline/'primitive.safetensors'),device='cuda'))
    optimizer = torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['steps'],eta_min=settings['final_learning_rate'])

    def mse():
        total = 0.
        network.eval()
        with torch.inference_mode():
            for begin in range(0,len(seconds),2048):
                residual = network(visual[begin:begin+2048],seconds[begin:begin+2048])-targets[begin:begin+2048]
                total += float(torch.sum(residual*residual))
        return total/targets.numel()

    report = {'schema':protocol['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sha(args.source/'retrieval.npz'),
              'training_gate_sha256':sha(args.source/'input-replay.json'),'warm_start_sha256':protocol['baseline_spoon_sha256'],
              'script_sha256':sha(Path(__file__)),'runtime_source_sha256':{'simulation_lab/primitive_policy.py':sha(ROOT/'simulation_lab/primitive_policy.py')},
              'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'settings':settings,'normalization':'Exact preserved v6 visual/action normalization; no re-estimation.',
              'environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__,
                             'openvino':version('openvino'),'gpu':torch.cuda.get_device_name()},
              'initial_mse':mse(),'steps':[],'completed_steps':0,'training_completed':False,'selection':None,
              'scope':'One warm-start fit on 39 physically replayed modified inputs; no physical candidate or Intel result yet.'}
    write(args.output/'training.json',report)
    started, stopped = time.perf_counter(), None
    for step in range(1,settings['steps']+1):
        if time.perf_counter()-started > protocol['budget']['maximum_training_minutes']*60:
            stopped = 'Training time budget reached.'
            break
        if torch.cuda.max_memory_reserved()/GIB > protocol['budget']['maximum_peak_torch_reserved_gib']:
            stopped = 'Declared Torch reserved-memory limit reached.'
            break
        network.train()
        indices = torch.randint(len(seconds),(settings['batch_size'],),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((network(visual[indices],seconds[indices])-targets[indices])**2)
        if not torch.isfinite(loss):
            stopped = 'Nonfinite training loss.'
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        report['completed_steps'] = step
        if step == 1 or step%1000 == 0:
            row = {'step':step,'loss':float(loss.detach()),'wall_seconds':time.perf_counter()-started,
                   'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/GIB}
            report['steps'].append(row)
            write(args.output/'training.json',report)
            print(json.dumps(row),flush=True)
    report['wall_seconds'] = time.perf_counter()-started
    report['peak_torch_reserved_gib'] = torch.cuda.max_memory_reserved()/GIB
    report['final_mse'] = mse()
    report['stop_reason'] = stopped
    complete = (report['completed_steps'] == settings['steps'] and stopped is None
                and report['wall_seconds'] <= protocol['budget']['maximum_training_minutes']*60
                and report['peak_torch_reserved_gib'] <= protocol['budget']['maximum_peak_torch_reserved_gib'])
    report['training_completed'] = complete
    used = sum(p.stat().st_size for base in (ROOT/'.run/spoon-release-v1',ROOT/'training/spoon_release_v1',
               ROOT/'models/spoon_release_v1',ROOT/'docs/robotics/evidence/spoon-release-v1') if base.exists() for p in base.rglob('*') if p.is_file())
    if used+16*1024**2 > protocol['budget']['maximum_data_gib_including_packages']*GIB:
        raise ValueError('Candidate export exceeds the cumulative data limit.')
    require_space(args.output,16*1024**2)
    folder = args.output/(f'step-{report["completed_steps"]:06d}' if complete else 'stopped-fit')
    folder.mkdir()
    save_file({name:value.detach().cpu().contiguous() for name,value in network.state_dict().items()},str(folder/'primitive.safetensors'))
    meta = deepcopy(metadata)
    meta.update(source_sha256=report['source_sha256'],action_preprocessing=info['action_preprocessing'],
                training_protocol=args.protocol.relative_to(ROOT).as_posix(),warm_start_sha256=protocol['baseline_spoon_sha256'],
                normalization_contract=report['normalization'],training_completed=complete,
                arguments={'script':Path(__file__).relative_to(ROOT).as_posix(),'source':args.source.relative_to(ROOT).as_posix(),
                           'output':args.output.relative_to(ROOT).as_posix(),**settings})
    write(folder/'primitive.json',meta)
    for name in ('visual.npz','talos_normalization.json'):
        require_space(folder,1024**2)
        (folder/name).write_bytes((baseline/name).read_bytes())
    report['checkpoint'] = {'path':folder.relative_to(ROOT).as_posix(),'sha256':sha(folder/'primitive.safetensors')}
    report['selection'] = report['checkpoint'] if complete else None
    write(args.output/'training.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('steps','settings','runtime_source_sha256','environment')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/spoon-release-v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'training/spoon_release_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'.run/spoon-release-v1/fit')
    run(parser.parse_args())
