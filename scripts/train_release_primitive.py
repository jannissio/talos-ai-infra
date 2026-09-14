"""Fit one declared release primitive after its complete physical input gate."""
import argparse
from copy import deepcopy
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
from safetensors.torch import load_file,save_file
from simulation_lab.primitive_policy import PrimitiveNet
from scripts.release_experiment import GIB,read,repository_path,sha,space,write_json


def run(args):
    p = read(args.protocol)
    source,baseline = repository_path(p['training_package']),repository_path(p['baseline'])
    output = args.output or repository_path(p['raw_root'])/'fit'
    if output.exists():raise FileExistsError('Preserve the existing fit.')
    info,gate,audit = (read(source/name) for name in ('retrieval.json','input-replay.json','audit.json'))
    if (not gate['complete'] or not gate['passed'] or not audit['gate_passed']
            or len(gate['episodes'])!=p['budget']['modified_training_episodes']
            or info['release_protocol_sha256']!=sha(args.protocol) or gate['source_sha256']!=sha(source/'retrieval.npz')
            or sha(baseline/'primitive.safetensors')!=p['baseline_sha256']):
        raise ValueError('The full revised-input physical gate and original warm start are required.')
    for entry in read(source/'manifest.json')['files']:
        if sha(source/entry['file'])!=entry['sha256']:raise ValueError('A packaged training input changed.')
    if not torch.cuda.is_available():raise RuntimeError('This declared fit requires the CUDA training environment.')
    settings = p['training']
    torch.set_num_threads(2);torch.manual_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    space(p,output,32*1024**2);output.mkdir(parents=True)
    with np.load(source/'retrieval.npz',allow_pickle=False) as saved:
        values={name:saved[name] for name in ('visual','actions','seconds','bounds')}
    metadata=read(baseline/'primitive.json')
    visual=torch.tensor((values['visual']-metadata['visual_mean'])/metadata['visual_std'],dtype=torch.float32,device='cuda')
    seconds=torch.tensor(values['seconds'],dtype=torch.float32,device='cuda')
    targets=torch.tensor((values['actions']-metadata['action_mean'])/metadata['action_std'],dtype=torch.float32,device='cuda')
    endpoint_weights=np.empty(len(seconds),dtype=np.float32)
    cursor,stages=0,[]
    for name,duration in info['durations_s'].items():
        count=round(duration*20)
        stages += [name]*count;cursor+=count
    one_episode=np.asarray([settings['stage_loss_weights'].get(name,settings['other_stage_loss_weight']) for name in stages],dtype=np.float32)
    for begin,end in values['bounds']:
        if end-begin!=cursor:raise ValueError('Training stages are not aligned.')
        endpoint_weights[begin:end]=one_episode
    if not np.isfinite(endpoint_weights).all() or np.min(endpoint_weights)<=0:raise ValueError('Training weights must be positive and finite.')
    weights=torch.tensor(endpoint_weights,device='cuda')
    network=PrimitiveNet().cuda().eval()
    network.load_state_dict(load_file(str(baseline/'primitive.safetensors'),device='cuda'))
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['steps'],eta_min=settings['final_learning_rate'])

    def mse():
        total,weighted,total_weight=0.,0.,0.
        network.eval()
        with torch.inference_mode():
            for begin in range(0,len(seconds),2048):
                losses=torch.mean((network(visual[begin:begin+2048],seconds[begin:begin+2048])-targets[begin:begin+2048])**2,dim=1)
                total += float(losses.sum())
                weighted += float((losses*weights[begin:begin+2048]).sum())
                total_weight += float(weights[begin:begin+2048].sum())
        return {'unweighted':total/len(seconds),'weighted':weighted/total_weight}

    report={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sha(source/'retrieval.npz'),
            'training_gate_sha256':sha(source/'input-replay.json'),'warm_start_sha256':p['baseline_sha256'],
            'script_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'scripts/release_experiment.py'),
            'runtime_source_sha256':{'simulation_lab/primitive_policy.py':sha(ROOT/'simulation_lab/primitive_policy.py')},
            'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'settings':settings,'normalization':'Exact preserved original visual/action normalization; no re-estimation.',
            'environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__,
                           'openvino':version('openvino'),'gpu':torch.cuda.get_device_name()},
            'initial_mse':mse(),'endpoints':len(seconds),'stage_endpoint_counts':{name:stages.count(name)*len(values['bounds']) for name in info['durations_s']},
            'steps':[],'completed_steps':0,'training_completed':False,'selection':None,
            'scope':'One declared weighted warm fit after physical input replay; no candidate physical or Intel result yet.'}
    write_json(output/'training.json',report)
    started,stopped=time.perf_counter(),None
    for step in range(1,settings['steps']+1):
        if time.perf_counter()-started>p['budget']['maximum_training_minutes']*60:
            stopped='Training time budget reached.';break
        if torch.cuda.max_memory_reserved()/GIB>p['budget']['maximum_peak_torch_reserved_gib']:
            stopped='Torch reserved-memory budget reached.';break
        network.train()
        indices=torch.randint(len(seconds),(settings['batch_size'],),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        losses=torch.mean((network(visual[indices],seconds[indices])-targets[indices])**2,dim=1)
        loss=(losses*weights[indices]).sum()/weights[indices].sum()
        if not torch.isfinite(loss):stopped='Nonfinite training loss.';break
        loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();schedule.step();report['completed_steps']=step
        if step==1 or step%1000==0:
            row={'step':step,'loss':float(loss.detach()),'wall_seconds':time.perf_counter()-started,
                 'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/GIB}
            report['steps'].append(row);write_json(output/'training.json',report,replace=True)
            print(json.dumps(row),flush=True)
    report.update(wall_seconds=time.perf_counter()-started,peak_torch_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                  final_mse=mse(),stop_reason=stopped)
    complete=(report['completed_steps']==settings['steps'] and stopped is None
              and report['wall_seconds']<=p['budget']['maximum_training_minutes']*60
              and report['peak_torch_reserved_gib']<=p['budget']['maximum_peak_torch_reserved_gib'])
    report['training_completed']=complete
    space(p,output,16*1024**2)
    folder=output/(f'step-{report["completed_steps"]:06d}' if complete else 'stopped-fit')
    folder.mkdir()
    save_file({name:value.detach().cpu().contiguous() for name,value in network.state_dict().items()},str(folder/'primitive.safetensors'))
    meta=deepcopy(metadata)
    meta.update(source_sha256=report['source_sha256'],action_preprocessing=info['action_preprocessing'],
                training_protocol=args.protocol.relative_to(ROOT).as_posix(),warm_start_sha256=p['baseline_sha256'],
                normalization_contract=report['normalization'],training_completed=complete,
                arguments={'script':Path(__file__).relative_to(ROOT).as_posix(),'source':source.relative_to(ROOT).as_posix(),
                           'output':output.resolve().relative_to(ROOT).as_posix(),**settings})
    write_json(folder/'primitive.json',meta)
    for name in ('visual.npz','talos_normalization.json'):
        space(p,folder,1024**2)
        with (folder/name).open('xb') as stream:stream.write((baseline/name).read_bytes())
    report['checkpoint']={'path':folder.resolve().relative_to(ROOT).as_posix(),'sha256':sha(folder/'primitive.safetensors')}
    report['selection']=report['checkpoint'] if complete else None
    write_json(output/'training.json',report,replace=True)
    print(json.dumps({key:value for key,value in report.items() if key not in ('steps','settings','runtime_source_sha256','environment','stage_endpoint_counts')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();args.protocol=args.protocol.resolve()
    run(args)
