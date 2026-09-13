"""Bounded observer adaptation; strict deployed geometry decides selection."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from scripts.train_rgb_servo_observer import load_data
from simulation_lab.rgb_bottle_observer import RgbBottleObserver
from simulation_lab.rgb_servo_network import BottleKeypointNet,observation_loss
from simulation_lab.storage import GIB,require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    require_space(path,4*1024**2)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_bytes((json.dumps(value,indent=2)+'\n').encode())
    pending.replace(path)


def score_checkpoint(checkpoint,datasets,gate):
    observer=RgbBottleObserver(checkpoint,device='cuda',minimum_views=2,rigid_geometry=True)
    results={}
    for configuration,(manifest,data) in datasets.items():
        rows=[]
        for index in range(data['states']):
            pixels=data['rgb'][index*3:(index+1)*3].permute(0,2,3,1).numpy()
            result=observer.observe(dict(zip(manifest['views'],pixels)),manifest['calibrations'])
            present=bool(data['present'][index])
            error=float(np.linalg.norm(np.asarray(result['keypoints_m'])-data['world_points'][index],axis=1).max()*1000) if present and result['status']=='observed' else None
            rows.append({'index':index,'observation':result,'scoring_only_present':present,'scoring_only_error_mm':error})
        present=sum(r['scoring_only_present'] for r in rows)
        accepted=sum(r['scoring_only_present'] and r['observation']['status']=='observed' for r in rows)
        absent=len(rows)-present
        false=sum(not r['scoring_only_present'] and r['observation']['status']=='observed' for r in rows)
        errors=[r['scoring_only_error_mm'] for r in rows if r['scoring_only_error_mm'] is not None]
        p95,maximum=(float(np.quantile(errors,.95)),max(errors)) if errors else (None,None)
        passed=bool(present and absent and errors and accepted/present>=gate['minimum_present_acceptance_fraction']
                    and false<=gate['maximum_false_accepted_absent'] and p95<=gate['maximum_accepted_p95_error_mm']
                    and maximum<=gate['maximum_accepted_error_mm'])
        results[configuration]={'states':len(rows),'present':present,'accepted_present':accepted,'absent':absent,
                                'false_accepted_absent':false,'accepted_error_p95_mm':p95,'accepted_error_max_mm':maximum,
                                'gate_passed':passed,'rows':rows}
    return results


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve prior fits; choose an unused output directory.')
    protocol=json.loads(args.protocol.read_text())
    if args.data!=ROOT/'.run/rgb-servo-observer-camera-v1' or not args.output.is_relative_to(args.data):
        raise ValueError('Keep the fit inside the declared experiment data directory.')
    if not torch.cuda.is_available():
        raise RuntimeError('This declared experiment requires CUDA.')
    if protocol['schema']!='talos.rgb-servo-observer-camera-training.v1':
        raise ValueError('Unexpected training protocol.')
    settings=protocol['training']
    warm=ROOT/protocol['warm_start']
    if sha(warm)!=protocol['warm_start_sha256']:
        raise ValueError('The preserved warm-start weights changed.')
    if sha(ROOT/protocol['motor']/'motor.safetensors')!=protocol['motor_sha256']:
        raise ValueError('The fixed motor checkpoint changed.')
    require_space(args.output,64*1024**2)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    manifests,parts,development={},{},{}
    for split in ('train','development'):
        pair=[]
        for configuration in protocol['camera_configurations']:
            folder=args.data/(split+'-'+configuration)
            manifest,data=load_data(folder)
            if (manifest['split']!=split or manifest['camera_configuration']!=configuration
                    or manifest['protocol_sha256']!=sha(args.protocol) or data['states']!=protocol[split+'_states'] or data['views']!=3):
                raise ValueError('Dataset split, configuration or protocol mismatch.')
            if manifest['generator_sha256']!=sha(ROOT/'scripts/collect_rgb_servo_camera_observer.py'):
                raise ValueError('Dataset generator changed after collection.')
            if manifest['sampled_states_sha256']!=sha(folder/'sampled-states.npz'):
                raise ValueError('Sampled states changed.')
            with np.load(folder/'sampled-states.npz',allow_pickle=False) as states:
                pair.append({name:states[name] for name in states.files})
            manifests[split+'-'+configuration]={'path':(folder/'manifest.json').relative_to(ROOT).as_posix(),
                'sha256':sha(folder/'manifest.json'),'sampled_states_sha256':manifest['sampled_states_sha256'],
                'shards':manifest['shards']}
            if split=='train':
                parts[configuration]=data
            else:
                development[configuration]=(manifest,data)
        if pair[0].keys()!=pair[1].keys() or any(not np.array_equal(pair[0][name],pair[1][name]) for name in pair[0]):
            raise ValueError('Paired camera datasets used different sampled states.')
    device=torch.device('cuda')
    train={key:torch.cat([parts[name][key] for name in protocol['camera_configurations']]).to(device)
           for key in ('rgb','mask','keypoints','visible')}
    del parts
    network=BottleKeypointNet().to(device)
    network.load_state_dict(load_file(str(warm),device='cuda'))
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['steps'],eta_min=settings['final_learning_rate'])
    names=['scripts/train_rgb_servo_camera_observer.py','scripts/collect_rgb_servo_camera_observer.py',
           'scripts/train_rgb_servo_observer.py','simulation_lab/rgb_bottle_observer.py','simulation_lab/rgb_servo_network.py',
           'simulation_lab/rgb_servo_cameras.py','simulation_lab/rgb_servo_geometry.py','simulation_lab/storage.py']
    report={'schema':protocol['schema'],'protocol':args.protocol.relative_to(ROOT).as_posix(),'protocol_sha256':sha(args.protocol),
            'source_sha256':{name:sha(ROOT/name) for name in names},'dataset_manifests':manifests,
            'warm_start_sha256':sha(warm),'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'settings':settings,'training_views':len(train['rgb']),'cuda_device':torch.cuda.get_device_name(),
            'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'torch':str(torch.__version__),'openvino':version('openvino')},
            'steps':[],'candidates':[],'completed_steps':0,'training_completed':False,
            'selection':None,'physical_trials':0,'fresh_perception_states_exposed':0,
            'scope':'New observer adaptation only; camera angles, decoder, rigid geometry and motor stay fixed. No physical or deployment claim.'}
    began=time.perf_counter()
    report['baseline_development']=score_checkpoint(warm,development,protocol['perception_gate'])
    write(args.output/'training.json',report)
    stop_reason=None
    for step in range(1,settings['steps']+1):
        if time.perf_counter()-began>protocol['budget']['maximum_training_minutes']*60:
            stop_reason='Declared training time budget reached.'
            break
        if torch.cuda.max_memory_reserved()/GIB>protocol['budget']['maximum_peak_vram_gib']:
            stop_reason='Declared Torch reserved-memory limit reached.'
            break
        network.train()
        indices=torch.randint(len(train['rgb']),(settings['batch_size'],),device=device)
        rgb=train['rgb'][indices].float()/255
        optimizer.zero_grad(set_to_none=True)
        loss,metrics=observation_loss(network(rgb),train['keypoints'][indices],train['mask'][indices],train['visible'][indices])
        if not torch.isfinite(loss):
            stop_reason='Nonfinite loss; no selection.'
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
        optimizer.step(); schedule.step()
        report['completed_steps']=step
        if step==1 or step%100==0:
            row={'step':step,'loss':float(loss.detach()),**metrics,'wall_seconds':time.perf_counter()-began,
                 'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/GIB,
                 'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/GIB}
            report['steps'].append(row)
            print(json.dumps(row),flush=True)
        if step in settings['candidate_steps']:
            require_space(args.output,32*1024**2)
            folders=[args.data,ROOT/'docs/robotics/evidence/rgb-servo-observer-camera-v1',ROOT/'models/bottle_servo_camera_v1']
            used=sum(p.stat().st_size for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file())
            used+=sum(p.stat().st_size for p in (ROOT/'.run/final-goal').glob('rgb-servo-observer-camera-v1*') if p.is_file())
            if used+32*1024**2>protocol['budget']['maximum_new_data_gib']*GIB:
                stop_reason='Declared cumulative data limit reached.'
                break
            checkpoint=args.output/f'step-{step:06d}.safetensors'
            save_file({name:value.detach().cpu().contiguous() for name,value in network.state_dict().items()},str(checkpoint))
            measured=score_checkpoint(checkpoint,development,protocol['perception_gate'])
            report['candidates'].append({'step':step,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                                         'sha256':sha(checkpoint),'development':measured})
            write(args.output/'training.json',report)
            print(json.dumps({'candidate':step,'development':{name:{k:v for k,v in result.items() if k!='rows'} for name,result in measured.items()}}),flush=True)
    report['wall_seconds']=time.perf_counter()-began
    report['peak_torch_allocated_gib']=torch.cuda.max_memory_allocated()/GIB
    report['peak_torch_reserved_gib']=torch.cuda.max_memory_reserved()/GIB
    complete=(report['completed_steps']==settings['steps'] and stop_reason is None
              and report['wall_seconds']<=protocol['budget']['maximum_training_minutes']*60
              and report['peak_torch_reserved_gib']<=protocol['budget']['maximum_peak_vram_gib'])
    report['training_completed']=complete
    report['stop_reason']=stop_reason or ('Post-fit resource limit exceeded.' if not complete else None)
    configuration=protocol['deployment_configuration']
    eligible=[row for row in report['candidates'] if row['development'][configuration]['gate_passed']] if complete else []
    if eligible:
        selected=min(eligible,key=lambda row:(-row['development'][configuration]['accepted_present'],row['development'][configuration]['accepted_error_p95_mm']))
        report['selection']={'step':selected['step'],'checkpoint':selected['checkpoint'],'checkpoint_sha256':selected['sha256'],
                             'protocol_sha256':sha(args.protocol),'development_gate_passed':True,
                             'camera_configuration':configuration,'source_sha256':report['source_sha256']}
        write(args.output/'selected.json',report['selection'])
    write(args.output/'training.json',report)
    print(json.dumps({key:report[key] for key in ('training_completed','completed_steps','stop_reason','wall_seconds',
                                                'peak_torch_allocated_gib','peak_torch_reserved_gib','selection')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-observer-camera-training-v1.json'))
    parser.add_argument('--data',type=Path,default=Path('.run/rgb-servo-observer-camera-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('protocol','data','output'):
        setattr(args,name,getattr(args,name).resolve())
    run(args)
