"""Offline prediction error with zero-latent inference, not a physical success score."""
import argparse,json,sys,gzip,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulation_lab.act_learning import load_policy,normalize,denormalize,episode_indices

def score(a):
    torch.set_num_threads(4)
    policy,stats=load_policy(a.checkpoint,a.device)
    root=Path(a.dataset)
    ds=LeRobotDataset('talos/local',root=root,delta_timestamps={'action':[i/20 for i in range(20)]},video_backend='pyav')
    lineage=json.loads((root/'talos_lineage.json').read_text())
    eligible=episode_indices(lineage,a.episode) if a.episode else list(range(len(ds)))
    indices=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(a.samples,len(eligible)),dtype=int))]
    phases=[];phase_scores={};cursor=0
    if a.phase_source:
        for row in lineage['episodes']:
            folder=Path(a.phase_source)/row['source']
            if hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=row['source_manifest_sha256']:
                raise ValueError('Phase source manifest differs from dataset lineage.')
            with gzip.open(folder/'actions.jsonl.gz','rt') as f:actions=[json.loads(s)['stage'] for s in f]
            with gzip.open(folder/'observations.jsonl.gz','rt') as f:obs=[json.loads(s) for s in f]
            labels=[actions[o['action_index']] for o in obs if not o['terminal']]
            if len(labels)!=row['frames']:raise ValueError('Phase/frame count mismatch.')
            phases.extend(labels)
    total=np.zeros(12);baseline_total=np.zeros(12);first_total=np.zeros(12);first_baseline=np.zeros(12);valid=first_valid=0
    for batch in DataLoader(Subset(ds,indices),batch_size=4,num_workers=0):
        inputs={k:v for k,v in batch.items() if k.startswith('observation.')}
        if a.blank_images:
            inputs={k:torch.zeros_like(v) if k.startswith('observation.images.') else v for k,v in inputs.items()}
        with torch.inference_mode():prediction=denormalize(policy.predict_action_chunk(normalize(inputs,stats,a.device)),stats,inputs['observation.state']).cpu()
        mask=(~batch['action_is_pad']).unsqueeze(-1)
        error=(prediction-batch['action']).abs()*mask
        baseline=(batch['observation.state'][:,:12,None].transpose(1,2)-batch['action']).abs()*mask
        total+=error.sum((0,1)).numpy();baseline_total+=baseline.sum((0,1)).numpy()
        first_total+=error[:,:5].sum((0,1)).numpy();first_baseline+=baseline[:,:5].sum((0,1)).numpy()
        first_valid+=int(mask[:,:5].sum())
        valid+=int(mask.sum())
        if phases:
            for j in range(len(error)):
                phase=phases[indices[cursor+j]]
                entry=phase_scores.setdefault(phase,{'observations':0,'valid':0,'first_valid':0,'joint_sum':np.zeros(12),'first_sum':np.zeros(12)})
                entry['observations']+=1;entry['valid']+=int(mask[j].sum());entry['first_valid']+=int(mask[j,:5].sum())
                entry['joint_sum']+=error[j].sum(0).numpy();entry['first_sum']+=error[j,:5].sum(0).numpy()
            cursor+=len(error)
    error=total/valid
    result={'metric':'offline zero-latent action prediction MAE, not task success',
            'checkpoint':a.checkpoint,'dataset':a.dataset,'sampled_observations':len(indices),
            'valid_future_targets':valid,'mae_rad_per_joint':error.tolist(),'left_arm_mae_rad':float(error[:5].mean()),
            'left_gripper_mae_rad':float(error[5]),'right_parked_mae_rad':float(error[6:].mean()),
            'first_five_left_arm_mae_rad':float((first_total/first_valid)[:5].mean()),
            'current_joint_position_reference_mae_rad':float((baseline_total/valid)[:5].mean()),
            'first_five_current_joint_reference_mae_rad':float((first_baseline/first_valid)[:5].mean()),
            'blank_images':a.blank_images,'selected_episode':a.episode,
            'phase_scores':{k:{'sampled_observations':v['observations'],
                'full_chunk_mae_rad_per_joint':(v['joint_sum']/v['valid']).tolist(),
                'first_five_mae_rad_per_joint':(v['first_sum']/v['first_valid']).tolist()} for k,v in phase_scores.items()},
            'phase_labels_policy_input':False,
            'dataset_lineage':lineage,
            'normalization':'Checkpoint training statistics; evaluation statistics unused.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=128);p.add_argument('--device',default='cuda')
    p.add_argument('--blank-images',action='store_true')
    p.add_argument('--episode');p.add_argument('--phase-source')
    a=p.parse_args()
    if a.samples<=0:p.error('--samples must be positive')
    score(a)
