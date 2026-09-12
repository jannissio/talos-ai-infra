"""Bounded ACT learning experiment; local artifacts only, no Hub or tracking upload."""
import argparse,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simulation_lab.act_learning import ACTPolicy,config,normalization,normalize,load_policy,episode_indices,learning_loss
from simulation_lab.storage import require_space,GIB

def train(a):
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device=a.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    root=Path(a.dataset).resolve();out=Path(a.output).resolve()
    if out.exists() and not a.resume:raise FileExistsError('Use a fresh output or --resume.')
    planned_saves=(a.steps+a.save_every-1)//a.save_every
    storage=require_space(out,int(planned_saves*.8*GIB))
    out.mkdir(parents=True,exist_ok=True)
    ds=LeRobotDataset('talos/local',root=root,delta_timestamps={'action':[i/20 for i in range(20)]},video_backend='pyav')
    stats=normalization(ds.meta.stats)
    lineage=json.loads((root/'talos_lineage.json').read_text())
    if a.resume or a.warm_start:
        origin=Path(a.resume or a.warm_start)
        origin_meta=json.loads((origin/'run.json').read_text())
        if origin_meta['dataset']!=lineage:
            raise ValueError('Resume/warm-start requires the same dataset lineage in this diagnostic.')
        if a.resume and (origin_meta['arguments'].get('objective','vae')!=a.objective or origin_meta['arguments'].get('episode')!=a.episode):
            raise ValueError('Resume must retain the objective and episode selection; use warm-start for a new diagnostic.')
        policy,stats=load_policy(origin,device)
    else:policy=ACTPolicy(config(device,not a.no_pretrained)).to(device)
    old_representation=stats.get('action_representation','absolute')
    if a.resume and old_representation!=a.action_representation:
        raise ValueError('Resume must preserve the action representation.')
    if a.action_representation=='relative' and old_representation!='relative':
        # Compute residual RMS from numeric columns only; no image copy or decoding.
        numeric=ds.hf_dataset.select_columns(['action','observation.state']).with_format('numpy')
        targets=np.asarray(numeric['action']);positions=np.asarray(numeric['observation.state'])[:,:12]
        chosen=episode_indices(lineage,a.episode) if a.episode else list(range(len(ds)))
        windows=[]
        offset=0
        for row in lineage['episodes']:
            end=offset+row['frames']
            if not a.episode or row['source']==a.episode:
                for i in range(offset,end):windows.extend(targets[i:min(i+20,end)]-positions[i])
            offset=end
        rms=np.maximum(np.sqrt(np.mean(np.square(windows),axis=0)),.02)
        stats['action']={'mean':[0.]*12,'std':rms.tolist()}
        stats['action_representation']='relative'
        torch.nn.init.zeros_(policy.model.action_head.weight);torch.nn.init.zeros_(policy.model.action_head.bias)
    elif a.action_representation!=old_representation:
        raise ValueError('Changing relative weights back to absolute needs an explicit new initialization.')
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=device=='cuda')
    start=0
    if a.resume:
        state=torch.load(Path(a.resume)/'training_state.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['step'];torch.set_rng_state(state['torch_rng'])
        if device=='cuda' and state.get('cuda_rng'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    selected=Subset(ds,episode_indices(lineage,a.episode)) if a.episode else ds
    if len(selected)<a.batch_size:raise ValueError('Dataset subset is smaller than a training batch.')
    loader=DataLoader(selected,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,drop_last=True)
    iterator=iter(loader);policy.train();began=time.perf_counter()
    metadata={'arguments':vars(a),'dataset':lineage,'torch':torch.__version__,'device':device,
              'storage_preflight':storage,'selected_frames':len(selected),
              'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
              'experiment':'familiar-start diagnostic, not evidence of generalization',
              'resume_note':'Restores model, optimizer, scaler and Torch RNG; shuffled data iteration restarts, so resume is not bitwise identical.'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    def save(step,batch):
        require_space(out,int(.8*GIB))
        folder=out/f'step-{step:06d}';folder.mkdir(exist_ok=True)
        policy.save_pretrained(folder)
        (folder/'talos_normalization.json').write_text(json.dumps(stats,indent=2))
        (folder/'run.json').write_text(json.dumps(metadata,indent=2))
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []},folder/'training_state.pt')
        policy.eval()
        with torch.no_grad():reference=policy.predict_action_chunk(batch).cpu()
        # Check serialized weights exactly without keeping two GPU models resident.
        from safetensors.torch import load_file
        restored=load_file(str(folder/'model.safetensors'))
        assert all(torch.equal(v.detach().cpu(),restored[k]) for k,v in policy.state_dict().items())
        np.save(folder/'reload_reference.npy',reference.numpy(),allow_pickle=False)
        policy.train();print('CHECKPOINT',folder,flush=True)
    updates=0
    for step in range(start+1,a.steps+1):
        try:raw=next(iterator)
        except StopIteration:iterator=iter(loader);raw=next(iterator)
        batch=normalize(raw,stats,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            loss,info=learning_loss(policy,batch,a.objective)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ACT loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        updates+=int(torch.isfinite(grad))
        if step==start+1 or step%a.log_every==0:
            record={'step':step,'loss':float(loss.detach()),'components':{k:float(v) for k,v in info.items()},
                    'gradient_norm':float(grad) if torch.isfinite(grad) else None,'successful_updates':updates,
                    'elapsed_s':time.perf_counter()-began,
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%a.save_every==0 or step==a.steps:save(step,batch)
    if not updates:raise RuntimeError('No finite optimizer updates; checkpoint is not trained.')
    print('Completed',updates,'finite updates; physical rollout is required to assess task success.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--log-every',type=int,default=20)
    source=p.add_mutually_exclusive_group();source.add_argument('--resume');source.add_argument('--warm-start')
    p.add_argument('--episode');p.add_argument('--objective',choices=['vae','inference-l1'],default='vae')
    p.add_argument('--action-representation',choices=['absolute','relative'],default='absolute')
    p.add_argument('--no-pretrained',action='store_true');a=p.parse_args()
    if min(a.steps,a.batch_size,a.save_every,a.log_every)<=0:p.error('Steps, batch size and log/save intervals must be positive.')
    train(a)
