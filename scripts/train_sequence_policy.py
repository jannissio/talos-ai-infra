"""Compact full-episode recurrent imitation, trained at the actual 5 Hz replanning rate."""
import argparse,json,hashlib,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from safetensors.torch import save_file
from simulation_lab.sequence_policy import SequenceNet
from simulation_lab.storage import require_space,GIB

def main(a):
    if min(a.steps,a.save_every)<=0 or a.lr<=0:raise ValueError('Positive training limits and learning rate required.')
    torch.set_num_threads(2);torch.manual_seed(19);np.random.seed(19)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    root=Path(__file__).resolve().parents[1]/'.run'
    used=sum(f.stat().st_size for p in root.glob('bottle-sequence*') for f in (p.rglob('*') if p.is_dir() else [p]) if f.is_file())
    planned=((a.steps+a.save_every-1)//a.save_every)*3*1024**2
    if used+planned>2*GIB:raise OSError('Sequence experiment exceeds its 2 GiB artifact budget.')
    require_space(out,planned);out.mkdir(parents=True)
    source=Path(a.source)
    with np.load(source/'retrieval.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    source_meta=json.loads((source/'retrieval.json').read_text())
    selected=[i for i,name in enumerate(source_meta['episodes']) if not a.episode or name==a.episode]
    if not selected:raise ValueError('Unknown training episode')
    state_mean=d['states'].mean(0);state_std=np.maximum(d['states'].std(0),.02)
    features=np.c_[d['visual'],(d['states']-state_mean)/state_std].astype(np.float32)
    rows=[];labels=[];masks=[]
    for ep in selected:
        start,end=d['bounds'][ep];indices=np.arange(start,end,4)
        future=indices[:,None]+np.arange(20)[None,:]
        mask=future<end;future=np.minimum(future,end-1)
        rows.append(features[indices]);labels.append(d['actions'][future] if a.action_mode=='absolute' else d['actions'][future]-d['states'][indices,None,:12]);masks.append(mask)
    all_labels=np.concatenate(labels)
    mean=all_labels.mean(axis=(0,1)) if a.action_mode=='absolute' else np.zeros(12,np.float32)
    scale=np.maximum(np.sqrt(np.mean((all_labels-mean)**2,axis=(0,1))),.02).astype(np.float32)
    length=max(map(len,rows));n=len(rows)
    x=np.zeros((n,length,56),np.float32);y=np.zeros((n,length,20,12),np.float32);mask=np.zeros((n,length,20,1),np.float32)
    for i,(f,l,m) in enumerate(zip(rows,labels,masks)):
        x[i,:len(f)]=f;y[i,:len(f)]=(l-mean)/scale;mask[i,:len(f),:,0]=m
    if a.executed_only:mask[:,:,5:]=0
    x,y,mask=[torch.from_numpy(v).to(a.device) for v in [x,y,mask]]
    net=SequenceNet(a.initial_context).to(a.device);optimizer=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
    if a.warm_start:
        previous=json.loads((Path(a.warm_start)/'sequence.json').read_text())
        if previous.get('action_mode','relative')!=a.action_mode:raise ValueError('Warm start requires unchanged action representation.')
        if previous['source_sha256']!=hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest():
            raise ValueError('Warm start requires an unchanged feature/data basis.')
        from safetensors.torch import load_file
        weights=load_file(str(Path(a.warm_start)/'sequence.safetensors'),device=a.device)
        if a.initial_context and weights['encoder.0.weight'].shape[1]==56:
            weights['encoder.0.weight']=torch.cat([weights['encoder.0.weight'],torch.zeros_like(weights['encoder.0.weight'])],dim=1)
        net.load_state_dict(weights)
        # Preserve absolute actions when expanding an episode subset changes label statistics.
        ratio=torch.tensor(np.asarray(previous['action_scale'])/scale,device=a.device,dtype=torch.float32).repeat(20)
        offset=torch.tensor((np.asarray(previous.get('action_mean',[0.]*12))-mean)/scale,device=a.device,dtype=torch.float32).repeat(20)
        with torch.no_grad():
            net.head[-1].weight.mul_(ratio[:,None]);net.head[-1].bias.mul_(ratio).add_(offset)
    meta={'state_mean':state_mean.tolist(),'state_std':state_std.tolist(),'action_scale':scale.tolist(),
          'action_mean':mean.tolist(),'action_mode':a.action_mode,
          'reconstruction_limit':source_meta['reconstruction_limit'],'training_episodes':[source_meta['episodes'][i] for i in selected],
          'source_sha256':hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest(),
          'replanning_hz':5,'physics_hz':200,'sequence_lengths':list(map(len,rows)),
          'arguments':vars(a),'architecture':f'GRU 128x2; 32 PCA RGB + 24 joint features; 20 {a.action_mode} joint targets'}
    (out/'run.json').write_text(json.dumps(meta,indent=2));began=time.perf_counter()
    def save(step):
        folder=out/f'step-{step:06d}';folder.mkdir()
        save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},str(folder/'sequence.safetensors'))
        np.savez_compressed(folder/'visual.npz',**{k:d[k] for k in ['mean','components','scale']})
        (folder/'sequence.json').write_text(json.dumps(meta,indent=2))
        (folder/'talos_normalization.json').write_text(json.dumps({'observation.state':{'mean':[0.]*24,'std':[1.]*24},'action':{'mean':[0.]*12,'std':[1.]*12},'images':{'mean':[0.]*3,'std':[1.]*3}}))
        print('CHECKPOINT',folder,flush=True)
    for step in range(1,a.steps+1):
        optimizer.zero_grad(set_to_none=True);inputs=x;targets=y;step_mask=mask
        if a.stutter:
            # Synthetic actuator pauses: repeat an observed pre-grasp state and its
            # command, rather than advancing the recurrent controller's phase.
            delay=torch.randint(0,11,(n,1),device=x.device)
            start=torch.randint(5,26,(n,1),device=x.device)
            time_index=torch.arange(length+10,device=x.device)[None,:].expand(n,-1)
            index=torch.where(time_index<start,time_index,torch.maximum(start,time_index-delay)).clamp(max=length-1)
            batch_index=torch.arange(n,device=x.device)[:,None]
            inputs=x[batch_index,index].clone();targets=y[batch_index,index].clone();step_mask=mask[batch_index,index].clone()
            step_mask*=((time_index-delay)<length)[:,:,None,None]
            paused=(time_index>start)&(time_index<=start+delay)
            inputs[:,:,44:50]=torch.where(paused[:,:,None],torch.tensor(-state_mean[12:18]/state_std[12:18],device=x.device),inputs[:,:,44:50])
        if a.joint_noise or a.visual_noise or a.velocity_noise:
            inputs=inputs.clone();targets=targets.clone()
            noise=torch.randn((*inputs.shape[:2],6),device=x.device)*a.joint_noise
            inputs[:,:,32:38]+=noise/torch.tensor(state_std[:6],device=x.device)
            if a.action_mode=='relative':targets[:,:,:,:6]-=noise[:,:,None,:]/torch.tensor(scale[:6],device=x.device)
            inputs[:,:,:32]+=torch.randn_like(inputs[:,:,:32])*a.visual_noise
            inputs[:,:,44:50]+=torch.randn_like(inputs[:,:,44:50])*a.velocity_noise/torch.tensor(state_std[12:18],device=x.device)
        if a.clean_start:
            inputs[:,0]=x[:,0];targets[:,0]=y[:,0]
        pred,_=net(inputs)
        loss=(((pred-targets)**2)*step_mask).sum()/(step_mask.sum()*12)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0:print(json.dumps({'step':step,'loss':float(loss),'gradient_norm':float(grad),'elapsed_s':time.perf_counter()-began}),flush=True)
        if step%a.save_every==0 or step==a.steps:save(step)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='.run/bottle-robustness-model');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--device',default='cuda');p.add_argument('--episode');p.add_argument('--warm-start')
    p.add_argument('--joint-noise',type=float,default=0);p.add_argument('--visual-noise',type=float,default=0)
    p.add_argument('--action-mode',choices=['absolute','relative'],default='relative')
    p.add_argument('--executed-only',action='store_true');p.add_argument('--velocity-noise',type=float,default=0)
    p.add_argument('--stutter',action='store_true');p.add_argument('--clean-start',action='store_true')
    p.add_argument('--initial-context',action='store_true');main(p.parse_args())
