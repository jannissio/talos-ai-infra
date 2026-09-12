"""Train-only RGB bottle features and stage-aligned upright demonstrations.

Stage labels align demonstration time offline. Runtime gets images/motor data
and internal progression only, never teacher stage or object position.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from simulation_lab.bottle_vision import bottle_features
from simulation_lab.retrieval_policy import ObservationRejected
from simulation_lab.storage import require_space

def run(a):
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    require_space(out,128*1024**2)
    rows=[];excluded=[];stage_order=['approach','descend','close','lift','hold','align','lower','release','retract','park','verify']
    for f in sorted(Path(a.dataset).glob('train-*')):
        m=json.loads((f/'manifest.json').read_text())
        if not m.get('training_eligible') or m['spec'].get('pose',{}).get('sideways'):continue
        images=[np.array(Image.open(f/(c+'.png'))) for c in ['overhead','left_wrist_cam','right_wrist_cam']]
        try:feature=bottle_features(images)
        except ObservationRejected as e:excluded.append({'episode':f.name,'reason':str(e)});continue
        with np.load(f/'trajectory.npz',allow_pickle=False) as z:traj={k:z[k].copy() for k in z.files}
        if set(traj['stages'])-set(stage_order):raise ValueError('Unexpected stage order')
        rows.append((f.name,feature,traj))
    durations={s:max(np.sum(t['stages']==s)/20 for _,_,t in rows) for s in stage_order}
    durations={s:np.ceil(d*20)/20 for s,d in durations.items()};bounds=[];actions=[];seconds=[];visual=[];cursor=0
    for name,feature,t in rows:
        act=[];times=[];offset=0.
        for stage in stage_order:
            indices=np.flatnonzero(t['stages']==stage)
            count=max(1,int(round(durations[stage]*20)));u=np.linspace(0,len(indices),count,endpoint=False)
            # Include the next stage's first endpoint to preserve continuity.
            ix=np.r_[indices,min(indices[-1]+1,len(t['actions20'])-1)]
            act.append(np.column_stack([np.interp(u,np.arange(len(ix)),t['actions20'][ix,j]) for j in range(12)]))
            times.append(offset+np.arange(count)/20);offset+=count/20
        act=np.concatenate(act);times=np.concatenate(times)
        actions.append(act);seconds.append(times);visual.append(np.repeat(feature[None],len(times),axis=0))
        bounds.append([cursor,cursor+len(times)]);cursor+=len(times)
    out.mkdir(parents=True)
    np.savez_compressed(out/'retrieval.npz',mean=np.zeros(32,dtype='float32'),components=np.eye(32,dtype='float32'),scale=np.ones(32,dtype='float32'),
        visual=np.concatenate(visual).astype('float32'),actions=np.concatenate(actions).astype('float32'),seconds=np.concatenate(seconds).astype('float32'),bounds=bounds)
    meta={'episodes':[n for n,_,_ in rows],'visual_encoder':'bottle_rgb_geometry','reconstruction_check':'initial_only','reconstruction_limit':.001,
          'durations_s':durations,'excluded':excluded,'scope':'RGB initial bottle location conditions a neural trajectory. Fixed-camera calibrated ROI; upright only. Offline demonstration-stage alignment is not a runtime stage input.'}
    (out/'retrieval.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'episodes':len(rows),'endpoints':cursor,'duration':sum(durations.values()),'excluded':excluded}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='.run/bottle-wide-compact-v1');p.add_argument('--output',required=True);run(p.parse_args())
