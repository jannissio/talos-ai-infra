"""Compare recorded-feature training with actual PNG/state recurrent inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from simulation_lab.sequence_policy import SequencePolicy
from simulation_lab.dataset import load_policy_episode
from simulation_lab.retrieval_policy import KEYS,image_vector

def main(a):
    torch.set_num_threads(2);policy=SequencePolicy(a.checkpoint,'cpu');meta=policy.meta
    source=Path(meta['arguments']['source'])
    info=json.loads((source/'retrieval.json').read_text());ep=info['episodes'].index(a.episode)
    with np.load(source/'retrieval.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    lineage=json.loads(Path('.run/bottle-robustness-data/fit-lineage.json').read_text())
    row=next(r for r in lineage['episodes'] if r['source']==a.episode)
    samples=load_policy_episode(row['folder']);start,end=d['bounds'][ep];indices=np.arange(start,end,4)
    features=np.c_[d['visual'][indices],(d['states'][indices]-meta['state_mean'])/meta['state_std']]
    with torch.inference_mode():
        raw,_=policy.net(torch.tensor(features,dtype=torch.float32)[None])
        offline=raw[0].numpy()*meta['action_scale']
        offline+=np.array(meta['action_mean']) if meta.get('action_mode')=='absolute' else d['states'][indices,None,:12]
        actual=[];feature_error=0.
        for j,index in enumerate(indices):
            sample=samples[index-start];images=[];batch={'observation.state':torch.tensor(d['states'][index])[None]}
            for key,path in zip(KEYS,sample['images'].values()):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB')).copy()
                images.append(rgb);batch[key]=torch.tensor(rgb).permute(2,0,1)[None].float()/255
            visual=(image_vector(images)-d['mean'])@d['components'].T/d['scale']
            feature_error=max(feature_error,float(np.max(np.abs(visual-d['visual'][index]))))
            actual.append(policy.predict_action_chunk(batch)[0].numpy())
    actual=np.array(actual);future=np.minimum(indices[:,None]+np.arange(5)[None,:],end-1)
    teacher=d['actions'][future]
    report={'episode':a.episode,'recorded_observations':len(indices),'feature_max_abs_error':feature_error,
        'full_sequence_vs_runtime_first5_max_rad':float(np.max(np.abs(offline[:,:5]-actual[:,:5]))),
        'recorded_first5_arm_mae_rad':float(np.mean(np.abs(actual[:,:5,:5]-teacher[:,:,:5]))),
        'recorded_first5_gripper_mae_rad':float(np.mean(np.abs(actual[:,:5,5]-teacher[:,:,5]))),
        'physical_success_claim':False}
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episode',default='upright-01');p.add_argument('--output',required=True);main(p.parse_args())
