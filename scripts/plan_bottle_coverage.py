"""Declare a small coverage experiment before collecting or selecting models."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulation_lab.storage import require_space


def protocol():
    original=np.random.default_rng(617131);poses={}
    for i in range(16,32):
        sideways=i%4==0
        poses[i]={'x':float(original.uniform(-.13,-.05) if sideways else original.uniform(-.1,.05)),
                  'y':float(original.uniform(-.15,-.07) if sideways else original.uniform(-.15,-.04)),
                  'yaw':float(original.uniform(.4,1.2) if sideways else original.uniform(-.4,.4)),
                  'sideways':sideways}
    centers=[poses[i] for i in (17,19,25)]
    train=[{'id':f'train-{j:03d}','seed':2026090001+i,'split':'training','pose':poses[i],
            'reconstructed_from':f'original-count32/train-{i:03d}'} for j,i in enumerate((17,19,25))]
    used=[(p['x'],p['y']) for p in centers]
    def offset(rng,center):
        for _ in range(10000):
            p=dict(center,x=float(center['x']+rng.uniform(-.015,.015)),y=float(center['y']+rng.uniform(-.015,.015)))
            if all(np.hypot(p['x']-x,p['y']-y)>=.003 for x,y in used):
                used.append((p['x'],p['y']));return p
        raise RuntimeError('Unable to generate separated local poses.')
    rng=np.random.default_rng(407024)
    for j in range(3,18):
        train.append({'id':f'train-{j:03d}','seed':2026091401+j,'split':'training','pose':offset(rng,centers[(j-3)%3])})
    for j in range(18,24):
        train.append({'id':f'train-{j:03d}','seed':2026091401+j,'split':'training'})
    rng=np.random.default_rng(407006)
    dev=[{'id':f'dev-{j:03d}','seed':2026091501+j,'split':'development','pose':offset(rng,centers[j%3])} for j in range(6)]
    rng=np.random.default_rng(407012)
    evaluation=[{'id':f'eval-{j:03d}','seed':2026091601+j,'split':'evaluation',
                 **({'pose':offset(rng,centers[(j-3)%3])} if j>=3 else {})} for j in range(12)]
    return {'version':1,'purpose':'Bounded upright bottle coverage; not full workspace coverage',
            'baseline':'models/bottle_visual','training':train,'development':dev,'evaluation':evaluation,
            'regression_seeds':[42,*range(2026091301,2026091311)],
            'training_budget':{'adam_steps':16000,'lbfgs_steps':400,'selection_checkpoints':['adam-16000','lbfgs-400'],
                               'selection':'Development physical success count, then mean placement error; tie retains Adam.'},
            'promotion':{'fresh_successes_min':10,'strictly_beat_baseline':True,'all_11_regressions_pass':True},
            'data_gate':'All three reconstructed teacher runs and their 200/20 Hz replays must pass before further collection/training.',
            'storage':{'output_budget_gib':1,'reserve_gib':10},
            'provenance':'First three training attempts retain original scene seeds. Remaining 21 use fresh seeds. Raw laptop states are absent; reconstructions are not byte-identical copies.',
            'pose_distribution':'Upright only; 15 mm axis offsets around three exposed positions; 3 mm minimum separation of explicit XY starts; yaw fixed per region. Six training and three evaluation task-preset anchors.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);require_space(out,1024**2)
    if out.exists():raise FileExistsError('The declared protocol is immutable; choose a fresh path.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(protocol(),indent=2)+'\n',encoding='utf-8')
    print(out)
