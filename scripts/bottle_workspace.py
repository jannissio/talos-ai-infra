"""Approach-planner coverage, explicitly distinct from full manipulation success."""
import sys,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.prepare_bottle_data import setup,save,ROOT
from simulation_lab.scene import HOME
from simulation_lab.dinner_autonomy import DinnerTask

def check(spec):
    _,model,data,layout=setup(spec)
    if layout['bottle_reset']['initial_penetration_m']>.001:
        return dict(spec,label='reset_overlap',reason='Initial bottle overlap exceeds 1 mm; not an approach candidate.')
    task=DinnerTask(model,data,layout)
    task.start(side=spec['arm'],object_id='bottle')
    task.update(np.array(HOME*2))
    label='approach_candidate' if task.active else 'tested_path_collision' if 'insufficient clearance' in task.message else 'planner_unresolved'
    return dict(spec,label=label,reason=task.message)

if __name__=='__main__':
    specs=[dict(x=round(x,4),y=round(y,4),yaw=a*math.pi/2,sideways=s,arm=arm)
           for x in np.arange(-.15,.151,.05) for y in np.arange(-.20,.001,.05)
           for s in (False,True) for a in (range(4) if s else [0]) for arm in ('left','right')]
    # 1 cm samples around one empirically useful approach family.
    specs += [dict(x=x,y=y,yaw=math.pi/4,sideways=True,arm='left')
              for x in [-.11,-.10,-.09] for y in [-.16,-.15,-.14]]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(check,specs,chunksize=4))
    save(ROOT/'docs/robotics/bottle-workspace.json',{
        'grid_step_m':.05,'local_refinement_m':.01,
        'scope':'Bottle approach planning on one dinner layout. Not a complete reachable set. Planner failure is not proof of physical impossibility. Full-rollout results are separate.',
        'samples':rows,'approach_candidates':sum(r['label']=='approach_candidate' for r in rows)})
    print(len(rows),'samples',sum(r['label']=='approach_candidate' for r in rows),'candidates')
