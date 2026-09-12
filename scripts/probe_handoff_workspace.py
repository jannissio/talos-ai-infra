"""Planning-only overlap probe. Does not claim physical handoff success."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scripts.evaluate_dinner_scene import load
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.scene import HOME

def main():
    m,base,layout=load(42)
    for _ in range(200):mujoco.mj_step(m,base)
    initial=base.qpos.copy();rows=[]
    for x in [-.08,-.04,0.,.04,.08,.12]:
        for y in [-.15,-.10,-.05,0.,.05]:
            d=mujoco.MjData(m);d.qpos[:]=initial;d.ctrl[:]=HOME*2
            adr=m.joint('bottle_free').qposadr[0];d.qpos[adr:adr+7]=[x,y,layout['table_z']+.001,1,0,0,0]
            mujoco.mj_forward(m,d)
            for _ in range(100):mujoco.mj_step(m,d)
            row={'xy':[x,y]}
            for arm in ['left','right']:
                task=DinnerTask(m,d,layout);targets=np.array(HOME*2);task.start(side=arm,object_id='bottle');task.update(targets)
                row[arm]={'planned':task.stage=='approach' and task.active,'message':task.message}
            rows.append(row)
    Path('.run/submission-work/handoff-reach-probe.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'overlap':[r['xy'] for r in rows if r['left']['planned'] and r['right']['planned']],
                      'left':sum(r['left']['planned'] for r in rows),'right':sum(r['right']['planned'] for r in rows),'tested':len(rows)}))

if __name__=='__main__':main()
