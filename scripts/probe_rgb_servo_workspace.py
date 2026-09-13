"""Declared bottle approach grid; no claim of full manipulation reachability."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene,HOME
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.storage import require_space

def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads(args.protocol.read_text()); gate=protocol['workspace_gate']
    require_space(args.output,32*1024**2)
    args.output.mkdir(parents=True)
    rows=[]; began=time.perf_counter()
    for x in gate['x_samples_m']:
        for y in gate['y_samples_m']:
            for arm in gate['arms']:
                require_space(args.output,1024**2)
                xml,layout=build_scene(seed=gate['scene_seed'],scenario='dinner',dinner_preset='task')
                model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
                data.qpos[:12]=HOME*2;data.ctrl[:]=HOME*2
                address=model.joint('bottle_free').qposadr[0]
                data.qpos[address:address+7]=[x,y,layout['table_z']+.001,1,0,0,0]
                mujoco.mj_forward(model,data)
                body=model.body('bottle').id
                overlap=max([-c.dist for c in data.contact if c.dist<0 and body in [model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]]],default=0.)
                for _ in range(300):mujoco.mj_step(model,data)
                data.time=0.
                row={'x_m':x,'y_m':y,'arm':arm,'initial_overlap_m':float(overlap),'settled_bottle_position_m':data.body('bottle').xpos.tolist()}
                if overlap>.001 or data.body('bottle').xmat[8]<.98:
                    row.update(status='invalid_start',reason='Initial overlap or non-upright settled bottle.')
                else:
                    task=DinnerTask(model,data,layout);task.start(side=arm,object_id='bottle')
                    q,v=data.qpos.copy(),data.qvel.copy();task.update(np.array(HOME*2))
                    assert np.array_equal(q,data.qpos) and np.array_equal(v,data.qvel)
                    assert model.neq==0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                    row.update(status='approach_candidate' if task.active else 'planner_rejected',reason=task.message)
                rows.append(row)
                result={'protocol':args.protocol.as_posix(),'scope':gate['interpretation'],'attempts':len(rows),'planned':48,
                        'approach_candidates':sum(r['status']=='approach_candidate' for r in rows),'rows':rows,'wall_seconds':time.perf_counter()-began}
                (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:row[k] for k in ['x_m','y_m','arm','status']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-bottle-v1.json'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
