"""Independently check goal-contact proofs, opening sweep and rigid grasp routes."""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put,arrays

def run(args):
    out=args.output.resolve();log=out.with_suffix('.log');source=args.source.resolve()
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        started=time.perf_counter();put(out/'source/scripts/verify_glass_goal_contact.py',Path(__file__).read_bytes())
        teacher_path=source/'source/simulation_lab/table_teacher.py';put(out/'source/simulation_lab/table_teacher.py',teacher_path.read_bytes())
        name='simulation_lab._goal_verification';spec=importlib.util.spec_from_file_location(name,teacher_path);teacher=importlib.util.module_from_spec(spec);sys.modules[name]=teacher;spec.loader.exec_module(teacher)
        from simulation_lab.scene import build_scene
        _,layout=build_scene(seed=2026114004,scenario='dinner',dinner_preset='task')
        model=mujoco.MjModel.from_xml_string((source/'scene.xml').read_text());results=[]
        for proof_path in sorted(source.glob('proof-*.json')):
            proof=json.loads(proof_path.read_text());data=mujoco.MjData(model)
            with np.load(proof_path.with_suffix('.npz')) as saved:data.qpos[:]=saved['qpos'];data.qvel[:]=0.;data.ctrl[:]=saved['ctrl']
            mujoco.mj_forward(model,data)
            task=teacher.TableTeacher(model,data,layout);task._select_item(proof['side'],next(x for x in layout['objects'] if x['id']=='glass'))
            row=dict(proof=proof_path.name,proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),opening=[],paths=[])
            q=data.qpos[task.offset:task.offset+5].copy();grip=float(data.qpos[task.offset+5]);scratch=mujoco.MjData(model);scratch.qpos[:]=data.qpos
            for opening in np.linspace(grip,.85,25):
                scratch.qpos[task.offset+5]=opening;mujoco.mj_forward(model,scratch)
                collision=task._collision(scratch,True,.0008);jaw_pen=0.
                for c in scratch.contact:
                    a,b=int(c.geom1),int(c.geom2)
                    if model.geom_bodyid[a]==task.body and b in task.jaw_geoms or model.geom_bodyid[b]==task.body and a in task.jaw_geoms:jaw_pen=max(jaw_pen,-float(c.dist))
                row['opening'].append(dict(grip=float(opening),collision=collision,jaw_penetration_mm=jaw_pen*1000,passed=collision is None and jaw_pen<.0005))
            task.ik.grasp_point=np.asarray(proof['object_origin_in_gripper'])
            rotation=np.asarray(proof['object_rotation_in_gripper']);reference=(np.zeros(3),rotation)
            task.ik.axis_local=rotation[:,2];task.ik.axis_target=np.array([0.,0.,1.]);task.ik.x_target=None;task.ik.soft_placement=True;task.ik.placement_solve=True
            for label,target in [('lift20',[.075,.10,.78]),('lift40',[.075,.10,.80]),('buffer_above',[.14425,-.02075,.80]),('buffer_contact',[.14425,-.02075,.76])]:
                record=dict(label=label,target=target)
                try:
                    end=task.ik.solve(np.asarray(target),q)
                    task._check_path([end],grip,True,carry=reference,support=task.base_geom if label=='buffer_contact' else None)
                    record.update(endpoint=True,q=end.tolist())
                    points=task._edge(q,end,grip,carry=reference,support=task.base_geom,allow_target=True)
                    record.update(direct_path=True,samples=len(points))
                    arrays(out/(proof_path.stem+'-'+label+'.npz'),q=points,closed_grip=np.array(grip),object_origin_in_gripper=task.ik.grasp_point,object_rotation_in_gripper=rotation)
                except teacher.PlanningError as exc:record.update(error=str(exc))
                row['paths'].append(record)
            row['opening_clear']=all(x['passed'] for x in row['opening']);results.append(row);put(out/proof_path.name,row)
            print(dict(proof=proof_path.name,opening_clear=row['opening_clear'],paths=row['paths']),flush=True)
        report=dict(scope='Independent reset-only kinematic/collision verification; no physical placement, hold, release or motor success.',preflight=preflight,wall_s=time.perf_counter()-started,results=results)
        put(out/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('.run/glass-goal-geometry-v3'));run(p.parse_args())
