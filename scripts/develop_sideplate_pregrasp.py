"""Contact-driven pregrasp geometry for the exposed inverted side plate.

Geometry phase reads the actual state after accepted actions00–03. It never
steps physics or changes the object pose. All hypothetical robot poses use a
separate scratch MjData. Physics is a separate explicit mode, if a route exists.
"""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
from mujoco import minimize
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put,arrays

def geometry(args):
    out=args.output.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,256*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        started=time.perf_counter();manifest={};source=args.scene.resolve()
        for name in ('scripts/develop_sideplate_pregrasp.py','simulation_lab/table_teacher.py','simulation_lab/autonomy.py','simulation_lab/dinner_autonomy.py','simulation_lab/random_dinner.py','simulation_lab/scene.py','simulation_lab/dinner.py'):
            path=source/'source'/name if name=='simulation_lab/table_teacher.py' else ROOT/name
            value=path.read_bytes();put(out/'source'/name,value);manifest[name]=hashlib.sha256(value).hexdigest()
        spec=importlib.util.spec_from_file_location('simulation_lab._sideplate_pregrasp',out/'source/simulation_lab/table_teacher.py');teacher=importlib.util.module_from_spec(spec);sys.modules[spec.name]=teacher;spec.loader.exec_module(teacher)
        from simulation_lab.scene import build_scene,HOME
        xml,layout=build_scene(seed=2026114004,scenario='dinner',dinner_preset='task');put(out/'scene.xml',xml.encode());put(out/'layout.json',layout)
        m=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(m);accepted=[]
        for index in range(4):
            record=json.loads((source/f'accepted-action-{index:02d}.json').read_text());assert record['status']=='succeeded' and record['continuous_main_scene_replay_exact']
            path=source/record['path']/'states.npz';accepted.append(dict(action=index,path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=record['frames']))
        with np.load(ROOT/accepted[-1]['path']) as a:data.qpos[:]=a['qpos'][-1];data.qvel[:]=a['qvel'][-1];data.ctrl[:]=a['ctrl'][-1]
        mujoco.mj_forward(m,data);arrays(out/'actual-post-action03-state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy());put(out/'prefix.json',accepted)
        task=teacher.TableTeacher(m,data,layout);task._select_item('left',next(o for o in layout['objects'] if o['id']=='side_plate'))
        scratch=mujoco.MjData(m);scratch.qpos[:]=data.qpos;scratch.qvel[:]=0.;scratch.ctrl[:]=data.ctrl
        body=data.body('side_plate');rotation=body.xmat.reshape(3,3);radial=body.xpos-data.body('left_shoulder').xpos;radial[2]=0;radial/=np.linalg.norm(radial)
        lo=np.r_[task.ik.lo,.45];hi=np.r_[task.ik.hi,1.70];rows=[];proofs=[];rng=np.random.default_rng(2026119101)
        def solve(point,g,initial):
            def residual(qs):
                values=[]
                for q in qs.T:
                    scratch.qpos[:6]=q;mujoco.mj_kinematics(m,scratch)
                    values.append(np.r_[scratch.geom_xpos[g]-point,.00001*(q-initial)])
                return np.asarray(values).T
            q,_=minimize.least_squares(initial,residual,bounds=(lo,hi),max_iter=160,verbose=0)
            scratch.qpos[:6]=q;mujoco.mj_forward(m,scratch)
            error=float(np.linalg.norm(scratch.geom_xpos[g]-point))
            if error>.00020:raise teacher.PlanningError(f'Moving-finger point IK residual {error*1000:.6f}mm')
            bad=task._collision(scratch,True,.0008)
            if bad:raise teacher.PlanningError('Robot clearance: '+bad)
            pen=0.
            for c in scratch.contact:
                a,b=int(c.geom1),int(c.geom2)
                if m.geom_bodyid[a]==task.body or m.geom_bodyid[b]==task.body:
                    other=b if m.geom_bodyid[a]==task.body else a
                    if other!=task.base_geom:pen=max(pen,-float(c.dist))
            if pen>.0003:raise teacher.PlanningError(f'Source plate/jaw penetration{pen*1000:.5f}mm')
            return q,error
        for sphere in ('left_moving_jaw_sph_tip1','left_moving_jaw_sph_tip2','left_moving_jaw_sph_tip3'):
            g=m.geom(sphere).id
            for angle in (0.,-.12,.12):
                ca,sa=np.cos(angle),np.sin(angle);direction=np.array([ca*radial[0]-sa*radial[1],sa*radial[0]+ca*radial[1],0.])
                local_direction=rotation.T@direction;local_direction[2]=0.;local_direction/=np.linalg.norm(local_direction)
                # Inverted rim's upper face is body-local z=.004; sphere stays
                # .1mm clear during geometric preflight before controlled touch.
                contact=body.xpos+rotation@(-local_direction*.050+np.array([0.,0.,.004]))
                center=contact+[0.,0.,float(m.geom_size[g,0])+.0001]
                for trial in range(args.starts):
                    seed=rng.uniform(lo,hi)
                    if trial<6:
                        seed=np.array([.40,(.2,.8,1.2)[trial%3],(-.2,.4)[trial//3],(-1.2,.0,1.2)[trial%3],(-1.5,1.5)[trial//3],1.4])
                    record=dict(sphere=sphere,angle=angle,start=trial,initial_q=seed.tolist(),contact_world=contact.tolist())
                    try:
                        q,error=solve(center,g,seed);record.update(contact_q=q.tolist(),error_mm=error*1000,endpoint=True)
                        route=[];state=q.copy()
                        for dz in np.linspace(0.,.035,19):
                            state,_=solve(center+[0,0,dz],g,state);route.append(state.copy())
                        # The arm approach route uses the solved open grip; the
                        # finger path itself checks its measured varying grip.
                        task._route(data.qpos[:5],route[-1][:5],float(route[-1][5]),iterations=100)
                        record.update(approach=True,contact_to_hover_q=np.asarray(route).tolist())
                        proofs.append(record);put(out/f'proof-{len(proofs):03d}.json',record);print('PROOF',record,flush=True)
                    except teacher.PlanningError as exc:record.update(error=str(exc))
                    rows.append(record)
                    if len(proofs)>=2:break
                print(dict(sphere=sphere,angle=angle,attempts=len(rows),proofs=len(proofs)),flush=True)
                if len(proofs)>=2:break
            if len(proofs)>=2:break
        put(out/'source-manifest.json',manifest);put(out/'report.json',dict(scope='Reset-only actual post-action03 geometry; no physics, object-pose edits or successful repositioning claim.',preflight=preflight,attempts=len(rows),proofs=len(proofs),rows=rows,wall_s=time.perf_counter()-started));print(dict(done=True,proofs=len(proofs),wall_s=time.perf_counter()-started),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,default=Path('.run/whole-table-development-v24-seed4004'));p.add_argument('--starts',type=int,default=12);geometry(p.parse_args())
