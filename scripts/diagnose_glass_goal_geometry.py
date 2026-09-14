"""Reset-only coupled grasp geometry at the unchanged glass goal; no physics steps."""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
from mujoco import minimize
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put,arrays


def run(args):
    out=args.output.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,256*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        started=time.perf_counter();manifest={}
        for name in ('scripts/diagnose_glass_goal_geometry.py','simulation_lab/table_teacher.py','simulation_lab/autonomy.py','simulation_lab/dinner_autonomy.py','simulation_lab/scene.py','simulation_lab/dinner.py','simulation_lab/glass_grasp_candidates.py','simulation_lab/side_plate_grasp_candidates.py'):
            value=(ROOT/name).read_bytes();put(out/'source'/name,value);manifest[name]=hashlib.sha256(value).hexdigest()
        put(out/'source-manifest.json',manifest)
        module_name='simulation_lab._glass_goal_snapshot';spec=importlib.util.spec_from_file_location(module_name,out/'source/simulation_lab/table_teacher.py');teacher=importlib.util.module_from_spec(spec);sys.modules[module_name]=teacher;spec.loader.exec_module(teacher)
        from simulation_lab.scene import build_scene,HOME
        xml,layout=build_scene(seed=2026114004,scenario='dinner',dinner_preset='task');put(out/'scene.xml',xml.encode())
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        source=Path('.run/whole-table-development-v18-seed4004/lookahead-01-034-00-glass/states.npz')
        with np.load(source) as saved:mujoco.mj_setState(model,data,saved['initial_integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
        data.qpos[:12]=HOME*2;data.qvel[:]=0.;data.ctrl[:]=HOME*2
        data.joint('glass_free').qpos[:]=[.075,.10,.76,1.,0.,0.,0.]
        mujoco.mj_forward(model,data)
        arrays(out/'reset-state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy())
        put(out/'reset-provenance.json',dict(source=source.as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),intervention='Goal-only geometric reset: glass at unchanged [.075,.10,.76], upright; both arms HOME. No mj_step call or physical placement success.'))
        center=data.body('glass').xpos.copy();rng=np.random.default_rng(2026117001)
        rows=[];proofs=[];allq=[]
        pads=[('tip',[-.008,0.,-.0982],[-.0123,-.076,.01875]),('middle',[-.0098,0.,-.0905],[-.0103,-.067,.01875])]
        for side in ('right','left'):
            task=teacher.TableTeacher(model,data,layout);task._select_item(side,next(x for x in layout['objects'] if x['id']=='glass'))
            scratch=mujoco.MjData(model);scratch.qpos[:]=data.qpos
            offset=task.offset;indices=np.arange(offset,offset+5);fixed=model.body(side+'_gripper').id;moving=model.body(side+'_moving_jaw_so101_v1').id
            lo=np.r_[task.ik.lo,model.jnt_range[offset+5,0]+.005,.018,-np.pi]
            hi=np.r_[task.ik.hi,min(.80,model.jnt_range[offset+5,1]-.005),.072,np.pi]
            for pad,fpoint,mpoint in pads:
                for family in ('diameter','rim'):
                    for sign in (1,-1):
                        def kin(v):
                            scratch.qpos[indices]=v[:5];scratch.qpos[offset+5]=v[5]
                            mujoco.mj_kinematics(model,scratch)
                            rf=scratch.xmat[fixed].reshape(3,3);rm=scratch.xmat[moving].reshape(3,3)
                            pf=scratch.xpos[fixed]+rf@fpoint;pm=scratch.xpos[moving]+rm@mpoint
                            radial=np.array([np.cos(v[7]),np.sin(v[7]),0.])
                            if family=='diameter':tf=center+radial*(.024*sign);tm=center-radial*(.024*sign)
                            else:tf=center+radial*(.01975 if sign==1 else .02425);tm=center+radial*(.02425 if sign==1 else .01975)
                            tf[2]+=v[6];tm[2]+=v[6]
                            return pf,pm,tf,tm,rf,rm
                        def residual(vs):
                            result=[]
                            for v in vs.T:
                                pf,pm,tf,tm,rf,rm=kin(v)
                                result.append(np.r_[pf-tf,pm-tm,.003*rf[2,0],.003*rm[2,0]])
                            return np.asarray(result).T
                        for trial in range(args.starts):
                            initial=rng.uniform(lo,hi)
                            if trial<3:initial[:5]=np.asarray(HOME[:5]);initial[4]=(-2.4,0.,2.4)[trial]
                            initial[5]=.4 if family=='diameter' else -.05;initial[6]=(.022,.045,.068)[trial%3]
                            v,_=minimize.least_squares(initial,residual,bounds=(lo,hi),max_iter=150,verbose=0)
                            pf,pm,tf,tm,rf,rm=kin(v);error=max(np.linalg.norm(pf-tf),np.linalg.norm(pm-tm))
                            row=dict(side=side,pad=pad,family=family,closing_sign=sign,trial=trial,contact_height=float(v[6]),bearing=float(v[7]),q=v[:6].tolist(),contact_error_mm=float(error*1000))
                            allq.append(v.copy())
                            if error<.00025:
                                mujoco.mj_forward(model,scratch)
                                collision=task._collision(scratch,True,.0008)
                                if args.refine and collision is None:
                                    original=v[:6].copy();glass_geoms=np.flatnonzero(model.geom_bodyid==task.body)
                                    pair_buffer=np.zeros(6)
                                    def exact_gaps(q):
                                        scratch.qpos[indices]=q[:5];scratch.qpos[offset+5]=q[5]
                                        mujoco.mj_forward(model,scratch)
                                        gaps=[]
                                        for group in (task.fixed_geoms,task.moving_geoms):
                                            gaps.append(min(mujoco.mj_geomDistance(model,scratch,int(a),int(b),.02,pair_buffer) for a in group for b in glass_geoms if ((model.geom_contype[a]&model.geom_conaffinity[b]) or (model.geom_contype[b]&model.geom_conaffinity[a]))))
                                        return np.asarray(gaps)
                                    def contact_residual(qs):
                                        return np.asarray([np.r_[exact_gaps(q),.0001*(q-original)] for q in qs.T]).T
                                    q,_=minimize.least_squares(original,contact_residual,bounds=(lo[:6],hi[:6]),max_iter=100,verbose=0)
                                    refined_gaps=exact_gaps(q)
                                    v[:6]=q
                                    pf,pm,tf,tm,rf,rm=kin(v);mujoco.mj_forward(model,scratch)
                                    collision=task._collision(scratch,True,.0008)
                                    row.update(refined_q=q.tolist(),exact_refined_gaps_mm=(refined_gaps*1000).tolist(),q=q.tolist())
                                contacts=[];jaw_pen=0.;other_pen=0.;jaw_gap={'fixed':.1,'moving':.1}
                                for contact in scratch.contact:
                                    a,b=int(contact.geom1),int(contact.geom2)
                                    if model.geom_bodyid[a]!=task.body and model.geom_bodyid[b]!=task.body:continue
                                    other=b if model.geom_bodyid[a]==task.body else a
                                    key='fixed' if other in task.fixed_geoms else 'moving' if other in task.moving_geoms else 'external'
                                    if key in jaw_gap:jaw_gap[key]=min(jaw_gap[key],float(contact.dist));jaw_pen=max(jaw_pen,-float(contact.dist))
                                    elif other!=task.base_geom:other_pen=max(other_pen,-float(contact.dist))
                                    contacts.append(dict(other=model.geom(other).name,distance_mm=float(contact.dist*1000),position=contact.pos.tolist()))
                                row.update(collision=collision,jaw_penetration_mm=jaw_pen*1000,other_penetration_mm=other_pen*1000,jaw_gaps_mm={k:v*1000 for k,v in jaw_gap.items()},contacts=contacts)
                                valid=not collision and jaw_pen<=.0005 and other_pen<=.00015 and all(x<.00025 for x in jaw_gap.values())
                                row['checked_goal_contact']=bool(valid)
                                if valid:
                                    reference=rf.T@(center-scratch.xpos[fixed]);relative_rotation=rf.T@np.eye(3)
                                    row.update(gripper_position=scratch.xpos[fixed].tolist(),gripper_rotation=rf.tolist(),object_origin_in_gripper=reference.tolist(),object_rotation_in_gripper=relative_rotation.tolist(),fixed_contact_world=pf.tolist(),moving_contact_world=pm.tolist())
                                    proofs.append(row)
                                    arrays(out/f'proof-{len(proofs):03d}.npz',qpos=scratch.qpos.copy(),qvel=scratch.qvel.copy(),ctrl=scratch.ctrl.copy())
                                    put(out/f'proof-{len(proofs):03d}.json',row)
                                    print(dict(proof=len(proofs),side=side,pad=pad,family=family,height=v[6],contact_error_mm=error*1000),flush=True)
                            rows.append(row)
                        print(dict(side=side,pad=pad,family=family,sign=sign,done=len(rows),proofs=len(proofs)),flush=True)
        arrays(out/'optimizer-solutions.npz',parameters=np.asarray(allq))
        report=dict(scope='Reset-only coupled geometry at unchanged glass final setting, not physics or source-route success.',attempts=len(rows),checked_goal_contacts=len(proofs),preflight=preflight,wall_s=time.perf_counter()-started,rows=rows)
        put(out/'report.json',report);print({k:v for k,v in report.items() if k!='rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=16)
    p.add_argument('--refine',action='store_true')
    run(p.parse_args())
