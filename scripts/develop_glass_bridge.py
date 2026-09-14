"""Reset-only intersection search for vertical-to-horizontal glass bridge buffers."""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put,arrays

def run(args):
    out=args.output.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        manifest={};started=time.perf_counter()
        for name in ('scripts/develop_glass_bridge.py','simulation_lab/table_teacher.py','simulation_lab/autonomy.py','simulation_lab/dinner_autonomy.py','simulation_lab/random_dinner.py','simulation_lab/scene.py','simulation_lab/dinner.py','simulation_lab/glass_grasp_candidates.py','simulation_lab/side_plate_grasp_candidates.py'):
            value=(ROOT/name).read_bytes();put(out/'source'/name,value);manifest[name]=hashlib.sha256(value).hexdigest()
        put(out/'source-manifest.json',manifest)
        name='simulation_lab._glass_bridge';spec=importlib.util.spec_from_file_location(name,out/'source/simulation_lab/table_teacher.py');teacher=importlib.util.module_from_spec(spec);sys.modules[name]=teacher;spec.loader.exec_module(teacher)
        if args.goal_seed:
            goal_proof=json.loads((ROOT/'.run/glass-goal-geometry-v3/proof-001.json').read_text())
            goal_seed=np.asarray(goal_proof['q'][:5]);put(out/'goal-seed-proof.json',goal_proof)
            class GoalSeedIK(teacher.TableIK):
                def solve(self,position,initial):
                    try:return super().solve(position,initial)
                    except teacher.PlanningError:
                        if self.axis_index!=1:raise
                        return super().solve(position,goal_seed)
            teacher.TableIK=GoalSeedIK
        from simulation_lab.scene import HOME
        source=ROOT/'.run/glass-placement-v5-middle-top-wrap'
        xml=(source/'scene.xml').read_text();put(out/'scene.xml',xml.encode());layout=json.loads((source/'layout.json').read_text())
        m=mujoco.MjModel.from_xml_string(xml)
        with np.load(source/'states.npz') as a:
            held_index=int(np.flatnonzero(a['stage']=='hold')[-1]);heldq=a['qpos'][held_index].copy();initial=a['initial_integration'].copy()
        held=mujoco.MjData(m);held.qpos[:]=heldq;mujoco.mj_forward(m,held)
        glass=m.body('glass').id;gripper=m.body('right_gripper').id;rot=held.xmat[gripper].reshape(3,3)
        relative_pos=rot.T@(held.xpos[glass]-held.xpos[gripper]);relative_rot=rot.T@held.xmat[glass].reshape(3,3)
        vertical=teacher.TableTeacher(m,held,layout);vertical._select_item('right',next(o for o in layout['objects'] if o['id']=='glass'))
        vertical.ik.grasp_point=relative_pos.copy();vref=(np.zeros(3),relative_rot);vertical.placement_free_yaw=True;vertical._place_axes(vref)
        vertical_support_offset=0.;vertical_target_quat=np.array([1.,0.,0.,0.])
        if args.tilted_buffer:
            vertical.placement_axis_target=held.xmat[glass].reshape(3,3)[:,2].copy();vertical._place_axes(vref)
            vertical_target_quat=held.joint('glass_free').qpos[3:].copy()
            from simulation_lab.random_dinner import body_bounds
            low,_=body_bounds(m,'glass',vertical_target_quat);vertical_support_offset=-low[2]
        vstart=heldq[6:11].copy();vgrip=float(heldq[11]);rows=[];proofs=[]
        original_candidates=teacher.grasp_candidates
        # Nearest useful extension first; all candidates are actual table coordinates.
        positions=[(x,y) for y in (-.025,-.005,.015,.035,.055,.075,.095) for x in (.04,0.,-.04,.08,.12,.16,.20,.24,.28,.32)] if args.goal_seed else [(x,y) for y in (.015,.035,.055,.075,.095,.115) for x in (.16,.20,.12,.24,.08)]
        if args.boundary:
            positions=[(x,y) for y in (-.12,-.08,-.05,.03,.05,.07,.09) for x in (-.10,-.06,-.02,.02,.06,.10,.14,.18,.22)]
        if args.fine:
            positions=[(x,y) for y in (.076,.078,.074,.080,.082) for x in (.22,.24,.20,.26,.18,.28,.16)]
        if args.margin:
            positions=[(.24,y) for y in (.090,.085,.080)]
        for xy in positions:
            center=np.r_[xy,.76];entry=dict(buffer=center.tolist(),vertical=[],horizontal=[])
            vertical_center=center+[0.,0.,vertical_support_offset]
            vertical_ok=None
            vertical_seeds=[vstart] if not args.boundary else [vstart,np.asarray(HOME[:5]),np.array([0.,-.7,.8,.2,2.4])]
            for height,vertical_seed in [(h,q) for h in (.04,.02,.065) for q in vertical_seeds]:
                try:
                    above=vertical.ik.solve(vertical_center+[0,0,height],vertical_seed);lower=vertical.ik.solve(vertical_center,above)
                    vertical._check_path([lower],vgrip,True,carry=vref,support=vertical.base_geom)
                    route=vertical._route(vstart,above,vgrip,carry=vref,allow_target=True,iterations=60)
                    vertical._edge(above,lower,vgrip,carry=vref,support=vertical.base_geom,allow_target=True)
                    vertical_ok=dict(height=height,above_q=above.tolist(),lower_q=lower.tolist(),route_samples=len(route),target_position=vertical_center.tolist(),target_quaternion=vertical_target_quat.tolist());entry['vertical'].append(dict(passed=True,**vertical_ok));break
                except teacher.PlanningError as exc:entry['vertical'].append(dict(height=height,error=str(exc)))
            if vertical_ok is None:rows.append(entry);continue
            for band in (.067,.072):
                data=mujoco.MjData(m);mujoco.mj_setState(m,data,initial,mujoco.mjtState.mjSTATE_INTEGRATION)
                data.qpos[:12]=HOME*2;data.qvel[:]=0.;data.ctrl[:]=HOME*2;data.joint('glass_free').qpos[:]=[*center,1.,0.,0.,0.];mujoco.mj_forward(m,data)
                def candidates(data,name):
                    point=data.body('glass').xpos+data.body('glass').xmat.reshape(3,3)@np.array([0.,0.,band])
                    candidate=teacher.GraspCandidate('bridge_horizontal_body',point,np.array([.017,0.,-.098]),1,np.array([0.,0.,1.]),None,.85,None,.65)
                    if args.exact_horizontal:
                        candidate.local_axis=np.asarray(goal_proof['object_rotation_in_gripper'])[:,2].copy()
                        candidate.local_tool_point=np.asarray(goal_proof['object_origin_in_gripper'])+candidate.local_axis*.072
                    return [candidate]
                teacher.grasp_candidates=candidates
                for side in ('right','left'):
                    task=teacher.TableTeacher(m,data,layout);task.start(object_id='glass',side=side,target=[.075,.10,.76],target_quaternion=[1.,0.,0.,0.]);targets=data.ctrl.copy();task.update(targets)
                    item=dict(side=side,band=band,status=task.status,stage=task.stage,search=task.search_log)
                    if task.stage=='approach':
                        try:
                            q=np.array(task.approach_descent[-1]);ref=task._carry_reference(q);task.placement_free_yaw=True;task._place_axes(ref)
                            _,lift=task._point_for_center(center+[0,0,.065],ref,q)
                            task._edge(q,lift,.455,carry=ref,support=task.base_geom,allow_target=True)
                            errors=[];path=None
                            for h in (.02,.04,.065):
                                try:
                                    _,above=task._point_for_center(task.destination_position+[0,0,h],ref,lift);_,lower=task._point_for_center(task.destination_position,ref,above)
                                    route=task._route(lift,above,.455,carry=ref,allow_target=True,iterations=60)
                                    task._edge(above,lower,.455,carry=ref,support=task.base_geom,allow_target=True)
                                    path=dict(height=h,source_q=q.tolist(),lift_q=lift.tolist(),above_q=above.tolist(),lower_q=lower.tolist(),route_samples=len(route));break
                                except teacher.PlanningError as exc:errors.append(str(exc))
                            if path is None:raise teacher.PlanningError('; '.join(errors))
                            item.update(passed=True,path=path);proof=dict(buffer=center.tolist(),vertical=vertical_ok,horizontal=item,scope='Reset-only geometry, not physical placement or grasp retention.')
                            proofs.append(proof);put(out/f'proof-{len(proofs):03d}.json',proof);print('PROOF',proof,flush=True)
                        except teacher.PlanningError as exc:item.update(error=str(exc))
                    entry['horizontal'].append(item)
                    if len(proofs)>=args.max_proofs:break
                if len(proofs)>=args.max_proofs:break
            rows.append(entry);print(dict(buffer=center.tolist(),checked=len(rows),proofs=len(proofs)),flush=True)
            if len(proofs)>=args.max_proofs:break
        teacher.grasp_candidates=original_candidates
        put(out/'report.json',dict(scope='Reset-only geometry intersection; no physics steps. Actual held vertical grasp from saved motor run; horizontal hypothetical buffer resets retain actual obstacles.',preflight=preflight,source=source.relative_to(ROOT).as_posix(),source_states_sha256=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest(),held_index=held_index,vertical_object_origin_in_tool=relative_pos.tolist(),vertical_object_rotation_in_tool=relative_rot.tolist(),checked_buffers=len(rows),proofs=len(proofs),rows=rows,wall_s=time.perf_counter()-started))
        print(dict(done=True,checked_buffers=len(rows),proofs=len(proofs),wall_s=time.perf_counter()-started),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-proofs',type=int,default=2);p.add_argument('--goal-seed',action='store_true');p.add_argument('--boundary',action='store_true');p.add_argument('--fine',action='store_true');p.add_argument('--exact-horizontal',action='store_true');p.add_argument('--tilted-buffer',action='store_true');p.add_argument('--margin',action='store_true');run(p.parse_args())
