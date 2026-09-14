"""Reset-only generic IK seed checks at the actual settled bridge and held glass."""
import contextlib,hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put
from simulation_lab.horizontal_glass_grasp_candidates import horizontal_glass_diameter_candidates

out=ROOT/'.run/glass-bridge-generic-seed-v1';log=out.with_suffix('.log')
if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(out,32*1024**2);out.mkdir(parents=True)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    source=ROOT/'.run/glass-bridge-physics-v3/leg2';manifest={}
    for path in (Path(__file__),ROOT/'simulation_lab/horizontal_glass_grasp_candidates.py',source/'source/simulation_lab/table_teacher.py'):
        payload=path.read_bytes();put(out/'source'/path.name,payload);manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(payload).hexdigest()
    put(out/'source-manifest.json',manifest)
    spec=importlib.util.spec_from_file_location('simulation_lab._generic_seed_check',source/'source/simulation_lab/table_teacher.py');teacher=importlib.util.module_from_spec(spec);sys.modules[spec.name]=teacher;spec.loader.exec_module(teacher)
    from simulation_lab.scene import HOME
    m=mujoco.MjModel.from_xml_string((source/'scene.xml').read_text());layout=json.loads((source/'layout.json').read_text())
    with np.load(source/'states.npz') as a:
        prefix=json.loads((source/'result.json').read_text())['prefix_motor_frames'];source_q=a['qpos'][prefix].copy()
        held_index=int(np.flatnonzero(a['stage']=='hold')[-1]);held_q=a['qpos'][held_index].copy()
    data=mujoco.MjData(m);data.qpos[:]=source_q;mujoco.mj_forward(m,data)
    task=teacher.TableTeacher(m,data,layout);task._select_item('right',next(o for o in layout['objects'] if o['id']=='glass'))
    candidate=horizontal_glass_diameter_candidates(data,'glass',teacher.GraspCandidate)[0]
    source_rows=[]
    for roll in (-2.4,0.,2.4):
        seed=np.array([0.,.8,.4,-1.2,roll]);row=dict(seed=seed.tolist())
        try:
            task._configure(candidate);q=task.ik.solve(candidate.point,seed);task._check_path([q],.85,True)
            task._precheck_transfer(q);task._configure(candidate)
            above=task.ik.solve(candidate.point+[0,0,.04],q)
            descent=task._cartesian(candidate.point+[0,0,.04],candidate.point,above,.85)
            approach=task._route(source_q[6:11],above,.85)
            row.update(passed=True,q=q.tolist(),above_q=above.tolist(),approach_samples=len(approach),descent_samples=len(descent))
        except teacher.PlanningError as exc:row.update(passed=False,error=str(exc))
        source_rows.append(row)
    held=mujoco.MjData(m);held.qpos[:]=held_q;mujoco.mj_forward(m,held)
    goal=teacher.TableTeacher(m,held,layout);goal._select_item('right',next(o for o in layout['objects'] if o['id']=='glass'));goal._configure(candidate)
    ref=goal._carry_reference();goal.placement_free_yaw=True;goal._place_axes(ref)
    goal_rows=[]
    for roll in (-2.4,0.,2.4):
        seed=np.array([0.,.8,.4,-1.2,roll]);row=dict(seed=seed.tolist())
        try:
            _,above=goal._point_for_center(np.array([.075,.10,.78]),ref,seed)
            _,lower=goal._point_for_center(np.array([.075,.10,.76]),ref,above)
            route=goal._route(held_q[6:11],above,float(held_q[11]),carry=ref,allow_target=True)
            goal._edge(above,lower,float(held_q[11]),carry=ref,support=goal.base_geom,allow_target=True)
            row.update(passed=True,above_q=above.tolist(),lower_q=lower.tolist(),route_samples=len(route))
        except teacher.PlanningError as exc:row.update(passed=False,error=str(exc))
        goal_rows.append(row)
    result=dict(scope='Reset-only checks from actual saved settled-source and held states; zero new physical trials, no scene-specific initial joint vector.',preflight=preflight,source_motor_trace=source.relative_to(ROOT).as_posix()+'/states.npz',source_states_sha256=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest(),source_state_index=prefix,held_state_index=held_index,source=source_rows,goal_from_measured_hold=goal_rows,geometry=dict(local_tool_point=candidate.local_tool_point.tolist(),local_axis=candidate.local_axis.tolist(),band=.067),passing_source=sum(x['passed'] for x in source_rows),passing_goal=sum(x['passed'] for x in goal_rows))
    put(out/'report.json',result);print(json.dumps(result,indent=2))
