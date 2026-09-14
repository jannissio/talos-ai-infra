"""Finite diagonal fork cross-section pinch geometry; no live physics writes."""
import math
import mujoco
import numpy as np
from .autonomy import PlanningError
from .scene import HOME,TABLE_Z
from .table_teacher import TableTeacher,GraspCandidate


def endpoint_family(data):
    for y in (-.015,0.,.032):
        z=.008 if y<.02 else .004
        point=data.body('fork').xpos+np.array([0.,y,z])
        for tx in (-.013,-.003,.005):
            for degrees in (-60,-45,-30,30,45,60):
                theta=math.radians(degrees)
                yield degrees,GraspCandidate(f'diagonal_{y}_{tx}_{degrees}',point,
                    np.array([tx,0.,-.100]),0,np.array([math.sin(theta),0.,math.cos(theta)]),None,.4)


def seeds_for(data,side,point):
    shoulder=data.body(side+'_shoulder').xpos
    pan=math.atan2(float(shoulder[0]-point[0]),float(shoulder[1]-point[1]))
    seeds=[]
    for roll in (0.,-2.4,2.4):
        q=np.array(HOME[:5]);q[4]=roll;seeds.append(q)
        seeds.append(np.array([0.,.8,.4,-1.2,roll]))
        seeds.append(np.array([pan,1.5,1.3,-.7,roll]))
    return seeds


def gap_check(task,data):
    model=task.model;object_geoms=[g for g in range(model.ngeom)
        if model.geom_bodyid[g]==task.body and (model.geom_contype[g] or model.geom_conaffinity[g])]
    closest={};segment=np.empty(6)
    for group,geoms in (('fixed',task.fixed_geoms),('moving',task.moving_geoms)):
        best={'gap_m':.005,'finger_geom':None,'object_geom':None}
        for finger in sorted(geoms):
            for obj in object_geoms:
                dist=float(mujoco.mj_geomDistance(model,data,finger,obj,.005,segment))
                if dist<best['gap_m']:
                    best={'gap_m':dist,'finger_geom':model.geom(finger).name or model.body(model.geom_bodyid[finger]).name,
                        'object_geom':model.geom(obj).name,'surface_points_world':segment.tolist()}
        closest[group]=best
    contacts=[];external=0.;finger_depth=0.
    for c in data.contact:
        a,b=int(c.geom1),int(c.geom2)
        if task.body not in (model.geom_bodyid[a],model.geom_bodyid[b]):continue
        other=b if model.geom_bodyid[a]==task.body else a
        if other in task.jaw_geoms:finger_depth=max(finger_depth,-float(c.dist))
        else:external=max(external,-float(c.dist))
        contacts.append({'a':model.geom(a).name or model.body(model.geom_bodyid[a]).name,
            'b':model.geom(b).name or model.body(model.geom_bodyid[b]).name,
            'distance_m':float(c.dist),'point_world':c.pos.tolist()})
    collision=task._collision(data,True,.00015)
    both=all(-.001<=p['gap_m']<=.0002 for p in closest.values())
    return {'both_fingers_within_geometric_gap':both,'closest_finger_surface':closest,
        'maximum_finger_penetration_mm':finger_depth*1000,'maximum_external_penetration_mm':external*1000,
        'unexpected_robot_contact':collision,'fork_contacts':contacts,
        'clear_endpoint':not collision and both and finger_depth<=.001 and external<=.001}


def diagonal_endpoint_gate(model,data,layout,record):
    """Inspect only provided scratch pose; no stepping, attachment or force."""
    rows=[];clear=[]
    for side in ('left','right'):
        task=TableTeacher(model,data,layout);task.start(side=side,object_id='fork')
        task._select_item(side,next(x for x in layout['objects'] if x['id']=='fork'))
        for tilt,candidate in endpoint_family(data):
            task._configure(candidate)
            row={'arm':side,'candidate':candidate.name,'tilt_deg':tilt,'point_m':candidate.point.tolist(),
                'local_tool_point_m':candidate.local_tool_point.tolist(),'closing_axis_world':candidate.axis_target.tolist(),
                'seed_attempts':[],'unique_solutions':[]}
            solutions=[]
            for index,initial in enumerate(seeds_for(data,side,candidate.point)):
                seed={'index':index,'initial':initial.tolist()}
                try:
                    q=task.ik.solve(candidate.point,initial);seed.update(solved=True,q=q.tolist())
                    duplicate=next((i for i,old in enumerate(solutions) if np.max(np.abs(q-old))<1e-4),None)
                    seed['duplicate_of_unique_solution']=duplicate
                    if duplicate is None:solutions.append(q.copy())
                except PlanningError as exc:
                    seed.update(solved=False,error=str(exc),last_point_error_mm=float(np.linalg.norm(task.ik.point(task.ik.data)-candidate.point)*1000))
                row['seed_attempts'].append(seed)
            for index,q in enumerate(solutions):
                solution={'q':q.tolist(),'grip_checks':[]}
                for grip in np.linspace(-.08,.4,25):
                    d=task.ik.data;d.qpos[:]=data.qpos;d.qvel[:]=0.;d.qpos[task.offset:task.offset+5]=q;d.qpos[task.offset+5]=grip
                    mujoco.mj_forward(model,d)
                    check=gap_check(task,d);check['grip_rad']=float(grip);solution['grip_checks'].append(check)
                    if check['clear_endpoint']:
                        clear.append({'record':len(rows),'solution':index,'grip_rad':float(grip),'arm':side,'candidate':candidate.name})
                row['unique_solutions'].append(solution)
            record(len(rows),row);rows.append(row)
    return rows,clear
