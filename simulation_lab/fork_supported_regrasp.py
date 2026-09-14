"""Unpromoted fork head pinch using clearance beneath inverted thin tines.

Planning poses are confined to scratch data. This does not classify scenes as
unreachable and does not establish a physical grasp by solving inverse kinematics.
"""
import math
import itertools
import mujoco
import numpy as np
from .autonomy import PlanningError
from .scene import HOME, TABLE_Z
from .random_dinner import body_bounds
from .table_teacher import TableTeacher, GraspCandidate, rotation_z, grasp_candidates
from .table_regrasp import copied_data


def head_candidates(data):
    body=data.body('fork');r=body.xmat.reshape(3,3)
    # Inverted fork head is 8 mm thick and tines 3 mm, while its table-supported
    # handle is 16 mm thick. The resulting gap is real model geometry.
    for y in (.032,.044,.024):
        for tool_x in (-.009,-.013):
            point=body.xpos+r@np.array([0.,y,.004])
            yield GraspCandidate(f'fork_under_head_{y}_{tool_x}',point,
                np.array([tool_x,0.,-.100]),0,np.array([0.,0.,1.]),None,.5,
                approach_direction=r[:,1].copy())


def head_grasp_gate(model,data,layout,*,full_routes=True):
    rows=[];found=[]
    for side in ('left','right'):
        task=TableTeacher(model,data,layout);task.start(side=side,object_id='fork')
        task._select_item(side,next(o for o in layout['objects'] if o['id']=='fork'))
        for candidate in head_candidates(data):
            for roll in (0.,-2.4,2.4):
                row={'arm':side,'candidate':candidate.name,'roll':roll,'check':'contact_ik'}
                try:
                    task._configure(candidate);initial=np.array(HOME[:5]);initial[4]=roll
                    q=task.ik.solve(candidate.point,initial)
                    row['check']='contact_clearance';task._check_path([q],candidate.open_grip,True)
                    row['check']='final_setting_geometry';task._precheck_transfer(q)
                    task._configure(candidate)
                    axis=task.ik.data.body(side+'_gripper').xmat.reshape(3,3)[:,2].copy()
                    # Actual solved palm bearing; approach from its side rather
                    # than prescribing a yaw unrelated to a five-joint arm.
                    task.ik.data.qpos[task.ik.indices]=q;mujoco.mj_forward(model,task.ik.data)
                    axis=task.ik.data.body(side+'_gripper').xmat.reshape(3,3)[:,2].copy()
                    hover=candidate.point+axis*.030
                    row['check']='hover_ik';above=task.ik.solve(hover,q)
                    row['check']='descent_clearance';descent=task._cartesian(hover,candidate.point,above,candidate.open_grip)
                    row['check']='route';approach=task._route(data.qpos[task.ik.indices],above,candidate.open_grip) if full_routes else None
                    row.update(planned=True,q=q.tolist(),point=candidate.point.tolist(),hover=hover.tolist(),axis=axis.tolist())
                    found.append({'side':side,'candidate':candidate,'q':q,'hover':hover,'descent':descent,'approach':approach,'roll':roll})
                    rows.append(row);return found,rows
                except PlanningError as exc:
                    row.update(planned=False,error=str(exc),nearest_q=task.ik.data.qpos[task.ik.indices].tolist())
                rows.append(row)
    return found,rows


class ForkHeadTeacher(TableTeacher):
    def _plan(self,targets):
        found,rows=head_grasp_gate(self.model,self.data,self.layout)
        self.search_log.extend(rows)
        if not found:raise PlanningError('Supported inverted-fork head pinch has no checked route.')
        plan=found[0];self._select_item(plan['side'],next(o for o in self.layout['objects'] if o['id']=='fork'))
        self._configure(plan['candidate']);self.chosen=plan['candidate'];self.chosen_roll=plan['roll']
        self.grasp=self.chosen.point.copy();self.hover=plan['hover'];self.approach_descent=plan['descent'];self.attempts=1
        self._move('approach',plan['approach'],self.open_grip,3.)


def buffer_poses(model,data):
    """Small tabletop support grid, yaw derived from the shoulder bearings.

    These are hypothetical static planning placements, never reached states.
    """
    destination=np.array([-.160,-.045,TABLE_Z]);flip=np.diag([1.,-1.,-1.])
    for side in ('left','right'):
        shoulder=data.body(side+'_shoulder').xpos
        target_bearing=math.atan2(destination[1]-shoulder[1],destination[0]-shoulder[0])
        for x,y in ((-.22,-.10),(-.16,-.10),(-.10,-.10),(.05,-.10),(.13,-.10),(.20,-.10)):
            source_bearing=math.atan2(y-shoulder[1],x-shoulder[0])
            yaw=source_bearing-target_bearing
            for turn in (0.,math.pi):
                r=rotation_z(yaw+turn)@flip;quat=np.empty(4);mujoco.mju_mat2Quat(quat,r.ravel())
                low,_=body_bounds(model,'fork',quat)
                yield {'center':np.array([x,y,TABLE_Z-low[2]]),'quaternion':quat,'yaw':yaw+turn,'receiver':side}


def side_candidates(data):
    body=data.body('fork');r=body.xmat.reshape(3,3)
    sign=np.sign(r[2,0])
    for y in (-.015,-.030,0.):
        for x in (.0035,.002):
            for closing_sign in (1,-1):
                yield GraspCandidate(f'side_handle_{y}_{x}_{closing_sign}',
                    body.xpos+r@np.array([sign*x,y,.008]),np.array([-.003,0.,-.100]),
                    2,np.array([0.,0.,1.]),closing_sign*r[:,2])


def plan_candidates(task,candidates,side):
    task._select_item(side,next(o for o in task.layout['objects'] if o['id']=='fork'))
    rows=[]
    for candidate in candidates:
        for roll in (0.,-2.4,2.4):
            row={'arm':side,'candidate':candidate.name,'roll':roll,'check':'contact_ik'}
            try:
                task._configure(candidate);initial=np.array(HOME[:5]);initial[4]=roll
                q=task.ik.solve(candidate.point,initial)
                row['check']='contact_clearance';task._check_path([q],candidate.open_grip,True)
                row['check']='final_setting_geometry';task._precheck_transfer(q)
                task._configure(candidate);hover=candidate.point+[0.,0.,.025]
                row['check']='hover_ik';above=task.ik.solve(hover,q)
                row['check']='descent_clearance';descent=task._cartesian(hover,candidate.point,above,candidate.open_grip)
                row['check']='route';approach=task._route(task.data.qpos[task.ik.indices],above,candidate.open_grip)
                row.update(planned=True,q=q.tolist());rows.append(row)
                return {'side':side,'candidate':candidate,'q':q,'hover':hover,'descent':descent,'approach':approach,'roll':roll},rows
            except PlanningError as exc:row.update(planned=False,error=str(exc))
            rows.append(row)
    return None,rows


def side_buffer_gate(model,data,layout):
    rows=[]
    for yaw in (0.,math.pi/2,-math.pi/2,math.pi):
        for sign in (1,-1):
            ry=np.array([[0.,0.,sign],[0.,1.,0.],[-sign,0.,0.]])
            r=rotation_z(yaw)@ry;quat=np.empty(4);mujoco.mju_mat2Quat(quat,r.ravel())
            low,_=body_bounds(model,'fork',quat)
            for x,y in ((-.16,-.045),(-.22,-.10),(-.10,-.10),(-.16,.03)):
                pos=np.array([x,y,TABLE_Z-low[2]])
                scratch=copied_data(model,data);scratch.joint('fork_free').qpos[:3]=pos;scratch.joint('fork_free').qpos[3:]=quat;mujoco.mj_forward(model,scratch)
                row={'yaw':yaw,'side_sign':sign,'center':pos.tolist(),'quaternion':quat.tolist(),'receiver_checks':[]}
                receiver=None
                for side in ('left','right'):
                    task=TableTeacher(model,scratch,layout);task.start(side=side,object_id='fork')
                    receiver,checks=plan_candidates(task,list(side_candidates(scratch)),side)
                    row['receiver_checks'].extend(checks)
                    if receiver:break
                if receiver:
                    row['receiver_planned']=True;row['donor_checks']=[]
                    for side in ('left','right'):
                        donor=TableTeacher(model,data,layout);donor.start(side=side,object_id='fork',target=pos,target_quaternion=quat)
                        plan,checks=plan_candidates(donor,grasp_candidates(data,'fork'),side)
                        row['donor_checks'].extend(checks)
                        if plan:
                            row['donor_planned']=True;rows.append(row)
                            return {'position':pos,'quaternion':quat,'donor':plan,'receiver':receiver},rows
                rows.append(row)
    return None,rows


def final_horizontal_gate(model,data,layout):
    """Reverse endpoint gate for a final face-up pinch; no physical closure."""
    scratch=copied_data(model,data);scratch.joint('fork_free').qpos[:]=[-.16,-.045,TABLE_Z,1.,0.,0.,0.];mujoco.mj_forward(model,scratch)
    rows=[];found=[]
    for side in ('left','right'):
        task=TableTeacher(model,scratch,layout);task.start(side=side,object_id='fork');task._select_item(side,next(o for o in layout['objects'] if o['id']=='fork'))
        for y in (-.015,.0,.032):
            z=.008 if y<.02 else .004
            point=np.array([-.16,-.045+y,TABLE_Z+z])
            for tx in (-.013,-.003,.005):
                for sign in (-1,1):
                    candidate=GraspCandidate(f'final_horizontal_{y}_{tx}_{sign}',point,np.array([tx,0.,-.100]),0,np.array([0.,0.,float(sign)]),None,.4)
                    row={'arm':side,'candidate':candidate.name,'point':point.tolist(),'ik_solved':False,'grip_checks':[]}
                    task._configure(candidate);q=None
                    shoulder=scratch.body(side+'_shoulder').xpos
                    folded_pan=math.atan2(float(shoulder[0]-point[0]),float(shoulder[1]-point[1]))
                    seeds=[]
                    for roll in (0.,-2.4,2.4):
                        initial=np.array(HOME[:5]);initial[4]=roll;seeds.append(initial)
                        seeds.append(np.array([0.,.8,.4,-1.2,roll]))
                        seeds.append(np.array([folded_pan,1.5,1.3,-.7,roll]))
                    row['initial_seeds']=len(seeds)
                    for initial in seeds:
                        try:q=task.ik.solve(point,initial);break
                        except PlanningError:pass
                    if q is None:
                        row['last_attempt_point_error_mm']=float(np.linalg.norm(task.ik.point(task.ik.data)-point)*1000)
                        row['last_attempt_q']=task.ik.data.qpos[task.ik.indices].tolist()
                        rows.append(row);continue
                    row['ik_solved']=True;row['q']=q.tolist()
                    for grip in np.linspace(-.08,.4,25):
                        d=task.ik.data;d.qpos[:]=scratch.qpos;d.qpos[task.offset:task.offset+5]=q;d.qpos[task.offset+5]=grip;mujoco.mj_forward(model,d)
                        collision=task._collision(d,True);finger_contact={'fixed':False,'moving':False};contacts=[]
                        for c in d.contact:
                            a,b=int(c.geom1),int(c.geom2)
                            if c.dist>=.0002 or task.body not in (model.geom_bodyid[a],model.geom_bodyid[b]):continue
                            other=b if model.geom_bodyid[a]==task.body else a
                            if other in task.fixed_geoms:finger_contact['fixed']=True
                            if other in task.moving_geoms:finger_contact['moving']=True
                            contacts.append({'a':model.geom(a).name,'b':model.geom(b).name,'penetration_mm':-float(c.dist)*1000})
                        good=not collision and all(finger_contact.values()) and all(c['penetration_mm']<1. for c in contacts)
                        row['grip_checks'].append({'grip':float(grip),'collision':collision,'contacts':contacts,'both_contact_geometry':all(finger_contact.values()),'candidate':good})
                        if good:found.append({'side':side,'q':q,'grip':float(grip),'candidate':candidate})
                    rows.append(row)
    return found,rows


def analytic_side_support(model):
    """Rigid box support hull, not physical settling/release verification."""
    verts=[];body=model.body('fork').id
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid]!=body or not model.geom_contype[gid]:continue
        assert model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX
        r=np.empty(9);mujoco.mju_quat2Mat(r,model.geom_quat[gid])
        for signs in itertools.product((-1,1),repeat=3):verts.append(model.geom_pos[gid]+r.reshape(3,3)@(model.geom_size[gid]*signs))
    verts=np.asarray(verts);com=model.body('fork').ipos;results=[]
    for sign in (-1,1):
        base=np.array([[0.,0.,sign],[0.,1.,0.],[-sign,0.,0.]])
        angles={-.2,0.,.2}
        for a,b in itertools.combinations(verts,2):
            delta=a-b
            if np.linalg.norm(delta[:2])<1e-10:continue
            angle=math.atan2(delta[0],delta[1])
            for candidate in (angle,angle+math.pi,angle-math.pi):
                if -.2<candidate<.2:angles.add(candidate)
        def energy(angle):
            r=base@rotation_z(angle);return -(verts@r.T)[:,2].min()+(r@com)[2]
        angle=min(angles,key=energy);r=base@rotation_z(angle);world=verts@r.T;low=world[:,2].min()
        points=sorted(set(tuple(p[:2]) for p in world if abs(p[2]-low)<1e-8))
        def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        lower=[];upper=[]
        for p in points:
            while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
            lower.append(p)
        for p in reversed(points):
            while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
            upper.append(p)
        hull=lower[:-1]+upper[:-1];c=(r@com)[:2]
        inside=len(hull)>=3 and all(cross(a,b,c)>=-1e-10 for a,b in zip(hull,hull[1:]+hull[:1]))
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,r.ravel())
        results.append({'side_sign':sign,'lengthwise_tilt_deg':math.degrees(angle),'quaternion':quat.tolist(),
            'body_origin_height_m':TABLE_Z-low,'support_hull_xy_relative':hull,'com_xy_relative':c.tolist(),
            'com_inside_theoretical_support_hull':bool(inside),'physical_settling_verified':False})
    return results
