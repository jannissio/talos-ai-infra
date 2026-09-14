"""Unpromoted reset-only fork support on the unchanged hollow bottle rim."""
import math
import mujoco
import numpy as np
from .autonomy import PlanningError
from .scene import HOME
from .random_dinner import body_bounds
from .table_teacher import TableTeacher,GraspCandidate,rotation_z
from .fork_diagonal_pinch import gap_check


class RimPlanningTeacher(TableTeacher):
    def _check_path(self,points,grip,allow_tube=False,carry=None,support=None):
        if not isinstance(support,frozenset):return super()._check_path(points,grip,allow_tube,carry,support)
        # Explicit intentional support is the unchanged entire bottle rim,
        # whose collision mesh consists of multiple separate primitive geoms.
        scratch=self.ik.data;scratch.qpos[:]=self.data.qpos;scratch.qvel[:]=0.
        for q in points:
            scratch.qpos[self.offset:self.offset+5]=q;scratch.qpos[self.offset+5]=grip
            mujoco.mj_kinematics(self.model,scratch);r=scratch.body(self.side+'_gripper').xmat.reshape(3,3)
            pose=scratch.joint('fork_free').qpos;pose[:3]=self.ik.point(scratch)+r@carry[0]
            mujoco.mju_mat2Quat(pose[3:],(r@carry[1]).ravel());mujoco.mj_forward(self.model,scratch)
            collision=self._collision(scratch,allow_tube)
            if collision:raise PlanningError('Supported lift robot collision: '+collision)
            for c in scratch.contact:
                a,b=int(c.geom1),int(c.geom2)
                if self.body not in (self.model.geom_bodyid[a],self.model.geom_bodyid[b]):continue
                other=b if self.model.geom_bodyid[a]==self.body else a
                if other in self.jaw_geoms:continue
                tolerance=.001 if other in support else .00015
                if c.dist < -tolerance:raise PlanningError('Supported lift external collision: '+str(self.model.geom(other).name))


def support_poses(model,data):
    bottle=data.body('bottle');bq=data.joint('bottle_free').qpos[3:]
    _,high=body_bounds(model,'bottle',bq);top=bottle.xpos[2]+high[2]
    com=model.body('fork').ipos
    for side in ('left','right'):
        vector=bottle.xpos[:2]-data.body(side+'_shoulder').xpos[:2];vector/=np.linalg.norm(vector)
        yaw=math.atan2(vector[0],-vector[1]);r=rotation_z(yaw)@np.diag([1.,-1.,-1.])
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,r.ravel());low,_=body_bounds(model,'fork',quat)
        center=np.r_[bottle.xpos[:2]-(r@com)[:2],top-low[2]+.001]
        yield {'receiver':side,'center':center,'quaternion':quat,'rotation':r,'bottle_collision_top_m':float(top),'yaw_rad':yaw}


def support_metrics(model,data,others):
    fork=model.body('fork').id;bottle=model.body('bottle').id
    load=0.;external=0.;bottle_depth=0.;points=[];contacts=[];wrench=np.empty(6);robot_collision=None;robot_depth=0.
    for index,c in enumerate(data.contact):
        a,b=int(c.geom1),int(c.geom2);ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
        if bottle in (ba,bb) and ba!=bb:bottle_depth=max(bottle_depth,-float(c.dist))
        robot=any((model.body(x).name or '').startswith(('left_','right_')) for x in (ba,bb))
        if robot and -float(c.dist)>robot_depth:
            robot_depth=-float(c.dist);robot_collision=[model.geom(a).name or model.body(ba).name,model.geom(b).name or model.body(bb).name]
        if fork not in (ba,bb):continue
        external=max(external,-float(c.dist));mujoco.mj_contactForce(model,data,index,wrench)
        if bottle in (ba,bb):
            load+=max(0.,float(wrench[0]))
            if wrench[0]>.001:points.append(tuple(c.pos[:2]))
        contacts.append({'a':model.geom(a).name,'b':model.geom(b).name,'distance_m':float(c.dist),'normal_force_n':float(wrench[0]),'point_world':c.pos.tolist()})
    points=sorted(set(points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=lower[:-1]+upper[:-1];com=data.body('fork').xipos[:2]
    inside=len(hull)>=3 and all(cross(a,b,com)>=-1e-10 for a,b in zip(hull,hull[1:]+hull[:1]))
    dof=int(model.joint('fork_free').dofadr[0]);speed=float(np.linalg.norm(data.qvel[dof:dof+3]));angular=float(np.linalg.norm(data.qvel[dof+3:dof+6]))
    moves={n:float(np.linalg.norm(data.body(n).xpos-pos))*1000 for n,pos in others.items()}
    stable=inside and load>.06 and speed<.003 and angular<.08 and external<=.001 and bottle_depth<=.001 and max(moves.values())<=4. and robot_depth<=.0008
    return {'bottle_support_normal_force_n':load,'contact_hull_xy':hull,'com_xy':com.tolist(),'com_inside_load_bearing_hull':inside,
        'fork_speed_mm_s':speed*1000,'fork_angular_speed_rad_s':angular,'fork_external_penetration_mm':external*1000,
        'bottle_external_penetration_mm':bottle_depth*1000,
        'robot_penetration_mm':robot_depth*1000,'robot_contact':robot_collision,'other_displacement_mm':moves,
        'stable':bool(stable),'contacts':contacts}


def receiver_gate(model,data,layout,side):
    task=RimPlanningTeacher(model,data,layout);task.start(side=side,object_id='fork')
    task._select_item(side,next(o for o in layout['objects'] if o['id']=='fork'))
    body=data.body('fork');r=body.xmat.reshape(3,3);rows=[]
    for y in (-.045,-.030,.040):
        for sign in (-1,1):
            point=body.xpos+r@np.array([0.,y,.008 if y<0 else .004])
            candidate=GraspCandidate(f'under_rim_{y}_{sign}',point,np.array([-.003,0.,-.100]),2,np.array([0.,0.,-1.]),sign*r[:,0],.4)
            for roll in (0.,-2.4,2.4):
                row={'receiver':side,'candidate':candidate.name,'roll':roll,'check':'contact_ik'}
                try:
                    task._configure(candidate);initial=np.array(HOME[:5]);initial[4]=roll
                    q=task.ik.solve(point,initial);row['q']=q.tolist()
                    row['check']='open_contact_clearance';task._check_path([q],.4,True)
                    row['check']='static_jaw_gap';row['jaw_checks']=[];grips=[]
                    for grip in np.linspace(-.08,.4,25):
                        d=task.ik.data;d.qpos[:]=data.qpos;d.qpos[task.offset:task.offset+5]=q;d.qpos[task.offset+5]=grip;mujoco.mj_forward(model,d)
                        check=gap_check(task,d);check['grip_rad']=float(grip);row['jaw_checks'].append(check)
                        if check['clear_endpoint']:grips.append(float(grip))
                    if not grips:raise PlanningError('No static two-finger gap/penetration configuration.')
                    row['check']='final_placement_endpoint';task._precheck_transfer(q)
                    task._configure(candidate);hover=point-[0.,0.,.025]
                    row['check']='underneath_hover_ik';above=task.ik.solve(hover,q)
                    row['check']='underneath_approach_clearance';descent=task._cartesian(hover,point,above,.4)
                    row['check']='park_to_underneath_route';approach=task._route(data.qpos[task.ik.indices],above,.4)
                    grip=grips[len(grips)//2];reference=task._carry_reference(q)
                    row['check']='lift_with_bottle_support_clearance'
                    lift=task._cartesian(point,point+[0.,0.,.04],q,grip,check=False)
                    bottle_geoms=frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('bottle').id)
                    task._check_path(lift[:2],grip,True,carry=reference,support=bottle_geoms)
                    task._check_path(lift[2:],grip,True,carry=reference)
                    row.update(planned=True,grip_rad=grip,hover=hover.tolist());rows.append(row)
                    return {'receiver':side,'candidate':candidate.name,'q':q.tolist(),'grip_rad':grip},rows
                except PlanningError as exc:row.update(planned=False,error=str(exc))
                rows.append(row)
    return None,rows
