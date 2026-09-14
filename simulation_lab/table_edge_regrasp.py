"""Geometry-first development of ordinary-support edge cutlery regrasp.

Hypothetical poses are written only to private planning MjData. A supported
overhang is temporary and never satisfies final full-table placement.
"""
import math
import mujoco
import numpy as np

from .autonomy import OPEN,PlanningError
from .scene import HOME,TABLE_Z
from .table_teacher import TableTeacher,GraspCandidate
from .table_regrasp import copied_data


def edge_candidates(model,data,name='spoon'):
    table=model.geom('table').id
    center=data.geom_xpos[table];half=model.geom_size[table]
    front=float(center[1]-half[1]);rotation=np.diag([-1.,1.,-1.])
    quaternion=np.array([0.,0.,1.,0.])
    for inward in (.020,.025,.015,.030):
        for x in np.linspace(float(center[0]-half[0]+.08),float(center[0]+half[0]-.08),13):
            yield {'edge':'front','edge_y':front,'inward_m':inward,
                'center':np.array([x,front+inward,TABLE_Z+.016]),
                'rotation':rotation,'quaternion':quaternion}


def assess_edge_receiver(model,data,layout,name='spoon',*,check_routes=True,poses=None,pitches=(0.,)):
    """Search two arms and outward approaches without claiming physical reach."""
    rows=[];found=[]
    item=next(x for x in layout['objects'] if x['id']==name)
    for candidate in edge_candidates(model,data,name) if poses is None else poses:
        scratch=copied_data(model,data)
        pose=scratch.joint(name+'_free').qpos
        pose[:3]=candidate['center'];pose[3:]=candidate['quaternion']
        mujoco.mj_forward(model,scratch)
        for side in ('left','right'):
            teacher=TableTeacher(model,scratch,layout);teacher._select_item(side,item)
            shoulder=scratch.body(side+'_shoulder').xpos
            for ly in (-.040,-.048):
                point=candidate['center']+candidate['rotation']@np.array([0.,ly,.008])
                radial=point-shoulder;radial[2]=0.;radial/=np.linalg.norm(radial)
                horizontal=[('front_normal',np.array([0.,-1.,0.])),('shoulder_radial',radial),('shoulder_inward',-radial)]
                directions=[(f'{label}_pitch_{pitch}',axis*math.cos(pitch)+np.array([0.,0.,math.sin(pitch)]),
                             np.array([0.,0.,math.cos(pitch)])-axis*math.sin(pitch))
                            for label,axis in horizontal for pitch in pitches if label!='shoulder_inward' or pitch>0.]
                for bearing,axis,closing in directions:
                    grasp=GraspCandidate(f'edge_{bearing}_{ly}',point,np.array([-.003,0.,-.100]),2,axis,closing)
                    teacher._configure(grasp)
                    row={'center':candidate['center'].tolist(),'edge_y':candidate['edge_y'],
                        'com_inside_edge_mm':float((scratch.body(name).xipos[1]-candidate['edge_y'])*1000),
                        'exposed_grasp_outside_edge_mm':float((candidate['edge_y']-point[1])*1000),
                        'side':side,'candidate':grasp.name,'axis':axis.tolist(),'point':point.tolist()}
                    row['shoulder_xy']=shoulder[:2].tolist()
                    row['shoulder_to_contact_world_bearing_deg']=math.degrees(math.atan2(float(point[1]-shoulder[1]),float(point[0]-shoulder[0])))
                    row['shoulder_to_contact_arm_pan_bearing_deg']=math.degrees(math.atan2(float(point[0]-shoulder[0]),float(point[1]-shoulder[1])))
                    try:
                        solutions=[]
                        folded_pan=math.atan2(float(shoulder[0]-point[0]),float(shoulder[1]-point[1]))
                        seeds=[]
                        for roll in (0.,-2.4,2.4):
                            initial=np.array(HOME[:5]);initial[4]=roll;seeds.append(initial)
                            seeds.append(np.array([folded_pan,1.5,1.3,-.7,roll]))
                            seeds.append(np.array([folded_pan,1.2,1.4,-1.2,roll]))
                        for initial in seeds:
                            try:solutions.append(teacher.ik.solve(point,initial))
                            except PlanningError:pass
                        if not solutions:raise PlanningError('No receiver IK solution from nine standard/folded-arm seeds.')
                        errors=[]
                        for q in solutions:
                            try:
                                teacher._check_path([q],OPEN,True)
                                hover=point+axis*.035
                                qhover=teacher.ik.solve(hover,q)
                                descent=teacher._cartesian(hover,point,qhover,OPEN)
                                route=teacher._route(scratch.qpos[teacher.ik.indices],qhover,OPEN) if check_routes else None
                                row.update(planned=True,q=q.tolist(),hover=hover.tolist())
                                found.append({'support_pose':candidate,'receiver':side,'grasp':grasp,'q':q,'approach':route,'descent':descent})
                                break
                            except PlanningError as exc:errors.append(str(exc))
                        if 'planned' not in row:raise PlanningError('; '.join(errors))
                    except PlanningError as exc:
                        row['error']=str(exc)
                        row['nearest_point']=teacher.ik.point(teacher.ik.data).tolist()
                        row['nearest_q']=teacher.ik.data.qpos[teacher.ik.indices].tolist()
                        r=teacher.ik.data.body(side+'_gripper').xmat.reshape(3,3)
                        row['tool_axis_error_deg']=float(np.degrees(np.arccos(np.clip(r[:,2]@axis,-1,1))))
                        row['closing_axis_error_deg']=float(np.degrees(np.arccos(np.clip(r[:,0]@closing,-1,1))))
                    rows.append(row)
        if found:return found,rows
    return found,rows


def supported_overhang(model,data,name='spoon'):
    """Actual table-contact polygon, support force, stability and overhang."""
    body=model.body(name).id;table=model.geom('table').id
    forces=0.;depth=0.;points=[];wrench=np.zeros(6)
    for index,c in enumerate(data.contact):
        a,b=int(c.geom1),int(c.geom2)
        if body not in (model.geom_bodyid[a],model.geom_bodyid[b]):continue
        other=b if model.geom_bodyid[a]==body else a
        depth=max(depth,-float(c.dist))
        if other==table:
            mujoco.mj_contactForce(model,data,index,wrench)
            forces+=max(0.,float(wrench[0]))
            if wrench[0]>.001:points.append(c.pos[:2].copy())
    # Convex hull of measured load-bearing contact positions.
    points=sorted(set(tuple(p) for p in points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    low=[];high=[]
    for p in points:
        while len(low)>1 and cross(low[-2],low[-1],p)<=0:low.pop()
        low.append(p)
    for p in reversed(points):
        while len(high)>1 and cross(high[-2],high[-1],p)<=0:high.pop()
        high.append(p)
    hull=low[:-1]+high[:-1];com=data.body(name).xipos[:2]
    inside=len(hull)>=3 and all(cross(a,b,com)>=-1e-10 for a,b in zip(hull,hull[1:]+hull[:1]))
    dof=int(model.joint(name+'_free').dofadr[0]);speed=np.linalg.norm(data.qvel[dof:dof+3]);angular=np.linalg.norm(data.qvel[dof+3:dof+6])
    return {'table_support_force_n':forces,'external_penetration_mm':depth*1000,'contact_polygon':hull,
        'com_xy':com.tolist(),'com_within_support_polygon':bool(inside),'speed_mm_s':float(speed*1000),
        'angular_speed_rad_s':float(angular),'supported_stable':bool(inside and forces>.06 and speed<.003 and angular<.08 and depth<=.001)}
