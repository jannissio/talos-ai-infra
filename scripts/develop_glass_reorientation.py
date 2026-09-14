"""Exposed glass reorientation diagnostic with immutable sources and exact replay."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put, arrays


def run(args):
    folder=args.output.resolve(); log=folder.with_suffix('.log')
    if folder.exists() or log.exists(): raise FileExistsError('Preserve both output and log.')
    preflight=require_space(folder,256*1024**2); folder.mkdir(parents=True)
    with log.open('x',encoding='utf8') as stream,contextlib.redirect_stdout(stream):
        started=time.perf_counter(); manifest={}
        for name in ('simulation_lab/table_teacher.py','simulation_lab/autonomy.py',
                'simulation_lab/dinner_autonomy.py','simulation_lab/random_dinner.py',
                'simulation_lab/dinner.py','simulation_lab/scene.py','simulation_lab/side_plate_grasp_candidates.py',
                'simulation_lab/glass_grasp_candidates.py',
                'scripts/develop_side_plate_grasp.py','scripts/develop_glass_reorientation.py',
                'scripts/develop_glass_placement.py','scripts/develop_glass_bridge_physics.py'):
            source=Path(args.teacher_source) if name=='simulation_lab/table_teacher.py' and args.teacher_source else ROOT/name
            payload=source.read_bytes();put(folder/'source'/name,payload)
            manifest[name]=hashlib.sha256(payload).hexdigest()
        put(folder/'source-manifest.json',manifest)
        module_name='simulation_lab._glass_snapshot'
        spec=importlib.util.spec_from_file_location(module_name,folder/'source/simulation_lab/table_teacher.py')
        teacher=importlib.util.module_from_spec(spec);sys.modules[module_name]=teacher;spec.loader.exec_module(teacher)
        horizontal_proof=None
        bridge_proof=None
        if getattr(args,'post_hold_bridge',None):
            bridge_proof=json.loads(Path(args.post_hold_bridge).read_text());put(folder/'post-hold-bridge-proof.json',bridge_proof)
        if getattr(args,'horizontal_proof',None):
            horizontal_proof=json.loads(Path(args.horizontal_proof).read_text());put(folder/'horizontal-proof.json',horizontal_proof)
            class GoalSeedIK(teacher.TableIK):
                def solve(self,position,initial):
                    try:return super().solve(position,initial)
                    except teacher.PlanningError:
                        if self.axis_index!=1:raise
                        return super().solve(position,np.asarray(horizontal_proof['q'][:5]))
            teacher.TableIK=GoalSeedIK
        from simulation_lab.scene import build_scene,HOME
        from simulation_lab.random_dinner import assess
        xml,layout=build_scene(seed=getattr(args,'seed',2026114002),scenario='dinner',dinner_preset='task')
        put(folder/'scene.xml',xml.encode());put(folder/'layout.json',layout)
        model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
        spec_state=mujoco.mjtState.mjSTATE_INTEGRATION
        source=Path(args.initial_from)
        with np.load(source/'states.npz') as archive:
            saved={key:archive[key] for key in ('initial_integration','ctrl','qpos','qvel','stage')}
        mujoco.mj_setState(model,data,saved['initial_integration'],spec_state);mujoco.mj_forward(model,data)
        prefix_replay=bool(getattr(args,'replay_prefix',False))
        if prefix_replay:
            for index,ctrl in enumerate(saved['ctrl']):
                data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
                assert np.array_equal(data.qpos,saved['qpos'][index+1]) and np.array_equal(data.qvel,saved['qvel'][index+1]),'Motor-only prefix replay diverged.'
        put(folder/'reset-provenance.json',dict(source=source.as_posix(),states_sha256=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest(),prefix_motor_replay_exact=prefix_replay,reset='Exact saved initial integration state; optional full saved motor prefix replay; no later state assignments.'))
        put(folder/'initial-geometry.json',assess(model,data))
        assert np.max(np.abs(model.actuator_forcerange))<=2.94+1e-8
        original_candidates=teacher.grasp_candidates
        family_spec=importlib.util.spec_from_file_location('frozen_glass_candidates',folder/'source/simulation_lab/glass_grasp_candidates.py')
        glass_family=importlib.util.module_from_spec(family_spec);family_spec.loader.exec_module(glass_family)
        def candidates(data,name):
            if horizontal_proof is not None:
                body=data.body(name);axis=np.asarray(horizontal_proof['object_rotation_in_gripper'])[:,2].copy()
                candidate=teacher.GraspCandidate('bridge_horizontal_proof001',body.xpos+body.xmat.reshape(3,3)@np.array([0.,0.,args.rim_height]),
                    np.asarray(horizontal_proof['object_origin_in_gripper'])+axis*.072,1,np.array([0.,0.,1.]),None,.85,None,.65)
                candidate.local_axis=axis;return [candidate]
            if getattr(args,'goal_contact',False):
                body=data.body(name)
                point=body.xpos+body.xmat.reshape(3,3)@np.array([0.,0.,getattr(args,'goal_contact_height',.072)])
                candidate=teacher.GraspCandidate('glass_goal_pitched_diameter',point,
                    np.array([.016844050389936402,-.0009278055682283223,-.0993204764644536]),
                    2,np.array([0.,0.,1.]),None,.85,None,args.torque)
                candidate.local_axis=np.array([-.009721405847552073,.5015516051577634,.8650731076805114])
                return [candidate]
            if getattr(args,'pitch',None) is not None:
                body=data.body(name);point=body.xpos+body.xmat.reshape(3,3)@np.array([0.,0.,args.rim_height])
                result=[]
                for side in ('right','left'):
                    radial=point-data.body(side+'_shoulder').xpos;radial[2]=0.;radial/=np.linalg.norm(radial)
                    long_axis=-radial*np.sin(args.pitch)+np.array([0.,0.,np.cos(args.pitch)])
                    closing=radial*np.cos(args.pitch)+np.array([0.,0.,np.sin(args.pitch)])
                    result.append(teacher.GraspCandidate('glass_pitched_wrap_'+side,point,np.array([args.tool_x,0.,args.tool_z]),2,long_axis,closing,.85,None,args.torque))
                return result
            if getattr(args,'refined_family',False):
                return glass_family.padded_glass_wrap_candidates(data,name,teacher.GraspCandidate)
            if getattr(args,'side_wrap',False):
                body=data.body(name);rotation=body.xmat.reshape(3,3)
                point=body.xpos+rotation@np.array([0.,0.,args.rim_height])
                return [teacher.GraspCandidate('glass_horizontal_wrap',point,
                    np.array([args.tool_x,0.,args.tool_z]),1,np.array([0.,0.,args.closing_sign]),
                    None,.85,None,args.torque)]
            if getattr(args,'rim_angle',None) is not None:
                angle=args.rim_angle
                body=data.body(name);rotation=body.xmat.reshape(3,3)
                radial=np.array([np.cos(angle),np.sin(angle),0.])
                point=body.xpos+rotation@(radial*args.rim_radius+[0.,0.,args.rim_height])
                return [teacher.GraspCandidate('glass_rim_diagnostic',point,
                    np.array([args.tool_x,0.,args.tool_z]),2,np.array([0.,0.,1.]),
                    args.closing_sign*(rotation@radial),teacher.OPEN,None,args.torque)]
            if args.grasp:
                selected=[x for x in original_candidates(data,name) if x.name==args.grasp]
                if args.torque is not None:
                    for x in selected:x.torque_limit_nm=args.torque
                for x in selected:
                    if args.tool_z is not None:x.local_tool_point[2]=args.tool_z
                    if args.tool_x is not None:x.local_tool_point[0]=args.tool_x
                return selected
            return original_candidates(data,name)
        teacher.grasp_candidates=candidates
        diagnostic=[]
        class PlaneIK(teacher.TableIK):
            horizontal_body_axis=None
            def solve(self,position,initial):
                if self.horizontal_body_axis is None:return super().solve(position,initial)
                weight=.05
                def kin(q):
                    self.data.qpos[self.indices]=q
                    mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
                    r=self.data.xmat[self.body].reshape(3,3)
                    return self.point(self.data),r,r@self.horizontal_body_axis
                def residual(qs):
                    values=[]
                    for q in qs.T:
                        point,r,axis=kin(q)
                        values.append(np.r_[point-position,weight*axis[2],args.roll_bias*r[2,0]])
                    return np.asarray(values).T
                def jacobian(q,_):
                    point,r,axis=kin(q.ravel())
                    mujoco.mj_jac(self.model,self.data,self.jp,self.jr,point,self.body)
                    return np.vstack((self.jp[:,self.indices],(-weight*teacher.skew(axis)@self.jr[:,self.indices])[2],(-args.roll_bias*teacher.skew(r[:,0])@self.jr[:,self.indices])[2]))
                q,_=teacher.minimize.least_squares(np.asarray(initial).copy(),residual,bounds=(self.lo,self.hi),jacobian=jacobian,max_iter=120,verbose=0)
                point,r,axis=kin(q)
                if np.linalg.norm(point-position)<.00025 and abs(axis[2])<np.sin(np.deg2rad(3)):
                    return q
                raise teacher.PlanningError('Horizontal-axis IK failed position or three-degree plane tolerance.')
        class GlassTeacher(teacher.TableTeacher):
            plane_active=False
            def _cartesian(self,start,end,initial,grip,check=True):
                try:return super()._cartesian(start,end,initial,grip,check)
                except teacher.PlanningError:
                    if self.stage!='release' or not getattr(args,'bridge_park_fallback',False):raise
                    path=self._route(initial,np.asarray(HOME[:5]),grip)
                    self.metrics['release_joint_space_park_fallback']=True
                    return path
            def _precheck_transfer(self,grasp_q):
                point=super()._precheck_transfer(grasp_q)
                if not getattr(args,'require_full_preflight',False):return point
                reference=self._carry_reference(grasp_q);self._place_axes(reference)
                source_center=self.data.body('glass').xpos.copy();errors=[]
                for height in (.065,.04,.10,.02):
                    record=dict(kind='complete_rigid_route_preflight',height=height,nominal_closed_grip=.44)
                    try:
                        _,lift=self._point_for_center(source_center+[0.,0.,.065],reference,grasp_q)
                        _,above=self._point_for_center(self.destination_position+[0.,0.,height],reference,lift)
                        _,lower=self._point_for_center(self.destination_position,reference,above)
                        self._edge(grasp_q,lift,.44,carry=reference,support=self.base_geom,allow_target=True)
                        route=self._route(lift,above,.44,carry=reference,allow_target=True)
                        self._edge(above,lower,.44,carry=reference,support=self.base_geom,allow_target=True)
                        record.update(passed=True,source_q=grasp_q.tolist(),lift_q=lift.tolist(),above_q=above.tolist(),lower_q=lower.tolist(),transfer_samples=len(route));diagnostic.append(record)
                        return point
                    except teacher.PlanningError as exc:
                        record.update(passed=False,error=str(exc));diagnostic.append(record);errors.append(str(exc))
                raise teacher.PlanningError('Required complete rigid carry route preflight failed: '+'; '.join(errors))
            def _finish(self,status,message,*remaining):
                if status=='succeeded' and self.plane_active:
                    plane_error=float(np.degrees(np.arcsin(np.clip(abs(self.data.body('glass').xmat.reshape(3,3)[2,2]),0.,1.))))
                    self.metrics['final_horizontal_plane_error_deg']=plane_error
                    if plane_error>3.:status='failed';message='Final glass axis exceeds three-degree horizontal-plane tolerance.'
                return super()._finish(status,message,*remaining)
            def _select_item(self,side,item):
                super()._select_item(side,item)
                if args.horizontal_plane:self.ik=PlaneIK(self.model,self.data,side)
                if args.side_axis:
                    self.placement_body_axis=2
                    self.placement_axis_target=self.destination_rotation[:,2].copy()
                    self.orientation_constraint='Horizontal glass body Z axis at specified bearing; axial roll free.'
            def _place_axes(self,reference):
                if self.plane_active:
                    self.ik.horizontal_body_axis=reference[1][:,2].copy()
                    self.ik.axis_local=reference[1][:,2].copy()
                    self.ik.axis_target=self.placement_axis_target.copy();self.ik.x_target=None
                else:super()._place_axes(reference)
            def _check_path(self,points,grip,allow_tube=False,carry=None,support=None):
                if not self.plane_active or not args.carry_clearance or carry is None or support is not None:
                    return super()._check_path(points,grip,allow_tube,carry,support)
                from simulation_lab.random_dinner import body_bounds
                for q in points:
                    super()._check_path([q],grip,allow_tube,carry,support)
                    pose=self.ik.data.joint('glass_free').qpos
                    low,_=body_bounds(self.model,'glass',pose[3:])
                    if pose[2]+low[2]<.76+args.carry_clearance:
                        raise teacher.PlanningError('Carried-glass table clearance below diagnostic margin.')
            def _lower(self):
                error=abs(self.data.body('glass').xmat.reshape(3,3)[2,2])
                count=self.metrics.get('measured_reorientation_corrections',0)
                if self.plane_active and error>np.sin(np.deg2rad(3)) and count<args.reorient_corrections:
                    self.metrics['measured_reorientation_corrections']=count+1
                    return self._place_route()
                return super()._lower()
            def update(self,targets):
                if args.carry_torque is not None and self.stage in ('hold','align','lower'):
                    self.grip_torque=args.carry_torque
                    self.metrics['carry_gripper_torque_limit_nm']=args.carry_torque
                if self.plane_active:
                    axis=self.data.body('glass').xmat.reshape(3,3)[:,2].copy();axis[2]=0.
                    if np.linalg.norm(axis)>1e-8:self.placement_axis_target=axis/np.linalg.norm(axis)
                return super().update(targets)
            def _place_route(self):
                if bridge_proof is not None and not self.metrics.get('post_hold_bridge_selected',False):
                    self.destination_position=np.asarray(bridge_proof['vertical']['target_position']).copy()
                    self.destination_quaternion=np.asarray(bridge_proof['vertical']['target_quaternion']).copy()
                    mujoco.mju_quat2Mat(self.destination_rotation.ravel(),self.destination_quaternion)
                    self.placement_axis_target=self.destination_rotation[:,2].copy()
                    self.destination={'id':'glass_place','position_m':self.destination_position.tolist()}
                    self.orientation_constraint='Explicit measured-tilt intermediate target; unchanged three-degree IK and physical guards.'
                    self._precheck_transfer(self.data.qpos[self.offset:self.offset+5].copy())
                    self.metrics['post_hold_bridge_selected']=True
                if args.post_hold_sideways:
                    self.plane_active=args.horizontal_plane
                    self.reference=self._carry_reference()
                    start=self.data.qpos[self.offset:self.offset+5].copy()
                    grip=float(self.data.qpos[self.offset+5])
                    from simulation_lab.random_dinner import body_bounds
                    positions=([.10,.02],[.05,-.06],[.15,-.04],[.20,.04],[.10,.12],[.25,.10],[.30,.0],[.0,-.10])
                    shoulder=self.data.body(self.side+'_shoulder').xpos
                    for position in positions:
                        radial=np.r_[np.asarray(position)-shoulder[:2],0.]
                        radial/=np.linalg.norm(radial)
                        for sign in ((1,) if self.plane_active else (1,-1)):
                            desired=radial*sign
                            self.placement_body_axis=2;self.placement_axis_target=desired
                            rotation=np.column_stack(([0.,0.,1.],np.cross(desired,[0.,0.,1.]),desired))
                            quat=np.empty(4);mujoco.mju_mat2Quat(quat,rotation.ravel())
                            low,_=body_bounds(self.model,'glass',quat)
                            self.destination_quaternion=quat;self.destination_rotation=rotation
                            self.destination_position=np.r_[position,.76-low[2]]
                            self.destination={'id':'glass_place','position_m':self.destination_position.tolist()}
                            self.orientation_constraint=('Glass body Z within three degrees of horizontal plane; bearing and axial roll free.' if self.plane_active else 'Horizontal glass body Z along chosen shoulder bearing; axial roll free.')
                            self._place_axes(self.reference)
                            for height in (.065,.10,.04):
                                center=self.destination_position+[0.,0.,height]
                                for index,initial in enumerate((start,np.asarray(HOME[:5]),np.array([0.,-.7,.8,.2,-2.4]),np.array([0.,-.7,.8,.2,2.4]))):
                                    record=dict(position=position,sign=sign,height=height,index=index)
                                    try:
                                        _,end=self._point_for_center(center,self.reference,initial)
                                        _,lower=self._point_for_center(self.destination_position,self.reference,end)
                                        self._check_path([lower],grip,True,carry=self.reference,support=self.base_geom)
                                        points=self._route(start,end,grip,carry=self.reference,allow_target=True)
                                        record.update(planned=True,q=end.tolist());diagnostic.append(record)
                                        self.metrics['post_hold_sideways_target']=self.destination_position.tolist()
                                        self.metrics['post_hold_sideways_axis']=desired.tolist()
                                        self._move('align',points,teacher.CLOSED,args.align_duration);return
                                    except teacher.PlanningError as exc:
                                        record.update(error=str(exc));diagnostic.append(record)
                    raise teacher.PlanningError('Post-hold sideways pose/path search exhausted; retained as unsolved.')
                try:return super()._place_route()
                except teacher.PlanningError as original:
                    if not args.branches:raise
                    self.reference=self._carry_reference();self._place_axes(self.reference)
                    start=self.data.qpos[self.offset:self.offset+5].copy()
                    grip=float(self.data.qpos[self.offset+5]); rng=np.random.default_rng(2026116001)
                    for height in (.065,.04,.10,.15):
                        center=self.destination_position+[0.,0.,height]
                        for index in range(args.branches):
                            initial=rng.uniform(self.ik.lo,self.ik.hi)
                            try:
                                _,end=self._point_for_center(center,self.reference,initial)
                                diagnostic.append(dict(height=height,index=index,ik=True,q=end.tolist()))
                                points=self._route(start,end,grip,carry=self.reference,allow_target=True)
                                self._move('align',points,teacher.CLOSED,4.);return
                            except teacher.PlanningError as exc:
                                axis=self.ik.data.xmat[self.ik.body].reshape(3,3)
                                axis=axis@self.ik.axis_local if self.ik.axis_local is not None else axis[:,self.ik.axis_index]
                                diagnostic.append(dict(height=height,index=index,error=str(exc),normal_error_deg=float(np.degrees(np.arccos(np.clip(np.dot(axis,self.ik.axis_target),-1,1)))),q=self.ik.data.qpos[self.offset:self.offset+5].tolist()))
                    raise teacher.PlanningError(str(original)+'; additional placement branches exhausted.')
        target_quat=([1.,0.,0.,0.] if args.mode=='upright' else data.joint('glass_free').qpos[3:].copy() if args.mode=='inverted' else np.array([np.sqrt(.5),0.,-np.sqrt(.5),0.]))
        if getattr(args,'explicit_target_quaternion',None) is not None:target_quat=np.asarray(args.explicit_target_quaternion)
        if args.side_axis:
            yaw=args.bearing
            # Body Z horizontal at bearing; body X vertical, axial roll unconstrained by our explicit axis choice.
            rz=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
            r=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
            target_quat=np.empty(4);mujoco.mju_mat2Quat(target_quat,(rz@r).ravel())
        targets=data.ctrl.copy(); task=GlassTeacher(model,data,layout)
        task.start(object_id='glass',side=args.arm,target=args.target,target_quaternion=target_quat)
        task.update(targets)
        put(folder/'parameters.json',{**vars(args),'output':str(folder),'preflight':preflight})
        put(folder/'search.json',task.search_log)
        initial=np.empty(mujoco.mj_stateSize(model,spec_state));mujoco.mj_getState(model,data,initial,spec_state)
        controls=[];qpos=[data.qpos.copy()];qvel=[data.qvel.copy()];stages=[task.stage];sensors=[];relative=[]
        previous_continuation=None
        if getattr(args,'continuation_from',None):
            with np.load(Path(args.continuation_from)/'states.npz') as archive:previous_continuation={key:archive[key] for key in ('ctrl','qpos','qvel')}
        last=None
        while task.active and len(controls)<24000:
            if task.stage!=last:print(dict(stage=task.stage,time=float(data.time)),flush=True);last=task.stage
            task.update(targets)
            tool=data.body(task.side+'_gripper');item=data.body('glass');r=tool.xmat.reshape(3,3)
            relative.append(np.r_[r.T@(item.xpos-tool.xpos),(r.T@item.xmat.reshape(3,3)).ravel(),r[:,2]])
            sensors.append([float(data.time),*task.metrics.get('finger_forces_n',[0.,0.]),task.metrics.get('external_contact_n',0.),task.metrics.get('destination_normal_error_deg',0.),task.metrics.get('lift_cm',0.)])
            if not task.active:break
            data.ctrl[:]=task.apply_gripper_limit(targets);controls.append(data.ctrl.copy())
            if previous_continuation is not None and len(controls)<=len(previous_continuation['ctrl']):
                index=len(controls)-1
                assert np.array_equal(data.ctrl,previous_continuation['ctrl'][index]) and np.array_equal(data.qpos,previous_continuation['qpos'][index]) and np.array_equal(data.qvel,previous_continuation['qvel'][index]),'Released-state continuation prefix diverged.'
            assert model.neq==0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            mujoco.mj_step(model,data);qpos.append(data.qpos.copy());qvel.append(data.qvel.copy());stages.append(task.stage)
        if task.active:task._finish('failed','Episode tick budget exhausted.',True)
        continuation_frames=len(qpos)
        if prefix_replay:
            initial=saved['initial_integration']
            controls=list(saved['ctrl'])+controls
            qpos=list(saved['qpos'])+qpos[1:];qvel=list(saved['qvel'])+qvel[1:]
            stages=list(saved['stage'])+stages[1:]
        arrays(folder/'states.npz',initial_integration=initial,ctrl=np.asarray(controls),qpos=np.asarray(qpos),qvel=np.asarray(qvel),stage=np.asarray(stages),sensors=np.asarray(sensors),relative=np.asarray(relative))
        put(folder/'placement-search.json',diagnostic)
        replay=mujoco.MjData(model);mujoco.mj_setState(model,replay,initial,spec_state);mujoco.mj_forward(model,replay)
        error=0.
        for i,ctrl in enumerate(controls):
            replay.ctrl[:]=ctrl;mujoco.mj_step(model,replay)
            error=max(error,float(np.max(np.abs(replay.qpos-qpos[i+1]))),float(np.max(np.abs(replay.qvel-qvel[i+1]))))
        result=dict(status=task.status,message=task.message,metrics=task.metrics,history=task.history,frames=len(qpos),continuation_frames=continuation_frames,prefix_motor_frames=len(saved['ctrl']) if prefix_replay else 0,replay_exact=error==0.,replay_max_state_error=error,demonstration_eligible=task.status=='succeeded' and error==0.,coverage_evaluation=False,wall_s=time.perf_counter()-started)
        if bridge_proof is not None:
            source_hold=ROOT/'.run/glass-placement-v5-middle-top-wrap/states.npz'
            with np.load(source_hold) as original:
                count=int(np.flatnonzero(original['stage']=='hold')[-1])
                result['proven_hold_prefix_frames']=count
                result['proven_hold_motor_prefix_exact']=bool(len(controls)>=count and np.array_equal(np.asarray(controls[:count]),original['ctrl'][:count]) and np.array_equal(np.asarray(qpos[:count+1]),original['qpos'][:count+1]) and np.array_equal(np.asarray(qvel[:count+1]),original['qvel'][:count+1]))
                result['proven_hold_source_sha256']=hashlib.sha256(source_hold.read_bytes()).hexdigest()
        if previous_continuation is not None:
            count=len(previous_continuation['ctrl'])
            result['released_state_prefix_motor_frames']=count
            result['released_state_prefix_exact']=bool(len(controls)>=count and np.array_equal(np.asarray(qpos[:count+1]),previous_continuation['qpos']) and np.array_equal(np.asarray(qvel[:count+1]),previous_continuation['qvel']))
        put(folder/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-source')
    p.add_argument('--initial-from',default='.run/whole-table-development-v16-seed4002/lookahead-01-024-00-glass')
    p.add_argument('--grasp',default='top_wrap_0.045_-3.14159');p.add_argument('--torque',type=float)
    p.add_argument('--tool-z',type=float);p.add_argument('--tool-x',type=float)
    p.add_argument('--mode',choices=('upright','inverted','sideways'),default='inverted')
    p.add_argument('--side-axis',action='store_true');p.add_argument('--bearing',type=float,default=0.)
    p.add_argument('--post-hold-sideways',action='store_true')
    p.add_argument('--horizontal-plane',action='store_true');p.add_argument('--roll-bias',type=float,default=0.)
    p.add_argument('--align-duration',type=float,default=4.)
    p.add_argument('--carry-torque',type=float)
    p.add_argument('--carry-clearance',type=float,default=0.)
    p.add_argument('--reorient-corrections',type=int,default=0)
    p.add_argument('--target',type=float,nargs=3,default=[.07928541907265485,-.020714581012188205,.834])
    p.add_argument('--branches',type=int,default=24);p.add_argument('--arm',choices=('right','left','auto'),default='right')
    args=p.parse_args()
    if args.torque is not None and not 0<args.torque<=2.94:p.error('Torque exceeds real model limits.')
    if args.carry_torque is not None and not 0<args.carry_torque<=2.94:p.error('Carry torque exceeds real model limits.')
    run(args)
