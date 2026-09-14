"""Development-only physical bimanual transfer; no attachment or live state writes.

All hypothetical transforms are confined to independent MuJoCo planning data.
IK feasibility is a search result, never a claim of physical reachability.
"""
import math
import mujoco
import numpy as np
from mujoco import minimize

from .autonomy import OPEN, PlanningError
from .dinner_autonomy import CLOSED
from .scene import HOME
from .table_teacher import TableTeacher, GraspCandidate


class RegraspReceiver(TableTeacher):
    """Open receiving fingers must not sweep through the held object."""
    def __init__(self,model,data,layout):
        super().__init__(model,data,layout)
        self.regrasp_open=True

    def _collision(self,data,allow_tube=False,penetration=.00015):
        ordinary=super()._collision(data,allow_tube,penetration)
        if ordinary:return ordinary
        if self.regrasp_open:
            for c in data.contact:
                a,b=int(c.geom1),int(c.geom2)
                if c.dist>=-penetration:continue
                if ((a in self.jaw_geoms and self.model.geom_bodyid[b]==self.body) or
                    (b in self.jaw_geoms and self.model.geom_bodyid[a]==self.body)):
                    return 'Open receiving fingers would push the held object before closure.'
        return None

    def contact_approach(self,point,initial):
        """Try a guarded straight final segment before a full joint route."""
        q=self.ik.solve(point,initial)
        self._check_path([q],OPEN,True)
        start=self.data.qpos[self.ik.indices].copy()
        for distance in (.025,.012):
            try:
                hover=point+self.ik.axis_target*distance
                qhover=self.ik.solve(hover,q)
                descent=self._cartesian(hover,point,qhover,OPEN)
                approach=self._route(start,qhover,OPEN,iterations=600)
                return np.concatenate((approach,descent)),q
            except PlanningError:pass
        return self._route(start,q,OPEN,allow_target=True,iterations=600),q


def copied_data(model, data):
    scratch = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    mujoco.mj_setState(model, scratch, state, spec)
    mujoco.mj_forward(model, scratch)
    return scratch


def shared_pose_search(donor, *, maximum=1, check_routes=True):
    """Find opposed tool-Z grasp poses, checking both real robot geometries."""
    model, data, layout = donor.model, donor.data, donor.layout
    name = donor.tube['id']; receiver_side = 'right' if donor.side == 'left' else 'left'
    reference = donor._carry_reference()
    initial = data.qpos[donor.offset:donor.offset+5].copy()
    donor_x = donor.ik.x_target.copy()
    donor_x[2]=0.; donor_x/=np.linalg.norm(donor_x)
    grip = float(data.qpos[donor.offset+5])
    rows, found = [], []
    for yaw,pitch in ((0.,-.6),(0.,-1.0),(0.,-1.3),(math.pi/2,-.6),(-math.pi/2,-.6)):
        c, s = math.cos(yaw), math.sin(yaw)
        desired_x = np.array([[c,-s,0],[s,c,0],[0,0,1]])@donor_x
        for z in (.91,.93,.95,.89,.87):
            for x, y in ((0.,-.235),(-.025,-.235),(.025,-.235),(0.,-.20),(0.,-.27)):
                center = np.array([x,y,z]); row = {'center':center.tolist(),'yaw_delta':yaw,'pitch':pitch}
                radial=center-data.body(donor.side+'_shoulder').xpos;radial[2]=0.;radial/=np.linalg.norm(radial)
                donor.ik.axis_target=radial*math.sin(pitch)+np.array([0.,0.,math.cos(pitch)])
                closing=desired_x-donor.ik.axis_target*np.dot(desired_x,donor.ik.axis_target)
                donor.ik.x_target=closing/np.linalg.norm(closing)
                try:
                    _, qdonor = donor._point_for_center(center, reference, initial)
                    donor._check_path([qdonor], grip, True, carry=reference)
                    shared = copied_data(model, data)
                    shared.qpos[donor.offset:donor.offset+5] = qdonor
                    mujoco.mj_kinematics(model, shared)
                    rotation = shared.body(donor.side+'_gripper').xmat.reshape(3,3)@reference[1]
                    joint = shared.joint(name+'_free').qpos
                    joint[:3] = center; mujoco.mju_mat2Quat(joint[3:], rotation.ravel())
                    mujoco.mj_forward(model, shared)
                    receiver = TableTeacher(model, shared, layout)
                    receiver._select_item(receiver_side, donor.tube)
                    attempts = []
                    for local_y in (-.040, .030, .043, -.025):
                        for sign in (-1,1):
                            receiver_axis=-donor.ik.axis_target.copy()
                            closing=sign*rotation[:,0].copy();closing-=receiver_axis*np.dot(closing,receiver_axis);closing/=np.linalg.norm(closing)
                            cand = GraspCandidate(f'under_{local_y}_{sign}', center+rotation@np.array([0.,local_y,.006]),
                                np.array([-.003,0.,-.100]),2,receiver_axis,closing)
                            receiver._configure(cand)
                            try:
                                solved=None
                                for roll in (0.,-2.4,2.4):
                                    seed=np.asarray(HOME[:5]);seed[4]=roll
                                    try:solved=receiver.ik.solve(cand.point,seed);break
                                    except PlanningError:pass
                                if solved is None:raise PlanningError('Receiver pose unsolved with three wrist-roll seeds.')
                                qreceiver=solved
                                receiver._check_path([qreceiver], OPEN, True)
                                hover = cand.point+receiver_axis*.025
                                qabove = receiver.ik.solve(hover,qreceiver)
                                descent = receiver._cartesian(hover,cand.point,qabove,OPEN)
                                path_donor = donor._route(initial,qdonor,grip,carry=reference,allow_target=True) if check_routes else None
                                path_receiver = receiver._route(shared.qpos[receiver.offset:receiver.offset+5],qabove,OPEN) if check_routes else None
                                row.update(planned=True, receiver_candidate=cand.name,qdonor=qdonor.tolist(),qreceiver=qreceiver.tolist())
                                found.append(dict(donor_path=path_donor,receiver_path=path_receiver,receiver_descent=descent,
                                    candidate=cand,receiver=receiver,center=center,donor_x=donor.ik.x_target.copy(),qdonor=qdonor,qreceiver=qreceiver))
                                rows.append(row)
                                if len(found)>=maximum:return found,rows
                            except PlanningError as exc:
                                attempts.append({'candidate':cand.name,'error':str(exc),
                                    'point':cand.point.tolist(),'nearest_point':receiver.ik.point(receiver.ik.data).tolist(),
                                    'q':receiver.ik.data.qpos[receiver.offset:receiver.offset+5].tolist()})
                    row['receiver_attempts']=attempts
                except PlanningError as exc:
                    row['error']=str(exc)
                    r=donor.ik.data.body(donor.side+'_gripper').xmat.reshape(3,3)
                    actual_center=donor.ik.point(donor.ik.data)+r@reference[0]
                    row.update(nearest_center=actual_center.tolist(),position_error_mm=float(np.linalg.norm(actual_center-center)*1000),
                        tool_z=r[:,2].tolist(),tool_x=r[:,0].tolist(),desired_x=donor.ik.x_target.tolist(),
                        q=donor.ik.data.qpos[donor.offset:donor.offset+5].tolist())
                rows.append(row)
    donor.ik.x_target=donor_x
    return found,rows


def coupled_pose_search(donor, *, check_routes=True):
    """Jointly solve the free shared pose; each arm has only five DOFs."""
    model,data=donor.model,donor.data
    scratch=copied_data(model,data)
    name=donor.tube['id'];other='right' if donor.side=='left' else 'left'
    receiver=RegraspReceiver(model,scratch,donor.layout);receiver._select_item(other,donor.tube)
    reference=donor._carry_reference(); local=np.array([-.003,0.,-.100])
    ids=np.r_[donor.ik.indices,receiver.ik.indices]
    low=np.r_[donor.ik.lo,receiver.ik.lo];high=np.r_[donor.ik.hi,receiver.ik.hi]
    # Search interior receiver wrist configurations so measured grasp drift
    # can be corrected later; the physical model's joint limits are unchanged.
    low[8]+=.12;high[8]-=.12
    rows=[]
    def pose(q):
        scratch.qpos[ids]=q;mujoco.mj_kinematics(model,scratch)
        rd=scratch.body(donor.side+'_gripper').xmat.reshape(3,3).copy()
        rr=scratch.body(other+'_gripper').xmat.reshape(3,3).copy()
        center=donor.ik.point(scratch)+rd@reference[0]
        receive=scratch.body(other+'_gripper').xpos+rr@local
        return rd,rr,center,receive
    for z in (.93,.91,.95,.89):
        target=np.array([0.,-.1961647,z])
        for ly in (-.040,.043,.030):
            local=np.array([.008 if ly>0. else -.003,0.,-.100])
            for sign in (-1,1):
                object_local=np.array([0.,ly,.008 if ly<0 else .004])
                def residual(qs):
                    results=[]
                    for q in qs.T:
                        rd,rr,center,receive=pose(q)
                        contact=center+rd@reference[1]@object_local
                        results.append(np.r_[receive-contact,.08*(rr[:,2]+rd[:,2]),
                            .08*(rr[:,0]-sign*rd[:,0]),.05*(center-target),.015*(rd[:,2]-[-.7,0.,.7])])
                    return np.array(results).T
                for roll in (0.,-2.4,2.4):
                    seed=np.r_[data.qpos[donor.ik.indices],[-1.5,.0,1.4,-1.5,roll]]
                    q,_=minimize.least_squares(np.clip(seed,low,high),residual,bounds=(low,high),max_iter=160,verbose=0)
                    rd,rr,center,receive=pose(q);contact=center+rd@reference[1]@object_local
                    row={'mode':'coupled','target':target.tolist(),'receiver_y':ly,'sign':sign,'roll':roll,
                         'center':center.tolist(),'position_error_mm':float(np.linalg.norm(receive-contact)*1000),
                         'opposed_axis_error_deg':float(np.degrees(np.arccos(np.clip(-rr[:,2]@rd[:,2],-1,1)))),
                         'closing_axis_error_deg':float(np.degrees(np.arccos(np.clip(rr[:,0]@(sign*rd[:,0]),-1,1))))}
                    try:
                        if row['position_error_mm']>.25 or row['opposed_axis_error_deg']>3 or row['closing_axis_error_deg']>3:
                            raise PlanningError('Coupled relative grasp pose remains unsolved.')
                        donor.ik.x_target=rd[:,0].copy();donor.ik.axis_target=rd[:,2].copy()
                        grip=float(data.qpos[donor.offset+5])
                        row['check_stage']='donor_endpoint'
                        donor._check_path([q[:5]],grip,True,carry=reference)
                        # Hold the donor and its hypothetical carried object in planning data only.
                        scratch.qpos[donor.ik.indices]=q[:5]
                        scratch.qpos[receiver.ik.indices]=data.qpos[receiver.ik.indices]
                        obj=scratch.joint(name+'_free').qpos;obj[:3]=center
                        mujoco.mju_mat2Quat(obj[3:],(rd@reference[1]).ravel());mujoco.mj_forward(model,scratch)
                        candidate=GraspCandidate(f'coupled_under_{ly}_{sign}',contact.copy(),local,2,rr[:,2].copy(),rr[:,0].copy())
                        receiver._configure(candidate)
                        row['check_stage']='receiver_endpoint'
                        receiver._check_path([q[5:]],OPEN,True)
                        # A Cartesian axial retreat can exceed wrist flex at
                        # this valid endpoint. Search the full joint approach.
                        row['q']=q.tolist();row['receiver_axis']=rr[:,2].tolist()
                        descent=np.array([q[5:].copy(),q[5:].copy()])
                        row['check_stage']='donor_route'
                        dp=donor._route(data.qpos[donor.ik.indices],q[:5],grip,carry=reference,allow_target=True) if check_routes else None
                        row['check_stage']='receiver_route'
                        rp=receiver._route(data.qpos[receiver.ik.indices],q[5:],OPEN,allow_target=True) if check_routes else None
                        row.update(planned=True,qdonor=q[:5].tolist(),qreceiver=q[5:].tolist());rows.append(row)
                        return [dict(donor_path=dp,receiver_path=rp,receiver_descent=descent,candidate=candidate,receiver=receiver,
                            center=center,donor_x=rd[:,0].copy(),qdonor=q[:5],qreceiver=q[5:],object_local=object_local.copy())],rows
                    except PlanningError as exc:row['error']=str(exc);rows.append(row)
    return [],rows


class TableRegrasp(TableTeacher):
    """Reusable donor pick and contact-verified airborne transfer controller.

    The receiver uses the common placement controller after the donor parks.
    Development status: physical success must be established by saved replay.
    """
    def __init__(self, model, data, layout):
        super().__init__(model,data,layout)
        self.transfer = None
        self.receiver = None
        self.receiver_targets = None
        self.handoff_log = []
        self.bimanual = False
        self.receiver_bad_grip_since = None

    def start(self, side='left', object_id='spoon', **kwargs):
        if object_id not in ('spoon','fork'):raise ValueError('Flat cutlery development only.')
        super().start(side=side,object_id=object_id,**kwargs)
        self.kind='bimanual_table_regrasp'

    def _precheck_transfer(self, grasp_q):
        # A donor cannot itself flip an inverted flat item near the table.
        # The receiver handoff is checked against the actual held reference.
        return None

    def apply_gripper_limit(self, targets):
        ctrl=super().apply_gripper_limit(targets)
        if self.receiver is not None and self.bimanual:
            ctrl=self.receiver.apply_gripper_limit(ctrl)
        return ctrl

    def action_metadata(self):
        return {'controller':'development_bimanual_table_regrasp','item':self.requested_object,
            'donor_arm':self.side,'receiver_arm':None if self.receiver is None else self.receiver.side,
            'receiver_candidate':None if self.transfer is None else self.transfer['candidate'].name,
            'physical_success':self.status=='succeeded',
            'support_contract':'Both donor fingers before transfer; both receiver fingers before donor release and throughout carry, with 0.18 s loss grace.',
            'other_arm_motion_exemption':'Both arms are intentionally scheduled during bimanual phases; both must park for final success.',
            'scope':'Unverified development dependency; never a whole-table coverage result.'}

    def _monitor_pair(self):
        all_jaws=self.jaw_geoms|self.receiver.jaw_geoms
        for task in (self,self.receiver):
            collision=task._collision(self.data,True,.0008)
            if collision:
                self.metrics['unexpected_collisions']+=1
                raise PlanningError('Bimanual unexpected contact: '+collision)
        depth=0.;external_force=0.;wrench=np.zeros(6)
        for index,contact in enumerate(self.data.contact):
            a,b=int(contact.geom1),int(contact.geom2)
            if self.body not in (self.model.geom_bodyid[a],self.model.geom_bodyid[b]):continue
            other=b if self.model.geom_bodyid[a]==self.body else a
            if other not in all_jaws:
                depth=max(depth,-float(contact.dist))
                mujoco.mj_contactForce(self.model,self.data,index,wrench)
                external_force+=max(0.,float(wrench[0]))
        self.metrics['external_support_excluding_partner_fingers_n']=external_force
        self.metrics['maximum_external_penetration_mm']=max(self.metrics.get('maximum_external_penetration_mm',0),depth*1000)
        if depth>.001:raise PlanningError('Bimanual item external penetration exceeds 1 mm.')
        disturbed=max((float(np.linalg.norm(self.data.body(n).xpos-p)) for n,p in self.others.items()),default=0.)
        self.metrics['other_object_max_displacement_mm']=max(self.metrics.get('other_object_max_displacement_mm',0),disturbed*1000)
        if disturbed>.004:raise PlanningError('Bimanual unrelated item moved more than 4 mm.')
        observations=self._observe(),self.receiver._observe()
        self.metrics['donor_external_including_partner_fingers_n']=self.metrics['external_contact_n']
        self.metrics['external_contact_n']=external_force
        return observations

    def update(self,targets):
        if not self.active:return
        try:
            if self.stage=='hold' and self._done_motion():
                found,rows=coupled_pose_search(self)
                self.handoff_log.extend(rows)
                if not found:raise PlanningError('No shared collision-free handoff pose/path solved; not proof of impossibility.')
                self.transfer=found[0]
                self.receiver=RegraspReceiver(self.model,self.data,self.layout)
                self.receiver._select_item(self.transfer['receiver'].side,self.tube)
                self.receiver._configure(self.transfer['candidate'])
                self.receiver.grip_torque=.15
                self.receiver.status='running'
                self.bimanual=True
                self.metrics['bimanual_scheduled']=True
                self.metrics['receiver_gripper_torque_limit_nm']=self.receiver.grip_torque
                self._move('shared_lift',self.transfer['donor_path'],CLOSED,5.)
                return
            if not self.bimanual:return super().update(targets)
            now=float(self.data.time)
            if now-self.started>180:raise PlanningError('Bimanual transfer timed out.')
            donor_obs,receiver_obs=self._monitor_pair()
            self.metrics['donor_finger_forces_n']=donor_obs[0]['fixed'],donor_obs[0]['moving']
            self.metrics['receiver_finger_forces_n']=receiver_obs[0]['fixed'],receiver_obs[0]['moving']
            if self.stage in ('donor_release','donor_retract','donor_park','receiver_place'):
                placement_supported=(self.stage=='receiver_place' and self.receiver.stage in ('release','retract','park','verify'))
                if not receiver_obs[3] and not placement_supported:
                    if self.receiver_bad_grip_since is None:self.receiver_bad_grip_since=now
                    if now-self.receiver_bad_grip_since>.18:raise PlanningError('Receiver lost two-finger support after handoff.')
                else:self.receiver_bad_grip_since=None
            if self.stage in ('shared_lift','shared_settle','shared_adjust','receiver_approach','receiver_descend','receiver_refine','receiver_close') and not donor_obs[3]:
                if self.bad_grip_since is None:self.bad_grip_since=now
                if now-self.bad_grip_since>.18:raise PlanningError('Donor lost two-finger support before handoff.')
            else:self.bad_grip_since=None
            if self.stage in ('shared_lift','shared_settle','shared_adjust','donor_release','donor_retract','donor_park'):
                targets[self.offset:self.offset+6]=self.motion.sample(now)
                done=self._done_motion()
            elif self.stage.startswith('receiver_'):
                targets[self.receiver.offset:self.receiver.offset+6]=self.receiver.motion.sample(now)
                done=self.receiver._done_motion()
            else:done=False
            if self.stage=='shared_lift' and done:
                q=self.data.qpos[self.ik.indices].copy()
                self._move('shared_settle',np.array([q,q]),CLOSED,8.)
            elif self.stage=='shared_settle' and done:
                found,rows=coupled_pose_search(self)
                self.handoff_log.extend(rows)
                if not found:raise PlanningError('No collision-free shared pose solved from the settled physical grasp.')
                self.transfer=found[0]
                self.receiver._configure(self.transfer['candidate'])
                self.receiver.grip_torque=.15
                self._move('shared_adjust',self.transfer['donor_path'],CLOSED,3.)
            elif self.stage=='shared_adjust' and done:
                # Reobserve the real held pose: contact compliance can change
                # the donor-to-object transform during a large wrist turn.
                r=self.data.body(self.tube['id']).xmat.reshape(3,3)
                point=self.data.body(self.tube['id']).xpos+r@self.transfer['object_local']
                self.receiver.ik.soft_placement=True
                # The bowl permits a free closing bearing. Keeping its old
                # six-dimensional pose would defeat five-DOF live correction.
                self.receiver.ik.x_target=None
                route,qreceive=self.receiver.contact_approach(point,self.transfer['qreceiver'])
                self.transfer['receiver_descent']=np.array([qreceive.copy(),qreceive.copy()])
                self.metrics['receiver_contact_reobserved_m']=point.tolist()
                self.receiver._move('approach',route,OPEN,5.)
                self._stage('receiver_approach')
            elif self.stage=='receiver_approach' and done:
                self.receiver._move('descend',self.transfer['receiver_descent'],OPEN,3.)
                self._stage('receiver_descend')
            elif self.stage=='receiver_descend' and done:
                r=self.data.body(self.tube['id']).xmat.reshape(3,3)
                point=self.data.body(self.tube['id']).xpos+r@self.transfer['object_local']
                qstart=self.data.qpos[self.receiver.ik.indices].copy()
                qreceive=self.receiver.ik.solve(point,qstart)
                route=self.receiver._route(qstart,qreceive,OPEN,allow_target=True)
                self.receiver._move('descend',route,OPEN,2.)
                self._stage('receiver_refine')
            elif self.stage=='receiver_refine' and done:
                self.receiver.regrasp_open=False
                q=self.data.qpos[self.receiver.offset:self.receiver.offset+5].copy()
                self.receiver._move('close',np.array([q,q]),CLOSED,4.)
                self._stage('receiver_close')
            elif self.stage=='receiver_close' and done:
                if not receiver_obs[3]:raise PlanningError('Receiver did not establish two-finger contact.')
                self.metrics['both_arms_contact_verified']=True
                q=self.data.qpos[self.offset:self.offset+5].copy()
                self._move('donor_release',np.array([q,q]),OPEN,3.)
            elif self.stage=='donor_release' and done:
                if not receiver_obs[3]:raise PlanningError('Receiver lost support during donor release.')
                p=self.ik.point(self.data)
                path=self._cartesian(p,p+self.ik.axis_target*.035,self.data.qpos[self.offset:self.offset+5],OPEN)
                self._move('donor_retract',path,OPEN,3.)
            elif self.stage=='donor_retract' and done:
                path=self._route(self.data.qpos[self.offset:self.offset+5],np.asarray(HOME[:5]),OPEN)
                self._move('donor_park',path,HOME[5],4.)
            elif self.stage=='donor_park' and done:
                if not receiver_obs[3]:raise PlanningError('Receiver lost support before placement.')
                self.receiver.parked=self.data.qpos[self.offset:self.offset+6].copy()
                self.receiver.metrics['other_arm_max_motion_deg']=0.
                self.receiver.started=now
                self.receiver._place_route()
                self._stage('receiver_place')
            elif self.stage=='receiver_place':
                self.receiver.update(targets)
                if not self.receiver.active:
                    self.metrics['receiver_placement']=self.receiver.metrics
                    self._finish(self.receiver.status,self.receiver.message,self.receiver.request_pause)
        except (PlanningError,np.linalg.LinAlgError) as exc:
            targets[:]=self.data.qpos[:12]
            self._finish('failed',str(exc),True)
