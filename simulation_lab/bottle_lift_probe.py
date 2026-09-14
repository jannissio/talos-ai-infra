"""Unpromoted COM-centered bottle pinch/lift component; physical motors only."""
import math
import numpy as np
from .autonomy import PlanningError
from .dinner_autonomy import CLOSED
from .scene import HOME
from .table_teacher import TableTeacher,GraspCandidate


def bottle_contact_region_violations(model,data,side,band,jaw_geoms,penetration=.00015):
    """Reject an action's unintended base/root contact, never the scene itself.

    The distal region and axial window are empirical grasp-family diagnostics.
    The combined predicate falsely rejects the physically successful seed4004
    outside_body_band_0.04 route's brief distal-tip/base contact. Do not promote
    it as a general action or scene rejection gate. It is used only by this
    unpromoted probe; physical acceptance gates stay intact.
    """
    body=model.body('bottle').id;gripper=data.body(side+'_gripper')
    rg=gripper.xmat.reshape(3,3);rb=data.body('bottle').xmat.reshape(3,3);rows=[]
    for c in data.contact:
        a,b=int(c.geom1),int(c.geom2)
        if c.dist>=-penetration or body not in (model.geom_bodyid[a],model.geom_bodyid[b]):continue
        object_geom=a if model.geom_bodyid[a]==body else b;other=b if object_geom==a else a
        if other not in jaw_geoms:continue
        local_gripper=rg.T@(c.pos-gripper.xpos);local_bottle=rb.T@(c.pos-data.body('bottle').xpos)
        reasons=[]
        if local_gripper[2]>-.075:reasons.append('proximal_finger_or_root')
        if abs(float(local_bottle[2])-band)>.025:reasons.append('outside_requested_axial_band')
        if reasons:rows.append({'reasons':reasons,'object_geom':model.geom(object_geom).name,
            'finger_geom':model.geom(other).name or model.body(model.geom_bodyid[other]).name,
            'contact_bottle_local_m':local_bottle.tolist(),'contact_gripper_local_m':local_gripper.tolist(),
            'penetration_mm':-float(c.dist)*1000})
    return rows


class BottleLiftProbe(TableTeacher):
    def start_probe(self,band=.070,pitch=math.radians(75),side='left',body_axis=False,tool_x=.017,opening=.85):
        quaternion=self.data.joint('bottle_free').qpos[3:].copy()
        self.start(object_id='bottle',side=side,target=self.data.body('bottle').xpos.copy(),target_quaternion=quaternion)
        self.band=float(band);self.pitch=float(pitch);self.kind='bottle_lift_component_probe'
        self.body_axis=body_axis
        self.tool_x=float(tool_x);self.probe_opening=float(opening)

    def _collision(self,data,allow_tube=False,penetration=.00015):
        ordinary=super()._collision(data,allow_tube,penetration)
        if ordinary:return ordinary
        violations=bottle_contact_region_violations(self.model,data,self.side,self.band,self.jaw_geoms,penetration)
        if violations:return 'Bottle contact-region violation: '+', '.join(violations[0]['reasons'])
        return None

    def _plan(self,targets):
        if np.max(np.abs(self.data.qpos[:12]-HOME*2))>.1:raise PlanningError('Both arms must be parked.')
        item=next(x for x in self.layout['objects'] if x['id']=='bottle')
        self._select_item(self.requested_side,item)
        body=self.data.body('bottle');point=body.xpos+body.xmat.reshape(3,3)@np.array([0.,0.,self.band])
        candidate=GraspCandidate(f'com_band_{self.band:.3f}_pitch_{self.pitch:.5f}_x_{self.tool_x:.3f}',point,np.array([self.tool_x,0.,-.098]),
            2,np.array([0.,0.,1.]),None,self.probe_opening,local_axis=np.array([0.,math.sin(self.pitch),math.cos(self.pitch)]))
        if self.body_axis:
            candidate.name=f'com_body_axis_{self.band:.3f}'
            candidate.axis_index=1;candidate.axis_target=body.xmat.reshape(3,3)[:,2].copy();candidate.local_axis=None
        for roll in (-2.4,0.,2.4):
            row={'candidate':candidate.name,'initial_roll':roll}
            try:
                self._configure(candidate);initial=np.array(HOME[:5]);initial[4]=roll
                self.ik.soft_placement=self.ik.placement_solve=True
                row['check_stage']='contact_ik'
                q=self.ik.solve(point,initial)
                row['check_stage']='contact_clearance'
                self._check_path([q],self.open_grip,True)
                hover=point+[0.,0.,.04]
                row['check_stage']='hover_ik'
                above=self.ik.solve(hover,q)
                row['check_stage']='descent_clearance'
                descent=self._cartesian(hover,point,above,self.open_grip)
                row['check_stage']='approach_route'
                approach=self._route(self.data.qpos[self.ik.indices],above,self.open_grip)
                self.chosen=candidate;self.chosen_roll=roll;self.grasp=point.copy();self.hover=hover
                self.approach_descent=descent;self.attempts=1
                row.update(planned=True,q=q.tolist());self.search_log.append(row)
                self._move('approach',approach,self.open_grip,3.)
                self.metrics.update(lift_only_component=True,desired_axial_band_m=self.band,pitch_rad=self.pitch)
                return
            except PlanningError as exc:
                row['error']=str(exc);row['nearest_q']=self.ik.data.qpos[self.ik.indices].tolist()
                row['position_error_mm']=float(np.linalg.norm(self.ik.point(self.ik.data)-point)*1000)
                self.search_log.append(row)
        raise PlanningError('COM-centered pinch approach remains unsolved.')

    def _lift(self):
        rotation=self.data.body(self.side+'_gripper').xmat.reshape(3,3)
        local_axis=np.eye(3)[:,self.ik.axis_index] if self.ik.axis_local is None else self.ik.axis_local
        self.ik.axis_target=rotation@local_axis
        self.ik.x_target=None;self.ik.soft_placement=self.ik.placement_solve=True
        start=self.ik.point(self.data).copy();q=self.data.qpos[self.ik.indices].copy()
        grip=float(self.data.qpos[self.offset+5]);reference=self._carry_reference();errors=[]
        toward=self.data.body(self.side+'_shoulder').xpos-start;toward[2]=0.;toward/=np.linalg.norm(toward)
        for shift in (np.array([0.,0.,.04]),toward*.025+[0.,0.,.04]):
            try:
                points=self._cartesian(start,start+shift,q,grip,check=False)
                self._check_path(points[:2],grip,True,carry=reference,support=self.base_geom)
                self._check_path(points[2:],grip,True,carry=reference)
                self.metrics['measured_axis_lift_shift_m']=shift.tolist()
                self._move('lift',points,CLOSED,4.);return
            except PlanningError as exc:errors.append(str(exc))
        raise PlanningError('Measured-axis lift geometry unsolved: '+'; '.join(errors))

    def _place_route(self):
        self._finish('succeeded','Lift-only component: bottle raised and held with both fingers and no external support. No transfer, release, placement or parked-arm finish is claimed.')
