"""Shared exact-state physical demonstration teacher for all seven loose items.

Geometry supplies training labels and planning checks, never learned inputs.
The live MjData is advanced only by motor commands and ordinary physics.
Every unsuccessful grasp/path search is unsolved, not proof of unreachability.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mujoco import minimize

from .autonomy import ArmIK, OPEN, PlanningError, skew
from .dinner import OBJECTS
from .dinner_autonomy import CLOSED, DinnerTask
from .glass_grasp_candidates import padded_glass_wrap_candidates
from .horizontal_glass_grasp_candidates import horizontal_glass_diameter_candidates, horizontal_glass_ik_seeds
from .random_dinner import body_bounds
from .scene import HOME, TABLE_Z
from .side_plate_grasp_candidates import reverse_rim_candidates
from .table_buffers import measured_glass_buffer


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def destination(layout, name):
    target = next((t for t in layout['targets'] if t['object_id'] == name), None)
    return np.asarray(target['position_m'] if target else [.10, -.115, TABLE_Z], dtype=float)


class TableIK(ArmIK):
    """Exact grasp IK and bounded MuJoCo least-squares held-object alignment.

    Optimize the measured object normal, including its offset in the fingers.
    A solution must meet 0.25 mm position and three-degree normal tolerances.
    Independent physical placement, contact and settling checks still apply.
    """
    soft_placement = False
    placement_solve = False
    axis_in_plane = False
    fallback_initials = ()

    def solve(self, position, initial):
        try:
            result = self._solve(position, initial)
            self.last_initial = np.asarray(initial).copy()
            return result
        except PlanningError as exc:
            first_error = exc
        for alternative in self.fallback_initials:
            try:
                result = self._solve(position, alternative)
                self.last_initial = np.asarray(alternative).copy()
                return result
            except PlanningError:
                pass
        raise first_error

    def _solve(self, position, initial):
        placement = self.placement_solve or self.soft_placement
        if not self.soft_placement and not self.axis_in_plane:
            try:return super().solve(position, initial)
            except PlanningError:pass
        # Preserve successful existing exact grasp solutions. For an unsolved
        # initial guess, use MuJoCo's bounded optimizer before giving up.
        weight = .05 if placement else .08
        def kinematics(q):
            self.data.qpos[self.indices] = q
            mujoco.mj_kinematics(self.model, self.data); mujoco.mj_comPos(self.model, self.data)
            rotation = self.data.xmat[self.body].reshape(3, 3)
            axis = rotation[:, self.axis_index] if self.axis_local is None else rotation@self.axis_local
            return self.point(self.data), rotation, axis
        def residual(qs):
            values = []
            for q in qs.T:
                point, rotation, axis = kinematics(q)
                normal_error = np.dot(axis, self.axis_target) if self.axis_in_plane else axis-self.axis_target
                error = np.r_[point-position, weight*normal_error]
                if self.x_target is not None:error = np.r_[error, weight*(rotation[:, 0]-self.x_target)]
                values.append(error)
            return np.array(values).T
        def jacobian(q, _):
            point, rotation, axis = kinematics(q.ravel())
            mujoco.mj_jac(self.model, self.data, self.jp, self.jr, point, self.body)
            axis_jac = -weight*skew(axis)@self.jr[:, self.indices]
            if self.axis_in_plane:axis_jac = self.axis_target@axis_jac
            jac = np.vstack((self.jp[:, self.indices], axis_jac))
            if self.x_target is not None:jac = np.vstack((jac, -weight*skew(rotation[:, 0])@self.jr[:, self.indices]))
            return jac
        q, _ = minimize.least_squares(np.asarray(initial).copy(), residual,
            bounds=(self.lo, self.hi), jacobian=jacobian, max_iter=100, verbose=0)
        point, rotation, axis = kinematics(q)
        tolerance = math.radians(3 if placement else .5)
        aligned = (abs(np.dot(axis, self.axis_target)) < math.sin(tolerance) if self.axis_in_plane
                   else np.dot(axis, self.axis_target) > math.cos(tolerance))
        if (np.linalg.norm(point-position) < .00025
                and aligned
                and (self.x_target is None or np.linalg.norm(rotation[:, 0]-self.x_target) < (.08 if placement else .01))):
            return q
        raise PlanningError('Gripper position/orientation tolerances not solved within joint limits.')


@dataclass
class GraspCandidate:
    name: str
    point: np.ndarray
    local_tool_point: np.ndarray
    axis_index: int
    axis_target: np.ndarray
    x_target: np.ndarray | None
    open_grip: float = OPEN
    approach_direction: np.ndarray | None = None
    torque_limit_nm: float | None = None
    local_axis: np.ndarray | None = None


def grasp_candidates(data, name):
    """Object-relative contact families transformed using privileged teacher pose."""
    body = data.body(name); center = body.xpos.copy(); r = body.xmat.reshape(3, 3)
    rows = reverse_rim_candidates(data, name, GraspCandidate, OPEN)
    rows += padded_glass_wrap_candidates(data, name, GraspCandidate)
    rows += horizontal_glass_diameter_candidates(data, name, GraspCandidate)
    def add(label, point, local, axis, direction, x=None, opening=OPEN, approach=None):
        rows.append(GraspCandidate(label, center+r@np.asarray(point), np.asarray(local), axis,
                                  np.asarray(direction), None if x is None else np.asarray(x), opening,
                                  None if approach is None else np.asarray(approach)))
    if name in ('plate', 'side_plate'):
        radius, z = (.061, .013) if name == 'plate' else (.048, .009)
        # For a pitched approach, the five-joint arm fixes the tool bearing.
        # Select contacts in each shoulder/object plane instead of demanding an
        # arbitrary six-dimensional pose from a five-dimensional mechanism.
        for side in ('left', 'right'):
            toward_arm = data.body(side+'_shoulder').xpos-center
            toward_arm[2] = 0.; toward_arm /= np.linalg.norm(toward_arm)
            for sign in (1, -1):
                radial = toward_arm*sign
                for pitch in (math.pi/6, math.pi/4, math.pi/3):
                    long_axis = radial*math.cos(pitch)+np.array([0., 0., math.sin(pitch)])
                    closing_axis = -radial*math.sin(pitch)+np.array([0., 0., math.cos(pitch)])
                    p = center+radial*radius+r[:, 2]*((OBJECTS[name]['size_m'][2]+.004)/2)
                    rows.append(GraspCandidate(f'pitched_face_{side}_{sign}_{pitch:.5f}', p,
                        np.array([-.0035, 0., -.100]), 2, long_axis, closing_axis, OPEN, long_axis, .2))
        for angle in np.linspace(-math.pi, math.pi, 12, endpoint=False):
            radial = np.array([math.cos(angle), math.sin(angle), 0.])
            world_radial = r@radial
            # The gripper comes from above for either resting face. Reorientation
            # to the task's upward face is a separate physically checked carry.
            if name == 'side_plate':
                # Clear the inner rim during descent. The moving finger then
                # brings the rim to the fixed finger during ordinary pinching.
                add(f'clear_rim_{angle:.5f}', radial*radius+[0, 0, .0105], [-.001, 0., -.100],
                    2, [0., 0., 1.], world_radial)
                rows[-1].torque_limit_nm = .8
            add(f'rim_{angle:.5f}', radial*radius+[0, 0, z], [-.003, 0., -.100],
                2, [0., 0., 1.], world_radial)
            if name == 'side_plate':
                add(f'low_rim_{angle:.5f}', radial*radius+[0, 0, .005], [-.003, 0., -.100],
                    2, [0., 0., 1.], world_radial)
            add(f'face_pinch_{angle:.5f}', radial*radius+[0, 0, (OBJECTS[name]['size_m'][2]+.004)/2],
                [.003, 0., -.092], 2, world_radial, [0., 0., 1.], approach=world_radial)
    elif name in ('fork', 'spoon'):
        for y in (-.015, -.030, 0.):
            for sign in (-1, 1):
                add(f'handle_{y}_{sign}', [0., y, .008], [-.003, 0., -.100],
                    2, [0., 0., 1.], sign*r[:, 0])
    else:
        bands = ([.128, .116, .070, .040] if name == 'bottle' else
                 [.045, .030, .060] if name == 'glass' else [.037, .048, .025])
        # Center a diameter wrap between the actual fingers. The older generic
        # 3 mm tool-X offset is suitable for a narrow neck, but places the fixed
        # finger inside a wider vessel. Keep that older family as a separate
        # retained option; these candidates use the physical wall radius.
        for z in bands:
            radius = .011 if name == 'bottle' and z > .10 else .024 if name in ('bottle', 'glass') else .025
            for sign in (1, -1):
                add(f'outside_body_band_{z}_{sign}', [0., 0., z], [radius-.007, 0., -.098],
                    1, sign*r[:, 2], opening=.85)
        if name == 'bottle':
            # Tilt the finger length while keeping the closing axis horizontal.
            # A tool-local up vector leaves the shoulder-plane bearing free;
            # prescribing world pitch and yaw separately overconstrains this
            # five-joint mechanism at offset contacts.
            for z in (.116, .128, .070, .040):
                radius = .011 if z > .10 else .024
                for pitch in (-math.pi/6, math.pi/6, -math.pi/3, math.pi/3):
                    add(f'inclined_bottle_band_{z}_{pitch:.5f}', [0., 0., z], [radius-.007, 0., -.098],
                        2, [0., 0., 1.], opening=.85)
                    rows[-1].local_axis = np.array([0., math.sin(pitch), math.cos(pitch)])
        # A vertical approach around the outside of the vessel reaches short
        # or inverted objects that a horizontal pinch cannot approach safely.
        # The open fixed finger clears the outside wall, rather than descending
        # through the solid base of an inverted cup.
        for z in bands:
            radius = .011 if name == 'bottle' and z > .10 else .024 if name in ('bottle', 'glass') else .025
            for angle in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                closing = np.array([math.cos(angle), math.sin(angle), 0.])
                if abs(r[2, 2]) < .8 and abs(np.dot(closing, r[:, 2])) > .4:continue
                add(f'top_wrap_{z}_{angle:.5f}', [0., 0., z], [radius-.006, 0., -.100],
                    2, [0., 0., 1.], closing, opening=.85)
                rows[-1].torque_limit_nm = .65
        if name == 'mug':
            for sign in (1, -1):
                add(f'handle_axis_{sign}', [.047, 0., .037], [.003, 0., -.092], 1, sign*r[:, 2])
        for z in bands:
            for sign in (1, -1):
                add(f'body_band_{z}_{sign}', [0., 0., z], [.003, 0., -.092], 1, sign*r[:, 2],
                    opening=.85 if abs(r[2, 2]) < .8 else OPEN)
        # Rim pinches offer narrower contact widths than wrapping the vessel.
        if name in ('mug', 'glass'):
            radius = .023 if name == 'mug' else .022
            height = .056 if name == 'mug' else .067
            for angle in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                radial = np.array([math.cos(angle), math.sin(angle), 0.])
                add(f'vessel_rim_{angle:.5f}', radial*radius+[0., 0., height], [-.003, 0., -.100],
                    2, [0., 0., 1.], r@radial)
    return rows


class TableTeacher(DinnerTask):
    """One shared physical pick, orient and place primitive, with arm/grasp search."""
    def __init__(self, model, data, layout):
        super().__init__(model, data, layout)
        self.search_log = []
        self.chosen = None
        self.chosen_roll = None
        self.place_override = None
        self.orientation_override = None
        self.placement_free_yaw = False
        self.allow_measured_buffer = False
        self.buffer_search = None
        self.source_pose_cache = {}
        self.excluded_grasps = set()
        self.route_rng = np.random.default_rng(2026115002)
        self.stages = ['planning', 'approach', 'descend', 'close', 'lift', 'hold',
                       'align', 'lower', 'release', 'retract', 'park', 'verify']

    def start(self, side='auto', object_id='bottle', target=None, target_quaternion=None,
              excluded_grasps=(), source_pose_cache=None, allow_measured_buffer=False, **kwargs):
        if self.active:raise ValueError('Cancel the current primitive first.')
        if object_id not in OBJECTS:raise ValueError('Unknown loose table item.')
        self.__init__(self.model, self.data, self.layout)
        self.requested_side, self.requested_object = side, object_id
        self.allow_measured_buffer = allow_measured_buffer
        self.place_override = None if target is None else np.asarray(target, dtype=float)
        if target_quaternion is not None:
            quaternion = np.asarray(target_quaternion, dtype=float)
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-8:
                raise ValueError('A finite nonzero target quaternion is required.')
            self.orientation_override = quaternion/np.linalg.norm(quaternion)
        self.excluded_grasps = set(tuple(row) for row in excluded_grasps)
        if source_pose_cache is not None:self.source_pose_cache = source_pose_cache
        self.kind = 'whole_table_teacher'; self.status = 'running'; self._stage('planning')

    def _select_item(self, side, item):
        super()._select_item(side, item)
        self.ik = TableIK(self.model, self.data, side)
        self.sideways = False  # This shared primitive monitors grip, not upright-only carrying.
        self.grip_torque = .5 if item['id'] in ('plate', 'side_plate') else .3 if item['id'] in ('fork', 'spoon') else .4
        self.default_grip_torque = self.grip_torque
        self.destination_position = destination(self.layout, item['id']) if self.place_override is None else self.place_override.copy()
        self.destination_rotation = rotation_z(-math.pi/2 if item['id'] == 'spoon' else 0.)
        self.destination_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(self.destination_quaternion, self.destination_rotation.ravel())
        if self.orientation_override is not None:
            self.destination_quaternion = self.orientation_override.copy()
            mujoco.mju_quat2Mat(self.destination_rotation.ravel(), self.destination_quaternion)
        sideways_target = (self.orientation_override is not None and
                           OBJECTS[item['id']]['kind'] in ('bottle', 'mug', 'glass') and
                           abs(self.destination_rotation[2, 2]) < .1)
        self.placement_in_plane = sideways_target and item['id'] in ('bottle', 'glass')
        self.placement_body_axis = 0 if sideways_target and not self.placement_in_plane else 2
        self.placement_axis_target = np.array([0., 0., 1.]) if sideways_target else self.destination_rotation[:, 2].copy()
        self.orientation_constraint = ('body Z within three degrees of horizontal plane; bearing and axial rotation free'
            if self.placement_in_plane else 'sideways with body X up and free table yaw'
            if sideways_target else 'target body Z normal; cutlery yaw also constrained')
        low, _ = body_bounds(self.model, item['id'], self.destination_quaternion)
        self.destination_position[2] = TABLE_Z-low[2]
        self.destination = {'id': item['id']+'_place', 'position_m': self.destination_position.tolist()}
        self.metrics['controller'] = 'shared_exact_state_physical_teacher'
        self.metrics['gripper_torque_limit_nm'] = self.grip_torque

    def _point_for_center(self, center, reference, initial):
        # Solve the rigidly held object's center directly in the robot model.
        # Alternating endpoint/offset solves can fail at workspace boundaries
        # even when the actual offset-point constraint is reachable.
        point = self.ik.grasp_point.copy()
        try:
            self.ik.grasp_point = point+reference[0]
            q = self.ik.solve(np.asarray(center), np.asarray(initial).copy())
        finally:
            self.ik.grasp_point = point
        self.ik.data.qpos[self.offset:self.offset+5] = q
        mujoco.mj_kinematics(self.model, self.ik.data)
        return self.ik.point(self.ik.data).copy(), q

    def _place_axes(self, reference):
        self.ik.placement_solve = True
        self.ik.soft_placement = True
        self.ik.axis_local = reference[1][:, self.placement_body_axis].copy()
        target_normal = self.placement_axis_target
        self.ik.axis_target = target_normal.copy(); self.ik.x_target = None
        self.ik.axis_in_plane = self.placement_in_plane
        if self.placement_in_plane:return
        # Reuse the arm's natural axis constraint when the physical grasp leaves
        # the item's normal within seven degrees of a tool axis. Enforcing that
        # tool axis vertically guarantees the same bounded body tilt without
        # overconstraining a contact-induced offset orientation.
        closest = int(np.argmax(np.abs(self.ik.axis_local)))
        natural_tolerance = 1 if self.tube['id'] in ('plate', 'side_plate') else 7
        if abs(self.ik.axis_local[closest]) > math.cos(math.radians(natural_tolerance)):
            sign = float(np.sign(self.ik.axis_local[closest]))
            self.ik.axis_local = None; self.ik.axis_index = closest
            self.ik.axis_target = target_normal*sign
            self.ik.soft_placement = False
            if closest == 2 and not self.placement_free_yaw:
                self.ik.x_target = self.data.body(self.side+'_gripper').xmat.reshape(3, 3)[:, 0].copy()
                self.ik.x_target -= target_normal*np.dot(target_normal, self.ik.x_target)
                self.ik.x_target /= np.linalg.norm(self.ik.x_target)
        name = self.tube['id']
        if name in ('fork', 'spoon'):
            target_r = self.destination_rotation
            self.ik.x_target = (target_r@reference[1].T)[:, 0]
            if self.ik.axis_local is None and self.ik.axis_index in (1, 2):
                self.ik.x_target -= target_normal*np.dot(target_normal, self.ik.x_target)
                self.ik.x_target /= np.linalg.norm(self.ik.x_target)

    def _precheck_transfer(self, grasp_q):
        # Reject an arm/grasp that cannot reach any destination approach before
        # touching the object. This is a search filter, never an invalid-start label.
        reference = self._carry_reference(grasp_q)
        errors = []
        for free_yaw in self._placement_yaw_modes():
            self.placement_free_yaw = free_yaw
            self._place_axes(reference)
            if (not free_yaw and self.ik.axis_local is None and self.ik.axis_index == 2
                    and self.tube['id'] not in ('fork', 'spoon')):
                self.ik.data.qpos[self.offset:self.offset+5] = grasp_q
                mujoco.mj_kinematics(self.model, self.ik.data)
                self.ik.x_target = self.ik.data.body(self.side+'_gripper').xmat.reshape(3, 3)[:, 0].copy()
                normal = self.placement_axis_target
                self.ik.x_target -= normal*np.dot(normal, self.ik.x_target)
                self.ik.x_target /= np.linalg.norm(self.ik.x_target)
            for height in (.065, .04, .10, .02):
                try:
                    point, above = self._point_for_center(self.destination_position+[0., 0., height], reference, grasp_q)
                    _, lower = self._point_for_center(self.destination_position, reference, above)
                    self._check_path([above], self.open_grip, True, carry=reference)
                    self._check_path([lower], self.open_grip, True, carry=reference, support=self.base_geom)
                    return point
                except PlanningError as exc:errors.append(str(exc))
        raise PlanningError('Grasp has no solved destination approach: '+'; '.join(errors))

    def _placement_yaw_modes(self):
        # Preserve the measured grasp yaw first, avoiding needless twisting.
        # Vessels have no required final table yaw; actual carried geometry,
        # including the mug handle, still goes through the same path checks.
        return (False, True) if self.tube['id'] in ('bottle', 'mug', 'glass') and not self.placement_in_plane else (False,)

    def _edge(self, start, end, grip, carry=None, support=None, allow_target=False):
        points = np.linspace(start, end, max(3, int(np.max(np.abs(end-start))/.035)+2))
        self._check_path(points, grip, allow_tube=allow_target, carry=carry, support=support)
        return points

    def _route(self, start, end, grip, carry=None, allow_target=False, iterations=180):
        """Direct checked edge, then a deterministic bidirectional RRT-Connect."""
        def edge(a, b):return self._edge(a, b, grip, carry, allow_target=allow_target)
        try:return edge(start, end)
        except PlanningError as direct_error:failure = str(direct_error)
        trees = [([np.asarray(start).copy()], [-1]), ([np.asarray(end).copy()], [-1])]
        def extend(tree, target):
            nodes, parents = tree
            nearest = int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
            delta = target-nodes[nearest]; length = np.linalg.norm(delta)
            q = nodes[nearest]+delta*min(1., .22/max(length, 1e-12))
            try:edge(nodes[nearest], q)
            except PlanningError:return None
            nodes.append(q); parents.append(nearest)
            return len(nodes)-1
        def branch(tree, i):
            out = []
            while i >= 0:out.append(tree[0][i]); i = tree[1][i]
            return out[::-1]
        for iteration in range(iterations):
            active = iteration % 2; other = 1-active
            sample = trees[other][0][0] if iteration % 5 == 0 else self.route_rng.uniform(self.ik.lo, self.ik.hi)
            i = extend(trees[active], sample)
            if i is None:continue
            for _ in range(48):
                j = extend(trees[other], trees[active][0][i])
                if j is None:break
                if np.linalg.norm(trees[other][0][j]-trees[active][0][i]) < 1e-8:
                    a, b = branch(trees[active], i), branch(trees[other], j)
                    chain = a+b[-2::-1] if active == 0 else b+a[-2::-1]
                    self.metrics['rrt_routes'] = self.metrics.get('rrt_routes', 0)+1
                    return np.concatenate([edge(a, b) for a, b in zip(chain[:-1], chain[1:])])
        raise PlanningError('Route search exhausted (not proof of impossibility): '+failure)

    def _configure(self, candidate):
        self.placement_free_yaw = False
        self.ik.placement_solve = False
        self.ik.axis_in_plane = False
        self.ik.soft_placement = False
        self.ik.grasp_point = candidate.local_tool_point.copy()
        self.ik.axis_index = candidate.axis_index
        self.ik.axis_local = None if candidate.local_axis is None else candidate.local_axis.copy()
        self.ik.axis_target = candidate.axis_target.copy()
        self.ik.x_target = candidate.x_target
        self.ik.fallback_initials = horizontal_glass_ik_seeds() if candidate.name.startswith('horizontal_glass_diameter_') else ()
        self.open_grip = candidate.open_grip
        self.grip_torque = self.default_grip_torque if candidate.torque_limit_nm is None else candidate.torque_limit_nm
        self.metrics['gripper_torque_limit_nm'] = self.grip_torque

    def _plan(self, targets):
        if np.max(np.abs(self.data.qpos[:12]-HOME*2)) > .1 or np.max(np.abs(self.data.qvel[:12])) > .12:
            raise PlanningError('Both arms must be parked before choosing the next object.')
        name = self.requested_object; item = next(o for o in self.layout['objects'] if o['id'] == name)
        candidates = grasp_candidates(self.data, name)
        sides = ['left', 'right'] if self.requested_side == 'auto' else [self.requested_side]
        sides.sort(key=lambda side: np.linalg.norm(self.data.body(name).xpos[:2]-[-.25 if side == 'left' else .25, -.235]))
        for side in sides:
            self._select_item(side, item)
            for candidate in candidates:
                if (side, candidate.name) in self.excluded_grasps:
                    self.search_log.append({'arm': side, 'candidate': candidate.name,
                                            'skipped': 'Earlier saved physical lookahead failed this grasp.'})
                    continue
                self._configure(candidate)
                for roll in (-2.4, 0., 2.4):
                    record = {'arm': side, 'candidate': candidate.name, 'initial_wrist_roll': roll}
                    if (side, candidate.name, roll) in self.excluded_grasps:
                        record['skipped'] = 'Earlier saved physical lookahead failed this wrist configuration.'
                        self.search_log.append(record)
                        continue
                    try:
                        self._configure(candidate)
                        initial = np.asarray(HOME[:5]); initial[4] = roll
                        cache_key = (id(self.model), name, side, candidate.name, roll,
                                     tuple(candidate.point), tuple(candidate.local_tool_point),
                                     candidate.axis_index, tuple(candidate.axis_target),
                                     None if candidate.local_axis is None else tuple(candidate.local_axis),
                                     None if candidate.x_target is None else tuple(candidate.x_target))
                        cached = self.source_pose_cache.get(cache_key)
                        record['source_pose_cache_hit'] = cached is not None
                        if cached is None:
                            try:
                                q = self.ik.solve(candidate.point, initial)
                                self.source_pose_cache[cache_key] = {'q': q.copy(), 'solver_initial': self.ik.last_initial.copy()}
                            except PlanningError as exc:
                                self.source_pose_cache[cache_key] = {'error': str(exc)}
                                raise
                        elif 'error' in cached:raise PlanningError(cached['error'])
                        else:q = cached['q'].copy()
                        solved = self.source_pose_cache[cache_key]
                        if 'solver_initial' in solved:record['source_ik_initial_joints'] = solved['solver_initial'].tolist()
                        # Different initial roll guesses can converge to the
                        # same joint solution. Retain that search result without
                        # rerunning an already failed physical configuration.
                        duplicate = False
                        for failed_roll in (-2.4, 0., 2.4):
                            if (side, candidate.name, failed_roll) not in self.excluded_grasps:continue
                            failed_key = cache_key[:4]+(failed_roll,)+cache_key[5:]
                            failed = self.source_pose_cache.get(failed_key, {})
                            if 'q' in failed and np.max(np.abs(q-failed['q'])) < 1e-5:
                                duplicate = True; break
                        if duplicate:
                            record['skipped'] = 'Source IK converged to an earlier saved failed configuration.'
                            self.search_log.append(record)
                            continue
                        self._check_path([q], self.open_grip, allow_tube=True)
                        self._precheck_transfer(q)
                        self._configure(candidate)
                        approach_axis = np.array([0., 0., 1.]) if candidate.approach_direction is None else candidate.approach_direction
                        self.hover = candidate.point+approach_axis*(.025 if name in ('fork', 'spoon') else .04)
                        above = self.ik.solve(self.hover, q)
                        descent = self._cartesian(self.hover, candidate.point, above, self.open_grip)
                        # Avoid an expensive RRT for every infeasible IK guess. It is
                        # used only after a complete collision-free descent exists.
                        approach = self._route(self.data.qpos[self.offset:self.offset+5], above, self.open_grip)
                        self.grasp = candidate.point.copy(); self.chosen = candidate; self.chosen_roll = roll
                        self.approach_descent = descent
                        record.update(planned=True, point_m=self.grasp.tolist(), hover_m=self.hover.tolist())
                        self.search_log.append(record)
                        self.attempts = 1; targets[:] = HOME*2
                        self._move('approach', approach, self.open_grip, 2.5)
                        return
                    except (PlanningError, np.linalg.LinAlgError) as exc:
                        record.update(planned=False, error=str(exc)); self.search_log.append(record)
        self.side = None
        raise PlanningError(f'No candidate grasp/approach solved for {name}; {len(self.search_log)} attempts retained.')

    def _lift(self):
        start = self.ik.point(self.data).copy(); grip = float(self.data.qpos[self.offset+5])
        carry = self._carry_reference(); errors = []
        for shift in ([0., 0., .065], [0., -.025, .065], [0., 0., .04]):
            try:
                points = self._cartesian(start, start+shift, self.data.qpos[self.offset:self.offset+5], grip, check=False)
                self._check_path(points[:2], grip, True, carry=carry, support=self.base_geom)
                self._check_path(points[2:], grip, True, carry=carry)
                self._move('lift', points, CLOSED, 3.); return
            except PlanningError as exc:errors.append(str(exc))
        raise PlanningError('Lift routes unsolved: '+'; '.join(errors))

    def _place_route(self):
        self.reference = self._carry_reference()
        if self.allow_measured_buffer and self.tube['id'] == 'glass' and self.buffer_search is None:
            previous_plane, previous_axis = self.placement_in_plane, self.placement_body_axis
            self.placement_in_plane, self.placement_body_axis = False, 2
            route = measured_glass_buffer(self, destination(self.layout, 'glass'), GraspCandidate)
            if route is not None:
                self._move('align', route, CLOSED, 4.)
                return
            self.placement_in_plane, self.placement_body_axis = previous_plane, previous_axis
        start = self.data.qpos[self.offset:self.offset+5].copy(); errors = []
        grip = float(self.data.qpos[self.offset+5])
        for free_yaw in self._placement_yaw_modes():
            self.placement_free_yaw = free_yaw
            self._place_axes(self.reference)
            for height in (.065, .10, .04, .02):
                center = self.destination_position+[0., 0., height]
                for initial in (start, np.asarray(HOME[:5]), np.array([0., -.7, .8, .2, -2.4])):
                    try:
                        _, end = self._point_for_center(center, self.reference, initial)
                        points = self._route(start, end, grip, carry=self.reference, allow_target=True)
                        self.metrics['placement_free_yaw'] = free_yaw
                        self._move('align', points, CLOSED, 4.); return
                    except PlanningError as exc:errors.append(str(exc))
        raise PlanningError('Placement pose/routes unsolved: '+'; '.join(errors))

    def _lower(self):
        reference = self._carry_reference()
        self._place_axes(reference)
        center = self.destination_position.copy()
        # Start from a supported pose for the *current* held orientation. Solving
        # the body origin at tabletop height first can be impossible for a tilted
        # dish even when the slightly higher, edge-supported pose is feasible.
        low, _ = body_bounds(self.model, self.tube['id'], self.data.joint(self.tube['id']+'_free').qpos[3:])
        center[2] = TABLE_Z-low[2]-.0003
        q = self.data.qpos[self.offset:self.offset+5].copy()
        for _ in range(3):
            point, q = self._point_for_center(center, reference, q)
            rotation = self.ik.data.xmat[self.ik.body].reshape(3, 3)@reference[1]
            quaternion = np.empty(4); mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            low, _ = body_bounds(self.model, self.tube['id'], quaternion)
            center[2] = TABLE_Z-low[2]-.0003
        point, _ = self._point_for_center(center, reference, q)
        points = self._cartesian(self.ik.point(self.data), point, self.data.qpos[self.offset:self.offset+5], CLOSED, check=False)
        self._check_path(points, float(self.data.qpos[self.offset+5]), True,
                         carry=self._carry_reference(), support=self.base_geom)
        self._move('lower', points, CLOSED, 4.)

    def update(self, targets):
        if not self.active:return
        try:
            if self.stage == 'planning':self._plan(targets); return
            now = float(self.data.time)
            if now-self.started > 110 or now-self.stage_started > self.motion.duration+6:
                raise PlanningError('Primitive timed out at '+self.stage)
            targets[self.offset:self.offset+6] = self.motion.sample(now)
            forces, lift, up, both = self._observe()
            alignment = float(np.dot(self.data.body(self.tube['id']).xmat.reshape(3, 3)[:, self.placement_body_axis],
                                     self.placement_axis_target))
            if self.placement_in_plane:alignment = math.sqrt(max(0., 1.-alignment*alignment))
            aligned = alignment > (math.cos(math.radians(3)) if self.placement_in_plane else .98)
            self.metrics['destination_normal_error_deg'] = math.degrees(math.acos(np.clip(alignment, -1., 1.)))
            collision = self._collision(self.data, True, .0008)
            if collision:
                self.metrics['unexpected_collisions'] += 1
                raise PlanningError('Unexpected contact: '+collision)
            penetration = 0.
            for contact in self.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if self.body not in (self.model.geom_bodyid[a], self.model.geom_bodyid[b]):continue
                other = b if self.model.geom_bodyid[a] == self.body else a
                if other not in self.jaw_geoms:penetration = max(penetration, -float(contact.dist))
            self.metrics['maximum_external_penetration_mm'] = max(
                self.metrics.get('maximum_external_penetration_mm', 0.), penetration*1000)
            if penetration > .001:raise PlanningError('The manipulated item penetrated external geometry by more than 1 mm.')
            if self.metrics['other_arm_max_motion_deg'] > 1.:
                raise PlanningError('The parked arm moved more than one degree.')
            disturbed = max((float(np.linalg.norm(self.data.body(n).xpos-p)) for n, p in self.others.items()), default=0.)
            self.metrics['other_object_max_displacement_mm'] = disturbed*1000
            if disturbed > .004:raise PlanningError('An unrelated item moved more than 4 mm.')
            if self.stage in ('lift', 'hold', 'align', 'lower') and lift > .012:
                if both:self.bad_grip_since = None
                else:
                    if self.bad_grip_since is None:self.bad_grip_since = now
                    if now-self.bad_grip_since > .18:raise PlanningError('Lost two-finger support.')
            done = self._done_motion()
            if self.stage == 'approach' and done:
                self._move('descend', self.approach_descent, self.open_grip, 3.)
            elif self.stage == 'descend' and done:
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('close', np.array([q, q]), CLOSED, 4.)
            elif self.stage == 'close' and done:
                if not both:raise PlanningError('Candidate did not establish two-finger contact.')
                self._lift()
            elif self.stage == 'lift' and done:
                if not both or lift < .025 or forces['external'] > .02:
                    raise PlanningError('Lift did not obtain independent finger support.')
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('hold', np.array([q, q]), CLOSED, 1.5)
            elif self.stage == 'hold':
                if not both or forces['external'] > .02:raise PlanningError('Hold lost independent support.')
                self.metrics['hold_verified_s'] = now-self.stage_started
                if done:self._place_route()
            elif self.stage == 'align':
                if forces['external'] > .10:raise PlanningError('The carried item contacted external support.')
                if done:self._lower()
            elif self.stage == 'lower':
                pose = self.data.joint(self.tube['id']+'_free').qpos
                low, _ = body_bounds(self.model, self.tube['id'], pose[3:])
                bottom = float(pose[2]+low[2])
                xy_error = np.linalg.norm(pose[:2]-self.destination_position[:2])
                if forces['base'] > .06 and abs(bottom-TABLE_Z) < .001 and xy_error < .01 and aligned:
                    point = self.ik.point(self.data)-self.data.body(self.side+'_gripper').xmat.reshape(3, 3)[:, 0]*.008
                    path = self._cartesian(self.ik.point(self.data), point,
                                           self.data.qpos[self.offset:self.offset+5], self.open_grip)
                    self._move('release', path, self.open_grip, 4.)
                elif done:
                    count = self.metrics.get('support_approach_corrections', 0)
                    gap = bottom-TABLE_Z
                    if count < 4 and 0. < gap < .012 and xy_error < .01 and aligned:
                        self.metrics['support_approach_corrections'] = count+1
                        reference = self._carry_reference(); self._place_axes(reference)
                        start = self.ik.point(self.data).copy()
                        end = start-[0., 0., min(.004, gap+.0003)]
                        path = self._cartesian(start, end, self.data.qpos[self.offset:self.offset+5], CLOSED, check=False)
                        self._check_path(path, float(self.data.qpos[self.offset+5]), True,
                                         carry=reference, support=self.base_geom)
                        self._move('lower', path, CLOSED, 1.5)
                    else:raise PlanningError('No measured tabletop support; release refused.')
            elif self.stage == 'release' and done:
                point = self.ik.point(self.data)
                try:
                    path = self._cartesian(point, point+[0., 0., .04],
                                           self.data.qpos[self.offset:self.offset+5], self.open_grip)
                except PlanningError as exc:
                    # After measured release, a Cartesian withdrawal can run
                    # into a wrist limit. Check the real open-hand joint route
                    # to park, treating the released item as an obstacle.
                    if forces['fixed']+forces['moving'] >= .02 or forces['base'] <= .06:
                        raise
                    path = self._route(self.data.qpos[self.offset:self.offset+5], np.asarray(HOME[:5]), self.open_grip)
                    self.metrics['release_joint_space_park_fallback'] = True
                    self.metrics['cartesian_release_retreat_failure'] = str(exc)
                self._move('retract', path, self.open_grip, 3.)
            elif self.stage == 'retract' and done:
                path = self._route(self.data.qpos[self.offset:self.offset+5], np.asarray(HOME[:5]), self.open_grip)
                self._move('park', path, HOME[5], 3.)
            elif self.stage == 'park' and done:
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('verify', np.array([q, q]), HOME[5], 1.)
            elif self.stage == 'verify':
                name = self.tube['id']; pos = self.data.body(name).xpos
                low, high = body_bounds(self.model, name, self.data.joint(name+'_free').qpos[3:])
                table_center = self.data.geom_xpos[self.base_geom, :2]
                table_half = self.model.geom_size[self.base_geom, :2]
                on_table = bool(np.all(pos[:2]+low[:2] >= table_center-table_half)
                                and np.all(pos[:2]+high[:2] <= table_center+table_half))
                dof = int(self.model.joint(name+'_free').dofadr[0])
                error = float(np.linalg.norm(pos[:2]-self.destination_position[:2]))
                z_error = abs(float(pos[2]-self.destination_position[2]))
                speed = float(np.linalg.norm(self.data.qvel[dof:dof+3]))
                angular = float(np.linalg.norm(self.data.qvel[dof+3:dof+6]))
                yaw_error = 0.
                if name in ('fork', 'spoon'):
                    r = self.data.body(name).xmat.reshape(3, 3)
                    yaw = math.atan2(r[1, 0], r[0, 0]); desired = math.atan2(self.destination_rotation[1, 0], self.destination_rotation[0, 0])
                    yaw_error = abs(math.atan2(math.sin(yaw-desired), math.cos(yaw-desired)))
                valid = (on_table and error < .008 and z_error < .004 and aligned and speed < .003 and angular < .08
                         and forces['base'] > .06 and forces['fixed']+forces['moving'] < .02
                         and yaw_error < .175 and np.max(np.abs(self.data.qpos[:12]-HOME*2)) < .035
                         and np.max(np.abs(self.data.qvel[:12])) < .12)
                self.metrics.update(placement_xy_error_mm=error*1000, placement_z_error_mm=z_error*1000,
                                    placement_yaw_error_deg=math.degrees(yaw_error), placement_speed_mm_s=speed*1000,
                                    complete_object_supported_within_table=on_table)
                if valid:
                    if self.stable_since is None:self.stable_since = now
                    if now-self.stable_since >= .5:
                        self._finish('succeeded', name+' physically lifted, held, oriented, placed and released; both arms parked.')
                else:self.stable_since = None
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc), True)
