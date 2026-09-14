"""Shared exact-state physical demonstration teacher for all seven loose items.

Geometry supplies training labels and planning checks, never learned inputs.
The live MjData is advanced only by motor commands and ordinary physics.
Every unsuccessful grasp/path search is unsolved, not proof of unreachability.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from .autonomy import OPEN, PlanningError
from .dinner import OBJECTS
from .dinner_autonomy import CLOSED, DinnerTask
from .random_dinner import body_bounds
from .scene import HOME, TABLE_Z


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def destination(layout, name):
    target = next((t for t in layout['targets'] if t['object_id'] == name), None)
    return np.asarray(target['position_m'] if target else [.10, -.115, TABLE_Z], dtype=float)


@dataclass
class GraspCandidate:
    name: str
    point: np.ndarray
    local_tool_point: np.ndarray
    axis_index: int
    axis_target: np.ndarray
    x_target: np.ndarray | None
    open_grip: float = OPEN


def grasp_candidates(data, name):
    """Object-relative contact families transformed using privileged teacher pose."""
    body = data.body(name); center = body.xpos.copy(); r = body.xmat.reshape(3, 3)
    rows = []
    def add(label, point, local, axis, direction, x=None, opening=OPEN):
        rows.append(GraspCandidate(label, center+r@np.asarray(point), np.asarray(local), axis,
                                  np.asarray(direction), None if x is None else np.asarray(x), opening))
    if name in ('plate', 'side_plate'):
        radius, z = (.061, .013) if name == 'plate' else (.048, .009)
        for angle in np.linspace(-math.pi, math.pi, 12, endpoint=False):
            radial = np.array([math.cos(angle), math.sin(angle), 0.])
            world_radial = r@radial
            # The gripper comes from above for either resting face. Reorientation
            # to the task's upward face is a separate physically checked carry.
            add(f'rim_{angle:.5f}', radial*radius+[0, 0, z], [-.003, 0., -.100],
                2, [0., 0., 1.], world_radial)
    elif name in ('fork', 'spoon'):
        for y in (-.015, -.030, 0.):
            for sign in (-1, 1):
                add(f'handle_{y}_{sign}', [0., y, .008], [-.003, 0., -.100],
                    2, [0., 0., 1.], sign*r[:, 0])
    else:
        bands = ([.128, .116, .070, .040] if name == 'bottle' else
                 [.045, .030, .060] if name == 'glass' else [.037, .048, .025])
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
        self.place_override = None
        self.route_rng = np.random.default_rng(2026115002)
        self.stages = ['planning', 'approach', 'descend', 'close', 'lift', 'hold',
                       'align', 'lower', 'release', 'retract', 'park', 'verify']

    def start(self, side='auto', object_id='bottle', target=None, **kwargs):
        if self.active:raise ValueError('Cancel the current primitive first.')
        if object_id not in OBJECTS:raise ValueError('Unknown loose table item.')
        self.__init__(self.model, self.data, self.layout)
        self.requested_side, self.requested_object = side, object_id
        self.place_override = None if target is None else np.asarray(target, dtype=float)
        self.kind = 'whole_table_teacher'; self.status = 'running'; self._stage('planning')

    def _select_item(self, side, item):
        super()._select_item(side, item)
        self.sideways = False  # This shared primitive monitors grip, not upright-only carrying.
        self.grip_torque = .5 if item['id'] in ('plate', 'side_plate') else .3 if item['id'] in ('fork', 'spoon') else .4
        self.destination_position = destination(self.layout, item['id']) if self.place_override is None else self.place_override.copy()
        low, _ = body_bounds(self.model, item['id'], np.array([1., 0., 0., 0.]))
        self.destination_position[2] = TABLE_Z-low[2]
        self.destination = {'id': item['id']+'_place', 'position_m': self.destination_position.tolist()}
        self.metrics['controller'] = 'shared_exact_state_physical_teacher'

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
        self.ik.grasp_point = candidate.local_tool_point.copy()
        self.ik.axis_index = candidate.axis_index
        self.ik.axis_local = None
        self.ik.axis_target = candidate.axis_target.copy()
        self.ik.x_target = candidate.x_target
        self.open_grip = candidate.open_grip

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
                self._configure(candidate)
                for roll in (-2.4, 0., 2.4):
                    record = {'arm': side, 'candidate': candidate.name, 'initial_wrist_roll': roll}
                    try:
                        initial = np.asarray(HOME[:5]); initial[4] = roll
                        q = self.ik.solve(candidate.point, initial)
                        self._check_path([q], self.open_grip, allow_tube=True)
                        self.hover = candidate.point+[0., 0., .025 if name in ('fork', 'spoon') else .04]
                        above = self.ik.solve(self.hover, q)
                        descent = self._cartesian(self.hover, candidate.point, above, self.open_grip)
                        # Avoid an expensive RRT for every infeasible IK guess. It is
                        # used only after a complete collision-free descent exists.
                        approach = self._route(self.data.qpos[self.offset:self.offset+5], above, self.open_grip)
                        self.grasp = candidate.point.copy(); self.chosen = candidate
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
        self.ik.axis_local = self.reference[1][:, 2].copy()
        self.ik.axis_target = np.array([0., 0., 1.]); self.ik.x_target = None
        name = self.tube['id']
        if name in ('fork', 'spoon'):
            target_r = rotation_z(-math.pi/2 if name == 'spoon' else 0.)
            self.ik.x_target = (target_r@self.reference[1].T)[:, 0]
        start = self.data.qpos[self.offset:self.offset+5].copy(); errors = []
        grip = float(self.data.qpos[self.offset+5])
        for height in (.065, .10, .04):
            center = self.destination_position+[0., 0., height]
            for initial in (start, np.asarray(HOME[:5]), np.array([0., -.7, .8, .2, -2.4])):
                try:
                    _, end = self._point_for_center(center, self.reference, initial)
                    points = self._route(start, end, grip, carry=self.reference, allow_target=True)
                    self._move('align', points, CLOSED, 4.); return
                except PlanningError as exc:errors.append(str(exc))
        raise PlanningError('Placement pose/routes unsolved: '+'; '.join(errors))

    def _lower(self):
        point, _ = self._point_for_center(self.destination_position-[0., 0., .0005],
                                          self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
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
            collision = self._collision(self.data, True, .0008)
            if collision:
                self.metrics['unexpected_collisions'] += 1
                raise PlanningError('Unexpected contact: '+collision)
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
                z_error = abs(float(self.data.body(self.tube['id']).xpos[2]-self.destination_position[2]))
                if forces['base'] > .06 and z_error < .003:
                    point = self.ik.point(self.data)-self.data.body(self.side+'_gripper').xmat.reshape(3, 3)[:, 0]*.008
                    path = self._cartesian(self.ik.point(self.data), point,
                                           self.data.qpos[self.offset:self.offset+5], self.open_grip)
                    self._move('release', path, self.open_grip, 4.)
                elif done:raise PlanningError('No measured tabletop support; release refused.')
            elif self.stage == 'release' and done:
                point = self.ik.point(self.data)
                path = self._cartesian(point, point+[0., 0., .04],
                                       self.data.qpos[self.offset:self.offset+5], self.open_grip)
                self._move('retract', path, self.open_grip, 3.)
            elif self.stage == 'retract' and done:
                path = self._route(self.data.qpos[self.offset:self.offset+5], np.asarray(HOME[:5]), self.open_grip)
                self._move('park', path, HOME[5], 3.)
            elif self.stage == 'park' and done:
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('verify', np.array([q, q]), HOME[5], 1.)
            elif self.stage == 'verify':
                name = self.tube['id']; pos = self.data.body(name).xpos
                dof = int(self.model.joint(name+'_free').dofadr[0])
                error = float(np.linalg.norm(pos[:2]-self.destination_position[:2]))
                z_error = abs(float(pos[2]-self.destination_position[2]))
                speed = float(np.linalg.norm(self.data.qvel[dof:dof+3]))
                angular = float(np.linalg.norm(self.data.qvel[dof+3:dof+6]))
                yaw_error = 0.
                if name in ('fork', 'spoon'):
                    r = self.data.body(name).xmat.reshape(3, 3)
                    yaw = math.atan2(r[1, 0], r[0, 0]); desired = -math.pi/2 if name == 'spoon' else 0.
                    yaw_error = abs(math.atan2(math.sin(yaw-desired), math.cos(yaw-desired)))
                valid = (error < .008 and z_error < .004 and up > .98 and speed < .003 and angular < .08
                         and forces['base'] > .06 and forces['fixed']+forces['moving'] < .02
                         and yaw_error < .175 and np.max(np.abs(self.data.qpos[:12]-HOME*2)) < .035
                         and np.max(np.abs(self.data.qvel[:12])) < .12)
                self.metrics.update(placement_xy_error_mm=error*1000, placement_z_error_mm=z_error*1000,
                                    placement_yaw_error_deg=math.degrees(yaw_error), placement_speed_mm_s=speed*1000)
                if valid:
                    if self.stable_since is None:self.stable_since = now
                    if now-self.stable_since >= .5:
                        self._finish('succeeded', name+' physically lifted, held, oriented, placed and released; both arms parked.')
                else:self.stable_since = None
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc), True)

