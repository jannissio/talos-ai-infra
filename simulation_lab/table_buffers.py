"""Measured glass-buffer search for the privileged demonstration teacher.

This is scratch-state geometry. A hypothetical settled regrasp is only a search
filter; the real release, settling, regrasp and final transfer must still pass.
"""
import math

import mujoco
import numpy as np

from .autonomy import PlanningError
from .dinner import OBJECTS
from .horizontal_glass_grasp_candidates import horizontal_glass_diameter_candidates, horizontal_glass_ik_seeds
from .random_dinner import body_bounds
from .scene import HOME, TABLE_Z


def _clear_buffer(task, center, quaternion):
    low, high = body_bounds(task.model, task.tube['id'], quaternion)
    low += center
    high += center
    table = task.model.geom('table').id
    table_center = task.data.geom_xpos[table, :2]
    half = task.model.geom_size[table, :2]
    if np.any(low[:2] < table_center-half+.006) or np.any(high[:2] > table_center+half-.006):
        return False
    for name in OBJECTS:
        if name == task.tube['id']:
            continue
        pose = task.data.joint(name+'_free').qpos
        other_low, other_high = body_bounds(task.model, name, pose[3:])
        if np.all(high[:2]+.006 > pose[:2]+other_low[:2]) and np.all(pose[:2]+other_high[:2]+.006 > low[:2]):
            return False
    return True


def _horizontal_continuation(task, center, final_target, candidate_type):
    # Explicit hypothetical settled placement in a separate MjData only. No
    # object/robot coordinates are assigned to the physically executing data.
    scratch = mujoco.MjData(task.model)
    scratch.qpos[:] = task.data.qpos
    scratch.qpos[:12] = HOME*2
    scratch.qvel[:] = 0.
    scratch.ctrl[:] = HOME*2
    scratch.joint('glass_free').qpos[:] = [*center, 1., 0., 0., 0.]
    mujoco.mj_forward(task.model, scratch)
    errors = []
    sides = [task.side, 'left' if task.side == 'right' else 'right']
    for side in sides:
        probe = type(task)(task.model, scratch, task.layout)
        probe.start(side=side, object_id='glass', target=final_target)
        probe._select_item(side, next(o for o in task.layout['objects'] if o['id'] == 'glass'))
        for candidate in horizontal_glass_diameter_candidates(scratch, 'glass', candidate_type):
            for seed in horizontal_glass_ik_seeds()[:2]:
                try:
                    probe._configure(candidate)
                    q = probe.ik.solve(candidate.point, seed)
                    probe._check_path([q], candidate.open_grip, True)
                    probe._precheck_transfer(q)
                    probe._configure(candidate)
                    hover = candidate.point+[0., 0., .04]
                    above = probe.ik.solve(hover, q)
                    descent = probe._cartesian(hover, candidate.point, above, candidate.open_grip)
                    approach = probe._route(scratch.qpos[probe.offset:probe.offset+5], above,
                                            candidate.open_grip, iterations=60)
                    return {'arm': side, 'candidate': candidate.name, 'source_q': q.tolist(),
                            'approach_samples': len(approach), 'descent_samples': len(descent),
                            'meaning': 'Hypothetical upright-settled source and destination geometry; physical continuation required.'}
                except PlanningError as exc:
                    errors.append(str(exc))
    raise PlanningError('Horizontal regrasp geometry unsolved: '+'; '.join(dict.fromkeys(errors)))


def measured_glass_buffer(task, final_target, candidate_type, maximum_candidates=256):
    """Select a checked temporary pose from a uniform grid over the actual table.

    Keep the measured small held tilt when an upright placement has no route.
    The fixed 20 mm search resolution is a development budget, not a reachability
    boundary. All examined outcomes and the unexamined count are retained.
    """
    if task.tube['id'] != 'glass' or task.data.body('glass').xmat[8] < math.cos(math.radians(15)):
        return None
    live_before = task.data.qpos.copy()
    model, data = task.model, task.data
    reference = task._carry_reference()
    start = data.qpos[task.offset:task.offset+5].copy()
    grip = float(data.qpos[task.offset+5])
    measured_quat = data.joint('glass_free').qpos[3:].copy()
    table = model.geom('table').id
    center, half = data.geom_xpos[table, :2], model.geom_size[table, :2]
    source = data.body('glass').xpos[:2]
    grid = .02
    xs = center[0]+np.arange(math.ceil(-half[0]/grid), math.floor(half[0]/grid)+1)*grid
    ys = center[1]+np.arange(math.ceil(-half[1]/grid), math.floor(half[1]/grid)+1)*grid
    positions = []
    for x in xs:
        for y in ys:
            p = np.array([x, y, TABLE_Z])
            if np.linalg.norm(p[:2]-source) < .06 or np.linalg.norm(p[:2]-final_target[:2]) < .03:
                continue
            if not _clear_buffer(task, p, np.array([1., 0., 0., 0.])):
                continue
            cost = np.linalg.norm(p[:2]-source)+np.linalg.norm(p[:2]-final_target[:2])
            positions.append((float(cost), p))
    positions.sort(key=lambda row: row[0])
    task.buffer_search = {'scope': 'Privileged measured-carry/hypothetical-settled-regrasp geometry only.',
                          'grid_spacing_m': grid, 'candidate_count': len(positions),
                          'maximum_examined': maximum_candidates, 'rows': [], 'selected': None}
    saved = (task.destination_position.copy(), task.destination_quaternion.copy(),
             task.destination_rotation.copy(), task.placement_axis_target.copy(),
             task.placement_free_yaw, task.orientation_constraint, task.destination.copy())
    original_reference = task.reference if hasattr(task, 'reference') else None
    try:
        for _, resting_center in positions[:maximum_candidates]:
            row = {'resting_center_m': resting_center.tolist(), 'placement_checks': []}
            task.buffer_search['rows'].append(row)
            # A future grasp is checked first to avoid repeating physical buffer
            # actions from which the requested final setting is still unsolved.
            try:
                continuation = _horizontal_continuation(task, resting_center, final_target, candidate_type)
                row['continuation'] = continuation
            except PlanningError as exc:
                row['error'] = str(exc)
                continue
            for label, quaternion in (('upright', np.array([1., 0., 0., 0.])), ('measured_held_tilt', measured_quat)):
                low, _ = body_bounds(model, 'glass', quaternion)
                target = resting_center.copy()
                target[2] = TABLE_Z-low[2]
                if not _clear_buffer(task, target, quaternion):
                    row['placement_checks'].append({'orientation': label, 'error': 'Actual tilted bounds lack buffer clearance.'})
                    continue
                task.destination_position = target
                task.destination_quaternion = quaternion.copy()
                task.destination_rotation = np.empty((3, 3))
                mujoco.mju_quat2Mat(task.destination_rotation.ravel(), quaternion)
                task.placement_axis_target = task.destination_rotation[:, 2].copy()
                task.placement_free_yaw = True
                task._place_axes(reference)
                for height in (.04, .02, .065):
                    check = {'orientation': label, 'height_m': height}
                    row['placement_checks'].append(check)
                    try:
                        _, above = task._point_for_center(target+[0., 0., height], reference, start)
                        _, lower = task._point_for_center(target, reference, above)
                        task._check_path([lower], grip, True, carry=reference, support=task.base_geom)
                        route = task._route(start, above, grip, carry=reference, allow_target=True, iterations=60)
                        task._edge(above, lower, grip, carry=reference, support=task.base_geom, allow_target=True)
                        check['passed_geometry'] = True
                        task.destination = {'id': 'glass_place', 'position_m': target.tolist()}
                        task.orientation_constraint = 'Measured temporary glass pose; unchanged physical settling and final-task checks.'
                        task.reference = reference
                        task.buffer_search['selected'] = {'position_m': target.tolist(), 'quaternion_wxyz': quaternion.tolist(),
                                                         'expected_settled_center_m': resting_center.tolist(),
                                                         'continuation': continuation}
                        task.metrics['measured_buffer_selected'] = True
                        task.metrics['measured_buffer_examined'] = len(task.buffer_search['rows'])
                        return route
                    except PlanningError as exc:
                        check['error'] = str(exc)
        return None
    finally:
        assert np.array_equal(data.qpos, live_before), 'Buffer planning changed live coordinates.'
        task.buffer_search['unexamined_candidates'] = len(positions)-len(task.buffer_search['rows'])
        if task.buffer_search['selected'] is None:
            (task.destination_position, task.destination_quaternion, task.destination_rotation,
             task.placement_axis_target, task.placement_free_yaw,
             task.orientation_constraint, task.destination) = saved
            if original_reference is not None:
                task.reference = original_reference
            task._place_axes(reference)
