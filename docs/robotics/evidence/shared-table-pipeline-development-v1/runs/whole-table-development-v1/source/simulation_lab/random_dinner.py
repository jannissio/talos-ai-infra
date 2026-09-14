"""Reset-only joint tabletop sampling, independent of learned policy coverage.

The arm-length bound is necessary geometry only; it does not certify a grasp,
collision-free path or task completion. Every rejected proposal is returned.
"""
from __future__ import annotations
import math
import mujoco
import numpy as np
from .dinner import OBJECTS
from .scene import HOME

FAMILIES = {name: ('upright', 'inverted', 'sideways') if spec['kind'] in ('bottle', 'mug', 'glass')
            else ('face_up', 'face_down') for name, spec in OBJECTS.items()}


def matrix(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def orientation(family, yaw, axial_roll=0.):
    z = np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
    tilt = 0. if family in ('upright', 'face_up') else math.pi/2 if family == 'sideways' else math.pi
    y = np.array([math.cos(tilt/2), 0., math.sin(tilt/2), 0.])
    q = np.empty(4); mujoco.mju_mulQuat(q, z, y)
    if family == 'sideways':
        spin = np.array([math.cos(axial_roll/2), 0., 0., math.sin(axial_roll/2)])
        combined = np.empty(4); mujoco.mju_mulQuat(combined, q, spin); q = combined
    return q


def half_extents(kind, size, rotation):
    """Exact axis-aligned support extents for the assets' convex primitives."""
    a = np.asarray(rotation); s = np.asarray(size)
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_BOX:return np.abs(a) @ s
    if kind == types.mjGEOM_SPHERE:return np.full(3, s[0])
    if kind == types.mjGEOM_CAPSULE:return np.full(3, s[0])+np.abs(a[:, 2])*s[1]
    if kind == types.mjGEOM_CYLINDER:return s[0]*np.sqrt(np.sum(a[:, :2]**2, axis=1))+s[1]*np.abs(a[:, 2])
    if kind == types.mjGEOM_ELLIPSOID:return np.sqrt((a*a) @ (s*s))
    raise ValueError('No exact support implementation for this collision primitive.')


def body_bounds(model, name, quaternion):
    body = model.body(name).id; rotation = matrix(quaternion)
    lows, highs = [], []
    for geom in np.flatnonzero(model.geom_bodyid == body):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
        center = rotation @ model.geom_pos[geom]
        extent = half_extents(model.geom_type[geom], model.geom_size[geom], rotation @ matrix(model.geom_quat[geom]))
        lows.append(center-extent); highs.append(center+extent)
    if not lows:raise ValueError('Object has no collision geometry.')
    return np.min(lows, axis=0), np.max(highs, axis=0)


def arm_outer_bounds(model, data):
    result = {}
    for side in ('left', 'right'):
        root = model.body(side+'_shoulder').id
        body = model.body(side+'_gripper').id
        radius = float(np.linalg.norm(model.site(side+'_gripperframe').pos))+.02
        while body != root:
            radius += float(np.linalg.norm(model.body_pos[body]))
            first, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
            radius += 2*sum(float(np.linalg.norm(model.jnt_pos[j])) for j in range(first, first+count))
            body = int(model.body_parentid[body])
            if not body:raise ValueError('Unexpected robot chain.')
        first = int(model.body_jntadr[root])
        radius += float(np.linalg.norm(model.jnt_pos[first]))
        result[side] = {'origin_m': data.body(root).xpos.copy().tolist(), 'radius_m': radius}
    return result


def draw(model, seed, *, maximum_proposals_per_item=256):
    """Draw seven objects together on the shared tabletop; no control is run."""
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
    for index, name in enumerate(OBJECTS):
        a = int(model.joint(name+'_free').qposadr[0])
        data.qpos[a:a+7] = [0., 0., 3.+index, 1., 0., 0., 0.]
    mujoco.mj_forward(model, data)
    table = model.geom('table').id
    table_xy = data.geom_xpos[table, :2].copy()
    table_half = model.geom_size[table, :2].copy()
    table_z = float(data.geom_xpos[table, 2]+model.geom_size[table, 2])
    bounds = arm_outer_bounds(model, data)
    rng = np.random.default_rng(seed)
    order = rng.permutation(list(OBJECTS)).tolist()
    proposals, objects = [], {}
    for name in order:
        family = str(rng.choice(FAMILIES[name])); yaw = float(rng.uniform(-math.pi, math.pi))
        axial = float(rng.uniform(-math.pi, math.pi)) if family == 'sideways' else 0.
        quat = orientation(family, yaw, axial)
        low, high = body_bounds(model, name, quat)
        center_offset = (low+high)/2; radius = float(np.linalg.norm((high-low)/2))
        limits = np.stack((table_xy-table_half-low[:2]+.002, table_xy+table_half-high[:2]-.002))
        if np.any(limits[1] <= limits[0]):raise ValueError('Object does not fit on the table.')
        address = int(model.joint(name+'_free').qposadr[0]); body = model.body(name).id
        accepted = False
        for attempt in range(maximum_proposals_per_item):
            xy = rng.uniform(limits[0], limits[1]); xyz = np.r_[xy, table_z-low[2]+.001]
            center = xyz+center_offset
            arms = [side for side, arm in bounds.items()
                    if np.linalg.norm(center-arm['origin_m']) <= arm['radius_m']+radius]
            row = {'object_id': name, 'attempt': attempt, 'position_m': xyz.tolist(), 'quaternion_wxyz': quat.tolist(),
                   'family': family, 'yaw_rad': yaw, 'axial_roll_rad': axial, 'candidate_arms': arms}
            if not arms:
                row['rejection'] = 'outside_conservative_arm_length_bound'
            else:
                data.qpos[address:address+7] = np.r_[xyz, quat]
                mujoco.mj_forward(model, data)
                collisions = [{'other_body': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[c.geom2] if model.geom_bodyid[c.geom1] == body else model.geom_bodyid[c.geom1])),
                    'penetration_m': -float(c.dist)} for c in data.contact
                    if body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                    and model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2] and c.dist < -1e-5]
                row['rejection'] = 'initial_geometric_intersection' if collisions else None
                row['collisions'] = collisions
            proposals.append(row)
            if row['rejection'] is None:
                objects[name] = row; accepted = True; break
        if not accepted:
            return data, {'seed': seed, 'generated': False, 'failed_item': name, 'order': order,
                          'objects': objects, 'proposals': proposals, 'arm_outer_bounds': bounds}
    data.qvel[:] = 0.; data.time = 0.; mujoco.mj_forward(model, data)
    return data, {'seed': seed, 'generated': True, 'order': order, 'objects': objects, 'proposals': proposals,
        'arm_outer_bounds': bounds, 'table_xy_bounds_m': [list(table_xy-table_half), list(table_xy+table_half)],
        'table_z': table_z, 'fixture': 'Original closed drawer and parked robot bases.',
        'scope': 'Reset geometry only; arm bounds do not certify a grasp or a collision-free route.'}


def assess(model, data):
    """Physical validity after settling, with no upright-only requirement."""
    table = model.geom('table').id; xy = data.geom_xpos[table, :2]; half = model.geom_size[table, :2]
    table_z = float(data.geom_xpos[table, 2]+model.geom_size[table, 2])
    rows = {}
    for name in OBJECTS:
        address = int(model.joint(name+'_free').qposadr[0]); dof = int(model.joint(name+'_free').dofadr[0])
        body = model.body(name).id; pose = data.qpos[address:address+7]
        low, high = body_bounds(model, name, pose[3:]); low += pose[:3]; high += pose[:3]
        contacts = [c for c in data.contact if body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                    and model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]]
        penetration = max((-float(c.dist) for c in contacts), default=0.)
        table_contact = any(table in (c.geom1, c.geom2) for c in contacts)
        speed = float(np.linalg.norm(data.qvel[dof:dof+3])); angular = float(np.linalg.norm(data.qvel[dof+3:dof+6]))
        up = float(data.body(name).xmat[8])
        reasons = []
        if penetration > .001:reasons.append('geometric_penetration')
        if np.any(low[:2] < xy-half-.001) or np.any(high[:2] > xy+half+.001):reasons.append('off_table')
        if not table_contact or low[2] < table_z-.001 or low[2] > table_z+.002:reasons.append('not_supported_on_table')
        if speed >= .003 or angular >= .08:reasons.append('not_settled')
        rows[name] = {'valid': not reasons, 'reasons': reasons, 'position_m': pose[:3].tolist(),
            'quaternion_wxyz': pose[3:].tolist(), 'speed_m_s': speed, 'angular_speed_rad_s': angular,
            'penetration_m': penetration, 'table_contact': table_contact,
            'realized_family': 'upright' if up > .9 else 'inverted' if up < -.9 else 'sideways' if abs(up) < .25 else 'tilted',
            'body_up_z': up, 'bounds_min_m': low.tolist(), 'bounds_max_m': high.tolist()}
    return {'valid': all(r['valid'] for r in rows.values()), 'objects': rows,
            'meaning': 'Stable supported geometry only; manipulation feasibility remains unmeasured.'}
