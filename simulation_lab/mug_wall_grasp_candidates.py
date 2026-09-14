"""Experimental deeper mug-wall pinch, derived from the existing DinnerTask.

The optional xyz+wxyz pose supplies only a grasp seed, never a state update.
Actual contact, source route, carried geometry and final placement still require
independent checks against the real model/state. No seed or position filtering.
"""
import math

import mujoco
import numpy as np


def mug_wall_grasp_candidates(data, name, candidate_type, *, pose=None, open_grip=.4):
    """Twelve full-circle wall contacts at radius23mm/height41mm, torque.35Nm.

    Tool Z follows body Z; tool X closes radially. Approach from +body Z, toward
    the open lip, before descending to the wall contact. The first azimuth is
    the legacy world closing direction projected onto the actual cross-section;
    later angles span 360deg. On an upright mug, candidate00 preserves the exact
    legacy world closing vector independently of the object's table yaw.
    """
    if name != 'mug':
        return []
    body = data.body(name)
    center, rotation = body.xpos.copy(), body.xmat.reshape(3, 3).copy()
    if pose is not None:
        values = np.asarray(pose, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)) or np.linalg.norm(values[3:]) < 1e-10:
            raise ValueError('Pose must be finite xyz plus nonzero wxyz quaternion')
        center = values[:3].copy()
        quaternion = values[3:] / np.linalg.norm(values[3:])
        mujoco.mju_quat2Mat(rotation.ravel(), quaternion)
    legacy = np.array([.55, -math.sqrt(1. - .55**2), 0.])
    projected = legacy - rotation[:, 2] * float(legacy @ rotation[:, 2])
    if np.linalg.norm(projected) < 1e-8:
        projected = rotation[:, 0].copy()
    local_projected = rotation.T @ projected
    phase = math.atan2(local_projected[1], local_projected[0])
    rows = []
    for index in range(12):
        angle = phase + index * math.pi / 6
        radial = rotation @ np.array([math.cos(angle), math.sin(angle), 0.])
        up = rotation[:, 2].copy()
        rows.append(candidate_type(f'mug_deep_wall_{index:02d}', center + .023 * radial + .041 * up,
                                   np.array([-.006, 0., -.092]), 2, up, radial,
                                   open_grip, up.copy(), .35))
    return rows
