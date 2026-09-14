"""Supplemental side-plate grasps, established in exposed physical diagnosis.

The fixed finger approaches outside the rim and the moving finger closes from
inside. The ordinary teacher still checks IK, collisions, contact, unsupported
hold, release and parking. This candidate family is not a coverage guarantee.
"""
import math
import numpy as np


def reverse_rim_candidates(data, name, candidate_type, opening):
    """Return outside-fixed-finger rim candidates for the existing teacher.

    Pass the teacher's GraspCandidate type explicitly to avoid a circular import.
    The angle-zero member passed seed 2026114001's full physical direct transfer
    in .run/side-plate-grasp-v10-reverse-a000-t080, with exact motor replay.
    Other poses/orientations remain candidates requiring independent execution.
    """
    if name != 'side_plate':
        return []
    body = data.body(name)
    center = body.xpos.copy()
    rotation = body.xmat.reshape(3, 3)
    rows = []
    for angle in np.arange(12)*math.pi/6:
        radial = np.array([math.cos(angle), math.sin(angle), 0.])
        rows.append(candidate_type(
            f'reverse_clear_rim_{angle:.5f}',
            center+rotation@(radial*.048+[0., 0., .0105]),
            np.array([-.001, 0., -.100]), 2, np.array([0., 0., 1.]),
            -(rotation@radial), opening, None, .8))
    return rows
