"""Horizontal diameter grasp from checked geometry and one full physical route.

The67mm-band, tool-Y-up contact passed the final leg from one exposed upright
glass bridge. Other sources/bands require full planning and physical validation.
No seed, object position, target, physics or joint-limit overrides are encoded.
"""
import numpy as np

# Body-up expressed in the gripper, from the collision-checked proof001 contact.
BODY_UP_IN_TOOL=np.array([-.0006212532843516971,.9999966397383193,.0025168544706080306])
TOOL_CONTACT_POINT=np.array([.01680914902460795,-.07192885098907144,-.09815831530628913])+BODY_UP_IN_TOOL*.072

def horizontal_glass_diameter_candidates(data,name,candidate_type):
    """Return both explicit bands, with the physically passing67mm band first."""
    if name!='glass':return []
    body=data.body(name);rotation=body.xmat.reshape(3,3);rows=[]
    for band in (.067,.072):
        candidate=candidate_type(f'horizontal_glass_diameter_{band:.3f}',
            body.xpos+rotation@np.array([0.,0.,band]),TOOL_CONTACT_POINT.copy(),
            1,np.array([0.,0.,1.]),None,.85,None,.65)
        candidate.local_axis=BODY_UP_IN_TOOL.copy();rows.append(candidate)
    return rows

def horizontal_glass_ik_seeds():
    """Generic alternate elbow/pitch branch; every solve still needs all checks.

    Rolls−2.4 and0 passed source/goal/path checks at the actual bridge states;
    +2.4 failed and is retained as a bounded unsolved search branch.
    """
    return [np.array([0.,.8,.4,-1.2,roll]) for roll in (-2.4,0.,2.4)]
