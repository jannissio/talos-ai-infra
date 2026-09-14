"""Supplemental glass wraps using a deeper portion of the finger pads.

One member passed the upright glass's full buffered placement from the exposed
post-mug joint seed2026114004 state. Every other start still requires the shared
teacher's full physical checks. Inverted/sideways glass remain unsolved.
"""
import math
import numpy as np


def padded_glass_wrap_candidates(data,name,candidate_type):
    """Return ordinary GraspCandidate instances without importing the teacher."""
    if name!='glass':return []
    body=data.body(name);center=body.xpos.copy();rotation=body.xmat.reshape(3,3)
    angles=[math.pi/4]+[x for x in np.linspace(-math.pi,math.pi,8,endpoint=False) if abs(x-math.pi/4)>1e-9]
    rows=[]
    for height in (.045,.030,.060):
        for angle in angles:
            closing=np.array([math.cos(angle),math.sin(angle),0.])
            if abs(rotation[2,2])<.8 and abs(np.dot(closing,rotation[:,2]))>.4:continue
            rows.append(candidate_type(f'padded_glass_wrap_{height}_{angle:.5f}',
                center+rotation@np.array([0.,0.,height]),np.array([.020,0.,-.090]),
                2,np.array([0.,0.,1.]),closing,.85,None,.65))
    return rows
