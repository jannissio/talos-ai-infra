"""Analytic fixed-height shoulder-sweep separation at the actual V24 state."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put

out=ROOT/'.run/sideplate-pregrasp-separation-v2';log=out.with_suffix('.log');source=ROOT/'.run/sideplate-pregrasp-geometry-v1';bound_source=ROOT/'.run/analytic-contact-reach-v5/4004/report.json'
if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(out,32*1024**2);out.mkdir(parents=True)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    xml=(source/'scene.xml').read_bytes();assert xml==(ROOT/'.run/analytic-contact-reach-v5/4004/scene.xml').read_bytes()
    model=mujoco.MjModel.from_xml_string(xml.decode());data=mujoco.MjData(model)
    with np.load(source/'actual-post-action03-state.npz') as a:data.qpos[:]=a['qpos'];data.qvel[:]=a['qvel'];data.ctrl[:]=a['ctrl']
    mujoco.mj_forward(model,data);bounds=json.loads(bound_source.read_text())['robot_bounds'];rows=[]
    for side in ('left','right'):
        pan=model.joint(side+'_shoulder_pan').id;lift=model.joint(side+'_shoulder_lift').id
        origin=data.xanchor[pan].copy();axis=data.xaxis[pan].copy();delta=data.xanchor[lift]-origin
        height=float(axis@delta);circle_center=origin+height*axis;radius=float(np.linalg.norm(delta-height*axis))
        body=data.body('side_plate');normal=body.xpos-circle_center;normal/=np.linalg.norm(normal)
        robot=[]
        for g in bounds[side]['geoms']:
            chain=g['joint_chain']
            if side+'_shoulder_lift' in chain:
                assert chain[1]==side+'_shoulder_lift'
                remaining=float(g['outer_radius_m']-g['joint_anchor_segment_lengths_m'][0])
                upper=radius*np.sqrt(max(0.,1-float(normal@axis)**2))+remaining
                kind='shoulder_lift_circle_plus_remaining_chain_ball'
            else:
                remaining=g['outer_radius_m'];upper=float(normal@(origin-circle_center))+remaining;kind='root_anchor_ball'
            robot.append(dict(geom_id=g['geom_id'],name=g['name'],remaining_chain_radius_m=remaining,projection_upper_m=upper,bound_type=kind))
        robot_upper=max(g['projection_upper_m'] for g in robot);object_rows=[]
        for g in range(model.ngeom):
            if model.geom_bodyid[g]!=body.id or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            a=data.geom_xmat[g].reshape(3,3).T@normal;s=model.geom_size[g]
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:extent=float(np.abs(a)@s)
            elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=float(s[0]*np.linalg.norm(a[:2])+s[1]*abs(a[2]))
            else:raise ValueError('Unexpected sideplate collision shape')
            lower=float(normal@(data.geom_xpos[g]-circle_center)-extent)
            object_rows.append(dict(name=model.geom(g).name,projection_lower_m=lower))
        object_lower=min(g['projection_lower_m'] for g in object_rows)
        margin=max(float(model.geom_margin[g['geom_id']]) for g in robot)+max(float(model.geom_margin[model.geom(g['name']).id]) for g in object_rows)+(float(max(model.pair_margin)) if model.npair else 0.)
        gap=object_lower-robot_upper-margin
        rows.append(dict(side=side,pan_anchor=origin.tolist(),pan_axis=axis.tolist(),shoulder_lift_circle_center=circle_center.tolist(),shoulder_lift_circle_radius_m=radius,fixed_axial_offset_m=height,normal=normal.tolist(),robot_projection_upper_m=robot_upper,object_projection_lower_m=object_lower,contact_margin_m=margin,strict_gap_m=gap,initial_direct_contact_excluded=gap>.00001,translation_4mm_fixed_orientation_excluded=gap>.00401,robot_geometries=robot,object_primitives=object_rows))
    manifest={}
    for path in (Path(__file__),source/'scene.xml',source/'actual-post-action03-state.npz',source/'prefix.json',bound_source):
        manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    put(out/'source.py',Path(__file__).read_bytes());put(out/'input-prefix.json',json.loads((source/'prefix.json').read_text()))
    report=dict(scope='Actual inverted side_plate after V24 accepted actions00–03; analytic initial direct robot-contact exclusion, not a global indirect-rearrangement impossibility claim.',preflight=preflight,physical_steps=0,model_or_object_pose_edits=False,sideplate_pose=data.joint('side_plate_free').qpos.tolist(),numerical_reserve_m=.00001,rows=rows,both_arms_direct_contact_excluded=all(r['initial_direct_contact_excluded'] for r in rows),hashes_sha256=manifest,derivation=['The shoulder-pan anchor and axis are fixed. The next hinge anchor sweeps a circle about that axis: its axial54.2mm component is fixed, while only its35.47mm radial component can turn with pan. Ignoring pan limits and allowing a complete circle is conservative.','For every arm collision geometry downstream of shoulder_lift, retain the previously derived triangle-inequality ball for all remaining anchor segments and terminal collision hull. Its center sweeps that circle. The support in unit direction n is radius_circle*sqrt(1-(n dot axis)^2)+radius_remaining_chain, relative to the circle center.','Root-level arm collision geometry is separately enclosed by its original root-anchor ball. Take the maximum support over every collision-enabled arm geometry, including complete fixed/moving jaw meshes.','Each current sideplate box/cylinder has an exact support minimum along the same normal. If every object primitive lies above the maximum robot support by more than contact margins plus10um, no arm collision point can touch that object in this pose. No optimizer result is used in this certificate.'],implication='The single-finger top-rim drag hypothesis cannot begin through direct arm contact at this measured pose. Enclosing-sphere overlap previously left this unresolved; it did not establish a reachable moving finger. No physical pregrasp alternatives are justified by these inputs. Object-assisted indirect rearrangement and changed future object poses are outside this certificate.')
    report=json.loads(json.dumps(report,default=lambda x:x.item()))
    put(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('rows',)}));print([(r['side'],r['strict_gap_m']) for r in rows])
