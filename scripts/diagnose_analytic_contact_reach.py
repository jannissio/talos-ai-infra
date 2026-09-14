"""Conservative analytic robot reach and separating-plane contact certificates.

No optimizer, sampling, model edits or physics steps are used.
"""
import contextlib,hashlib,itertools,json,os,sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from simulation_lab.random_dinner import arm_outer_bounds,body_bounds
from simulation_lab.dinner import OBJECTS
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put,arrays

T=mujoco.mjtGeom
def collision(m,g):return bool(m.geom_contype[g] or m.geom_conaffinity[g])
def support(m,d,g,n):
    """Exact support radius of each collision primitive in world direction n."""
    a=d.geom_xmat[g].reshape(3,3).T@n;s=m.geom_size[g];kind=m.geom_type[g]
    if kind==T.mjGEOM_BOX:return float(np.abs(a)@s)
    if kind==T.mjGEOM_SPHERE:return float(s[0])
    if kind==T.mjGEOM_CAPSULE:return float(s[0]+abs(a[2])*s[1])
    if kind==T.mjGEOM_CYLINDER:return float(s[0]*np.linalg.norm(a[:2])+s[1]*abs(a[2]))
    if kind==T.mjGEOM_ELLIPSOID:return float(np.linalg.norm(s*a))
    if kind==T.mjGEOM_MESH:
        mesh=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        return float(np.max(v@a))
    raise ValueError('Unsupported collision type')

def farthest_radius(m,d,g,origin):
    """Exact for listed primitives/mesh hull, enclosing bound for ellipsoid."""
    c=d.geom_xpos[g];r=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];v=r.T@(c-origin);kind=m.geom_type[g]
    if kind==T.mjGEOM_BOX:return float(np.linalg.norm(np.abs(v)+s))
    if kind==T.mjGEOM_SPHERE:return float(np.linalg.norm(v)+s[0])
    if kind==T.mjGEOM_CAPSULE:return float(max(np.linalg.norm(v+[0,0,s[1]]),np.linalg.norm(v-[0,0,s[1]]))+s[0])
    if kind==T.mjGEOM_CYLINDER:return float(np.hypot(np.linalg.norm(v[:2])+s[0],abs(v[2])+s[1]))
    if kind==T.mjGEOM_ELLIPSOID:return float(np.linalg.norm(v)+max(s))
    if kind==T.mjGEOM_MESH:
        mesh=int(m.geom_dataid[g]);vertices=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        return float(np.max(np.linalg.norm(vertices+v,axis=1)))
    raise ValueError('Unsupported collision type')

def point_distance(m,d,g,point):
    """Exact closed-volume distance, except documented ellipsoid enclosing box."""
    v=d.geom_xmat[g].reshape(3,3).T@(point-d.geom_xpos[g]);s=m.geom_size[g];kind=m.geom_type[g]
    if kind==T.mjGEOM_BOX:return float(np.linalg.norm(np.maximum(np.abs(v)-s,0))), 'exact_box'
    if kind==T.mjGEOM_SPHERE:return max(0.,float(np.linalg.norm(v)-s[0])), 'exact_sphere'
    if kind==T.mjGEOM_CAPSULE:return max(0.,float(np.linalg.norm(np.r_[v[:2],max(abs(v[2])-s[1],0)])-s[0])), 'exact_capsule'
    if kind==T.mjGEOM_CYLINDER:return float(np.linalg.norm([max(np.linalg.norm(v[:2])-s[0],0),max(abs(v[2])-s[1],0)])), 'exact_cylinder'
    if kind==T.mjGEOM_ELLIPSOID:return float(np.linalg.norm(np.maximum(np.abs(v)-s,0))), 'conservative_enclosing_oriented_box'
    raise ValueError('Object distance implementation unavailable')

def closest_direction(m,d,g,point):
    """Analytic closest-point direction; any numerical direction still defines a valid support bound."""
    rotation=d.geom_xmat[g].reshape(3,3);v=rotation.T@(point-d.geom_xpos[g]);s=m.geom_size[g];kind=m.geom_type[g]
    if kind==T.mjGEOM_BOX:y=np.clip(v,-s,s)
    elif kind==T.mjGEOM_SPHERE:y=v*min(1.,s[0]/max(np.linalg.norm(v),1e-30))
    elif kind==T.mjGEOM_CAPSULE:
        center=np.array([0.,0.,np.clip(v[2],-s[1],s[1])]);delta=v-center;y=center+delta*min(1.,s[0]/max(np.linalg.norm(delta),1e-30))
    elif kind==T.mjGEOM_CYLINDER:y=np.r_[v[:2]*min(1.,s[0]/max(np.linalg.norm(v[:2]),1e-30)),np.clip(v[2],-s[1],s[1])]
    elif kind==T.mjGEOM_ELLIPSOID:
        if np.sum((v/s)**2)<=1:y=v
        else:
            lo=0.;hi=float(np.linalg.norm(s*v))
            for _ in range(100):
                mid=(lo+hi)/2
                if np.sum((s*v/(mid+s*s))**2)>1:lo=mid
                else:hi=mid
            y=s*s*v/(hi+s*s)
    else:raise ValueError('Unsupported object shape')
    delta=rotation@(y-v);length=np.linalg.norm(delta)
    return delta/max(length,1e-30),float(length)

def robot_bounds(m,d):
    rows={}
    for side in ('left','right'):
        root=m.body(side+'_shoulder').id;root_joint=int(m.body_jntadr[root]);origin=d.xanchor[root_joint].copy();geoms=[]
        ancestor=int(m.body_parentid[root])
        while ancestor:
            assert m.body_jntnum[ancestor]==0,'Reach-ball origin must be fixed.'
            ancestor=int(m.body_parentid[ancestor])
        for g in range(m.ngeom):
            if not collision(m,g):continue
            body=int(m.geom_bodyid[g]);chain=[];ancestor=body
            while ancestor and ancestor!=root:chain.append(ancestor);ancestor=int(m.body_parentid[ancestor])
            if ancestor!=root:continue
            chain.append(root);joints=[]
            for b in reversed(chain):
                assert m.body_jntnum[b]<=1,'Derivation audited for one hinge per body.'
                for j in range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])):
                    assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
                    joints.append(j)
            assert joints[0]==root_joint
            distances=[float(np.linalg.norm(d.xanchor[b]-d.xanchor[a])) for a,b in zip(joints[:-1],joints[1:])]
            terminal=farthest_radius(m,d,g,d.xanchor[joints[-1]])
            name=m.geom(g).name or f'unnamed_geom_{g}';bname=m.body(body).name
            fixed=bool(name.startswith(side+'_fixed_jaw') or (bname==side+'_gripper' and m.geom_group[g]==4));moving=bname==side+'_moving_jaw_so101_v1'
            geoms.append(dict(geom_id=g,name=name,body=bname,ordinary_jaw=fixed or moving,fixed_jaw=fixed,moving_jaw=moving,joint_chain=[m.joint(j).name for j in joints],joint_anchor_segment_lengths_m=distances,terminal_geometry_radius_m=terminal,outer_radius_m=sum(distances)+terminal,margin_m=float(m.geom_margin[g])))
        rows[side]=dict(origin_m=origin.tolist(),all_arm_radius_m=max(x['outer_radius_m'] for x in geoms),jaw_radius_m=max(x['outer_radius_m'] for x in geoms if x['ordinary_jaw']),fixed_jaw_radius_m=max(x['outer_radius_m'] for x in geoms if x['fixed_jaw']),moving_jaw_radius_m=max(x['outer_radius_m'] for x in geoms if x['moving_jaw']),geoms=geoms)
    return rows

out=ROOT/'.run/analytic-contact-reach-v5';log=out.with_suffix('.log')
if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    manifest={}
    for name in ('scripts/diagnose_analytic_contact_reach.py','simulation_lab/random_dinner.py','simulation_lab/scene.py','simulation_lab/dinner.py','simulation_lab/autonomy.py'):
        payload=(ROOT/name).read_bytes();put(out/'source'/name,payload);manifest[name]=hashlib.sha256(payload).hexdigest()
    results=[]
    for version,seed in ((17,4001),(16,4002),(18,4004)):
        source=ROOT/f'.run/whole-table-development-v{version}-seed{seed}';folder=out/str(seed)
        model_source=ROOT/{4001:'.run/side-plate-grasp-v10-reverse-a000-t080',4002:'.run/glass-reorientation-v1-branches-inverted',4004:'.run/glass-placement-v5-middle-top-wrap'}[seed]
        comparisons={}
        for name in ('scene.py','dinner.py','random_dinner.py'):
            current=(ROOT/'simulation_lab'/name).read_bytes();prior=(model_source/'source/simulation_lab'/name).read_bytes();comparisons[name]=hashlib.sha256(current).hexdigest()==hashlib.sha256(prior).hexdigest()
        assert all(comparisons.values()),'Original/current model generation differs; do not silently rebuild.'
        xml,layout=build_scene(seed=2026110000+seed,scenario='dinner',dinner_preset='task');assert xml.encode()==(model_source/'scene.xml').read_bytes(),'Model XML differs from preserved actual-run model.'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        with np.load(source/'initial-state.npz') as a:d.qpos[:]=a['qpos'];d.qvel[:]=a['qvel'];d.ctrl[:]=a['ctrl']
        mujoco.mj_forward(m,d);put(folder/'scene.xml',xml.encode());arrays(folder/'actual-initial-state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
        arrays(folder/'compiled-geometry-inputs.npz',body_parentid=m.body_parentid,body_pos=m.body_pos,body_quat=m.body_quat,body_jntadr=m.body_jntadr,body_jntnum=m.body_jntnum,jnt_type=m.jnt_type,jnt_pos=m.jnt_pos,jnt_axis=m.jnt_axis,jnt_range=m.jnt_range,geom_bodyid=m.geom_bodyid,geom_type=m.geom_type,geom_pos=m.geom_pos,geom_quat=m.geom_quat,geom_size=m.geom_size,geom_group=m.geom_group,geom_contype=m.geom_contype,geom_conaffinity=m.geom_conaffinity,geom_margin=m.geom_margin,geom_dataid=m.geom_dataid,mesh_vert=m.mesh_vert,mesh_vertadr=m.mesh_vertadr,mesh_vertnum=m.mesh_vertnum,actual_anchor_positions=d.xanchor,actual_geom_positions=d.geom_xpos,actual_geom_rotations=d.geom_xmat)
        # All hinge positions may be arbitrary: anchor-to-anchor lengths and
        # last-anchor-to-collision-hull radii are invariant in this serial chain.
        bounds=robot_bounds(m,d);sampler=arm_outer_bounds(m,d);objects=[]
        for name in OBJECTS:
            body=m.body(name).id;gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and collision(m,g)];sides={}
            for side,bound in bounds.items():
                origin=np.asarray(bound['origin_m']);delta=d.xpos[body]-origin;n=delta/np.linalg.norm(delta)
                primitive_rows=[]
                for g in gs:
                    distance,method=point_distance(m,d,g,origin)
                    lower=float(n@(d.geom_xpos[g]-origin)-support(m,d,g,-n))
                    pn,exact_distance=closest_direction(m,d,g,origin);optimal_lower=float(pn@(d.geom_xpos[g]-origin)-support(m,d,g,-pn))
                    primitive_rows.append(dict(geom=m.geom(g).name,distance_lower_bound_m=distance,distance_method=method,separating_projection_min_m=lower,primitive_normal_world=pn.tolist(),primitive_support_lower_bound_m=optimal_lower,analytic_closest_distance_m=exact_distance,margin_m=float(m.geom_margin[g])))
                min_projection=min(x['separating_projection_min_m'] for x in primitive_rows);minimum_distance=min(x['distance_lower_bound_m'] for x in primitive_rows)
                padding=max(x['margin_m'] for x in primitive_rows)+max(x['margin_m'] for x in bound['geoms'])+(max(m.pair_margin) if m.npair else 0.)
                gap=min_projection-bound['all_arm_radius_m']-padding
                primitive_lower=min(x['primitive_support_lower_bound_m'] for x in primitive_rows)
                bodyradius=max(farthest_radius(m,d,g,d.xpos[body]) for g in gs)
                sides[side]=dict(normal_world=n.tolist(),origin_m=origin.tolist(),body_origin_distance_m=float(np.linalg.norm(delta)),minimum_collision_volume_distance_lower_bound_m=minimum_distance,separating_projection_min_m=min_projection,all_arm_outer_radius_m=bound['all_arm_radius_m'],jaw_outer_radius_m=bound['jaw_radius_m'],contact_margin_padding_m=padding,strict_all_arm_gap_m=gap,strict_jaw_gap_m=min_projection-bound['jaw_radius_m']-padding,current_pose_contact_excluded=gap>1e-5,translation_4mm_fixed_orientation_excluded=gap>.00401,arbitrary_rotation_and_translation_4mm_excluded=np.linalg.norm(delta)-bodyradius-bound['all_arm_radius_m']-padding>.00401,object_all_orientation_radius_m=bodyradius,primitives=primitive_rows)
                sides[side].update(primitive_support_lower_bound_m=primitive_lower,fixed_jaw_radius_m=bound['fixed_jaw_radius_m'],moving_jaw_radius_m=bound['moving_jaw_radius_m'],fixed_jaw_gap_m=primitive_lower-bound['fixed_jaw_radius_m']-padding,moving_jaw_gap_m=primitive_lower-bound['moving_jaw_radius_m']-padding,two_finger_grasp_excluded=primitive_lower-min(bound['fixed_jaw_radius_m'],bound['moving_jaw_radius_m'])-padding>1e-5)
                sides[side].update(primitive_planes_all_arm_gap_m=primitive_lower-bound['all_arm_radius_m']-padding,primitive_planes_current_pose_contact_excluded=primitive_lower-bound['all_arm_radius_m']-padding>1e-5,two_finger_4mm_translation_fixed_orientation_excluded=primitive_lower-min(bound['fixed_jaw_radius_m'],bound['moving_jaw_radius_m'])-padding>.00401,two_finger_arbitrary_rotation_4mm_translation_excluded=np.linalg.norm(delta)-bodyradius-min(bound['fixed_jaw_radius_m'],bound['moving_jaw_radius_m'])-padding>.00401)
            low,high=body_bounds(m,name,d.joint(name+'_free').qpos[3:]);center=d.xpos[body]+(low+high)/2;boxradius=float(np.linalg.norm((high-low)/2))
            objects.append(dict(object=name,pose=d.joint(name+'_free').qpos.tolist(),both_arms_current_pose_excluded=all(x['current_pose_contact_excluded'] for x in sides.values()),both_arms_with_4mm_translation_excluded=all(x['translation_4mm_fixed_orientation_excluded'] for x in sides.values()),both_arms_arbitrary_rotation_and_4mm_translation_excluded=all(x['arbitrary_rotation_and_translation_4mm_excluded'] for x in sides.values()),sampler_enclosing_aabb_sphere=dict(center_m=center.tolist(),radius_m=boxradius,overlap_arms=[side for side,b in sampler.items() if np.linalg.norm(center-b['origin_m'])<=b['radius_m']+boxradius]),sides=sides))
            objects[-1]['both_arms_two_finger_grasp_excluded']=all(x['two_finger_grasp_excluded'] for x in sides.values())
            objects[-1]['both_arms_primitive_planes_contact_excluded']=all(x['primitive_planes_current_pose_contact_excluded'] for x in sides.values())
            objects[-1]['both_arms_two_finger_4mm_translation_fixed_orientation_excluded']=all(x['two_finger_4mm_translation_fixed_orientation_excluded'] for x in sides.values())
            objects[-1]['both_arms_two_finger_arbitrary_rotation_4mm_translation_excluded']=all(x['two_finger_arbitrary_rotation_4mm_translation_excluded'] for x in sides.values())
        report=dict(seed=2026110000+seed,input=source.relative_to(ROOT).as_posix(),input_state_sha256=hashlib.sha256((source/'initial-state.npz').read_bytes()).hexdigest(),model_source=model_source.relative_to(ROOT).as_posix(),model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),preserved_model_xml_identical=True,unchanged_source_comparisons=comparisons,robot_bounds=bounds,existing_sampler_bounds=sampler,objects=objects)
        report=json.loads(json.dumps(report,default=lambda x:x.item()))
        put(folder/'report.json',report);results.append(report)
        print(seed,[(x['object'],x['both_arms_current_pose_excluded'],{k:round(v['strict_all_arm_gap_m']*1000,3) for k,v in x['sides'].items()}) for x in objects],flush=True)
    put(out/'source-manifest.json',manifest)
    put(out/'report.json',dict(scope='Analytic current-pose contact exclusion only. All rotations at every serial hinge are allowed in the outer bound; joint limits and obstacles can only shrink it. No optimizer failure is evidence. Arbitrary prior object relocation, tool-mediated rearrangement or rotation is not excluded unless separately certified.',preflight=preflight,derivation=['For each collision geometry, list serial hinge anchors from the fixed shoulder-pan anchor. Distances between adjacent anchors are invariant. Any geometry point lies within the sum of these lengths plus its farthest distance from the terminal anchor, by the triangle inequality. Taking the maximum covers every collision-enabled arm geometry, a superset of allowed finger contacts.','For a unit normal n, each object primitive lies on or beyond n dot(center-root) minus its exact support in direction -n. The union lower support is the minimum over primitives. If it strictly exceeds the complete robot outer radius plus contact margins and10um numerical reserve, the entire object and every arm collision surface are separated by a plane.','A stronger union certificate allows a different support plane per convex object primitive. Normals come from analytic nearest points; ellipsoid direction uses100 bisections of its monotone one-dimensional nearest-point equation. Any resulting unit direction remains a valid exact-support bound, independent of root-finding convergence. Every primitive must be separated from the enclosing robot ball.','Fixed/moving jaw groups exactly match the current teacher contact grouping, filtered to collision-enabled geometry. Ordinary two-finger grasp requires both groups; a strict bound excluding either group excludes that arm ordinary two-finger grasp. This is weaker than excluding all arm contact and is labeled separately.','Existing sampler adds a bounding sphere of the object AABB to a site-derived arm ball. Overlap is only a permissive screen, not a feasible-contact certificate. The new larger arm ball does not by itself prove the older smaller radius unsound; the older site-plus20mm assumption lacks the full collision-hull derivation provided here.'],results=results))
