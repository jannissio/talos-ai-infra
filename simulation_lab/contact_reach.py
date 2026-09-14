"""Conservative initial direct-contact separation for the compiled Talos arms.

``initial_direct_contact_separation(model, data, item, side, pose=None)``
returns a separating-plane certificate or an unresolved result. ``data`` must
have current forward kinematics. ``pose`` optionally supplies an object's world
[x,y,z,qw,qx,qy,qz], evaluated algebraically without changing model or data.
All yaw angles and rigid object orientations are supported. No physics, IK,
sampling, contact eligibility changes, or writes to model/data occur here.

The shoulder-pan anchor/axis must be fixed. Its next hinge anchor sweeps a full
circle: fixed axial component plus the rotating perpendicular component. Every
subsequent collision point lies within a ball of radius equal to the remaining
anchor-segment lengths plus the terminal geometry radius. Triangle inequality
makes this an outer bound for ALL hinge angles, ignoring limits and obstacles.
The circle-plus-ball support is rho*sqrt(1-(n.axis)**2)+R for unit n. Root-level
geometry instead uses its root-anchor ball. Exact object support functions then
give a plane gap; a positive gap beyond margins and numerical reserve proves
initial direct arm contact impossible. Overlap says only "unresolved".

This is NOT a task-impossibility certificate: prior direct or indirect object
rearrangement changes the pose. It is not a grasp/route feasibility test, and
must not retroactively remove trials or change a sampling protocol silently.
Unsupported articulated objects or robot chains raise ValueError, never a
certificate. Ordinary collision geometry and explicit-pair geometry are covered.
"""
import mujoco
import numpy as np


_T = mujoco.mjtGeom


def _collision_geometries(model):
    ids = set(np.flatnonzero(model.geom_contype | model.geom_conaffinity).tolist())
    ids.update(int(g) for g in model.pair_geom1)
    ids.update(int(g) for g in model.pair_geom2)
    return ids


def _vertices(model, geom):
    mesh = int(model.geom_dataid[geom])
    first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return model.mesh_vert[first:first + count]


def _support(model, geom, direction):
    """Local shape support; direction is unit length, including for meshes."""
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == _T.mjGEOM_BOX:
        return float(np.abs(direction) @ size)
    if kind == _T.mjGEOM_SPHERE:
        return float(size[0])
    if kind == _T.mjGEOM_CAPSULE:
        return float(size[0] + abs(direction[2]) * size[1])
    if kind == _T.mjGEOM_CYLINDER:
        return float(size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2]))
    if kind == _T.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(size * direction))
    if kind == _T.mjGEOM_MESH:
        return float(np.max(_vertices(model, geom) @ direction))
    raise ValueError(f"Unsupported bounded collision shape: geom {geom}")


def _radius(model, data, geom, anchor):
    """Farthest radius, with a conservative enclosing sphere for ellipsoids."""
    offset = data.geom_xmat[geom].reshape(3, 3).T @ (data.geom_xpos[geom] - anchor)
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == _T.mjGEOM_BOX:
        return float(np.linalg.norm(np.abs(offset) + size))
    if kind == _T.mjGEOM_SPHERE:
        return float(np.linalg.norm(offset) + size[0])
    if kind == _T.mjGEOM_CAPSULE:
        cap = np.array([0., 0., size[1]])
        return float(max(np.linalg.norm(offset + cap), np.linalg.norm(offset - cap)) + size[0])
    if kind == _T.mjGEOM_CYLINDER:
        return float(np.hypot(np.linalg.norm(offset[:2]) + size[0], abs(offset[2]) + size[1]))
    if kind == _T.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(offset) + max(size))
    if kind == _T.mjGEOM_MESH:
        return float(np.max(np.linalg.norm(_vertices(model, geom) + offset, axis=1)))
    raise ValueError(f"Unsupported bounded collision shape: geom {geom}")


def _descendant_path(model, body, root):
    path = []
    while body != root and body:
        path.append(body)
        body = int(model.body_parentid[body])
    return list(reversed(path + [root])) if body == root else None


def arm_contact_envelope(model, data, side):
    """Build a serial-hinge outer envelope from compiled geometry, read-only.

    Returned dictionaries contain independent lists/scalars. Recompute if the
    compiled model changes. Joint configuration may change freely afterwards.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    pan = model.joint(side + "_shoulder_pan").id
    lift = model.joint(side + "_shoulder_lift").id
    root = int(model.jnt_bodyid[pan])
    ancestor = int(model.body_parentid[root])
    while ancestor:
        if model.body_jntnum[ancestor]:
            raise ValueError("Shoulder-pan anchor must be fixed in world")
        ancestor = int(model.body_parentid[ancestor])
    origin = data.xanchor[pan].copy()
    axis = data.xaxis[pan].copy()
    axis /= np.linalg.norm(axis)
    delta = data.xanchor[lift] - origin
    height = float(axis @ delta)
    center = origin + height * axis
    radial = float(np.linalg.norm(delta - height * axis))
    rows = []
    for geom in sorted(_collision_geometries(model)):
        path = _descendant_path(model, int(model.geom_bodyid[geom]), root)
        if path is None:
            continue
        joints = []
        for body in path:
            count, first = int(model.body_jntnum[body]), int(model.body_jntadr[body])
            if count > 1:
                raise ValueError("Envelope requires at most one hinge per body")
            if count:
                if model.jnt_type[first] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Envelope requires serial hinges")
                joints.append(first)
        if not joints or joints[0] != pan or (len(joints) > 1 and joints[1] != lift):
            raise ValueError("Unexpected shoulder branch; envelope not certified")
        segments = [float(np.linalg.norm(data.xanchor[b] - data.xanchor[a]))
                    for a, b in zip(joints[:-1], joints[1:])]
        terminal = _radius(model, data, geom, data.xanchor[joints[-1]])
        downstream = len(joints) > 1
        remaining = sum(segments[1:] if downstream else segments) + terminal
        rows.append(dict(geom_id=geom, name=model.geom(geom).name,
                         circle_centered=downstream, radius_m=remaining,
                         joint_chain=[model.joint(j).name for j in joints],
                         anchor_segment_lengths_m=segments,
                         terminal_radius_m=terminal,
                         margin_m=float(model.geom_margin[geom])))
    if not rows:
        raise ValueError("Arm contains no bounded collision geometry")
    return dict(side=side, root_anchor=origin.tolist(), axis=axis.tolist(),
                circle_center=center.tolist(), circle_radius_m=radial,
                fixed_axial_offset_m=height, geoms=rows)


def initial_direct_contact_separation(model, data, item, side, *, pose=None,
                                      numerical_reserve_m=1e-5):
    """Test one arm against a rigid item pose; return gap and proof inputs.

    ``excluded`` applies ONLY to initial direct contact by the named arm. Both
    arms must be separately excluded to rule out an initial direct-contact
    start. False is unresolved, not reachable. No solver failure is consulted.
    An optional pose is a geometric query, never a reset or placement result.
    """
    if not np.isfinite(numerical_reserve_m) or numerical_reserve_m < 1e-5:
        raise ValueError("Numerical reserve must be at least 10 micrometers")
    envelope = arm_contact_envelope(model, data, side)
    body = model.body(item).id
    center = np.asarray(envelope["circle_center"])
    old_position, old_rotation = data.xpos[body], data.xmat[body].reshape(3, 3)
    position, rotation = old_position, old_rotation
    if pose is not None:
        values = np.asarray(pose, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("pose must be finite xyz plus wxyz quaternion")
        quaternion = values[3:].copy()
        if np.linalg.norm(quaternion) < 1e-12:
            raise ValueError("Pose quaternion must be nonzero")
        quaternion /= np.linalg.norm(quaternion)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        position, rotation = values[:3], matrix.reshape(3, 3)
    normal = np.asarray(position - center).copy()
    if np.linalg.norm(normal) < 1e-12:
        normal = np.array([0., 0., 1.])
    normal /= np.linalg.norm(normal)
    axis = np.asarray(envelope["axis"])
    circle_support = envelope["circle_radius_m"] * np.sqrt(max(0., 1. - float(normal @ axis) ** 2))
    root_support = float(normal @ (np.asarray(envelope["root_anchor"]) - center))
    uppers = [(circle_support if row["circle_centered"] else root_support) + row["radius_m"]
              for row in envelope["geoms"]]
    robot_upper = max(uppers)
    objects = []
    for geom in sorted(_collision_geometries(model)):
        path = _descendant_path(model, int(model.geom_bodyid[geom]), body)
        if path is None:
            continue
        if any(model.body_jntnum[b] for b in path[1:]):
            raise ValueError("Item must have a rigid collision subtree")
        relative_position = old_rotation.T @ (data.geom_xpos[geom] - old_position)
        relative_rotation = old_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
        geom_position = position + rotation @ relative_position
        geom_rotation = rotation @ relative_rotation
        lower = float(normal @ (geom_position - center) - _support(model, geom, geom_rotation.T @ -normal))
        objects.append(dict(geom_id=geom, name=model.geom(geom).name,
                            projection_lower_m=lower, margin_m=float(model.geom_margin[geom])))
    if not objects:
        raise ValueError("Item contains no bounded collision geometry")
    object_lower = min(row["projection_lower_m"] for row in objects)
    margin = (max(row["margin_m"] for row in objects)
              + max(row["margin_m"] for row in envelope["geoms"])
              + max([0.] + [float(v) for v in model.pair_margin]))
    gap = object_lower - robot_upper - margin
    return dict(item=item, side=side, excluded=bool(gap > numerical_reserve_m),
                classification="initial_direct_contact_separated" if gap > numerical_reserve_m else "unresolved",
                gap_m=float(gap), numerical_reserve_m=numerical_reserve_m,
                normal_world=normal.tolist(), robot_projection_upper_m=float(robot_upper),
                object_projection_lower_m=object_lower, contact_margin_m=margin,
                envelope=envelope, object_primitives=objects,
                pose_override_is_geometry_only=pose is not None)
