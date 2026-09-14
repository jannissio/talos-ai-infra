"""Resolve mug base/lip direction and handle roll with guarded RGB-D inputs."""
import numpy as np
import mujoco

from .table_observation import unproject


def _view_points(rgb, depth, calibration, table_z):
    rgb = np.asarray(rgb, dtype=float)
    red, green, blue = rgb.transpose(2, 0, 1)
    xyz = unproject(depth, calibration["projection"])
    mask = (np.all(np.isfinite(rgb), axis=-1) & np.all(np.isfinite(xyz), axis=-1) &
            (green > 1.5 * red) & (green > 1.015 * blue) &
            (blue > 1.3 * red) & (green > 45) &
            (xyz[..., 2] > table_z + .002) & (xyz[..., 2] < table_z + .09) &
            (np.abs(xyz[..., 0]) < .5) & (xyz[..., 1] > -.33) & (xyz[..., 1] < .47))
    return xyz[mask]


def resolve(rgb_views, depth_views, calibrations, table_z, cylinder_fit, thresholds):
    """Return one static-geometry pose or retain both endpoint/roll hypotheses."""
    counts = {"rgb_views": len(rgb_views), "depth_views": len(depth_views),
              "calibrations": len(calibrations)}
    if len(set(counts.values())) != 1:
        return {"status": "refused", "reason": "view_count_mismatch", "view_counts": counts}
    if counts["rgb_views"] == 0:
        return {"status": "refused", "reason": "no_camera_views", "view_counts": counts}
    if not np.isfinite(table_z):
        return {"status": "refused", "reason": "nonfinite_table_height"}
    if cylinder_fit.get("status") != "estimated_with_axis_direction_ambiguity":
        return {"status": "refused", "reason": "cylinder_fit_unavailable"}
    required_thresholds = ("minimum_axis_direction_point_margin", "minimum_handle_outlier_points",
                           "minimum_handle_direction_concentration", "minimum_handle_sign_score_margin")
    if any(name not in thresholds or not np.isfinite(thresholds[name]) for name in required_thresholds):
        return {"status": "refused", "reason": "invalid_nonfinite_or_missing_threshold"}
    for index, (rgb, depth, calibration) in enumerate(zip(rgb_views, depth_views, calibrations)):
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        projection = np.asarray(calibration.get("projection", []), dtype=float)
        position = np.asarray(calibration.get("position", []), dtype=float)
        if rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
            return {"status": "refused", "reason": "invalid_rgb_depth_shape", "view_index": index}
        if projection.shape != (3, 4) or position.shape != (3,):
            return {"status": "refused", "reason": "invalid_calibration_shape", "view_index": index}
        if not np.all(np.isfinite(projection)) or not np.all(np.isfinite(position)):
            return {"status": "refused", "reason": "nonfinite_calibration", "view_index": index}
    views = [_view_points(rgb, depth, calibration, table_z)
             for rgb, depth, calibration in zip(rgb_views, depth_views, calibrations)]
    if not any(len(points) for points in views):
        return {"status": "refused", "reason": "no_teal_surface_points",
                "view_point_counts": [int(len(points)) for points in views]}
    cameras = [np.asarray(calibration["position"], dtype=float) for calibration in calibrations]
    midpoint = np.asarray(cylinder_fit["cylinder_midpoint_m"], dtype=float)
    unsigned_axis = np.asarray(cylinder_fit["axis_unit_vector_sign_ambiguous"], dtype=float)
    radius = float(cylinder_fit["static_radius_m"])
    length = float(cylinder_fit["static_axis_length_m"])
    if (midpoint.shape != (3,) or unsigned_axis.shape != (3,) or
            not np.all(np.isfinite(midpoint)) or not np.all(np.isfinite(unsigned_axis)) or
            not np.isfinite(radius) or not np.isfinite(length) or radius <= 0 or length <= 0):
        return {"status": "refused", "reason": "invalid_nonfinite_cylinder_geometry"}
    axis_norm = float(np.linalg.norm(unsigned_axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        return {"status": "refused", "reason": "invalid_zero_cylinder_axis"}
    unsigned_axis /= axis_norm
    direction_rows = []
    for sign in (1, -1):
        axis = unsigned_axis * sign
        base_evidence = lip_interior = 0
        by_view = []
        for points, camera in zip(views, cameras):
            delta = points - midpoint
            axial = delta @ axis
            radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
            base_facing = float(np.dot(camera - midpoint, -axis)) > 0
            lip_facing = float(np.dot(camera - midpoint, axis)) > 0
            base_count = int(np.sum((axial < -length / 2 + .010) & (radial < radius - .007))) if base_facing else 0
            lip_count = int(np.sum((axial > length / 2 - .010) & (radial < radius - .007))) if lip_facing else 0
            base_evidence += base_count
            lip_interior += lip_count
            by_view.append({"base_facing": base_facing, "lip_facing": lip_facing,
                            "base_disk_interior_points": base_count,
                            "open_lip_interior_points": lip_count})
        direction_rows.append({"axis_sign": sign, "axis_unit_vector": axis.tolist(),
                               "body_origin_m": (midpoint - axis * length / 2).tolist(),
                               "base_disk_evidence": base_evidence,
                               "open_lip_interior_evidence": lip_interior,
                               "camera_consistency_score": base_evidence - lip_interior,
                               "views": by_view})
    direction_rows.sort(key=lambda row: row["camera_consistency_score"], reverse=True)
    axis_margin = direction_rows[0]["camera_consistency_score"] - direction_rows[1]["camera_consistency_score"]
    axis = np.asarray(direction_rows[0]["axis_unit_vector"])

    all_points = np.concatenate(views)
    delta = all_points - midpoint
    axial = delta @ axis
    radial_vectors = delta - axial[:, None] * axis
    radial = np.linalg.norm(radial_vectors, axis=1)
    handle = (radial > radius + .004) & (np.abs(axial) < length / 2 + .004)
    handle_count = int(handle.sum())
    if handle_count == 0:
        return {"status": "refused", "reason": "no_handle_outlier_points",
                "axis_direction_hypotheses": direction_rows,
                "axis_direction_margin": axis_margin, "handle_outlier_points": 0,
                "handle_direction_concentration": 0.0, "roll_hypotheses": []}
    directions = radial_vectors[handle] / radial[handle, None]
    weights = radial[handle] - radius
    vector = np.sum(directions * weights[:, None], axis=0) / max(float(weights.sum()), 1e-12)
    concentration = float(np.linalg.norm(vector))
    handle_axis = vector / max(concentration, 1e-12)
    roll_rows = [{"handle_sign": sign, "handle_axis_unit_vector": (handle_axis * sign).tolist(),
                  "point_consistency_score": float(np.mean(directions @ (handle_axis * sign)))}
                 for sign in (1, -1)]
    roll_rows.sort(key=lambda row: row["point_consistency_score"], reverse=True)
    roll_margin = roll_rows[0]["point_consistency_score"] - roll_rows[1]["point_consistency_score"]

    if axis_margin < thresholds["minimum_axis_direction_point_margin"]:
        return {"status": "refused", "reason": "base_lip_camera_consistency_margin_too_small",
                "axis_direction_hypotheses": direction_rows, "axis_direction_margin": axis_margin,
                "roll_hypotheses": roll_rows}
    if handle_count < thresholds["minimum_handle_outlier_points"] or concentration < thresholds["minimum_handle_direction_concentration"]:
        return {"status": "refused", "reason": "handle_direction_evidence_too_weak",
                "axis_direction_hypotheses": direction_rows, "axis_direction_margin": axis_margin,
                "handle_outlier_points": handle_count, "handle_direction_concentration": concentration,
                "roll_hypotheses": roll_rows}
    if roll_margin < thresholds["minimum_handle_sign_score_margin"]:
        return {"status": "refused", "reason": "handle_roll_sign_margin_too_small",
                "axis_direction_hypotheses": direction_rows, "axis_direction_margin": axis_margin,
                "roll_hypotheses": roll_rows, "roll_margin": roll_margin}
    body_x = np.asarray(roll_rows[0]["handle_axis_unit_vector"])
    body_x -= axis * np.dot(axis, body_x)
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(axis, body_x)
    rotation = np.column_stack((body_x, body_y, axis))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return {"status": "resolved_static_geometry_pose_hypothesis",
            "body_origin_m": direction_rows[0]["body_origin_m"],
            "body_quaternion_wxyz": quaternion.tolist(),
            "axis_direction_hypotheses": direction_rows, "axis_direction_margin_points": axis_margin,
            "handle_outlier_points": handle_count, "handle_direction_concentration": concentration,
            "roll_hypotheses": roll_rows, "roll_score_margin": roll_margin,
            "inference_note": "Base disk/open lip select body-Z sign; exterior teal handle points select body-X roll."}
