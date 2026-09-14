"""Robust static-geometry mug fit from calibrated full-resolution RGB-D views."""
import math

import numpy as np

from .table_observation import unproject


def teal_points(rgb_views, depth_views, calibrations, table_z):
    rows = []
    for rgb, depth, calibration in zip(rgb_views, depth_views, calibrations):
        rgb = np.asarray(rgb, dtype=float)
        red, green, blue = rgb.transpose(2, 0, 1)
        xyz = unproject(depth, calibration["projection"])
        teal = ((green > 1.5 * red) & (green > 1.015 * blue) &
                (blue > 1.3 * red) & (green > 45))
        within_table = ((xyz[..., 2] > table_z + .002) & (xyz[..., 2] < table_z + .09) &
                        (np.abs(xyz[..., 0]) < .5) & (xyz[..., 1] > -.33) &
                        (xyz[..., 1] < .47))
        rows.append(xyz[teal & within_table])
    points = np.concatenate(rows)
    # Merge repeat observations of the same surface without depending on view count.
    voxels = np.rint(points / .001).astype(np.int64)
    _, keep = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(keep)]


def _trimmed_mean(values, fraction=.7):
    count = max(1, int(len(values) * fraction))
    return float(np.partition(values, count - 1)[:count].mean())


def fit_sideways_mug(points, table_z, radius_m=.025, length_m=.064):
    """Fit the cylindrical vessel, leaving its axial sign explicitly ambiguous."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 200:
        return {"status": "refused", "reason": "insufficient_teal_surface_points",
                "point_count": int(len(points))}
    axis_center_z = table_z + radius_m
    usable = points[np.abs(points[:, 2] - axis_center_z) <= radius_m + .003]
    if len(usable) < 150:
        return {"status": "refused", "reason": "insufficient_points_near_supported_cylinder",
                "point_count": int(len(points)), "usable_point_count": int(len(usable))}
    best = None
    for yaw in np.linspace(0, math.pi, 181, endpoint=False):
        axis = np.array([math.cos(yaw), math.sin(yaw), 0.])
        lateral_axis = np.array([-axis[1], axis[0], 0.])
        lateral = usable @ lateral_axis
        dz = usable[:, 2] - axis_center_z
        expected = np.sqrt(np.maximum(radius_m ** 2 - dz ** 2, 0))
        lo, hi = np.quantile(lateral, [.02, .98])
        centers = np.arange(lo - radius_m, hi + radius_m + .0005, .001)
        residuals = np.abs(np.abs(lateral[:, None] - centers[None, :]) - expected[:, None])
        scores = np.array([_trimmed_mean(residuals[:, i]) for i in range(len(centers))])
        lateral_center = float(centers[int(np.argmin(scores))])
        radial_error = np.abs(np.abs(lateral - lateral_center) - expected)
        wall = usable[radial_error < .004]
        if len(wall) < 100:
            continue
        axial = wall @ axis
        q02, q98 = np.quantile(axial, [.02, .98])
        axial_center = float((q02 + q98) / 2)
        axial_span = float(q98 - q02)
        axial_excess = np.maximum(np.abs(axial - axial_center) - length_m / 2, 0)
        score = (_trimmed_mean(radial_error) + .25 * abs(axial_span - length_m)
                 + .5 * _trimmed_mean(axial_excess))
        candidate = (score, yaw, axis, lateral_axis, lateral_center, axial_center,
                     axial_span, radial_error, len(wall))
        if best is None or score < best[0]:
            best = candidate
    if best is None:
        return {"status": "refused", "reason": "no_supported_cylinder_candidate",
                "point_count": int(len(points))}
    score, yaw, axis, lateral_axis, lateral_center, axial_center, span, residual, inliers = best
    midpoint = axis * axial_center + lateral_axis * lateral_center
    midpoint[2] = axis_center_z
    endpoints = [midpoint - axis * length_m / 2, midpoint + axis * length_m / 2]
    return {"status": "estimated_with_axis_direction_ambiguity",
            "input_point_count": int(len(points)), "wall_inlier_count": int(inliers),
            "static_radius_m": radius_m, "static_axis_length_m": length_m,
            "axis_yaw_rad_modulo_pi": float(yaw), "axis_unit_vector_sign_ambiguous": axis.tolist(),
            "cylinder_midpoint_m": midpoint.tolist(),
            "body_origin_candidates_m": [point.tolist() for point in endpoints],
            "radial_trimmed_mean_error_mm": _trimmed_mean(residual) * 1000,
            "observed_axial_span_m": span, "fit_score": float(score),
            "ambiguities": ["body_axis_direction_and_base_endpoint", "upright_versus_inverted_not_applicable_to_sideways_fit",
                            "rotation_about_cylindrical_axis_without_handle_model"],
            "handle_treatment": "Teal handle points are robust outliers to the trimmed cylinder residual."}
