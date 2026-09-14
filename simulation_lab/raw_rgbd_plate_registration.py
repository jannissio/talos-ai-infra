"""Known-circle registration for face-down plates in calibrated raw RGB-D."""
import numpy as np

from .table_observation import unproject


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for row, column in zip(*np.nonzero(mask)):
        if seen[row, column]:
            continue
        seen[row, column] = True
        stack, pixels = [(int(row), int(column))], []
        while stack:
            r, c = stack.pop(); pixels.append((r, c))
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                rr, cc = r + dr, c + dc
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True; stack.append((rr, cc))
        result.append(np.asarray(pixels, dtype=int))
    return result


def register(rgb_views, depth_views, calibrations, table_z, protocol):
    """Return plate origins and face hypotheses without resolving circular yaw."""
    counts = {"rgb_views": len(rgb_views), "depth_views": len(depth_views),
              "calibrations": len(calibrations)}
    if len(set(counts.values())) != 1 or not counts["rgb_views"]:
        return {"status": "refused", "reason": "missing_or_mismatched_views",
                "view_counts": counts, "items": {}}
    candidates = []
    for view_index, (rgb, depth, calibration) in enumerate(zip(rgb_views, depth_views, calibrations)):
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        projection = np.asarray(calibration.get("projection", []), dtype=float)
        if rgb.shape != (*depth.shape, 3) or depth.ndim != 2 or projection.shape != (3, 4):
            return {"status": "refused", "reason": "invalid_view_shape_or_calibration",
                    "view_index": view_index, "items": {}}
        xyz = unproject(depth, projection)
        high, low = rgb.max(axis=2), rgb.min(axis=2)
        mask = (np.all(np.isfinite(xyz), axis=2) &
                (low >= protocol["white_minimum_channel"]) &
                ((high - low) <= protocol["white_maximum_channel_range"]) &
                (xyz[..., 2] >= table_z + protocol["minimum_height_m"]) &
                (xyz[..., 2] <= table_z + protocol["maximum_height_m"]) &
                (np.abs(xyz[..., 0]) <= protocol["table_abs_x_m"]) &
                (xyz[..., 1] >= protocol["table_y_min_m"]) &
                (xyz[..., 1] <= protocol["table_y_max_m"]))
        for component_index, pixels in enumerate(_components(mask)):
            if len(pixels) < protocol["minimum_component_pixels"]:
                continue
            points = xyz[pixels[:, 0], pixels[:, 1]]
            low_xy = np.quantile(points[:, :2], protocol["extent_quantile"], axis=0)
            high_xy = np.quantile(points[:, :2], 1 - protocol["extent_quantile"], axis=0)
            span = high_xy - low_xy
            candidates.append({
                "view_index": view_index, "component_index": component_index,
                "point_count": int(len(points)),
                "circle_center_xy_m": ((low_xy + high_xy) / 2).tolist(),
                "fitted_visible_radius_m": float(span.mean() / 2),
                "axis_ratio": float(span.min() / max(span.max(), 1e-12)),
                "horizontal_surface_z_m": float(np.median(points[:, 2])),
                "fit_method": "opposed_quantile_extents_of_calibrated_horizontal_disk"
            })
    items = {}
    assigned = {}
    for name, geometry in protocol["items"].items():
        rows = []
        for candidate in candidates:
            radius_error = abs(candidate["fitted_visible_radius_m"] - geometry["visible_base_radius_m"])
            z_error = abs(candidate["horizontal_surface_z_m"] - (table_z + geometry["body_height_m"]))
            if (radius_error <= protocol["radius_tolerance_m"] and
                    z_error <= protocol["surface_height_tolerance_m"] and
                    candidate["axis_ratio"] >= protocol["minimum_axis_ratio"]):
                row = dict(candidate)
                row["static_radius_error_m"] = float(radius_error)
                row["face_down_surface_height_error_m"] = float(z_error)
                rows.append(row)
                assigned.setdefault((candidate["view_index"], candidate["component_index"]), []).append(name)
        rows.sort(key=lambda row: (row["view_index"], row["static_radius_error_m"]))
        if len({row["view_index"] for row in rows}) < protocol["minimum_agreeing_views"]:
            items[name] = {"status": "refused", "reason": "insufficient_circle_matches",
                           "matching_candidates": rows}
            continue
        centers = np.asarray([row["circle_center_xy_m"] for row in rows])
        center = np.median(centers, axis=0)
        disagreement = float(np.max(np.linalg.norm(centers - center, axis=1)))
        z = float(np.median([row["horizontal_surface_z_m"] for row in rows]))
        down_error = abs(z - (table_z + geometry["body_height_m"]))
        up_error = abs(z - (table_z + geometry["face_up_base_surface_height_m"]))
        hypotheses = [
            {"orientation": "face_down", "body_z_axis_world": [0.0, 0.0, -1.0],
             "surface_height_error_m": float(down_error)},
            {"orientation": "face_up", "body_z_axis_world": [0.0, 0.0, 1.0],
             "surface_height_error_m": float(up_error)},
        ]
        hypotheses.sort(key=lambda row: row["surface_height_error_m"])
        margin = hypotheses[1]["surface_height_error_m"] - hypotheses[0]["surface_height_error_m"]
        if disagreement > protocol["maximum_view_center_disagreement_m"]:
            items[name] = {"status": "refused", "reason": "circle_centers_disagree_across_views",
                           "matching_candidates": rows, "center_disagreement_m": disagreement,
                           "orientation_hypotheses": hypotheses}
        elif margin < protocol["minimum_orientation_error_margin_m"]:
            items[name] = {"status": "refused", "reason": "face_orientation_ambiguous",
                           "matching_candidates": rows, "orientation_hypotheses": hypotheses,
                           "orientation_error_margin_m": float(margin)}
        else:
            items[name] = {"status": "partial_pose", "body_origin_m": [*center.tolist(), z],
                           "body_z_axis_world": hypotheses[0]["body_z_axis_world"],
                           "orientation": hypotheses[0]["orientation"],
                           "yaw_hypotheses": "all_table_yaws_due_to_circular_symmetry",
                           "orientation_hypotheses": hypotheses,
                           "orientation_error_margin_m": float(margin),
                           "center_disagreement_m": disagreement,
                           "matching_candidates": rows}
    conflicts = [{"view_index": key[0], "component_index": key[1], "item_hypotheses": names}
                 for key, names in assigned.items() if len(names) > 1]
    if conflicts:
        for names in (row["item_hypotheses"] for row in conflicts):
            for name in names:
                items[name] = {"status": "refused", "reason": "plate_identity_not_distinguishable",
                               "identity_conflicts": conflicts,
                               "prior_hypothesis": items[name]}
    return {"status": "complete", "contract": "calibrated_raw_rgbd_plus_static_plate_geometry_only",
            "uses_body_pose": False, "uses_segmentation_ids": False,
            "all_white_disk_candidates": candidates, "identity_conflicts": conflicts, "items": items}
