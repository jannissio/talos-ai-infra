"""Conservative known-asset registration from a saved table heightmap.

Inference consumes fused calibrated RGB-D arrays plus static appearance and size
descriptions. It never consumes body poses, geom IDs, or segmentation images.
"""
from __future__ import annotations

import math
import numpy as np

from .table_observation import TableGrid


MATERIAL_RGB = {
    "plate": (84, 130, 194),
    "side_plate": (207, 143, 71),
    "mug": (15, 135, 125),
    "glass": (158, 214, 227),
    "bottle": (140, 82, 33),
    "fork": (184, 199, 212),
    "spoon": (184, 199, 212),
}
MAX_HEIGHT_M = {"plate": .035, "side_plate": .025, "mug": .085,
                "glass": .095, "bottle": .155, "fork": .026, "spoon": .026}
MIN_PIXELS = {"plate": 45, "side_plate": 35, "mug": 80,
              "glass": 35, "bottle": 60, "fork": 18, "spoon": 18}
GEOMETRY = {
    "plate": ((.024, .033, .033, "face_up_or_down"),),
    "side_plate": ((.012, .026, .026, "face_up_or_down"),),
    "mug": ((.064, .013, .020, "upright_or_inverted"), (.050, .013, .032, "sideways")),
    "glass": ((.074, .012, .016, "upright_or_inverted"), (.048, .012, .037, "sideways")),
    "bottle": ((.140, .012, .016, "upright_or_inverted"), (.048, .012, .045, "sideways")),
}


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rows = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, points = [(int(r), int(c))], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        rows.append(np.asarray(points, dtype=int))
    return rows


def _candidate(points, heightmap, grid):
    rc = points.astype(float)
    center_rc = rc.mean(axis=0)
    centered = (rc - center_rc)[:, [1, 0]]
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    yaw = math.atan2(major[1], major[0])
    spread = np.sqrt(np.maximum(values, 0)) * grid.pixel_m
    heights = heightmap[points[:, 0], points[:, 1]]
    xy = (center_rc[[1, 0]] + .5) * grid.pixel_m + [grid.x_min, grid.y_min]
    return {"pixels": int(len(points)), "center_pixel_rc": center_rc.tolist(),
            "center_xy_m": xy.tolist(), "maximum_height_m": float(heights.max()),
            "mean_height_m": float(heights.mean()), "pca_yaw_rad_modulo_pi": float(yaw),
            "minor_major_spread_m": spread.tolist(),
            "elongation": float(np.sqrt(max(values) / max(min(values), 1e-9)))}


def _geometry_score(name, candidate):
    height = candidate["maximum_height_m"]
    minor, major = candidate["minor_major_spread_m"]
    scores = []
    for expected_height, expected_minor, expected_major, family in GEOMETRY[name]:
        score = (abs(height - expected_height) / .012 + abs(minor - expected_minor) / .012
                 + abs(major - expected_major) / .018)
        scores.append((score, family))
    score, family = min(scores)
    return float(score), family


def register(policy_image, observed, grid_values):
    """Estimate visible planar pose evidence, retaining ambiguity/refusal reasons."""
    image = np.asarray(policy_image)
    observed = np.asarray(observed)
    grid = TableGrid(**grid_values)
    if image.shape != (grid.rows, grid.columns, 6) or observed.shape != image.shape[:2]:
        raise ValueError("Saved policy image/grid shapes disagree.")
    rgb, heightmap = image[..., :3], image[..., 3]
    chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 1)
    elevated = observed & (heightmap > .002)
    results = {}
    shared_steel = None
    for name, color in MATERIAL_RGB.items():
        if name in ("fork", "spoon") and shared_steel is not None:
            continue
        reference = np.asarray(color, dtype=float)
        reference /= reference.sum()
        distance = np.linalg.norm(chromaticity - reference, axis=2)
        # Broad enough for illumination; geometry filters reject robot/cabinet regions.
        mask = elevated & (heightmap <= MAX_HEIGHT_M[name]) & (distance < .075)
        candidates = [_candidate(points, heightmap, grid) for points in _components(mask)
                      if len(points) >= MIN_PIXELS[name]]
        if name in ("fork", "spoon"):
            candidates = [row for row in candidates if row["elongation"] >= 2.5]
            candidates.sort(key=lambda row: row["minor_major_spread_m"][0], reverse=True)
            shared_steel = candidates
            if len(candidates) < 2:
                reason = ("shared_steel_material_has_fewer_than_two_separable_elongated_components; "
                          "fork/spoon identity and pose are not observable")
                results["fork"] = {"status": "refused", "reason": reason,
                                   "candidate_count": len(candidates), "candidates": candidates}
                results["spoon"] = {"status": "refused", "reason": reason,
                                    "candidate_count": len(candidates), "candidates": candidates}
            elif len(candidates) > 2:
                reason = "shared_steel_material_has_more_than_two_plausible_elongated_components"
                for item in ("fork", "spoon"):
                    results[item] = {"status": "refused", "reason": reason,
                                     "candidate_count": len(candidates), "candidates": candidates}
            else:
                # The spoon bowl is wider than the fork head in the static assets.
                for item, candidate in zip(("spoon", "fork"), candidates):
                    results[item] = {"status": "partial_pose", "candidate": candidate,
                        "position_xy_m": candidate["center_xy_m"],
                        "orientation_family": "face_up_or_down",
                        "long_axis_yaw_rad_modulo_pi": candidate["pca_yaw_rad_modulo_pi"],
                        "yaw_rad_modulo_pi": None,
                        "unobserved": ["body_origin_z_m", "body_x_yaw_rad",
                                       "long_axis_direction", "face_up_versus_down"],
                        "reason": "Width distinguishes the two steel components; pose remains symmetry-ambiguous",
                        "candidate_count": 2, "alternatives": []}
            continue
        if not candidates:
            results[name] = {"status": "refused", "reason": "no_color_depth_component",
                             "candidate_count": 0, "candidates": []}
            continue
        for candidate in candidates:
            candidate["static_geometry_score"], candidate["geometry_family"] = _geometry_score(name, candidate)
        candidates.sort(key=lambda row: row["static_geometry_score"])
        best = candidates[0]
        if best["static_geometry_score"] > 2.0:
            results[name] = {"status": "refused", "reason": "no_component_matches_static_geometry",
                             "candidate_count": len(candidates), "candidates": candidates}
            continue
        if name in ("plate", "side_plate") and sum(
                row["static_geometry_score"] <= 2.0 for row in candidates) > 1:
            results[name] = {"status": "refused",
                "reason": "multiple_round_low_profile_components_match_plate_geometry",
                "candidate_count": len(candidates), "candidates": candidates}
            continue
        if len(candidates) > 1 and candidates[1]["static_geometry_score"] < best["static_geometry_score"] + .35:
            results[name] = {"status": "refused", "reason": "multiple_components_match_static_geometry",
                             "candidate_count": len(candidates), "candidates": candidates}
            continue
        family = best["geometry_family"]
        yaw_observable = name == "mug" and best["elongation"] > 1.25
        results[name] = {"status": "partial_pose", "candidate": best,
            "position_xy_m": best["center_xy_m"], "orientation_family": family,
            "yaw_rad_modulo_pi": best["pca_yaw_rad_modulo_pi"] if yaw_observable else None,
            "unobserved": (["body_origin_z_m"] + ([] if yaw_observable else ["yaw_rad"]) +
                           (["upright_versus_inverted"] if family == "upright_or_inverted" else [])),
            "reason": "RGB-D component supports only the reported partial pose",
            "candidate_count": len(candidates), "alternatives": candidates[1:]}
    return {"contract": "calibrated_rgbd_and_static_asset_appearance_geometry_only",
            "uses_body_pose": False, "uses_segmentation_ids": False,
            "estimates": results}
