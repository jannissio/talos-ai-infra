import json
import unittest
from unittest.mock import patch

import numpy as np

from simulation_lab.raw_rgbd_mug_full_pose_v2 import resolve


CYLINDER = {
    "status": "estimated_with_axis_direction_ambiguity",
    "cylinder_midpoint_m": [0.0, 0.0, 0.8],
    "axis_unit_vector_sign_ambiguous": [1.0, 0.0, 0.0],
    "static_radius_m": 0.025,
    "static_axis_length_m": 0.064,
}
THRESHOLDS = {
    "minimum_axis_direction_point_margin": 20,
    "minimum_handle_outlier_points": 50,
    "minimum_handle_direction_concentration": 0.8,
    "minimum_handle_sign_score_margin": 1.0,
}
CALIBRATION = {"projection": np.hstack((np.eye(3), np.zeros((3, 1)))),
               "position": np.array([0.0, 0.0, 1.0])}


def _assert_strict_json(value):
    assert "NaN" not in json.dumps(value, allow_nan=False)


class RawRgbdMugFullPoseV2Tests(unittest.TestCase):
    def test_refuses_mug_absent_observation_without_nonfinite_json(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        depth = np.ones((3, 4), dtype=float)
        result = resolve([rgb], [depth], [CALIBRATION], 0.75, CYLINDER, THRESHOLDS)
        self.assertEqual(result, {"status": "refused", "reason": "no_teal_surface_points",
                                  "view_point_counts": [0]})
        _assert_strict_json(result)

    def test_refuses_handle_absent_surface_without_nonfinite_json(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        depth = np.ones((1, 1), dtype=float)
        # Cylinder-wall/end evidence is present, but every point remains within the
        # known body radius, so there is no exterior handle direction to average.
        body_only = np.repeat([[[-0.031, 0.0, 0.8]]], 70, axis=0).reshape(70, 3)
        with patch("simulation_lab.raw_rgbd_mug_full_pose_v2._view_points", return_value=body_only):
            result = resolve([rgb], [depth], [CALIBRATION], 0.75, CYLINDER, THRESHOLDS)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["reason"], "no_handle_outlier_points")
        self.assertEqual(result["handle_outlier_points"], 0)
        self.assertEqual(result["handle_direction_concentration"], 0.0)
        self.assertEqual(result["roll_hypotheses"], [])
        _assert_strict_json(result)

    def test_refuses_mismatched_view_counts_before_zip_truncation(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        result = resolve([rgb], [], [CALIBRATION], 0.75, CYLINDER, THRESHOLDS)
        self.assertEqual(result, {"status": "refused", "reason": "view_count_mismatch",
                                  "view_counts": {"rgb_views": 1, "depth_views": 0,
                                                  "calibrations": 1}})
        _assert_strict_json(result)


if __name__ == "__main__":
    unittest.main()
