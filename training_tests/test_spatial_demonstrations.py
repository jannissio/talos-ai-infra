"""Contract tests for the strict physical-demonstration adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation_lab.spatial_demonstrations import (
    SpatialDemonstrationDataset, _deduplicate, _orientation, _pixel, build_inventory,
    inspect_run, sha256)
from simulation_lab.table_observation import TableGrid


class SpatialDemonstrationTests(unittest.TestCase):
    def test_grid_labels_refuse_clipping(self):
        grid = TableGrid()
        self.assertEqual(_pixel(grid, [grid.x_min + .001, grid.y_min + .001, .8], "pick"), [0, 0])
        with self.assertRaisesRegex(ValueError, "pick_out_of_grid"):
            _pixel(grid, [grid.x_min - .001, grid.y_min, .8], "pick")

    def test_orientation_keeps_yaw_and_alternative_family(self):
        row = _orientation([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        self.assertAlmostEqual(row["table_yaw_rad"], np.pi / 2)
        self.assertEqual(row["orientation_family"], "upright")

    def test_missing_report_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            samples, exclusions = inspect_run(Path(folder) / "still-running")
        self.assertEqual(samples, [])
        self.assertEqual(exclusions[0]["reason"], "incomplete_run_missing_report")

    def test_unaccepted_success_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            (run / "report.json").write_text(json.dumps({"actions": [], "physical_lookahead": [{
                "path": "attempt", "status": "succeeded", "demonstration_eligible": True,
                "replay_exact": True}]}), encoding="utf-8")
            inventory = build_inventory([run])
        self.assertEqual(inventory["sample_count"], 0)
        self.assertEqual(inventory["unique_exact_content_sample_count"], 0)
        self.assertEqual(inventory["exclusions"][0]["reason"],
                         "not_accepted_into_continuous_main_scene")
        self.assertFalse(inventory["training_readiness"])

    def test_explicit_unfinished_cutoff_does_not_read_later_report(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "late-run"
            run.mkdir()
            (run / "report.json").write_text('{"actions": [{"status": "succeeded"}]}',
                                              encoding="utf-8")
            inventory = build_inventory([], excluded_runs=[run])
        self.assertEqual(inventory["sample_count"], 0)
        self.assertEqual(inventory["exclusions"][0]["reason"],
                         "explicitly_excluded_as_unfinished_at_inventory_cutoff")

    def test_dedup_prefers_accepted_action_and_identical_trace(self):
        base = {"run": "r", "whole_table_claim": False}
        samples = [
            {**base, "sample_id": "independent-same-action", "episode": "i1",
             "workflow_origin": "independent_lookahead", "state_control_group": "t2",
             "action_label_group": "same"},
            {**base, "sample_id": "accepted", "episode": "a",
             "workflow_origin": "accepted_continuous", "state_control_group": "t1",
             "action_label_group": "same"},
            {**base, "sample_id": "independent-same-trace", "episode": "i2",
             "workflow_origin": "independent_lookahead", "state_control_group": "t1",
             "action_label_group": "different"},
        ]
        exclusions = []
        self.assertEqual([row["sample_id"] for row in _deduplicate(samples, exclusions)], ["accepted"])
        self.assertEqual({row["reason"] for row in exclusions}, {
            "duplicate_state_control_example",
            "independent_lookahead_duplicates_accepted_action_label"})

    def test_loader_keeps_feedback_outside_policy_input_and_verifies_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run-a"
            observation = run / "episode" / "observation.npz"
            observation.parent.mkdir(parents=True)
            np.savez_compressed(observation, policy_image=np.zeros((2, 3, 6), np.float32),
                                observed=np.ones((2, 3), bool))
            rel = "episode/observation.npz"
            inventory = {"schema_version": "talos_spatial_demonstration_v2", "samples": [{
                "sample_id": "one", "run": run.name,
                "workflow_origin": "accepted_continuous", "whole_table_claim": False,
                "input": {"policy_image": rel, "instruction": "place the plate"},
                "privileged_teacher_label": {"action": {"pick_contact_pixel_rc": [0, 0]},
                    "item_pose_keyframes": [{"item_body_origin_m": [0, 0, 0]}]},
                "robot_feedback": {"keyframes": [{"robot_joint_position": [0]}]},
                "provenance": {"source_files_sha256": {rel: sha256(observation)}}}]}
            dataset = SpatialDemonstrationDataset(inventory, [run])
            sample = dataset[0]
            self.assertEqual(set(sample["policy_input"]), {"image", "observed", "instruction"})
            self.assertNotIn("robot_feedback", sample["policy_input"])
            self.assertNotIn("item_body_origin_m", sample["robot_feedback"]["keyframes"][0])
            self.assertIn("item_body_origin_m",
                          sample["privileged_teacher_label"]["item_pose_keyframes"][0])
            observation.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "observation_source_hash_mismatch"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
