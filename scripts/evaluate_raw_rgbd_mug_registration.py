"""Freeze a raw RGB-D mug fit, then score it against privileged state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from simulation_lab.raw_rgbd_mug_registration import fit_sideways_mug, teal_points
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    payload = (json.dumps(value, indent=2) + "\n").encode()
    require_space(path, len(payload) + 1024)
    with path.open("xb") as stream:
        stream.write(payload)


def run(observation_dir, state_path, run_report_path, protocol_path, output):
    output, log = output.resolve(), output.with_suffix(".log")
    if output.exists() or log.exists():
        raise FileExistsError("Output and log must be unused; preserve every attempt.")
    preflight = require_space(output, 4 * 1024**2)
    output.mkdir(parents=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arrays_path, metadata_path = observation_dir / "observation.npz", observation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        points = teal_points(arrays["rgb"], arrays["depth_m"], metadata["calibration"],
                            metadata["grid"]["table_z"])
    estimate = fit_sideways_mug(points, metadata["grid"]["table_z"],
                                protocol["static_mug"]["radius_m"],
                                protocol["static_mug"]["axis_length_m"])
    frozen = {"scope": protocol["scope"],
              "inference_contract": "three_raw_calibrated_rgb_metric_depth_views_and_static_mug_geometry_only",
              "uses_privileged_pose": False, "uses_renderer_segmentation": False,
              "estimate": estimate, "preflight": preflight,
              "inputs_sha256": {"observation.npz": sha(arrays_path),
                                "metadata.json": sha(metadata_path), "protocol.json": sha(protocol_path)}}
    write(output / "estimate-before-truth.json", frozen)

    report = json.loads(run_report_path.read_text(encoding="utf-8"))
    xml, _ = build_scene(seed=int(report["seed"]), scenario="dinner", dinner_preset="task")
    model, data = mujoco.MjModel.from_xml_string(xml), None
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
    mujoco.mj_forward(model, data)
    truth = data.body("mug")
    endpoints = np.asarray(estimate.get("body_origin_candidates_m", []))
    errors = np.linalg.norm(endpoints - truth.xpos, axis=1) * 1000 if len(endpoints) else np.array([])
    rotation = truth.xmat.reshape(3, 3)
    truth_axis = rotation[:, 2]
    inferred_axis = np.asarray(estimate.get("axis_unit_vector_sign_ambiguous", [np.nan] * 3))
    alignment = abs(float(np.dot(truth_axis, inferred_axis)))
    axis_error = math.degrees(math.acos(np.clip(alignment, -1, 1))) if np.isfinite(alignment) else None
    score = {"scope": protocol["scope"], "truth_used_only_after_frozen_estimate": True,
             "status": estimate["status"], "body_origin_candidate_errors_mm": errors.tolist(),
             "best_ambiguous_body_origin_error_mm": float(errors.min()) if len(errors) else None,
             "axis_error_deg_ignoring_sign": axis_error,
             "passed_declared_position_gate": bool(len(errors) and errors.min() <= protocol["gate"]["body_origin_mm"]),
             "passed_declared_axis_gate": bool(axis_error is not None and axis_error <= protocol["gate"]["axis_deg"]),
             "full_rigid_pose_recovered": False,
             "limitations": ["One exposed scene only; this is not generalization.",
                             "Body-axis direction and rotation about the cylindrical axis remain ambiguous."],
             "source_sha256": {"module": sha(ROOT / "simulation_lab/raw_rgbd_mug_registration.py"),
                               "script": sha(Path(__file__))}}
    write(output / "score-after-truth.json", score)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({key: score[key] for key in
            ("status", "best_ambiguous_body_origin_error_mm", "axis_error_deg_ignoring_sign",
             "full_rigid_pose_recovered")}) + "\n")
    print(json.dumps({key: score[key] for key in
        ("status", "best_ambiguous_body_origin_error_mm", "axis_error_deg_ignoring_sign",
         "full_rigid_pose_recovered")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--run-report", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.observation, args.state, args.run_report, args.protocol, args.output)
