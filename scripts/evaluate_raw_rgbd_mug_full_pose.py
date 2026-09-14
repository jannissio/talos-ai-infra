"""Freeze raw RGB-D mug base/lip and handle inference, then score truth."""
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

from simulation_lab.raw_rgbd_mug_full_pose import resolve
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    payload = (json.dumps(value, indent=2) + "\n").encode()
    require_space(path, len(payload) + 1024)
    with path.open("xb") as stream: stream.write(payload)


def run(observation_dir, state_path, run_report_path, cylinder_estimate_path, protocol_path, output):
    output, log = output.resolve(), output.with_suffix(".log")
    if output.exists() or log.exists(): raise FileExistsError("Output and log must be unused.")
    preflight = require_space(output, 4 * 1024**2); output.mkdir(parents=True)
    arrays_path, metadata_path = observation_dir / "observation.npz", observation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cylinder = json.loads(cylinder_estimate_path.read_text(encoding="utf-8"))["estimate"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        estimate = resolve(arrays["rgb"], arrays["depth_m"], metadata["calibration"],
                           metadata["grid"]["table_z"], cylinder, protocol["ambiguity_gates"])
    frozen = {"scope": protocol["scope"],
              "inference_contract": "raw_calibrated_rgbd_rays_plus_static_mug_base_lip_handle_geometry_only",
              "uses_privileged_pose": False, "uses_renderer_segmentation": False,
              "estimate": estimate, "preflight": preflight,
              "inputs_sha256": {"observation.npz": sha(arrays_path), "metadata.json": sha(metadata_path),
                                "cylinder-estimate.json": sha(cylinder_estimate_path),
                                "protocol.json": sha(protocol_path)}}
    write(output / "estimate-before-truth.json", frozen)

    report = json.loads(run_report_path.read_text(encoding="utf-8"))
    xml, _ = build_scene(seed=int(report["seed"]), scenario="dinner", dinner_preset="task")
    model, data = mujoco.MjModel.from_xml_string(xml), None; data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = state["qpos"]; data.qvel[:] = state["qvel"]
    mujoco.mj_forward(model, data); truth = data.body("mug")
    if estimate["status"] == "resolved_static_geometry_pose_hypothesis":
        position_error = float(np.linalg.norm(np.asarray(estimate["body_origin_m"]) - truth.xpos) * 1000)
        predicted = np.empty(9); mujoco.mju_quat2Mat(predicted, np.asarray(estimate["body_quaternion_wxyz"]))
        relative = predicted.reshape(3, 3).T @ truth.xmat.reshape(3, 3)
        rotation_error = math.degrees(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    else: position_error = rotation_error = None
    score = {"scope": protocol["scope"], "truth_used_only_after_frozen_estimate": True,
             "status": estimate["status"], "body_origin_error_mm": position_error,
             "full_rotation_error_deg": rotation_error,
             "position_gate_passed": bool(position_error is not None and position_error <= protocol["score_gates"]["body_origin_mm"]),
             "rotation_gate_passed": bool(rotation_error is not None and rotation_error <= protocol["score_gates"]["rotation_deg"]),
             "one_exposed_scene_only": True,
             "source_sha256": {"module": sha(ROOT / "simulation_lab/raw_rgbd_mug_full_pose.py"),
                               "script": sha(Path(__file__))}}
    write(output / "score-after-truth.json", score)
    with log.open("x", encoding="utf-8") as stream: stream.write(json.dumps(score) + "\n")
    print(json.dumps({key: score[key] for key in ("status", "body_origin_error_mm", "full_rotation_error_deg",
                                                   "position_gate_passed", "rotation_gate_passed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path); parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--run-report", required=True, type=Path); parser.add_argument("--cylinder-estimate", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); run(args.observation, args.state, args.run_report, args.cylinder_estimate, args.protocol, args.output)
