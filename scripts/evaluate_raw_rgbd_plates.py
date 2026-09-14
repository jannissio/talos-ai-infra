"""Freeze raw RGB-D known-circle plate estimates, then score privileged truth."""
import argparse, hashlib, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from simulation_lab.raw_rgbd_plate_registration import register
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require_space(path, len(payload) + 1024)
    with path.open("xb") as stream: stream.write(payload)


def run(observation, state_path, run_report_path, protocol_path, output):
    output, log = output.resolve(), output.with_suffix(".log")
    if output.exists() or log.exists(): raise FileExistsError("Output and log must be unused.")
    preflight = require_space(output, 4 * 1024**2); output.mkdir(parents=True)
    arrays_path, metadata_path = observation / "observation.npz", observation / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        estimate = register(arrays["rgb"], arrays["depth_m"], metadata["calibration"],
                            metadata["grid"]["table_z"], protocol)
    frozen = {"scope": protocol["scope"], "truth_available_to_inference": False,
              "estimate": estimate, "preflight": preflight,
              "inputs_sha256": {"observation.npz": sha(arrays_path),
                                "metadata.json": sha(metadata_path), "protocol.json": sha(protocol_path)}}
    write_new(output / "estimates-before-truth.json", frozen)

    report = json.loads(run_report_path.read_text(encoding="utf-8"))
    xml, _ = build_scene(seed=int(report["seed"]), scenario="dinner", dinner_preset="task")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as saved:
        data.qpos[:] = saved["qpos"]; data.qvel[:] = saved["qvel"]
    mujoco.mj_forward(model, data)
    scores = {}
    for name, item in estimate["items"].items():
        row = {"status": item["status"], "truth_used_only_after_frozen_estimate": True}
        if item["status"] == "partial_pose":
            truth = data.body(name); predicted = np.asarray(item["body_origin_m"])
            row["body_origin_error_mm"] = float(np.linalg.norm(predicted - truth.xpos) * 1000)
            axis = np.asarray(item["body_z_axis_world"]); truth_axis = truth.xmat.reshape(3, 3)[:, 2]
            row["body_axis_error_deg"] = math.degrees(math.acos(np.clip(axis @ truth_axis, -1, 1)))
            row["position_gate_passed"] = row["body_origin_error_mm"] <= protocol["score_gates"]["body_origin_mm"]
            row["axis_gate_passed"] = row["body_axis_error_deg"] <= protocol["score_gates"]["body_axis_deg"]
        scores[name] = row
    score = {"scope": protocol["scope"], "truth_used_only_after_frozen_estimates": True,
             "scores": scores, "full_rotation_recovered": False,
             "yaw_limitation": "Circular plates expose no body yaw.",
             "source_sha256": {"module": sha(ROOT / "simulation_lab/raw_rgbd_plate_registration.py"),
                               "script": sha(Path(__file__))}}
    write_new(output / "score-after-truth.json", score)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(score, allow_nan=False) + "\n")
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", required=True, type=Path); p.add_argument("--state", required=True, type=Path)
    p.add_argument("--run-report", required=True, type=Path); p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); a = p.parse_args()
    run(a.observation, a.state, a.run_report, a.protocol, a.output)
