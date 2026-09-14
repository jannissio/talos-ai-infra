"""Register seven known dinner items in one saved RGB-D scene, then score."""
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

from simulation_lab.rgbd_item_registration import register
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def _write(path, value):
    payload = (json.dumps(value, indent=2) + "\n").encode()
    require_space(path, len(payload) + 1024)
    with path.open("xb") as stream:
        stream.write(payload)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(observation_dir, state_path, report_path, output):
    output, log = output.resolve(), output.with_suffix(".log")
    if output.exists() or log.exists():
        raise FileExistsError("Output and log must both be unused; preserve every result.")
    preflight = require_space(output, 4 * 1024**2)
    output.mkdir(parents=True)
    arrays_path, metadata_path = observation_dir / "observation.npz", observation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        estimates = register(arrays["policy_image"], arrays["observed"], metadata["grid"])
    estimates.update(scope="One exposed saved scene; geometric observability prototype, not control.",
                     input_sha256={"observation.npz": _sha(arrays_path), "metadata.json": _sha(metadata_path)},
                     preflight=preflight)
    # Freeze inference before privileged state/report is opened for scoring.
    _write(output / "estimates-before-truth.json", estimates)

    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    xml, layout = build_scene(seed=int(run_report["seed"]), scenario="dinner", dinner_preset="task")
    model, data = mujoco.MjModel.from_xml_string(xml), None
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
    mujoco.mj_forward(model, data)
    scores = {}
    for name, estimate in estimates["estimates"].items():
        truth = data.body(name)
        rotation = truth.xmat.reshape(3, 3)
        up = float(rotation[2, 2])
        truth_family = ("upright_or_face_up" if up > .9 else
                        "inverted_or_face_down" if up < -.9 else "sideways_or_tilted")
        row = {"status": estimate["status"], "truth_used_only_after_estimate": True,
               "truth_orientation_family": truth_family}
        if estimate["status"] == "partial_pose":
            error = np.linalg.norm(np.asarray(estimate["position_xy_m"]) - truth.xpos[:2])
            row["center_xy_error_mm"] = float(error * 1000)
            row["center_xy_within_15mm"] = bool(error <= .015)
            estimated_family = estimate["orientation_family"]
            row["orientation_family_compatible"] = bool(
                (estimated_family == "sideways" and truth_family == "sideways_or_tilted") or
                (estimated_family == "upright_or_inverted" and truth_family != "sideways_or_tilted") or
                (estimated_family == "face_up_or_down" and truth_family != "sideways_or_tilted"))
            if estimate["yaw_rad_modulo_pi"] is not None:
                truth_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                delta = estimate["yaw_rad_modulo_pi"] - truth_yaw
                delta = abs(math.atan2(math.sin(2 * delta), math.cos(2 * delta)) / 2)
                row["yaw_modulo_pi_error_deg"] = math.degrees(delta)
            if estimate.get("long_axis_yaw_rad_modulo_pi") is not None:
                truth_long_axis_yaw = math.atan2(rotation[1, 1], rotation[0, 1])
                delta = estimate["long_axis_yaw_rad_modulo_pi"] - truth_long_axis_yaw
                delta = abs(math.atan2(math.sin(2 * delta), math.cos(2 * delta)) / 2)
                row["long_axis_yaw_modulo_pi_error_deg"] = math.degrees(delta)
        scores[name] = row
    observed = sum(row["status"] == "partial_pose" for row in estimates["estimates"].values())
    usable = sum(row.get("center_xy_within_15mm", False) for row in scores.values())
    report = {"scope": estimates["scope"], "seven_item_full_rigid_pose_recovered": False,
              "partial_pose_count": observed, "refusal_count": 7 - observed,
              "scored_planar_positions_within_15mm": usable,
              "inference_contract": estimates["contract"], "scores": scores,
              "limitations": ["Circular known assets do not uniquely expose table yaw.",
                 "Upright and inverted symmetric vessels can be observationally ambiguous.",
                 "Fork and spoon share a material and can merge with fixed geometry in the fused map.",
                 "One exposed scene is not a generalization result."],
              "source_sha256": {"registration_module": _sha(ROOT / "simulation_lab/rgbd_item_registration.py"),
                                "evaluation_script": _sha(Path(__file__))}}
    _write(output / "score-after-truth.json", report)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"partial_pose_count": observed, "refusal_count": 7-observed,
                                 "seven_item_full_rigid_pose_recovered": False}) + "\n")
    print(json.dumps({"partial_pose_count": observed, "refusal_count": 7-observed,
                      "seven_item_full_rigid_pose_recovered": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--run-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.observation, args.state, args.run_report, args.output)
