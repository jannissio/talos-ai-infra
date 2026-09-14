"""Strict adapter from completed physical teacher runs to spatial-policy samples.

Policy inputs are only a saved calibrated RGB-D heightmap and an instruction.
Privileged teacher labels and robot feedback are returned in separate fields.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from .scene import build_scene
from .table_observation import TableGrid


SCHEMA_VERSION = "talos_spatial_demonstration_v2"
REQUIRED_STAGES = ("approach", "descend", "close", "lift", "hold", "align",
                   "lower", "release", "retract", "park", "verify")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _grid(metadata: dict) -> TableGrid:
    fields = asdict(TableGrid())
    values = metadata.get("grid")
    if not isinstance(values, dict) or set(values) != set(fields):
        raise ValueError("observation_grid_schema_mismatch")
    return TableGrid(**values)


def _pixel(grid: TableGrid, xyz, label: str) -> list[int]:
    rc = np.asarray(grid.pixel(xyz), dtype=int)
    if rc.shape != (2,) or not (0 <= rc[0] < grid.rows and 0 <= rc[1] < grid.columns):
        raise ValueError(f"{label}_out_of_grid")
    return rc.tolist()


def _model_for(report: dict, nq: int):
    xml, _ = build_scene(seed=int(report["seed"]), scenario="dinner", dinner_preset="task")
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nq != nq:
        raise ValueError("model_schema_mismatch")
    return model


def _orientation(quaternion) -> dict:
    quaternion = np.asarray(quaternion, dtype=float)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    matrix = matrix.reshape(3, 3)
    up = float(matrix[2, 2])
    family = "upright" if up > .9 else "inverted" if up < -.9 else "sideways_or_tilted"
    return {"quaternion_wxyz": quaternion.tolist(),
            "table_yaw_rad": float(math.atan2(matrix[1, 0], matrix[0, 0])),
            "body_up_z": up, "orientation_family": family}


def _trace_hash(states) -> str:
    """Canonical state/control hash independent of NPZ container encoding."""
    digest = hashlib.sha256()
    for name in ("qpos", "qvel", "ctrl", "stage"):
        value = np.ascontiguousarray(states[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _keyframes(model, states, item: str, arm: str) -> tuple[list[dict], list[dict]]:
    qpos, qvel, ctrl, stages = (states[k] for k in ("qpos", "qvel", "ctrl", "stage"))
    if len(qpos) != len(qvel) or len(stages) != len(qpos) or len(ctrl) + 1 != len(qpos):
        raise ValueError("motor_trace_shape_mismatch")
    names = [str(value) for value in stages]
    missing = [stage for stage in REQUIRED_STAGES if stage not in names]
    if missing:
        raise ValueError("missing_keyframe_stages:" + ",".join(missing))
    indices = [0] + [i for i in range(1, len(names)) if names[i] != names[i - 1]]
    if indices[-1] != len(names) - 1:
        indices.append(len(names) - 1)
    data = mujoco.MjData(model)
    arm_offset = 0 if arm == "left" else 6
    robot_rows, item_rows = [], []
    for index in indices:
        data.qpos[:] = qpos[index]
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        command = ctrl[index] if index < len(ctrl) else None
        gripper = data.body(arm + "_gripper")
        body = data.body(item)
        common = {"stage": names[index], "frame_index": int(index),
                  "elapsed_s": float(index * model.opt.timestep)}
        robot_rows.append({
            **common,
            "robot_joint_position": qpos[index, :12].tolist(),
            "robot_joint_velocity": qvel[index, :12].tolist(),
            "next_motor_command": None if command is None else command[:12].tolist(),
            "active_arm": arm,
            "active_gripper_joint_position": float(qpos[index, arm_offset + 5]),
            "next_active_gripper_command": None if command is None else float(command[arm_offset + 5]),
            "gripper_body_origin_m": gripper.xpos.tolist(),
            "gripper_body_quaternion_wxyz": gripper.xquat.tolist(),
        })
        item_rows.append({
            **common,
            "item_body_origin_m": body.xpos.tolist(),
            "item_body_quaternion_wxyz": body.xquat.tolist(),
        })
    return robot_rows, item_rows


def _reason(row: dict, *, require_continuous: bool) -> str | None:
    if row.get("status") != "succeeded":
        return "physical_action_not_successful"
    if row.get("demonstration_eligible") is not True:
        return "not_demonstration_eligible"
    if row.get("replay_exact") is not True or row.get("replay_max_state_error") != 0.0:
        return "motor_replay_not_exact"
    if require_continuous and row.get("continuous_main_scene_replay_exact") is not True:
        return "not_accepted_into_continuous_main_scene"
    return None


def inspect_run(run: Path, *, instruction_template: str = "place the {item}",
                include_independent_lookahead: bool = False,
                temporary_instruction_template: str =
                "move the {item} to temporary table cell row {row} column {column}") -> tuple[list[dict], list[dict]]:
    """Return portable sample records and explicit exclusions for one run folder."""
    run = Path(run)
    report_path = run / "report.json"
    if not report_path.is_file():
        return [], [{"run": run.name, "episode": None, "reason": "incomplete_run_missing_report"}]
    report = _read_json(report_path)
    exclusions = []
    accepted_paths = {row.get("path") for row in report.get("actions", [])}
    independent = [row for row in report.get("physical_lookahead", [])
                   if row.get("path") not in accepted_paths and row.get("status") == "succeeded"]
    if not include_independent_lookahead:
        for row in independent:
            exclusions.append({"run": run.name, "episode": row.get("path"),
                               "reason": "not_accepted_into_continuous_main_scene"})
    samples = []
    candidates = [(row, "accepted_continuous") for row in report.get("actions", [])]
    if include_independent_lookahead:
        candidates.extend((row, "independent_lookahead") for row in independent)
    for row, workflow_origin in candidates:
        reason = _reason(row, require_continuous=workflow_origin == "accepted_continuous")
        episode_name = row.get("path")
        if reason:
            exclusions.append({"run": run.name, "episode": episode_name, "reason": reason})
            continue
        try:
            episode = run / episode_name
            action_path, states_path = episode / "action.json", episode / "states.npz"
            observation_dir = episode / "observation-000-approach"
            observation_path = observation_dir / "observation.npz"
            metadata_path = observation_dir / "metadata.json"
            required = (action_path, states_path, observation_path, metadata_path,
                        episode / "result.json", run / "reset-recipe.json", report_path)
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise ValueError("missing_episode_files:" + ",".join(missing))
            action, metadata = _read_json(action_path), _read_json(metadata_path)
            if action.get("policy_observation") != "calibrated_rgbd_only":
                raise ValueError("policy_observation_contract_mismatch")
            if metadata.get("contract") != "calibrated_rgb_and_metric_optical_axis_depth_only":
                raise ValueError("camera_observation_contract_mismatch")
            grid = _grid(metadata)
            with np.load(observation_path, allow_pickle=False) as observation:
                if set(observation.files) != {"policy_image", "observed", "rgb", "depth_m"}:
                    raise ValueError("observation_array_schema_mismatch")
                image = observation["policy_image"]
                observed = observation["observed"]
                if image.shape != (grid.rows, grid.columns, 6) or image.dtype != np.float32:
                    raise ValueError("policy_image_shape_or_dtype_mismatch")
                if observed.shape != (grid.rows, grid.columns) or observed.dtype != np.bool_:
                    raise ValueError("observed_mask_shape_or_dtype_mismatch")
            pick = action.get("pick_point_m")
            place = action.get("place_body_origin_m")
            if np.asarray(pick).shape != (3,) or np.asarray(place).shape != (3,):
                raise ValueError("action_point_schema_mismatch")
            with np.load(states_path, allow_pickle=False) as states:
                model = _model_for(report, states["qpos"].shape[1])
                trace_hash = _trace_hash(states)
                robot_keyframes, item_keyframes = _keyframes(model, states, action["item"], action["arm"])
            source_orientation = item_keyframes[0]["item_body_quaternion_wxyz"]
            placed_orientation = item_keyframes[-1]["item_body_quaternion_wxyz"]
            declared_orientation = action.get("place_body_quaternion_wxyz")
            provenance = {path.relative_to(run).as_posix(): sha256(path) for path in required}
            source_snapshot = {path.relative_to(run).as_posix(): sha256(path)
                               for path in sorted((run / "source").rglob("*")) if path.is_file()}
            scene_hash = sha256(run / "reset-recipe.json")
            content_hash = hashlib.sha256((json.dumps(action, sort_keys=True) + trace_hash +
                provenance[observation_path.relative_to(run).as_posix()]).encode()).hexdigest()
            action_hash = hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()
            sample_id = hashlib.sha256((run.name + "/" + episode_name + "/" +
                                        provenance[observation_path.relative_to(run).as_posix()]).encode()).hexdigest()[:20]
            pick_pixel = _pixel(grid, pick, "pick_contact")
            place_pixel = _pixel(grid, place, "place_body_origin")
            temporary_goal = row.get("role") == "relay_or_clearance"
            template = temporary_instruction_template if temporary_goal else instruction_template
            instruction = template.format(item=action["item"], row=place_pixel[0],
                column=place_pixel[1], x_m=float(place[0]), y_m=float(place[1]))
            samples.append({
                "schema_version": SCHEMA_VERSION, "sample_id": sample_id,
                "run": run.name, "episode": episode_name,
                "workflow_origin": workflow_origin, "whole_table_claim": False,
                "dependency_group": "reset_recipe_sha256:" + scene_hash,
                "exact_content_group": "action_trace_observation_sha256:" + content_hash,
                "state_control_group": "canonical_state_control_sha256:" + trace_hash,
                "action_label_group": "canonical_action_label_sha256:" + action_hash,
                "input": {"instruction": instruction,
                          "instruction_goal": ("temporary_body_origin_label" if temporary_goal
                                               else "final_body_origin_label"),
                          "policy_image": observation_path.relative_to(run).as_posix(),
                          "observed_mask": observation_path.relative_to(run).as_posix(),
                          "contract": "calibrated_rgbd_heightmap_and_instruction_only",
                          "grid": metadata["grid"]},
                "privileged_teacher_label": {
                  "contract": "labels_only_never_policy_observation",
                  "action": {
                    "item": action["item"], "arm": action["arm"], "role": row.get("role"),
                    "grasp_candidate": action.get("grasp_candidate"),
                    "pick_contact_world_xyz_m": pick, "pick_contact_pixel_rc": pick_pixel,
                    "place_body_origin_world_xyz_m": place, "place_body_origin_pixel_rc": place_pixel,
                    "source_item_orientation": _orientation(source_orientation),
                    "declared_place_body_quaternion_wxyz": declared_orientation,
                    "actual_final_item_orientation": _orientation(placed_orientation),
                    "orientation_constraint": action.get("orientation_constraint"),
                    "contact_vs_body_origin": "Pick is a surface contact. Place is the target item body origin. Neither is the gripper body origin."
                  },
                  "item_pose_keyframes": item_keyframes,
                },
                "robot_feedback": {"contract": "robot_state_only_separate_from_policy_observation",
                                   "motor_replay_exact": True, "keyframes": robot_keyframes},
                "provenance": {"source_files_sha256": provenance,
                               "recording_source_snapshot_sha256": source_snapshot,
                               "run_report_sha256": sha256(report_path),
                               "source_scene_sha256": scene_hash},
            })
        except (KeyError, TypeError, ValueError, OSError, mujoco.FatalError) as exc:
            exclusions.append({"run": run.name, "episode": episode_name,
                               "reason": str(exc) or type(exc).__name__})
    return samples, exclusions


def _deduplicate(samples: list[dict], exclusions: list[dict]) -> list[dict]:
    """Prefer accepted actions; collapse identical traces and their lookahead labels."""
    samples.sort(key=lambda sample: sample["workflow_origin"] != "accepted_continuous")
    unique, seen_traces, accepted_actions = [], {}, {}
    for sample in samples:
        trace = sample["state_control_group"]
        action = sample["action_label_group"]
        duplicate_of, reason = seen_traces.get(trace), "duplicate_state_control_example"
        if duplicate_of is None and sample["workflow_origin"] == "independent_lookahead":
            duplicate_of = accepted_actions.get(action)
            reason = "independent_lookahead_duplicates_accepted_action_label"
        if duplicate_of is not None:
            exclusions.append({"run": sample["run"], "episode": sample["episode"],
                               "reason": reason, "duplicate_of": duplicate_of})
            continue
        seen_traces[trace] = sample["sample_id"]
        if sample["workflow_origin"] == "accepted_continuous":
            accepted_actions[action] = sample["sample_id"]
        unique.append(sample)
    return unique


def build_inventory(runs: Iterable[Path], *, instruction_template: str = "place the {item}",
                    excluded_runs: Iterable[Path] = (), include_independent_lookahead: bool = False,
                    temporary_instruction_template: str =
                    "move the {item} to temporary table cell row {row} column {column}") -> dict:
    samples, exclusions = [], []
    run_names = []
    for run in runs:
        run = Path(run)
        run_names.append(run.name)
        accepted, refused = inspect_run(run, instruction_template=instruction_template,
            include_independent_lookahead=include_independent_lookahead,
            temporary_instruction_template=temporary_instruction_template)
        samples.extend(accepted)
        exclusions.extend(refused)
    for run in excluded_runs:
        run = Path(run)
        run_names.append(run.name)
        exclusions.append({"run": run.name, "episode": None,
                           "reason": "explicitly_excluded_as_unfinished_at_inventory_cutoff"})
    samples = _deduplicate(samples, exclusions)
    groups = {sample["dependency_group"] for sample in samples}
    content_groups = {sample["exact_content_group"] for sample in samples}
    counts = {}
    origin_counts = {}
    for sample in samples:
        item = sample["privileged_teacher_label"]["action"]["item"]
        counts[item] = counts.get(item, 0) + 1
        origin = sample["workflow_origin"]
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    return {"schema_version": SCHEMA_VERSION,
            "scope": "Adapter inventory only; no model fit, promotion, or generalization claim.",
            "whole_table_claim": False,
            "includes_independent_lookahead": bool(include_independent_lookahead),
            "training_readiness": False,
            "training_readiness_reason": "No jointly randomized all-seven-object table has passed and successful coverage is incomplete.",
            "runs_inspected": run_names, "sample_count": len(samples),
            "sample_count_by_item": counts, "sample_count_by_workflow_origin": origin_counts,
            "independent_scene_group_count": len(groups),
            "unique_exact_content_sample_count": len(content_groups),
            "exclusion_count": len(exclusions), "exclusions": exclusions, "samples": samples}


class SpatialDemonstrationDataset:
    """Lazy NumPy loader; run_roots explicitly resolve portable inventory paths."""
    def __init__(self, inventory: dict | Path, run_roots: Iterable[Path]):
        self.inventory = _read_json(Path(inventory)) if isinstance(inventory, (str, Path)) else inventory
        if self.inventory.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported_spatial_demonstration_schema")
        self.roots = {Path(root).name: Path(root) for root in run_roots}

    def __len__(self):
        return len(self.inventory["samples"])

    def __getitem__(self, index):
        row = self.inventory["samples"][index]
        root = self.roots.get(row["run"])
        if root is None:
            raise KeyError("No explicit run root for " + row["run"])
        path = root / row["input"]["policy_image"]
        expected = row["provenance"]["source_files_sha256"][row["input"]["policy_image"]]
        if sha256(path) != expected:
            raise ValueError("observation_source_hash_mismatch")
        with np.load(path, allow_pickle=False) as archive:
            policy_image = archive["policy_image"].copy()
            observed = archive["observed"].copy()
        return {"policy_input": {"image": policy_image,
                                 "observed": observed,
                                 "instruction": row["input"]["instruction"]},
                "privileged_teacher_label": row["privileged_teacher_label"],
                "robot_feedback": row["robot_feedback"],
                "sample_id": row["sample_id"], "workflow_origin": row["workflow_origin"],
                "whole_table_claim": row["whole_table_claim"]}
