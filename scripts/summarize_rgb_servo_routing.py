"""Audit every frozen routing trial, including refusals and physical failures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(batch, freeze_path):
    summary = json.loads((batch / 'summary.json').read_text())
    frozen = json.loads((batch / 'frozen-inputs.json').read_text())
    selection = json.loads(freeze_path.read_text())
    for key in ('protocol_sha256', 'routing_sha256', 'source_sha256', 'model_manifest_sha256'):
        if selection[key] != frozen[key]:
            raise ValueError('Freeze mismatch: ' + key)
    seeds = selection['evaluation_seeds']
    root = Path(__file__).resolve().parents[1]
    model_path = root / 'models/bottle_servo_v1/experiment.json'
    if sha(model_path) != frozen['model_manifest_sha256']:
        raise ValueError('Packaged model manifest differs from freeze.')
    model = json.loads(model_path.read_text())
    for name in ('observer', 'motor'):
        if sha(model_path.parent / (name + '.safetensors')) != model[name + '_sha256']:
            raise ValueError('Packaged checkpoint bytes differ: ' + name)
    for name, digest in model['openvino_artifact_sha256'].items():
        if sha(model_path.parent / 'openvino' / name) != digest:
            raise ValueError('Packaged OpenVINO bytes differ: ' + name)
    modes = ('nominal', 'nominal-frozen', 'push-live', 'push-frozen')
    expected = {(seed, mode) for seed in seeds for mode in modes}
    rows = summary['rows']
    if (summary['attempted'] != len(expected) or summary['planned'] != len(expected)
            or len(rows) != len(expected) or {(r['seed'], r['case']) for r in rows} != expected):
        raise ValueError('Missing, duplicate or unexpected trials.')
    if frozen['seeds'] != seeds or frozen['split'] != 'evaluation':
        raise ValueError('Incorrect evaluation split.')
    reports = {}
    frames = 0
    reasons = {mode: Counter() for mode in modes}
    for row in rows:
        path = batch / row['report']
        if sha(path) != row['report_sha256']:
            raise ValueError('Report hash mismatch: ' + str(path))
        report = json.loads(path.read_text())
        key = (row['seed'], row['case'])
        reports[key] = report
        if report['seed'] != row['seed'] or report['split'] != 'evaluation':
            raise ValueError('Report identity mismatch.')
        wanted_mode = 'frozen' if row['case'].endswith('frozen') else 'live'
        if report['mode'] != wanted_mode:
            raise ValueError('Ablation mode mismatch.')
        for field in ('protocol_sha256', 'routing_sha256'):
            if report[field] != frozen[field]:
                raise ValueError('Report input mismatch: ' + field)
        for field in ('observer_sha256', 'motor_sha256', 'openvino_artifact_sha256'):
            if report[field] != model[field]:
                raise ValueError('Report model differs from packaged weights: ' + field)
        for name, digest in report['source_sha256'].items():
            if frozen['source_sha256'][name] != digest:
                raise ValueError('Source differs from freeze: ' + name)
            source = path.parent / 'evaluated-source' / name
            if not source.exists():
                source = batch.parent / 'source-blobs' / (digest + '.py')
            if sha(source) != digest:
                raise ValueError('Saved source differs from report: ' + name)
        for field in ('physics_state_writes_during_control', 'hidden_forces', 'inverse_solver_calls_during_control'):
            if report[field] != 0:
                raise ValueError('Forbidden runtime assistance: ' + field)
        physical = report.get('physical') or {}
        passed = report['status'] == 'succeeded' and bool(physical.get('passed'))
        if row['passed'] != passed:
            raise ValueError('Summary disagrees with physical outcome.')
        if passed:
            metrics = physical['metrics']
            if (metrics['placement_error_mm'] > 8 or not metrics['both_arms_parked']
                    or metrics['unexpected_collisions'] != 0
                    or report['cumulative_non_target_displacement_m'] > .004):
                raise ValueError('Success conflicts with acceptance limits.')
            if len(report['completed_legs']) != len(report['route']['legs']):
                raise ValueError('Successful route has an incomplete leg.')
        else:
            reasons[row['case']][report.get('message', report['status'])] += 1
        states_path = path.parent / 'states.npz'
        if states_path.exists():
            with np.load(states_path, allow_pickle=False) as states:
                length = len(states['time'])
                pre_motion_refusal = (report.get('simulation_seconds') == 0
                                      and report['status'] != 'succeeded'
                                      and not report.get('completed_legs'))
                if ((not length and not pre_motion_refusal)
                        or any(len(states[name]) != length for name in states.files)):
                    raise ValueError('Incomplete synchronized state arrays.')
                for name in ('qpos', 'qvel', 'time', 'targets'):
                    if not np.isfinite(states[name]).all():
                        raise ValueError('Nonfinite state recording.')
                if np.any(np.diff(states['time']) <= 0):
                    raise ValueError('State time is not strictly increasing.')
                frames += length
        elif report['status'] != 'invalid_start':
            raise ValueError('Missing state recording for a valid-start trial.')
    counts = {mode: sum(r['passed'] for r in rows if r['case'] == mode) for mode in modes}
    if counts != summary['passed_by_case']:
        raise ValueError('Summary counts disagree with reports.')
    lookup = {(r['seed'], r['case']): r['passed'] for r in rows}
    paired = [s for s in seeds if lookup[s, 'nominal'] and lookup[s, 'nominal-frozen']]
    paired_live = sum(lookup[s, 'push-live'] for s in paired)
    paired_frozen = sum(lookup[s, 'push-frozen'] for s in paired)
    push_deltas = []
    for seed in seeds:
        baseline = reports[seed, 'nominal']
        for mode in modes:
            if (reports[seed, mode]['initial_pose'] != baseline['initial_pose']
                    or reports[seed, mode]['destination_m'] != baseline['destination_m']):
                raise ValueError('Counterfactual starting scene or goal differs.')
        for mode in ('nominal', 'nominal-frozen'):
            if reports[seed, mode].get('disclosed_push_ticks', 0) != 0:
                raise ValueError('Undisturbed case received a push.')
        live, frozen_push = reports[seed, 'push-live'], reports[seed, 'push-frozen']
        if live.get('disclosed_push_ticks', 0) != frozen_push.get('disclosed_push_ticks', 0):
            raise ValueError('Paired force durations differ.')
        a, b = live.get('push_translation_before_grasp_mm'), frozen_push.get('push_translation_before_grasp_mm')
        if a is not None and b is not None:
            push_deltas.append(float(np.max(np.abs(np.asarray(a) - b))))
    gate = counts['nominal'] >= 8 and counts['push-live'] >= 8 and len(paired) >= 4 and paired_live > paired_frozen
    return {
        'schema': 'talos.rgb-servo-routing-audit.v2',
        'all_trials_verified': len(rows), 'seeds_per_condition': len(seeds),
        'passed_by_case': counts, 'synchronized_state_frames': frames,
        'paired_nominal_pass_seeds': paired,
        'pushed_passes_in_paired_subset': {'live': paired_live, 'frozen': paired_frozen},
        'maximum_paired_push_translation_difference_mm': max(push_deltas, default=None),
        'failures_by_case': {mode: dict(reasons[mode]) for mode in modes},
        'promotion_gate_passed': gate,
        'status': 'experimental; promotion requires review' if gate else 'experimental; not promoted',
        'freeze_sha256': sha(freeze_path),
        'scope': 'Every declared seed retained. Frozen images reset per relay leg. Force schedule is paired; resulting trajectories need not be bit-identical. This finite experiment does not establish arbitrary-workspace coverage.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--freeze', type=Path, default=Path('docs/robotics/experiments/rgb-servo-final-evaluation-v2.json'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.batch, args.freeze), indent=2))
