"""Audit the frozen four-condition experiment without rerunning or selecting trials."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder, freeze):
    specification = json.loads(freeze.read_text())
    batch = json.loads((folder/'summary.json').read_text())
    modes = ('nominal', 'nominal-frozen', 'push-live', 'push-frozen')
    expected = {(seed, mode) for seed in specification['evaluation_seeds'] for mode in modes}
    rows = batch['rows']
    if len(rows) != len(expected) or {(r['seed'], r['case']) for r in rows} != expected:
        raise ValueError('Missing, duplicate or undeclared final cases.')
    if batch['attempted'] != len(expected) or batch['planned'] != len(expected):
        raise ValueError('The final batch has not completed.')
    records = {}
    for row in rows:
        path = folder/row['report']
        if digest(path) != row['report_sha256']:
            raise ValueError(f'Changed report: {row["report"]}')
        report = json.loads(path.read_text())
        for key in ('observer_sha256', 'motor_sha256'):
            if report[key] != specification[key]:
                raise ValueError(f'Changed model: {key}')
        if report.get('openvino_artifact_sha256') != specification['openvino_artifact_sha256']:
            raise ValueError('Changed inference export.')
        for source, sha in report['source_sha256'].items():
            if sha != specification['source_sha256'][source]:
                raise ValueError(f'Changed runtime source: {source}')
            snapshot = path.parent/'evaluated-source'/source
            if not snapshot.exists():
                snapshot = folder/'evaluated-source'/source
            if digest(snapshot) != sha:
                raise ValueError(f'Changed source snapshot: {source}')
        passed = bool((report.get('physical') or {}).get('passed'))
        if passed != row['passed']:
            raise ValueError('Summary outcome does not match physical monitor.')
        records[(row['seed'], row['case'])] = report
    results = {}
    for mode in modes:
        group = [r for r in rows if r['case'] == mode]
        failures = Counter()
        for row in group:
            if row['passed']:
                continue
            message = row.get('message') or ''
            category = ('pre_motion_refusal' if row.get('simulation_seconds') == 0 else
                        'rgb_unavailable' if 'RGB estimate was unavailable' in message else
                        'motor_progress_timeout' if 'timed out' in message else 'physical_acceptance_failed')
            failures[category] += 1
        results[mode] = {'passed': sum(r['passed'] for r in group), 'attempted': len(group),
                         'failure_categories': dict(failures)}
    paired = []
    for seed in specification['evaluation_seeds']:
        trial = {mode: records[(seed, mode)] for mode in modes}
        outcome = {mode: bool((trial[mode].get('physical') or {}).get('passed')) for mode in modes}
        live, frozen = trial['push-live'], trial['push-frozen']
        if live['disclosed_push_ticks'] != frozen['disclosed_push_ticks']:
            raise ValueError('Paired push durations differ.')
        left, right = live['push_translation_before_grasp_mm'], frozen['push_translation_before_grasp_mm']
        push_delta = max(abs(a-b) for a, b in zip(left, right)) if left is not None and right is not None else None
        if (left is None) != (right is None) or live['disclosed_push_protocol'] != frozen['disclosed_push_protocol']:
            raise ValueError('Paired push protocols differ.')
        # The identical force schedule is the controlled input. Floating-point
        # contact trajectories can differ; report the measured difference.
        paired.append({'seed': seed, 'outcomes': outcome,
                       'both_unperturbed_controls_pass': outcome['nominal'] and outcome['nominal-frozen'],
                       'push_applied': live['disclosed_push_ticks'] > 0,
                       'push_translation_mm': left, 'paired_push_max_difference_mm': push_delta,
                       'live_max_counterfactual_action_delta_rad': live['image_counterfactual_max_action_delta_rad']})
    comparable = [pair for pair in paired if pair['both_unperturbed_controls_pass']]
    return {'schema': 'talos.rgb-servo-final-audit.v1', 'freeze_sha256': digest(freeze),
            'raw_summary_sha256': digest(folder/'summary.json'), 'source_and_artifact_hashes_verified': True,
            'planned': len(expected), 'attempted': len(rows), 'results': results, 'paired': paired,
            'unperturbed_control_subset': {'seeds': [p['seed'] for p in comparable], 'attempted': len(comparable),
                 'push_live_passed': sum(p['outcomes']['push-live'] for p in comparable),
                 'push_frozen_passed': sum(p['outcomes']['push-frozen'] for p in comparable)},
            'decision': 'Experimental only; keep the six-skill submission baseline. Broad bottle coverage is not established.',
            'interpretation': 'All 12 seeds remain in each denominator. The subset with both unperturbed controls passing contains only one seed; its live-versus-frozen difference is a narrow case result, not general robustness.',
            'runtime': 'OpenVINO CPU FP32 on AMD Ryzen 7 5800X; not an Intel hardware verification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, default=Path('docs/robotics/experiments/rgb-servo-final-evaluation-v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(args.input, args.freeze)
    require_space(args.output, 1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'paired'}))
