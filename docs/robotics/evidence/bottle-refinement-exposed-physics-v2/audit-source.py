"""Retain the complete exposed-grid physical diagnosis and the stopped harness."""
from collections import Counter
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.bottle_refinement_experiment import read, sha, space, write
from simulation_lab.scene import HOME


def run():
    protocol = ROOT/'docs/robotics/experiments/bottle-refinement-exposed-physics-v2.json'
    p = read(protocol)
    raw = ROOT/p['raw_root']
    output = ROOT/p['evidence_package']
    first = ROOT/'.run/bottle-refinement-exposed-physics-v1'
    summary = read(raw/'physical/exposed/summary.json')
    assert summary['all_outcomes_retained'] and len(summary['rows']) == 58 and not summary['promotion']
    if output.exists():
        raise FileExistsError('Preserve the preceding physical evidence package.')
    required = sum(file.stat().st_size for folder in (raw, first) for file in folder.rglob('*') if file.is_file())
    preflight = space(p, output, required+64*1024**2)
    output.mkdir(parents=True)
    copied = {}
    def copy(source, target):
        space(p, target, source.stat().st_size+1024**2)
        write(target, source.read_bytes())
        copied[target.relative_to(output).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source_root, destination_root in ((raw, output), (first, output/'stopped-harness-v1')):
        for source in source_root.rglob('*'):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            if source.name == 'scene.xml':
                relative = relative.parent/'source-scene.xml'
            copy(source, destination_root/relative)
    copy(protocol, output/'protocol.json')
    copy(ROOT/'docs/robotics/experiments/bottle-refinement-exposed-physics-v1.json', output/'stopped-harness-v1/protocol.json')
    for version in (1, 2):
        source = ROOT/'.run/final-goal'/f'bottle-refinement-exposed-physics-v{version}.log'
        copy(source, output/'console'/source.name)
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        write(source.parent/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
    spec = read(raw/'physical/integration.json')
    for name, digest in spec['frozen_sha256'].items():
        if name.startswith('scripts/') or name.startswith('simulation_lab/'):
            frozen = output/'physical/frozen-source'/name
            assert sha(frozen) == digest, name
        else:
            assert sha(ROOT/name) == digest, name
    rows, frames, observations, successful_legs = [], 0, 0, 0
    for case in p['cases']:
        record = {'case': case, 'conditions': {}}
        baseline_folder = ROOT/p['coverage_evidence']/(case['id']+'-learned')
        teacher_folder = ROOT/p['coverage_evidence']/(case['id']+'-teacher')
        baseline, teacher = read(baseline_folder/'report.json'), read(teacher_folder/'report.json')
        with np.load(baseline_folder/'states.npz', allow_pickle=False) as z:
            baseline_q, baseline_v = z['qpos'][0].copy(), z['qvel'][0].copy()
        record['preset_baseline'] = baseline['status']
        record['programmed_comparison'] = teacher['status']
        for mode in p['modes']:
            folder = output/'physical/exposed'/(case['id']+'-'+mode)
            report = read(folder/'report.json')
            original_row = next(row for row in summary['rows'] if row['case'] == case['id'] and row['mode'] == mode)
            assert report['status'] == original_row['status'] and sha(folder/'report.json') == original_row['report_sha256']
            assert report['integration_sha256'] == sha(raw/'physical/integration.json')
            assert report['protocol_sha256'] == sha(protocol) and report['status'] != 'harness_error'
            assert report['setup']['identical_to_original_qpos_qvel']
            model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
            data = mujoco.MjData(model)
            with np.load(folder/'states.npz', allow_pickle=False) as z:
                state = {key: z[key] for key in z.files}
            n = len(state['time']); frames += n
            assert n == report['trace_frames'] and n > 0
            assert state['qpos'].shape == (n, model.nq) and state['qvel'].shape == (n, model.nv)
            assert state['targets'].shape == (n, 12)
            assert all(np.isfinite(state[key]).all() for key in ('qpos', 'qvel', 'time', 'targets'))
            assert np.all(np.diff(state['time']) > 0)
            assert np.array_equal(state['qpos'][0], baseline_q) and np.array_equal(state['qvel'][0], baseline_v)
            assert bool(report['setup']['valid']) == bool(baseline['setup']['valid'])
            data.qpos[:] = state['qpos'][-1]; data.qvel[:] = state['qvel'][-1]
            mujoco.mj_forward(model, data)
            if report['status'] == 'succeeded':
                assert report['physical']['passed'] and report['completed_legs']
                assert len(report['completed_legs']) == len(report['route']['legs'])
                assert all(report[key] == 0 for key in ('physics_state_writes_during_control', 'hidden_forces', 'inverse_solver_calls_during_control'))
                assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
                assert np.max(np.abs(data.qvel[:12])) < .12
                assert np.linalg.norm(data.body('bottle').xpos[:2]-report['destination_m']) < .008
                assert data.body('bottle').xmat[8] > .98
                dof = model.joint('bottle_free').dofadr[0]
                assert np.linalg.norm(data.qvel[dof:dof+3]) < .003
                assert np.linalg.norm(data.qvel[dof+3:dof+6]) < .08
                for leg in report['completed_legs']:
                    metric = leg['physical']['metrics']
                    assert leg['physical']['passed'] and not leg['physical']['failure']
                    assert metric['hold_verified_s'] >= 1.49 and metric['max_lift_cm'] > 5
                    assert metric['placement_error_mm'] < 8 and metric['placement_z_error_mm'] < 4
                    assert metric['stable_release_and_park_s'] >= .5 and metric['both_arms_parked']
                    assert metric['unexpected_collisions'] == 0 and metric['other_object_max_displacement_m'] <= .004
                    assert metric['longest_unsupported_gap_s'] <= .18
                    successful_legs += 1
            observed = read(folder/'observations.json')
            observations += len(observed)
            errors = []
            for item in observed:
                if item['observation']['status'] == 'observed':
                    actual = np.linalg.norm(np.array(item['observation']['grasp_point_m'])-item['scoring_only_grasp_point_m'])*1000
                    assert abs(actual-item['scoring_only_error_mm']) < 1e-10
                    errors.append(float(actual))
                else:
                    assert item['scoring_only_error_mm'] is None
            for item in read(folder/'image-counterfactuals.json'):
                if 'maximum_delta_rad' in item:
                    delta = np.max(np.abs(np.array(item['live_action'])-item['initial_image_action']))
                    assert abs(delta-item['maximum_delta_rad']) < 1e-12
            record['conditions'][mode] = {'status': report['status'], 'valid': report['setup']['valid'],
                'message': report['message'], 'route': report.get('route', {}).get('kind') if report.get('route') else None,
                'completed_legs': len(report.get('completed_legs', [])), 'trace_frames': n,
                'rgb_observations': len(observed), 'accepted_rgb': len(errors),
                'accepted_rgb_p95_error_mm': float(np.quantile(errors, .95)) if errors else None,
                'accepted_rgb_maximum_error_mm': max(errors) if errors else None,
                'report_sha256': sha(folder/'report.json'), 'identical_original_start_verified': True}
        rows.append(record)
    def counts(items):
        return {'planned': len(items), 'valid': sum(row['conditions']['live']['valid'] for row in items),
            'live': dict(Counter(row['conditions']['live']['status'] for row in items)),
            'frozen': dict(Counter(row['conditions']['frozen']['status'] for row in items)),
            'preset_baseline_successes': sum(row['preset_baseline'] == 'succeeded' for row in items),
            'programmed_successes': sum(row['programmed_comparison'] == 'succeeded' for row in items)}
    total = counts(rows)
    by_kind = {kind: counts([row for row in rows if row['case']['kind'] == kind]) for kind in sorted({row['case']['kind'] for row in rows})}
    audit = {'schema': p['schema'], 'protocol_sha256': sha(protocol), 'cases': rows, 'total': total,
        'by_kind': by_kind, 'all_58_outcomes_retained': True, 'all_original_starts_verified': True,
        'trace_frames': frames, 'rgb_observations': observations, 'verified_successful_route_legs': successful_legs,
        'source_copies': copied, 'preflight': preflight, 'promotion': False,
        'verification': 'All file hashes, matching initial qpos/qvel, trace dimensions and finite values, every reported successful-leg threshold and final-state geometry/parking checked. Every RGB scoring error and image-counterfactual delta independently recomputed. 20 Hz traces do not independently reconstruct each 200 Hz contact query.',
        'scope': spec['scope']}
    write(output/'audit.json', audit)
    write(output/'audit-source.py', Path(__file__).read_bytes())
    table = '\n'.join(f"| {kind} | {item['planned']} | {item['valid']} | {item['live'].get('succeeded',0)} | {item['frozen'].get('succeeded',0)} | {item['preset_baseline_successes']} | {item['programmed_successes']} |" for kind, item in by_kind.items())
    text = f'''# Refined bottle control: exposed physical coverage

All **58 live/frozen rollouts across 29 bottle cases** finish with every outcome retained. There are **{total['valid']} valid starts** per condition. Live control succeeds in **{total['live'].get('succeeded',0)}**, frozen images in **{total['frozen'].get('succeeded',0)}**, the preserved preset controller in **{total['preset_baseline_successes']}**, and the separate programmed comparison in **{total['programmed_successes']}**. All starts exactly match the original comparison traces.

| Case family | Planned | Valid | Live successes | Frozen successes | Preset successes | Programmed successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

The controller uses the frozen coarse/refined RGB observer, the V5 neural motor map and unchanged V2 R1 route selection. It may use one arm or a table-supported release/park/regrasp transfer. Live images update approach/descent; carrying follows the closure-time observed offset and motor feedback. The frozen condition uses the initial image separately for each leg. No exact object pose, teacher action, inverse solver or hidden force generates learned motor targets.

These are **training-exposed diagnostic cases**, with one recorded preceding-task context and one varied factor at a time. They are not new-scene generalization, combined arrangements or a promotion test. The preceding synthetic perception gate remains failed; its reserved physical stages remain unexposed. A new bounded physical protocol must be frozen before any fresh success claim.

The independent audit retains **{frames:,} state frames**, **{observations:,} RGB observations** and **{successful_legs} successful route legs**. All invalid starts and failures remain in the planned denominators. The stopped V1 harness (implicit destination, before control) and its source are retained alongside V2; the repaired destination binding was checked against all 29 original monitors. Sources, model hashes, portable scenes, original reports and representative views are available in the manifest. No selected browser model is changed by this diagnosis.
'''
    write(output/'README.md', text.encode())
    records = {file.relative_to(output).as_posix(): sha(file) for file in output.rglob('*') if file.is_file()}
    write(output/'manifest.json', {'schema': p['schema'], 'files': records})
    print({'files': len(records), 'frames': frames, 'observations': observations, 'total': total}, flush=True)


if __name__ == '__main__':
    run()
