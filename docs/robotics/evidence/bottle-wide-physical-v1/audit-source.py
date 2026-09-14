"""Preserve and independently audit every frozen wider-bottle physical result."""
from collections import Counter
import math
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


def check_physical(physical):
    assert physical['passed'] and not physical['failure']
    metric = physical['metrics']
    assert metric['hold_verified_s'] >= 1.49 and metric['max_lift_cm'] > 5
    assert metric['placement_error_mm'] < 8 and metric['placement_z_error_mm'] < 4
    assert metric['stable_release_and_park_s'] >= .5 and metric['both_arms_parked']
    assert metric['unexpected_collisions'] == 0 and metric['other_object_max_displacement_m'] <= .004
    assert metric['longest_unsupported_gap_s'] <= .18


def run():
    protocol = ROOT/'docs/robotics/experiments/bottle-wide-physical-v1.json'
    p = read(protocol)
    raw = ROOT/p['raw_root']
    output = ROOT/p['evidence_package']
    stages = {split: read(raw/'physical'/split/'gate.json') for split in ('development', 'evaluation')}
    declared = [('development', seed, mode) for seed in p['physical']['development_seeds'] for mode in p['development_modes']]
    declared += [('evaluation', seed, mode) for seed in p['physical']['evaluation_seeds'] for mode in p['evaluation_modes']]
    declared += [('regression', seed, 'live') for seed in p['regression_seeds']]
    rows = stages['development']['rows'] + stages['evaluation']['rows']
    assert len(rows) == len(declared) == 118
    assert {(r['split'], r['seed'], r['mode']) for r in rows} == set(declared)
    assert all(r['status'] != 'harness_error' and r['exit_code'] == 0 for r in rows)
    assert stages['development']['passed']
    freeze = read(raw/'physical/evaluation-freeze.json')
    spec = read(raw/'physical/integration.json')
    assert freeze['development_gate_sha256'] == sha(raw/'physical/development/gate.json')
    assert freeze['integration_sha256'] == sha(raw/'physical/integration.json')
    assert freeze['protocol_sha256'] == sha(protocol) == spec['protocol_sha256']
    if output.exists():
        raise FileExistsError('Preserve the wider-bottle evidence package.')
    required = sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())
    preflight = space(p, output, required+64*1024**2)
    output.mkdir(parents=True)
    copied = {}
    def copy(source, target):
        space(p, target, source.stat().st_size+1024**2)
        write(target, source.read_bytes())
        copied[target.relative_to(output).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source in raw.rglob('*'):
        if not source.is_file():
            continue
        relative = source.relative_to(raw)
        if source.name == 'scene.xml':
            relative = relative.parent/'source-scene.xml'
        copy(source, output/relative)
    copy(protocol, output/'protocol.json')
    for split in ('development', 'evaluation'):
        source = ROOT/'.run/final-goal'/f'bottle-wide-physical-v1-{split}.log'
        copy(source, output/'console'/source.name)
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        write(source.parent/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
    for name, digest in spec['frozen_sha256'].items():
        frozen = output/'physical/frozen-source'/name
        assert sha(frozen if frozen.exists() else ROOT/name) == digest, name
    records, initial, regenerated = [], {}, set()
    frames = observations = route_legs = 0
    for row in rows:
        split, seed, mode = row['split'], row['seed'], row['mode']
        folder = output/'physical'/split/f'{seed}-{mode}'
        report = read(folder/'report.json')
        assert sha(folder/'report.json') == row['report_sha256']
        assert report['status'] == row['status'] and report['setup']['valid'] == row['valid']
        assert report['integration_sha256'] == sha(raw/'physical/integration.json')
        assert report['protocol_sha256'] == sha(protocol)
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        data = mujoco.MjData(model)
        with np.load(folder/'states.npz', allow_pickle=False) as z:
            state = {key: z[key] for key in z.files}
        n = len(state['time']); frames += n
        assert n == report['trace_frames'] and n > 0
        assert state['qpos'].shape == (n, model.nq) and state['qvel'].shape == (n, model.nv)
        assert state['targets'].shape == (n, 12)
        assert all(np.isfinite(state[key]).all() for key in ('qpos', 'qvel', 'time', 'targets'))
        assert np.all(np.diff(state['time']) > 0) and state['time'][0] == 0.
        key = (split, seed)
        if key in initial:
            assert np.array_equal(state['qpos'][0], initial[key][0])
            assert np.array_equal(state['qvel'][0], initial[key][1])
        else:
            initial[key] = (state['qpos'][0], state['qvel'][0])
            # Independently reconstruct the reset from the portable scene and
            # declared seed, without calling the evaluation setup function.
            address = int(model.joint('bottle_free').qposadr[0])
            source = data.qpos[address:address+7].copy()
            if split != 'regression':
                rng = np.random.default_rng(seed)
                b = p['physical']['workspace_m']
                x, y, yaw = rng.uniform(*b['x']), rng.uniform(*b['y']), rng.uniform(*b['yaw'])
                source = np.array([x, y, report['layout']['table_z']+.001,
                                   math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
                assert report['destination_m'] == p['physical']['destinations_xy_m'][seed % 2]
            assert np.array_equal(source, report['setup']['requested_source_qpos'])
            data.qpos[:12] = HOME*2
            data.ctrl[:] = HOME*2
            data.qpos[address:address+7] = source
            mujoco.mj_forward(model, data)
            body = model.body('bottle').id
            penetration = max((-float(c.dist) for c in data.contact
                if model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]
                and body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])), default=0.)
            assert penetration == report['setup']['initial_penetration_m']
            for _ in range(spec['settle_steps']):
                mujoco.mj_step(model, data)
            data.time = 0.
            mujoco.mj_forward(model, data)
            assert np.array_equal(data.qpos, state['qpos'][0]), (split, seed, 'regenerated qpos')
            assert np.array_equal(data.qvel, state['qvel'][0]), (split, seed, 'regenerated qvel')
            dof = int(model.joint('bottle_free').dofadr[0])
            speed = float(np.linalg.norm(data.qvel[dof:dof+3]))
            valid = (penetration <= .001 and data.body('bottle').xmat[8] > .98
                     and data.body('bottle').xpos[2] >= report['layout']['table_z']-.004 and speed < .005)
            assert bool(valid) == report['setup']['valid']
            regenerated.add(key)
        data.qpos[:] = state['qpos'][-1]; data.qvel[:] = state['qvel'][-1]
        mujoco.mj_forward(model, data)
        if report['status'] == 'succeeded':
            check_physical(report['physical'])
            assert all(report[k] == 0 for k in ('physics_state_writes_during_control', 'hidden_forces', 'inverse_solver_calls_during_control'))
            assert report['cumulative_non_target_displacement_m'] <= .004
            assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
            assert np.max(np.abs(data.qvel[:12])) < .12
            assert np.linalg.norm(data.body('bottle').xpos[:2]-report['destination_m']) < .008
            assert data.body('bottle').xmat[8] > .98
            dof = model.joint('bottle_free').dofadr[0]
            assert np.linalg.norm(data.qvel[dof:dof+3]) < .003
            assert np.linalg.norm(data.qvel[dof+3:dof+6]) < .08
            if mode != 'baseline':
                assert len(report['completed_legs']) == len(report['route']['legs']) > 0
                for leg in report['completed_legs']:
                    check_physical(leg['physical']); route_legs += 1
        observed = read(folder/'observations.json'); observations += len(observed)
        errors = []
        for item in observed:
            if item['observation']['status'] == 'observed':
                error = np.linalg.norm(np.array(item['observation']['grasp_point_m'])-item['scoring_only_grasp_point_m'])*1000
                assert abs(error-item['scoring_only_error_mm']) < 1e-10
                errors.append(float(error))
            else:
                assert item['scoring_only_error_mm'] is None
        for item in read(folder/'image-counterfactuals.json'):
            if 'maximum_delta_rad' in item:
                delta = np.max(np.abs(np.array(item['live_action'])-item['initial_image_action']))
                assert abs(delta-item['maximum_delta_rad']) < 1e-12
        records.append({**row, 'trace_frames': n, 'rgb_observations': len(observed),
                        'accepted_rgb': len(errors), 'route': report.get('route'),
                        'accepted_rgb_p95_error_mm': float(np.quantile(errors, .95)) if errors else None,
                        'accepted_rgb_maximum_error_mm': max(errors) if errors else None})
    counts = {split: {mode: dict(Counter(r['status'] for r in records if r['split'] == split and r['mode'] == mode))
                     for mode in (['live'] if split == 'regression' else p[split+'_modes'])}
              for split in ('development', 'evaluation', 'regression')}
    requirements = {}
    for split, gate in stages.items():
        current = [r for r in records if r['split'] == split]
        totals = {mode: sum(r['mode'] == mode and r['status'] == 'succeeded' for r in current) for mode in p[split+'_modes']}
        limits = p[split+'_gate']
        live = [r for r in current if r['mode'] == 'live']
        valid = sum(r['valid'] for r in live)
        per_goal = [sum(r['status'] == 'succeeded' and r['destination_m'] == goal for r in live) for goal in p['physical']['destinations_xy_m']]
        assert totals == gate['counts'] and valid == gate['valid_starts'] and per_goal == gate['live_successes_per_goal']
        checks = {'total_live': totals['live'] >= limits['minimum_live_successes'],
                  'per_destination': min(per_goal) >= limits['minimum_live_per_destination']}
        if split == 'development':
            checks['baseline_improvement'] = totals['live'] > totals['baseline']
        else:
            checks.update(valid_fraction=totals['live']/max(1, valid) >= limits['minimum_live_fraction_of_valid'],
                baseline_improvement=totals['live'] >= totals['baseline']+limits['minimum_more_successes_than_baseline'],
                live_image_improvement=totals['live'] >= totals['frozen']+limits['minimum_more_successes_than_frozen'],
                exposed_regressions=sum(r['status'] == 'succeeded' for r in records if r['split'] == 'regression') == limits['required_exposed_regression_successes'])
        assert all(checks.values()) == gate['passed']
        requirements[split] = checks
    audit = {'schema': p['schema'], 'protocol_sha256': sha(protocol), 'all_118_outcomes_retained': True,
             'gate_passed': stages['evaluation']['passed'], 'requirements': requirements, 'counts': counts,
             'cases': records, 'trace_frames': frames, 'rgb_observations': observations,
             'verified_successful_route_legs': route_legs, 'independently_regenerated_starts': len(regenerated),
             'all_paired_initial_states_identical': True, 'source_copies': copied, 'preflight': preflight,
             'verification': 'Every file hash, all 46 independently regenerated initial states, all paired starts, finite traces, every successful physical threshold and terminal geometry/parking verified. Every RGB scoring error and counterfactual delta recomputed. 20 Hz traces do not reconstruct every 200 Hz contact query.',
             'standalone_perception_gate_remains_failed': True, 'production_integration_verified': False,
             'scope': spec['scope']}
    assert len(regenerated) == 46
    write(output/'audit.json', audit)
    write(output/'audit-source.py', Path(__file__).read_bytes())
    final = stages['evaluation']; dev = stages['development']
    text = f'''# Wider bottle physical evaluation V1

The unchanged candidate passes **{final['counts']['live']}/32** fresh live-camera trials, compared with **{final['counts']['frozen']}/32** using a frozen image and **{final['counts']['baseline']}/32** using the original preset controller. There are **{final['valid_starts']} valid starts**; all invalid starts remain in planned denominators. Live successes by destination are **{final['live_successes_per_goal']}** out of 16 each. The frozen final gate is **{'PASS' if final['passed'] else 'FAIL'}**.

Development was **{dev['counts']['live']}/8 live** versus **{dev['counts']['baseline']}/8 baseline**. All six exposed regression outcomes remain alongside the final comparison. All **118** physical outcomes, **{frames:,}** state frames, **{observations:,}** RGB observations and frozen sources are preserved; the independent audit reproduces all **46** initial states exactly and recomputes both gates.

Sources span X [-0.14,0.16], Y [-0.18,-0.06] metres and yaw +/-0.6 radians. Requested destinations are [0.10,-0.115] and [0.20,0.05]. Fresh original scene seeds vary other-object starts, mass +/-8%, friction +/-10% and lighting 0.92..1.06. The middle destination [0.16,-0.06] remains unsupported. This finite experiment does not certify every reachable position, combined changed cabinet arrangements, unfamiliar objects, or arbitrary instructions.

The original coarse/refined RGB observer, V5 neural motor and V2 R1 route controller stay fixed throughout. Live images update approach/descent and the closure-time carry offset; subsequent transport uses motor feedback. A route may include table-supported release, parking and regrasping by the other arm. The frozen condition refreshes its initial image only between completed legs. No exact object pose, teacher action, inverse solver or hidden force supplies learned targets.

The preceding standalone synthetic perception protocol remains **failed** (94.956% present acceptance; 4.427 mm maximum), and its original physical seeds remain unexposed. This separate complete-task result does not change those stricter perception limits. Passing this physical experiment permits separately guarded production integration and regressions; this package alone does not change the browser or hosted controller.

Reproduction: use the archived source and declared seeds in `protocol.json`. The existing exact observer inputs and both candidates are in `training/bottle_refinement_v1` and `models/bottle_refinement_v1`; the motor, routing and original model hashes are bound by `physical/integration.json`. No new fitting occurred in this experiment.
'''
    write(output/'README.md', text.encode())
    files = {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}
    write(output/'manifest.json', {'schema': p['schema'], 'files': files})
    print({'files': len(files), 'frames': frames, 'observations': observations,
           'regenerated_starts': len(regenerated), 'counts': counts, 'gate_passed': final['passed']}, flush=True)


if __name__ == '__main__':
    run()
