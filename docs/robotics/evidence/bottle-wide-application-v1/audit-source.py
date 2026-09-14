"""Archive and audit frozen production-entry and paired wider dinner outcomes."""
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
from scripts.bottle_refinement_experiment import read, sha, write
from scripts.package_bottle_wide import check_physical
from scripts.package_spoon_release import physical_criteria
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


def run():
    protocol = ROOT/'docs/robotics/experiments/bottle-wide-application-v1.json'
    p = read(protocol); raw = ROOT/p['raw_root']
    output = ROOT/'docs/robotics/evidence/bottle-wide-application-v1'
    stages = []
    for name in ('entry', 'development', 'evaluation'):
        if (raw/name/'gate.json').exists():
            stages.append((name, read(raw/name/'gate.json')))
        else:break
    assert stages, 'A completed gate is required.'
    assert stages[-1][0] == 'evaluation' or not stages[-1][1]['passed'], 'An authorized next stage remains unfinished.'
    if output.exists():raise FileExistsError('Preserve the production evidence archive.')
    required = sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())
    preflight = require_space(output, required+32*1024**2)
    copied = {}

    def copy(source, target):
        payload = source.read_bytes()
        if source.suffix in ('.json', '.log', '.py'):
            for actual, replacement in ((str(ROOT), '.'), (str(Path.home()), '<USER_HOME>')):
                for old in (actual.replace('\\', '\\\\'), actual, actual.replace('\\', '/')):
                    payload = payload.replace(old.encode(), replacement.encode())
        require_space(target, len(payload)+1024**2)
        write(target, payload)
        copied[target.relative_to(output).as_posix()] = {'source': source.relative_to(ROOT).as_posix(),
            'original_sha256': sha(source), 'copied_sha256': sha(target), 'sanitized': payload != source.read_bytes()}

    for source in raw.rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            relative = source.relative_to(raw)
            if source.name == 'scene.xml':relative = relative.parent/'source-scene.xml'
            copy(source, output/relative)
    for stage, _ in stages:
        source = ROOT/'.run/final-goal'/f'bottle-wide-application-v1-{stage}.log'
        copy(source, output/'console'/source.name)
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        require_space(source.parent/'scene.xml', 1024**2)
        write(source.parent/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
    for name, digest in p['source_sha256'].items():
        assert sha(output/'frozen-source'/name) == digest, name
    assert sha(output/'protocol.json') == sha(protocol)
    frames = observations = legs = 0
    starts = {}; records = []; gate_checks = {}
    for stage, gate in stages:
        planned = p['entry_cases'] if stage == 'entry' else [
            {'id': f'{seed}-{mode}', 'seed': seed, 'mode': mode}
            for seed in p[stage+'_seeds'] for mode in p['paired_modes']]
        assert gate['protocol_sha256'] == sha(protocol) and gate['all_planned_outcomes_retained']
        assert {r['case']['id'] for r in gate['rows']} == {r['id'] for r in planned}
        counts = Counter(); bottles = 0
        for row in gate['rows']:
            case = row['case']; folder = output/stage/case['id']; r = read(folder/'report.json')
            assert sha(folder/'report.json') == row['report_sha256']
            assert r['protocol_sha256'] == sha(protocol) and r['case'] == case
            assert r['status'] == row['status'] and r['passed'] == row['passed'] and row['exit_code'] == 0
            assert r['status'] != 'harness_error'
            counts[case['mode']] += int(r['passed'])
            bottles += int(case['mode'] == 'candidate' and r['bottle_completed'])
            with np.load(folder/'states.npz', allow_pickle=False) as z:state = {k: z[k] for k in z.files}
            n = len(state['time']); frames += n
            assert n == r['trace_frames']
            regenerated = False
            if n:
                model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
                assert state['qpos'].shape == (n, model.nq) and state['qvel'].shape == (n, model.nv)
                assert state['targets'].shape == (n, 12) and np.all(np.diff(state['time']) > 0)
                assert all(np.isfinite(state[k]).all() for k in ('qpos', 'qvel', 'time', 'targets'))
                with np.load(folder/'initial.npz', allow_pickle=False) as initial:
                    assert np.array_equal(initial['qpos'], state['qpos'][0])
                    assert np.array_equal(initial['qvel'], state['qvel'][0])
                key = (case['surface'], case['seed'], case['preset'])
                if key in starts:
                    assert np.array_equal(starts[key][0], state['qpos'][0])
                    assert np.array_equal(starts[key][1], state['qvel'][0])
                else:
                    starts[key] = (state['qpos'][0], state['qvel'][0])
                # Rebuild the reset directly from the portable XML and RNG recipe.
                data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
                data.joint('drawer_slide').qpos[0] = r['layout']['drawer']['initial_open_m']
                if case['preset'] == 'wide_rectangle':
                    rng = np.random.default_rng(case['seed'])
                    x, y, yaw = rng.uniform(-.14, .16), rng.uniform(-.18, -.06), rng.uniform(-.6, .6)
                    adr = int(model.joint('bottle_free').qposadr[0])
                    data.qpos[adr:adr+7] = [x, y, r['layout']['table_z']+.001, math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                mujoco.mj_forward(model, data)
                for _ in range(200 if case['preset'] == 'upright' else 300):mujoco.mj_step(model, data)
                data.time = 0.; mujoco.mj_forward(model, data)
                assert np.array_equal(data.qpos, state['qpos'][0]), (stage, case['id'], 'regenerated qpos')
                assert np.array_equal(data.qvel, state['qvel'][0]), (stage, case['id'], 'regenerated qvel')
                regenerated = True
                data.qpos[:] = state['qpos'][-1]; data.qvel[:] = state['qvel'][-1]; mujoco.mj_forward(model, data)
                if r['passed']:
                    assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
                    assert np.max(np.abs(data.qvel[:12])) < .12
                    if case['instruction'] == 'Set the table':assert data.joint('drawer_slide').qpos[0] >= .105
            else:
                assert not r['passed'], 'A success must retain a physical trace.'
            full = case['instruction'] == 'Set the table'
            if r['passed']:
                assert r['task']['status'] == 'succeeded'
                assert all(r[k] == 0 for k in ('physics_state_writes_during_control', 'hidden_forces', 'equality_constraints'))
                if full:
                    completed = r['task']['completed_steps']
                    assert all(s in completed for s in ('plate', 'mug', 'drawer', 'fork', 'spoon'))
                for skill in r['task'].get('results', []):
                    if not full and case['mode'] == 'candidate':continue
                    if skill.get('policy_mode') == 'learned_bottle_wide':
                        assert skill['teacher_updates'] == skill['inverse_solver_calls_during_control'] == 0
                        check_physical({'passed': True, 'failure': None, 'metrics': skill['metrics']})
                    else:physical_criteria(skill)
            bottle_path = folder/'bottle-runtime.json'; case_observations = 0
            if bottle_path.exists():
                for bottle in read(bottle_path)['tasks']:
                    case_observations += len(bottle['observations'])
                    for leg in bottle['completed_legs']:check_physical(leg['physical']); legs += 1
                    if bottle['snapshot']['status'] == 'succeeded':
                        assert bottle['snapshot']['completed_route_legs'] == bottle['snapshot']['total_route_legs'] == len(bottle['completed_legs'])
                        assert np.linalg.norm(data.body('bottle').xpos[:2]-bottle['snapshot']['destination_xy_m']) < .008
            observations += case_observations
            records.append({**row, 'trace_frames': n, 'rgb_observations': case_observations,
                'reset_independently_regenerated': regenerated})
        if stage == 'entry':passed = counts['candidate'] == len(planned) == 8
        else:
            limits = p[stage+'_gate']
            passed = counts['candidate'] >= limits['minimum_candidate_full_successes'] and bottles >= limits['minimum_candidate_bottle_successes']
            passed &= counts['candidate'] > counts['baseline'] if stage == 'development' else counts['candidate'] >= counts['baseline']+limits['minimum_more_full_successes_than_baseline']
        assert bool(passed) == gate['passed'] and bottles == gate['candidate_bottle_successes']
        assert all(counts[mode] == value for mode, value in gate['counts'].items())
        gate_checks[stage] = {'passed': bool(passed), 'counts': dict(counts), 'bottles': bottles}
    audit = {'schema': p['schema'], 'audit_passed': True, 'gates': gate_checks, 'rows': records,
        'state_frames': frames, 'rgb_observations': observations, 'completed_bottle_legs': legs,
        'distinct_paired_reset_groups': len(starts), 'preflight': preflight, 'source_copies': copied,
        'scope': 'Actual local/public entry points and fresh paired full dinners with wider bottle starts. Other objects and cabinet remain in the existing narrow distribution. All outcomes and 20 Hz traces retained. Reset states and final parked/placement geometry are independently checked; saved per-skill monitor thresholds are checked, but 20 Hz traces do not independently reconstruct every 200 Hz contact event. No live browser/cloud/Intel or arbitrary-arrangement claim.'}
    write(output/'audit.json', audit); write(output/'audit-source.py', Path(__file__).read_bytes())
    write(output/'README.md', ('# Wider bottle application and full dinner checks\n\n'
        + '\n'.join(f"- {name}: {value['counts']}; candidate bottle successes {value['bottles']}; gate {'PASS' if value['passed'] else 'FAIL'}." for name, value in gate_checks.items())
        + f'\n\nAll {len(records)} outcomes, {frames:,} state frames, {observations:,} bottle RGB observations and {legs} completed bottle route legs are retained. The audit independently regenerates every recorded reset, checks identical paired starts and recomputes every completed gate. Portable scenes, frozen production sources and all failures are included.\n\n'
        + audit['scope']+'\n\nA passing result permits a separate browser/cloud deployment check. Repository and Space stay private; the user alone presses final Submit.\n').encode())
    write(output/'manifest.json', {'schema': p['schema'], 'files': {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}})
    print({'audit_passed': True, 'gates': gate_checks, 'state_frames': frames, 'rgb_observations': observations}, flush=True)


if __name__ == '__main__':run()
