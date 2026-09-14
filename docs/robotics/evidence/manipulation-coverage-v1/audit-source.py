"""Audit and retain every paired case in the declared manipulation coverage map."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    if not isinstance(payload, bytes):
        payload = (json.dumps(payload, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def category(report):
    if report['status'] in ('succeeded', 'invalid_start', 'harness_error'):
        return report['status']
    message = report.get('task', {}).get('message', report.get('message', '')).lower()
    if any(word in message for word in ('camera', 'observation', 'ambiguous', 'missing', 'rgb')):
        return 'perception_or_trained_support'
    if 'track' in message:
        return 'motor_tracking'
    if any(word in message for word in ('finger', 'grasp', 'tilt', 'support')):
        return 'grasp_or_retention'
    if any(word in message for word in ('collision', 'contact', 'clearance')):
        return 'collision_or_clearance'
    if 'reach' in message:
        return 'planner_reach_not_demonstrated'
    if 'closed' in message:
        return 'drawer_closed_only_contract'
    if 'timeout' in message or 'time' in message:
        return 'goal_or_progress_timeout'
    return 'other_failure'


def run():
    raw = ROOT/'.run/manipulation-coverage-v1'
    output = ROOT/'docs/robotics/evidence/manipulation-coverage-v1'
    if output.exists():
        raise FileExistsError('Preserve completed coverage packages.')
    protocol, batch = read(raw/'protocol.json'), read(raw/'summary.json')
    assert batch['all_declared_rollouts_completed'] and batch['harness_errors'] == 0
    assert read(raw/'anchor-gate.json')['passed']
    assert len(batch['rows']) == 200 and len(protocol['cases']) == 100
    expected = {(c['id'], mode) for c in protocol['cases'] for mode in protocol['controllers']}
    assert {(r['case'], r['controller']) for r in batch['rows']} == expected
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    total = sum(p.stat().st_size for p in raw.rglob('*') if p.is_file())
    preflight = require_space(output, total+64*1024**2)
    output.mkdir(parents=True)
    copy_map = {}
    for source in raw.rglob('*'):
        if source.is_file():
            name = source.relative_to(raw).as_posix()
            if name.endswith('/scene.xml'):
                name = name[:-len('scene.xml')]+'source-scene.xml'
            write(output/name, source.read_bytes())
            copy_map[name] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    rows, frames = [], 0
    for case in protocol['cases']:
        records, starts = {}, {}
        for controller in protocol['controllers']:
            folder = output/(case['id']+'-'+controller)
            report = read(folder/'report.json')
            assert report['case'] == case and report['controller'] == controller
            assert report['status'] != 'harness_error'
            tree = ET.parse(folder/'source-scene.xml')
            tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', folder).replace('\\', '/'))
            write(folder/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
            model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
            with np.load(folder/'states.npz', allow_pickle=False) as trace:
                n = len(trace['time']); frames += n
                assert n == report['trace_frames'] and n > 0
                assert trace['qpos'].shape == (n, model.nq) and trace['qvel'].shape == (n, model.nv)
                assert trace['targets'].shape == (n, 12)
                assert all(np.isfinite(trace[k]).all() for k in ('qpos', 'qvel', 'time', 'targets'))
                assert np.all(np.diff(trace['time']) > 0)
                starts[controller] = (trace['qpos'][0].copy(), trace['qvel'][0].copy())
                data.qpos[:] = trace['qpos'][-1]; data.qvel[:] = trace['qvel'][-1]
                mujoco.mj_forward(model, data)
                if report['status'] == 'succeeded':
                    assert report['setup']['valid']
                    assert all(report[k] == 0 for k in ('physics_state_writes_during_control', 'hidden_forces', 'equality_constraints'))
                    assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
                    assert np.max(np.abs(data.qvel[:12])) < .12
                    if case['skill'] == 'drawer':
                        assert float(data.joint('drawer_slide').qpos[0]) >= .105
                    else:
                        targets = report['layout']['targets']
                        target = next((t['position_m'] for t in targets if t['object_id'] == case['skill']), None)
                        if target is None:
                            initial = next(o['initial_position_m'] for o in report['layout']['objects'] if o['id'] == case['skill'])
                            target = np.asarray(initial)+[.07, -.04, 0.]
                        position = data.body(case['skill']).xpos
                        assert np.linalg.norm(position[:2]-np.asarray(target)[:2]) < .008
                        assert data.body(case['skill']).xmat[8] > .98
            records[controller] = {'status': report['status'], 'failure_category': category(report),
                'valid_start': report['setup']['valid'], 'message': report.get('task', {}).get('message'),
                'report_sha256': sha(folder/'report.json'), 'source_xy_m': report['setup']['settled_source_xyz_m'][:2]}
        assert all(np.array_equal(a, b) for a, b in zip(starts['learned'], starts['teacher'])), case['id']
        assert records['learned']['valid_start'] == records['teacher']['valid_start']
        rows.append({'case': case, 'outcomes': records, 'identical_initial_qpos_qvel': True,
                     'feasible_but_learned_failed': records['teacher']['status'] == 'succeeded' and records['learned']['status'] != 'succeeded'})
    def summarize(records):
        valid = [r for r in records if r['outcomes']['learned']['valid_start']]
        return {'planned_cases': len(records), 'valid_starts': len(valid), 'invalid_starts': len(records)-len(valid),
            'learned_successes': sum(r['outcomes']['learned']['status'] == 'succeeded' for r in records),
            'teacher_successes': sum(r['outcomes']['teacher']['status'] == 'succeeded' for r in records),
            'teacher_feasible_but_learned_failed': sum(r['feasible_but_learned_failed'] for r in records),
            'learned_failure_categories': dict(Counter(r['outcomes']['learned']['failure_category'] for r in records))}
    skills = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon']
    diagnostic = [r for r in rows if r['case']['kind'] != 'anchor']
    audit = {'schema': protocol['schema'], 'protocol_sha256': sha(raw/'protocol.json'),
        'all_200_outcomes_retained': True, 'all_100_paired_initial_states_identical': True,
        'physical_traces_verified': 200, 'trace_frames': frames, 'storage_preflight': preflight,
        'all_cases': summarize(rows), 'diagnostic_without_anchors': summarize(diagnostic),
        'by_skill_without_anchors': {s: summarize([r for r in diagnostic if r['case']['skill'] == s]) for s in skills},
        'by_kind_without_anchors': {k: summarize([r for r in diagnostic if r['case']['kind'] == k]) for k in sorted({r['case']['kind'] for r in diagnostic})},
        'cases': rows, 'source_copies': copy_map,
        'scope': 'Exposed diagnostic grid, isolated skills after one recorded context. Exact-state controller outcomes are a feasibility lower bound, not learned success or proof that rejected points are unreachable. Sampled physical states independently confirm final geometry/parking; 20 Hz records do not independently reconstruct every 200 Hz contact observation.'}
    write(output/'audit.json', audit)
    write(output/'audit-source.py', Path(__file__).read_bytes())
    table = '\n'.join(f"| {s} | {a['planned_cases']} | {a['valid_starts']} | {a['learned_successes']} | {a['teacher_successes']} | {a['teacher_feasible_but_learned_failed']} |" for s, a in audit['by_skill_without_anchors'].items())
    summary = audit['diagnostic_without_anchors']
    text = f'''# Wider manipulation coverage: first diagnostic

All **100 cases / 200 paired rollouts** finish with no harness errors. The six learned anchor checks pass before broader exposure. Both controllers receive identical initial physical states in every case. All source versions, reports, failures and **{frames:,} recorded frames** are retained.

The following totals exclude the six anchors. There are **{summary['planned_cases']} diagnostic cases**, **{summary['valid_starts']} valid starts**, and **{summary['invalid_starts']} invalid arrangements**. The learned system succeeds in **{summary['learned_successes']}**; the separate programmed controller demonstrates physical success in **{summary['teacher_successes']}**. There are **{summary['teacher_feasible_but_learned_failed']}** cases that are physically demonstrated by the programmed controller but fail under learned control.

| Skill | Planned | Valid starts | Learned successes | Programmed successes | Programmed pass / learned fail |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

[The frozen protocol](protocol.json) contains every exact coordinate, orientation, destination and drawer configuration. [The independent audit](audit.json) contains every paired outcome and failure category. The [current broader goal](../../BROADER_MANIPULATION_PLAN.md) gives the operational target and next development steps.

These are isolated skill tests in one preceding-task context, with one factor changed at a time. They do not test arbitrary clutter combinations or independent randomized full sequences. Invalid resets are retained and reported separately; perception and planner refusals stay in the valid denominator. A failed exact-state controller does not prove a point unreachable. The mug uses the selected live correction in this isolated diagnostic harness; its public interface still permits that option only for full table setting. Destinations passed to physical scoring are not secretly supplied to an unchanged neural policy.

This result confirms that the narrow submission demonstration does not satisfy wider manipulation coverage. The existing models and demo are preserved as the fallback. There is no candidate promotion from this measurement alone.
'''
    write(output/'README.md', text.encode())
    records = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'manifest.json', {'schema': protocol['schema'], 'files': records})
    print(json.dumps({'files': len(records), 'frames': frames, 'diagnostic': summary}), flush=True)


if __name__ == '__main__':
    run()
