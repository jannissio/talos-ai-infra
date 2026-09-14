"""Preserve the exposed clearance diagnosis and reproduce all saved-action replays."""
from collections import Counter
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.measure_manipulation_coverage import read, sha, put
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.policy_control import apply_targets
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


def run():
    protocol = ROOT/'docs/robotics/experiments/drawer-clearance-diagnostic-v1.json'
    p = read(protocol); raw = ROOT/p['raw_root']
    evidence = ROOT/'docs/robotics/evidence/drawer-clearance-diagnostic-v1'
    output = evidence/'episodes'
    gate = read(raw/'gate.json')
    assert gate['all_42_outcomes_retained'] and len(gate['rows']) == 42
    assert gate['passed'] and gate['recovered_plate_failures'] == 29 and gate['planning_recoveries'] == 0
    if output.exists() or evidence.exists():
        raise FileExistsError('Preserve both drawer input and evidence packages.')
    required = sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())
    preflight = require_space(output, required+64*1024**2)
    copied = {}
    def copy(source, target):
        require_space(target, source.stat().st_size+1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        copied[target.relative_to(ROOT).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source in raw.rglob('*'):
        if not source.is_file():
            continue
        relative = source.relative_to(raw)
        destination = output if relative.parts[0] in ('training', 'development') else evidence
        if source.name == 'scene.xml':
            relative = relative.parent/'source-scene.xml'
        copy(source, destination/relative)
    copy(ROOT/'.run/final-goal/drawer-clearance-diagnostic-v1.log', evidence/'collection.log')
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        target = source.parent/'scene.xml'; require_space(target, 1024**2)
        with target.open('xb') as stream:
            stream.write(ET.tostring(tree.getroot(), encoding='utf-8'))
    for name, digest in p['source_sha256'].items():
        archived = evidence/'frozen-source'/name
        assert sha(archived if archived.exists() else ROOT/name) == digest, name
    records = []; total_frames = replay_frames = replayed = 0
    for row in gate['rows']:
        folder = output/row['split']/str(row['seed']); r = read(folder/'report.json')
        assert sha(folder/'report.json') == row['report_sha256']
        assert r['status'] == row['status'] and r['training_eligible'] == row['training_eligible']
        assert r['status'] != 'harness_error' and r['protocol_sha256'] == sha(protocol)
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        with np.load(folder/'teacher-states.npz', allow_pickle=False) as z:
            state = {k: z[k] for k in z.files}
        count = len(state['time']); total_frames += count
        assert count > 0 and np.all(np.diff(state['time']) > 0)
        assert state['qpos'].shape == (count, model.nq) and state['qvel'].shape == (count, model.nv)
        assert all(np.isfinite(state[k]).all() for k in ('qpos', 'qvel', 'time', 'targets'))
        with np.load(folder/'initial-state.npz', allow_pickle=False) as z:
            initial = z['integration']
        data = mujoco.MjData(model); mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        assert np.array_equal(data.qpos, state['qpos'][0]) and np.array_equal(data.qvel, state['qvel'][0])
        with np.load(folder/'original-initial-state.npz', allow_pickle=False) as z:
            reset = mujoco.MjData(model)
            mujoco.mj_setState(model, reset, z['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
        original = ROOT/p['original_inputs']/row['split']/str(row['seed'])/'teacher-states.npz'
        with np.load(original, allow_pickle=False) as z:
            assert np.array_equal(reset.qpos, z['qpos'][0]) and np.array_equal(reset.qvel, z['qvel'][0])
        adr = int(model.joint('plate_free').qposadr[0]); dof = int(model.joint('plate_free').dofadr[0])
        reset.qpos[adr:adr+2] = p['clearance_plate_xy_m']; reset.qvel[dof:dof+6] = 0.
        mujoco.mj_forward(model, reset)
        for _ in range(p['clearance_settle_steps']):mujoco.mj_step(model, reset)
        reset.time = 0.; mujoco.mj_forward(model, reset)
        reproduced = np.empty_like(initial)
        mujoco.mj_getState(model, reset, reproduced, mujoco.mjtState.mjSTATE_INTEGRATION)
        assert np.array_equal(reproduced, initial), (row['seed'], 'reset intervention')
        observation = read(folder/'initial-observation.json')
        points = np.stack((data.site('drawer_handle_grasp').xpos.copy(),
                           data.body('cutlery_cabinet').xpos + data.body('cutlery_cabinet').xmat.reshape(3, 3) @ np.array([0., 0., .102])))
        assert np.array_equal(points, observation['training_only_keypoints_world_m'])
        for name, value in observation['views'].items():
            assert sha(folder/(name+'.png')) == value['image_sha256']
            projection = np.asarray(value['calibration']['projection'])
            homogeneous = np.c_[points, np.ones(len(points))] @ projection.T
            projected = homogeneous[:, :2]/homogeneous[:, 2:3]
            assert np.max(np.abs(projected-value['training_only_keypoints_px'])) < 1e-10
        if r['training_eligible']:
            assert r['teacher_outcome']['status'] == 'succeeded' and r['teacher_physical']['passed']
            assert r['replayed_saved_float32_endpoints'] and r['replay']['passed']
            with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
                endpoints, indices = z['actions20'], z['action_indices']
            assert endpoints.dtype == np.float32 and endpoints.shape == (len(indices), 12)
            with np.load(folder/'teacher-actions.npz', allow_pickle=False) as z:
                expected = np.clip(z['targets'][indices], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]).astype('float32')
            assert np.array_equal(endpoints, expected)
            with np.load(folder/'replay-states.npz', allow_pickle=False) as z:
                replay = {k: z[k] for k in z.files}
            check = DinnerPhysicalMonitor(model, data, r['layout'], 'drawer', 'left')
            cursor = 0
            for tick in range(int(indices[-1])+1+300):
                target = np.array([np.interp(tick, indices, endpoints[:, j]) for j in range(12)])
                failure = check.update()
                assert failure is None, (row['seed'], failure)
                if tick % 10 == 0:
                    assert np.array_equal(data.qpos, replay['qpos'][cursor]), (row['seed'], tick, 'qpos')
                    assert np.array_equal(data.qvel, replay['qvel'][cursor]), (row['seed'], tick, 'qvel')
                    assert float(data.time) == replay['time'][cursor]
                    assert np.array_equal(target, replay['targets'][cursor])
                    cursor += 1
                data.ctrl[:] = apply_targets(model, data, target, p['gripper_cap_nm'], 0)
                mujoco.mj_step(model, data)
            check.update()
            assert check.succeeded and check.report() == r['replay']
            assert np.array_equal(data.qpos, replay['qpos'][-1]) and np.array_equal(data.qvel, replay['qvel'][-1])
            assert np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) < .035
            assert float(data.joint('drawer_slide').qpos[0]) >= .105
            replayed += 1; replay_frames += len(replay['time'])
        reason = r.get('teacher_physical', {}).get('failure') or r.get('teacher_outcome', {}).get('message')
        records.append({**row, 'teacher_stop_reason': reason, 'teacher_state_frames': count,
                        'saved_action_replay_reproduced': bool(r['training_eligible'])})
        if len(records) % 16 == 0:
            print({'audited_cases': len(records), 'exact_replays': replayed}, flush=True)
    assert replayed == 30
    totals = {split: dict(Counter(r['status'] for r in records if r['split'] == split)) for split in ('training', 'development')}
    failure_reasons = dict(Counter(r['teacher_stop_reason'] for r in records if not r['training_eligible']))
    audit = {'schema': p['schema'], 'diagnostic_gate_passed': True, 'all_42_outcomes_retained': True,
             'exactly_reproduced_eligible_replays': replayed, 'teacher_state_frames': total_frames,
             'replay_state_frames': replay_frames, 'outcomes': totals, 'failure_reasons': failure_reasons,
             'rows': records, 'preflight': preflight, 'source_copies': copied,
             'verification': 'Every hash, initial state/keypoint label, saved float32 endpoint and all 30 eligible 200 Hz physical replays reproduced, including every saved qpos/qvel/target/time array and full physical monitor reports.',
             'fit_performed': False, 'promotion': False}
    put(evidence/'audit.json', audit)
    copy(Path(__file__), evidence/'audit-source.py')
    text = f'''# Drawer path clearance diagnosis

All **42** exposed cases are retained: one passing anchor, all **29** prior plate-obstruction failures and all **12** pose-planning failures. Moving only the plate during the declared reset to [-0.035,-0.155] recovers all 29 obstruction cases. The anchor remains successful; the 12 planning failures remain. No thresholds, teacher or torque cap changed.

The audit exactly reconstructs every original and intervention reset and all **30** successful saved-action replays. It retains **{total_frames:,}** teacher and **{replay_frames:,}** replay frames, full-rate actions, all images, labels, reports and failures.

This is an exposed causal diagnosis with an explicit reset intervention. The robot has not learned to clear the plate, and the earlier 80-case demonstration input gate stays failed. No fitting, fresh manipulation claim or promotion follows from this package. The result motivates obstacle-aware task order and deliberate clearing in the shared whole-table pipeline.
'''
    for target in (output/'README.md', evidence/'README.md'):
        require_space(target, 1024**2)
        with target.open('xb') as stream:
            stream.write(text.encode())
    for folder in (output, evidence):
        put(folder/'manifest.json', {'schema': p['schema'], 'files': {f.relative_to(folder).as_posix(): sha(f) for f in folder.rglob('*') if f.is_file()}})
    print({'outcomes': totals, 'reproduced_replays': replayed, 'failure_reasons': failure_reasons,
           'teacher_frames': total_frames, 'replay_frames': replay_frames}, flush=True)


if __name__ == '__main__':
    run()
