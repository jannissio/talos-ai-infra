"""Frozen compact drawer demonstration gate with exact float32-action replay."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from scripts.diagnose_drawer_configuration import ConfigurationDrawerTeacher
from scripts.measure_manipulation_coverage import setup, read, sha, put
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.policy_control import apply_targets
from simulation_lab.policy_cameras import camera_argument
from simulation_lab.rgb_servo_cameras import calibration, project
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def arrays(path, **values):
    require_space(path, sum(np.asarray(v).nbytes for v in values.values())+1024**2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def declare(path):
    parent = ROOT/'docs/robotics/experiments/manipulation-coverage-v1.json'
    diagnosis = ROOT/'docs/robotics/experiments/drawer-configuration-diagnostic-v1.json'
    d = read(diagnosis); audit_path = ROOT/d['evidence_package']/'audit.json'; audit = read(audit_path)
    assert audit['all_15_outcomes_retained'] and audit['physical_successes'] == 5
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab').glob('*.py')]
    names += ['scripts/collect_drawer_configurations.py', 'scripts/diagnose_drawer_configuration.py', 'scripts/measure_manipulation_coverage.py']
    names += [parent.relative_to(ROOT).as_posix(), diagnosis.relative_to(ROOT).as_posix(), audit_path.relative_to(ROOT).as_posix()]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab/assets/so101').rglob('*') if p.is_file()]
    original = read(parent)
    names += [(Path(original['context'])/name).as_posix() for name in original['context_sha256']]
    p = {'schema': 'talos.drawer-demonstrations.v1', 'declared_on': '2026-09-14',
         'raw_root': '.run/drawer-demonstrations-v1', 'parent_protocol': parent.relative_to(ROOT).as_posix(),
         'teacher_diagnosis': diagnosis.relative_to(ROOT).as_posix(),
         'source_sha256': {n: sha(ROOT/n) for n in names},
         'source_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'training_seeds': list(range(2026112801, 2026112865)),
         'development_seeds': list(range(2026112901, 2026112917)),
         'configuration': {'cabinet_x': [-.29, -.235], 'cabinet_y': [.17, .19],
                           'cabinet_yaw_rad': [-.05, .28], 'initial_opening_m': [0., .065]},
         'sampling': 'Independent uniform X, Y, yaw and opening, in that RNG order from each declared seed. No resampling. These are candidate support ranges, not asserted feasible everywhere.',
         'context': 'Restore the same exposed preceding-skill context as the broader diagnostic, then change the cabinet and cutlery reset geometry. The scene seed varies mass/friction/light through the original scene generator; preceding-object coordinates remain the recorded context. This is isolated drawer control, not full-sequence generalization.',
         'teacher': 'Unchanged cabinet-aligned contact-only teacher from the 15-case diagnosis. Exact state and inverse kinematics are training-only. No controller/model fitting belongs to this collection protocol.',
         'maximum_workers': 1, 'maximum_simulated_seconds_per_rollout': 85., 'maximum_wall_seconds_per_case': 180.,
         'storage_budget_bytes': 2*1024**3, 'reserve_gib': 10, 'gripper_cap_nm': .15,
         'input_gate': {'minimum_training_eligible': 32, 'minimum_development_eligible': 8,
                        'require_all_planned_cases': True, 'maximum_harness_errors': 0},
         'eligibility': 'Only a passing teacher AND independent unchanged DinnerPhysicalMonitor AND passing replay of the exact saved float32 20 Hz targets. Retain full-rate targets and both state traces for every attempted replay; never hide invalid starts, planning failures or replay failures.',
         'perception_record': 'Three initial 320x240 RGB workspace views, camera calibrations, and scoring/training-only handle/roof keypoint labels. No learned model receives these privileged labels at inference. A separately gated observer and motor fit are still required.',
         'next_gate': 'Require >=32/64 training and >=8/16 development replay-eligible episodes, all outcomes retained and no harness error before any fit. Select no neural architecture/checkpoint from an undisclosed physical test. No reserved fresh physical evaluation seeds are exposed by this input collection.'}
    if path.exists() or (ROOT/p['raw_root']).exists():
        raise FileExistsError('Preserve declarations and raw demonstrations.')
    p['preflight'] = require_space(ROOT/p['raw_root'], p['storage_budget_bytes'])
    put(path, p)
    print({'planned_episodes': 80, 'protocol_sha256': sha(path)}, flush=True)


def initial_images(model, data, folder):
    points = np.stack((data.site('drawer_handle_grasp').xpos.copy(),
                       data.body('cutlery_cabinet').xpos + data.body('cutlery_cabinet').xmat.reshape(3, 3) @ np.array([0., 0., .102])))
    model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    observations = {}
    try:
        for name in ('overhead', 'table_left', 'table_right'):
            renderer.update_scene(data, camera=camera_argument(name), scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            image = renderer.render().copy()
            if image.mean() < 10:
                raise ValueError('Initial camera is black.')
            matrix = calibration(renderer)
            target = folder/(name+'.png'); require_space(target, 1024**2)
            with target.open('xb') as stream:
                Image.fromarray(image).save(stream, format='PNG')
            observations[name] = {'calibration': matrix, 'image_sha256': sha(target),
                                  'training_only_keypoints_px': project(points, matrix['projection']).tolist()}
    finally:
        renderer.close()
    put(folder/'initial-observation.json', {'views': observations, 'training_only_keypoints_world_m': points.tolist(),
                                           'point_order': ['drawer_handle_center', 'cabinet_roof_center']})


def episode(p, protocol_path, split, seed):
    folder = ROOT/p['raw_root']/split/str(seed)
    if folder.exists():
        raise FileExistsError('Preserve every drawer episode.')
    preflight = require_space(folder, 64*1024**2); folder.mkdir(parents=True)
    rng = np.random.default_rng(seed); bounds = p['configuration']
    x, y, yaw, opening = [float(rng.uniform(*bounds[k])) for k in ('cabinet_x', 'cabinet_y', 'cabinet_yaw_rad', 'initial_opening_m')]
    case = {'id': str(seed), 'skill': 'drawer', 'kind': 'combined_configuration',
            'cabinet_xy': [x, y], 'cabinet_yaw_rad': yaw, 'drawer_open_m': opening}
    report = {'schema': p['schema'], 'protocol_sha256': sha(protocol_path), 'seed': seed, 'split': split,
              'case': case, 'preflight': preflight, 'status': 'harness_error', 'training_eligible': False,
              'teacher': 'exact-state contact-only demonstration', 'state_writes_during_control': 0,
              'external_forces': 0, 'equality_constraints': 0}
    trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    replay_trace = {k: [] for k in ('qpos', 'qvel', 'time', 'targets')}
    actions, stages = [], []
    started = time.perf_counter(); task = None
    try:
        parent = deepcopy(read(ROOT/p['parent_protocol'])); parent['source_seed'] = seed
        model, data, layout, valid = setup(parent, case, folder)
        report.update(setup=valid, layout=layout)
        initial = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, initial, STATE)
        arrays(folder/'initial-state.npz', integration=initial)
        initial_images(model, data, folder)
        targets = data.ctrl.copy()
        if valid['valid']:
            task = ConfigurationDrawerTeacher(model, data, layout); task.start()
            monitor = DinnerPhysicalMonitor(model, data, layout, 'drawer', 'left')
            finished_at = None
            for tick in range(int(p['maximum_simulated_seconds_per_rollout']/model.opt.timestep)+1):
                q, v = data.qpos.copy(), data.qvel.copy()
                if task.active:
                    task.update(targets)
                if not np.array_equal(q, data.qpos) or not np.array_equal(v, data.qvel):
                    raise ValueError('Teacher wrote authoritative state.')
                failure = monitor.update()
                terminal = bool(failure) or task.status == 'failed' or (task.status == 'succeeded' and monitor.succeeded)
                if not task.active:
                    finished_at = float(data.time) if finished_at is None else finished_at
                    terminal |= float(data.time)-finished_at >= 1.
                if tick % 10 == 0 or terminal:
                    for key, value in zip(trace, (q, v, float(data.time), task.stage, targets.copy())):
                        trace[key].append(value)
                if terminal:
                    break
                if tick % 2000 == 0:
                    require_space(folder, 64*1024**2)
                if time.perf_counter()-started > p['maximum_wall_seconds_per_case']:
                    raise TimeoutError('Declared drawer episode wall-time limit.')
                actions.append(targets.copy()); stages.append(task.stage)
                data.ctrl[:] = task.apply_gripper_limit(targets)
                if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise ValueError('Hidden force or constraint.')
                mujoco.mj_step(model, data)
            report.update(teacher_outcome=task.snapshot(), teacher_physical=monitor.report(),
                          status='teacher_passed' if task.status == 'succeeded' and monitor.succeeded else 'teacher_failed')
            if actions:
                arrays(folder/'teacher-actions.npz', targets=np.asarray(actions), stage=np.asarray(stages))
            if report['status'] == 'teacher_passed':
                indices = np.unique(np.r_[np.arange(0, len(actions), 10), len(actions)-1])
                endpoints = np.clip(np.asarray(actions)[indices], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]).astype('float32')
                arrays(folder/'trajectory.npz', actions20=endpoints, action_indices=indices, stage=np.asarray(stages)[indices])
                replay = mujoco.MjData(model); mujoco.mj_setState(model, replay, initial, STATE); mujoco.mj_forward(model, replay)
                check = DinnerPhysicalMonitor(model, replay, layout, 'drawer', 'left')
                for tick in range(int(indices[-1])+1+300):
                    target = np.asarray([np.interp(tick, indices, endpoints[:, j]) for j in range(12)])
                    failure = check.update()
                    if tick % 10 == 0 or failure:
                        for key, value in zip(replay_trace, (replay.qpos.copy(), replay.qvel.copy(), float(replay.time), target.copy())):
                            replay_trace[key].append(value)
                    if failure:
                        break
                    if time.perf_counter()-started > p['maximum_wall_seconds_per_case']:
                        raise TimeoutError('Declared drawer replay wall-time limit.')
                    replay.ctrl[:] = apply_targets(model, replay, target, p['gripper_cap_nm'], 0)
                    if model.neq or np.any(replay.xfrc_applied) or np.any(replay.qfrc_applied):
                        raise ValueError('Hidden replay force or constraint.')
                    mujoco.mj_step(model, replay)
                check.update()
                if not replay_trace['time'] or replay_trace['time'][-1] != float(replay.time):
                    for key, value in zip(replay_trace, (replay.qpos.copy(), replay.qvel.copy(), float(replay.time), target.copy())):
                        replay_trace[key].append(value)
                report.update(replay=check.report(), replayed_saved_float32_endpoints=True,
                              training_eligible=bool(check.succeeded), status='eligible' if check.succeeded else 'replay_failed')
        else:
            report['status'] = 'invalid_start'
            for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time), 'invalid_start', targets.copy())):
                trace[key].append(value)
    except Exception as exc:
        report.update(status='harness_error', training_eligible=False, error_type=type(exc).__name__, message=str(exc))
        if task is not None:
            report['teacher_outcome'] = task.snapshot()
    finally:
        arrays(folder/'teacher-states.npz', **{k: np.asarray(v) for k, v in trace.items()})
        if replay_trace['time']:
            arrays(folder/'replay-states.npz', **{k: np.asarray(v) for k, v in replay_trace.items()})
        report['wall_seconds'] = time.perf_counter()-started
        put(folder/'report.json', report)
    row = {k: report[k] for k in ('seed', 'split', 'status', 'training_eligible', 'wall_seconds')}
    row.update(message=report.get('message', report.get('teacher_outcome', {}).get('message')), report_sha256=sha(folder/'report.json'))
    print(row, flush=True)
    return row


def run(path):
    p = read(path); raw = ROOT/p['raw_root']
    if raw.exists():
        raise FileExistsError('Preserve the original collection.')
    for name, digest in p['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen source or evidence changed: '+name)
    require_space(raw, p['storage_budget_bytes']); raw.mkdir(parents=True)
    put(raw/'protocol.json', p)
    for name in p['source_sha256']:
        if name.endswith('.py'):
            target = raw/'frozen-source'/name; require_space(target, (ROOT/name).stat().st_size+1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((ROOT/name).read_bytes())
    torch.set_num_threads(1)
    rows = []
    for split in ('training', 'development'):
        for seed in p[split+'_seeds']:
            row = episode(p, path, split, seed); rows.append(row)
            if row['status'] == 'harness_error':
                put(raw/'stopped.json', {'reason': 'Harness error; retain attempted data, later seeds remain unexposed.', 'rows': rows})
                return 1
    counts = {split: sum(r['training_eligible'] for r in rows if r['split'] == split) for split in ('training', 'development')}
    passed = len(rows) == 80 and all(counts[s] >= p['input_gate']['minimum_'+s+'_eligible'] for s in counts)
    put(raw/'gate.json', {'schema': p['schema'], 'protocol_sha256': sha(path), 'passed': passed,
                         'all_80_outcomes_retained': len(rows) == 80, 'eligible': counts, 'rows': rows,
                         'fit_performed': False, 'promotion': False})
    print({'input_gate_passed': passed, 'eligible': counts}, flush=True)
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.declare:
        declare(args.declare)
    elif args.protocol:
        raise SystemExit(run(args.protocol))
    else:
        parser.error('Use --declare or --protocol.')
