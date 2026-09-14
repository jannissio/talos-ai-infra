"""Develop complete jointly randomized physical workflows; preserve every attempt.

Runs an explicitly privileged demonstration teacher, not a learned controller.
Each output is immutable and contains the exact tested source and motor trace.
Development seeds are exposed; this command does not evaluate held-out success.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image

from simulation_lab.dinner import OBJECTS
from simulation_lab.random_dinner import assess, body_bounds, draw
from simulation_lab.scene import HOME, build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_observation import TableObserver
from simulation_lab.table_teacher import TableTeacher, destination


def put(path, value):
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:stream.write(payload)


def arrays(path, **values):
    require_space(path, sum(np.asarray(v).nbytes for v in values.values())+1024**2)
    with path.open('xb') as stream:np.savez_compressed(stream, **values)


def observe(observer, data, path):
    require_space(path, 24*1024**2); path.mkdir(parents=True)
    obs = observer.observe(data)
    arrays(path/'observation.npz', policy_image=obs['policy_image'], observed=obs['observed'],
           rgb=np.stack([v['rgb'] for v in obs['views']]), depth_m=np.stack([v['depth_m'] for v in obs['views']]))
    put(path/'metadata.json', {'time_s': obs['time_s'], 'grid': obs['grid'],
        'calibration': [v['calibration'] for v in obs['views']], 'contract': obs['observation_contract']})
    with (path/'heightmap.png').open('xb') as stream:Image.fromarray(obs['rgb']).save(stream, format='PNG')


def world_bounds(model, data, name):
    pose = data.joint(name+'_free').qpos
    low, high = body_bounds(model, name, pose[3:])
    return low+pose[:3], high+pose[:3]


def blockers(model, data, name, target):
    low, high = body_bounds(model, name, np.array([1., 0., 0., 0.]))
    if name == 'spoon':
        quat = np.array([np.sqrt(.5), 0., 0., -np.sqrt(.5)])
        low, high = body_bounds(model, name, quat)
    low += target; high += target
    rows = []
    for other in OBJECTS:
        if other == name:continue
        olow, ohigh = world_bounds(model, data, other)
        if np.all(high[:2]+.004 > olow[:2]) and np.all(ohigh[:2]+.004 > low[:2]):rows.append(other)
    return rows


def candidates(model, data, layout, remaining):
    direct = [(name, destination(layout, name), 'final_setting') for name in remaining
              if not blockers(model, data, name, destination(layout, name))]
    direct.sort(key=lambda row: min(np.linalg.norm(data.body(row[0]).xpos[:2]-[x, -.235]) for x in (-.25, .25)))
    if direct:return direct
    # A cyclic obstruction calls for an explicit temporary relocation, sampled
    # over actual tabletop geometry. It is never a reset-only teleport.
    table = model.geom('table').id; half = model.geom_size[table, :2]; center = data.geom_xpos[table, :2]
    options = []
    for name in remaining:
        low, high = body_bounds(model, name, np.array([1., 0., 0., 0.]))
        xs = np.arange(center[0]-half[0]-low[0]+.015, center[0]+half[0]-high[0]-.015, .065)
        ys = np.arange(center[1]-half[1]-low[1]+.015, center[1]+half[1]-high[1]-.015, .065)
        for x in xs:
            for y in ys:
                target = np.array([x, y, layout['table_z']])
                if np.linalg.norm(target[:2]-data.body(name).xpos[:2]) < .07:continue
                if blockers(model, data, name, target):continue
                cost = np.linalg.norm(target[:2]-data.body(name).xpos[:2])
                options.append((cost, name, target, 'temporary_clearance'))
    return [(name, target, role) for _, name, target, role in sorted(options, key=lambda r: r[0])[:24]]


def execute(model, data, layout, task, targets, folder, observer):
    require_space(folder, 384*1024**2); folder.mkdir(parents=True)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(model, state_spec)); mujoco.mj_getState(model, data, initial, state_spec)
    controls, qpos, qvel, stages = [], [data.qpos.copy()], [data.qvel.copy()], [task.stage]
    # The selected path was planned with real geometry; no action is applied yet.
    put(folder/'search.json', task.search_log)
    put(folder/'action.json', {'item': task.tube['id'], 'arm': task.side,
        'grasp_candidate': task.chosen.name, 'pick_point_m': task.grasp.tolist(),
        'place_body_origin_m': task.destination_position.tolist(),
        'teacher_observation': 'privileged_state', 'policy_observation': 'calibrated_rgbd_only'})
    stage_before = None; observations = 0; tick = 0
    while task.active and tick < 24000:
        if task.stage != stage_before:
            observe(observer, data, folder/f'observation-{observations:03d}-{task.stage}')
            stage_before = task.stage; observations += 1
            print({'item': task.tube['id'], 'arm': task.side, 'stage': task.stage,
                   'simulated_s': round(float(data.time), 3)}, flush=True)
        task.update(targets)
        if not task.active:break
        data.ctrl[:] = task.apply_gripper_limit(targets)
        controls.append(data.ctrl.copy())
        assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy()); stages.append(task.stage)
        tick += 1
    if task.active:task._finish('failed', 'Development episode tick budget exhausted.', True)
    values = {'initial_integration': initial, 'ctrl': np.asarray(controls),
              'qpos': np.asarray(qpos), 'qvel': np.asarray(qvel), 'stage': np.asarray(stages)}
    arrays(folder/'states.npz', **values)
    # Independently replay the saved actual motor commands, without the teacher.
    replay = mujoco.MjData(model); mujoco.mj_setState(model, replay, initial, state_spec)
    mujoco.mj_forward(model, replay)
    exact = True; max_error = 0.
    for i, ctrl in enumerate(values['ctrl']):
        replay.ctrl[:] = ctrl; mujoco.mj_step(model, replay)
        discrepancy = max(float(np.max(np.abs(replay.qpos-values['qpos'][i+1]))),
                          float(np.max(np.abs(replay.qvel-values['qvel'][i+1]))))
        max_error = max(max_error, discrepancy); exact &= discrepancy == 0.
    result = {'item': task.tube['id'], 'arm': task.side, 'status': task.status, 'message': task.message,
        'metrics': task.metrics, 'history': task.history, 'frames': len(qpos), 'observations': observations,
        'replay_exact': bool(exact), 'replay_max_state_error': max_error,
        'demonstration_eligible': task.status == 'succeeded' and bool(exact)}
    put(folder/'result.json', result); print(result, flush=True)
    return result


def run(folder, seed, maximum_actions, only_item=None, lookahead_attempts=6):
    if folder.exists():raise FileExistsError('Preserve every development attempt.')
    preflight = require_space(folder, 8*1024**3); folder.mkdir(parents=True)
    started = time.perf_counter(); report = {'seed': seed, 'preflight': preflight,
        'scope': 'Exposed development of a shared exact-state physical teacher; not learned evaluation.',
        'whole_table_complete': False, 'actions': [], 'planning_failures': [], 'remaining': list(OBJECTS),
        'only_item_diagnostic': only_item, 'maximum_actions': maximum_actions,
        'lookahead_attempts_per_item': lookahead_attempts, 'physical_lookahead': [],
        'lookahead_contract': 'Privileged training teacher searches independent copied physics states. Only a fully passing action is replayed through motor commands in the continuously evolving main scene. Failed searches remain; this is not learned online recovery.'}
    observer = None
    try:
        for name in ('simulation_lab/table_teacher.py', 'simulation_lab/table_observation.py',
                     'simulation_lab/random_dinner.py', 'simulation_lab/autonomy.py',
                     'simulation_lab/dinner_autonomy.py', 'scripts/develop_whole_table.py'):
            put(folder/'source'/name, (ROOT/name).read_bytes())
        xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml); data, recipe = draw(model, seed)
        put(folder/'reset-recipe.json', recipe)
        report['generated'] = recipe['generated']
        if not recipe['generated']:return
        for _ in range(600):mujoco.mj_step(model, data)
        initial_geometry = assess(model, data); put(folder/'initial-geometry.json', initial_geometry)
        report['valid_initial_geometry'] = initial_geometry['valid']
        arrays(folder/'initial-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
        if not initial_geometry['valid']:return
        observer = TableObserver(model); observe(observer, data, folder/'initial-observation')
        targets = np.asarray(HOME*2); remaining = list(OBJECTS)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        for action_index in range(maximum_actions):
            selected = None
            options = candidates(model, data, layout, remaining)
            if only_item:options = [(only_item, destination(layout, only_item), 'isolated_diagnostic')]
            for attempt_index, (name, target, role) in enumerate(options):
                excluded = []
                for grasp_attempt in range(lookahead_attempts):
                    initial = np.empty(mujoco.mj_stateSize(model, state_spec))
                    mujoco.mj_getState(model, data, initial, state_spec)
                    trial = mujoco.MjData(model); mujoco.mj_setState(model, trial, initial, state_spec)
                    mujoco.mj_forward(model, trial)
                    trial_targets = targets.copy()
                    task = TableTeacher(model, trial, layout)
                    task.start(object_id=name, target=target, excluded_grasps=excluded)
                    task.update(trial_targets)
                    if not task.active:
                        failure = {'action_index': action_index, 'attempt_index': attempt_index,
                            'grasp_attempt': grasp_attempt, 'item': name, 'role': role,
                            'message': task.message, 'search': task.search_log}
                        report['planning_failures'].append(failure)
                        put(folder/f'planning-{action_index:02d}-{attempt_index:03d}-{grasp_attempt:02d}.json', failure)
                        print({k: v for k, v in failure.items() if k != 'search'}, flush=True)
                        break
                    attempt_path = folder/f'lookahead-{action_index:02d}-{attempt_index:03d}-{grasp_attempt:02d}-{name}'
                    result = execute(model, trial, layout, task, trial_targets, attempt_path, observer)
                    result.update(role=role, path=attempt_path.relative_to(folder).as_posix(),
                                  candidate=task.chosen.name, action_index=action_index)
                    report['physical_lookahead'].append(result)
                    if result['demonstration_eligible']:
                        selected = result, attempt_path, trial_targets; break
                    excluded.append((task.side, task.chosen.name))
                if selected is not None:break
            if selected is None:
                report['stop_reason'] = 'No current candidate action solved; remaining items are unsolved, not classified impossible.'
                break
            result, attempt_path, trial_targets = selected
            values = np.load(attempt_path/'states.npz')
            # Main scene continuity: no resetting or assigning qpos/qvel here.
            assert np.array_equal(data.qpos, values['qpos'][0])
            assert np.array_equal(data.qvel, values['qvel'][0])
            for i, ctrl in enumerate(values['ctrl']):
                data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
                assert np.array_equal(data.qpos, values['qpos'][i+1]), 'Main-scene position replay diverged.'
                assert np.array_equal(data.qvel, values['qvel'][i+1]), 'Main-scene velocity replay diverged.'
            result['continuous_main_scene_replay_exact'] = True
            targets[:] = trial_targets
            report['actions'].append(result)
            put(folder/f'accepted-action-{action_index:02d}.json', result)
            if result['role'] == 'final_setting':remaining.remove(result['item'])
            if only_item:break
            if not remaining:break
        report['remaining'] = remaining
        report['final_geometry'] = assess(model, data)
        final_checks = {}
        for name in OBJECTS:
            row = report['final_geometry']['objects'][name]
            target = destination(layout, name)
            xy_error = float(np.linalg.norm(data.body(name).xpos[:2]-target[:2]))
            r = data.body(name).xmat.reshape(3, 3); yaw_error = 0.
            if name in ('fork', 'spoon'):
                angle = math.atan2(r[1, 0], r[0, 0])-(-math.pi/2 if name == 'spoon' else 0.)
                yaw_error = abs(math.atan2(math.sin(angle), math.cos(angle)))
            final_checks[name] = {'passed': bool(row['valid'] and xy_error < .008 and r[2, 2] > .98 and yaw_error < .175),
                                  'xy_error_mm': xy_error*1000, 'yaw_error_rad': yaw_error}
        report['final_task_checks'] = final_checks
        report['whole_table_complete'] = not remaining and only_item is None and all(r['passed'] for r in final_checks.values())
        observe(observer, data, folder/'final-observation')
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if observer is not None:observer.close()
        report['wall_s'] = time.perf_counter()-started
        put(folder/'report.json', report)
        print({k: v for k, v in report.items() if k not in ('actions', 'physical_lookahead', 'planning_failures', 'final_geometry')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026114001)
    parser.add_argument('--maximum-actions', type=int, default=14)
    parser.add_argument('--only-item', choices=tuple(OBJECTS))
    parser.add_argument('--lookahead-attempts', type=int, default=6)
    args = parser.parse_args(); run(args.output, args.seed, args.maximum_actions, args.only_item, args.lookahead_attempts)
