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
from simulation_lab.direct_contact_dinner import draw_candidate
from simulation_lab.random_dinner import assess, body_bounds, draw, orientation
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


def arrangement_key(data):
    """Coarse physical-state memory for avoiding repeated temporary placements."""
    values = []
    for name in OBJECTS:
        body = data.body(name); rotation = body.xmat.reshape(3, 3)
        values.extend(np.rint(body.xpos/.01).astype(int).tolist())
        values.extend(np.rint(rotation[:, (0, 2)].ravel()/.1).astype(int).tolist())
    return tuple(values)


def blockers(model, data, name, target, quaternion=None):
    quat = np.array([np.sqrt(.5), 0., 0., -np.sqrt(.5)]) if name == 'spoon' else np.array([1., 0., 0., 0.])
    low, high = body_bounds(model, name, quat if quaternion is None else quaternion)
    low += target; high += target
    rows = []
    for other in OBJECTS:
        if other == name:continue
        olow, ohigh = world_bounds(model, data, other)
        if np.all(high[:2]+.004 > olow[:2]) and np.all(ohigh[:2]+.004 > low[:2]):rows.append(other)
    return rows


def blocked_relations(model, data, layout, remaining):
    return {(name, other) for name in remaining
            for other in blockers(model, data, name, destination(layout, name))}


def orientation_progress(before, after, name, accepted_relays, introduced):
    """Bounded task progress from a physically verified temporary placement.

    Every final setting needs body Z up; cutlery yaw is separately enforced at
    final placement. A moved item may temporarily block a goal, but a relay may
    not introduce an unrelated blocker. This is not a completion certificate.
    """
    angle = lambda data: math.degrees(math.acos(np.clip(data.body(name).xmat[8], -1., 1.)))
    old, new = angle(before), angle(after)
    improved = (old-new > 30. and accepted_relays < 2
                and all(blocker == name for _, blocker in introduced))
    return {'before_error_deg': old, 'after_error_deg': new, 'improvement_deg': old-new,
            'minimum_improvement_deg': 30., 'previous_orientation_relays': accepted_relays,
            'maximum_orientation_relays_per_item': 2, 'eligible': improved}


def candidates(model, data, layout, remaining):
    direct = [(name, destination(layout, name), 'final_setting', None) for name in remaining
              if not blockers(model, data, name, destination(layout, name))]
    direct.sort(key=lambda row: min(np.linalg.norm(data.body(row[0]).xpos[:2]-[x, -.235]) for x in (-.25, .25)))
    # A cyclic obstruction or disjoint arm workspaces can require a temporary
    # placement. Search the actual table; no per-object success rectangle.
    table = model.geom('table').id; half = model.geom_size[table, :2]; center = data.geom_xpos[table, :2]
    options = []
    for name in remaining:
        nominal = np.array([np.sqrt(.5), 0., 0., -np.sqrt(.5)]) if name == 'spoon' else np.array([1., 0., 0., 0.])
        orientations = [('final_orientation', nominal)]
        current = data.joint(name+'_free').qpos[3:].copy()
        normal = data.body(name).xmat.reshape(3, 3)[:, 2]
        if normal[2] < .999 or name in ('fork', 'spoon'):
            orientations.append(('preserved_orientation', current))
        if OBJECTS[name]['kind'] in ('bottle', 'mug', 'glass'):
            for angle in (-math.pi, -math.pi/2, 0., math.pi/2):
                orientations.append((f'sideways_{angle}', orientation('sideways', angle, math.pi)))
        for orientation_role, quaternion in orientations:
            low, high = body_bounds(model, name, quaternion)
            xs = np.arange(center[0]-half[0]-low[0]+.015, center[0]+half[0]-high[0]-.015, .065)
            ys = np.arange(center[1]-half[1]-low[1]+.015, center[1]+half[1]-high[1]-.015, .065)
            for x in xs:
                for y in ys:
                    target = np.array([x, y, layout['table_z']-low[2]])
                    if np.linalg.norm(target[:2]-data.body(name).xpos[:2]) < .07:continue
                    if blockers(model, data, name, target, quaternion):continue
                    source = data.body(name).xpos[:2]; goal = destination(layout, name)[:2]
                    if np.linalg.norm(target[:2]-goal) < .02:continue
                    cost = np.linalg.norm(target[:2]-source)+np.linalg.norm(target[:2]-goal)
                    cost += .5*abs(target[0])  # Prefer a potential two-arm regrasp area.
                    options.append((cost, name, target, 'relay_or_clearance', quaternion, orientation_role))
    selected = []
    for name in remaining:
        for family in dict.fromkeys(r[5] for r in options if r[1] == name):
            count = 2 if family.startswith('sideways_') else 4
            selected.extend(sorted((r for r in options if r[1] == name and r[5] == family), key=lambda r: r[0])[:count])
    return direct+[(name, target, role, quaternion) for _, name, target, role, quaternion, _ in sorted(selected, key=lambda r: r[0])]


def execute(model, data, layout, task, targets, folder, observer):
    require_space(folder, 384*1024**2); folder.mkdir(parents=True)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(model, state_spec)); mujoco.mj_getState(model, data, initial, state_spec)
    controls, qpos, qvel, stages = [], [data.qpos.copy()], [data.qvel.copy()], [task.stage]
    # The selected path was planned with real geometry; no action is applied yet.
    put(folder/'search.json', task.search_log)
    def action_metadata():
        return {'item': task.tube['id'], 'arm': task.side,
            'grasp_candidate': task.chosen.name, 'pick_point_m': task.grasp.tolist(),
            'initial_wrist_roll': task.chosen_roll,
            'gripper_point_local_m': task.chosen.local_tool_point.tolist(),
            'gripper_constraint_axis_local': None if task.chosen.local_axis is None else task.chosen.local_axis.tolist(),
            'place_body_origin_m': task.destination_position.tolist(),
            'place_body_quaternion_wxyz': task.destination_quaternion.tolist(),
            'orientation_constraint': task.orientation_constraint,
            'teacher_observation': 'privileged_state', 'policy_observation': 'calibrated_rgbd_only'}
    put(folder/'action-plan.json', action_metadata())
    stage_before = None; observations = 0; tick = 0
    while task.active and tick < 24000:
        phase = (task.stage, float(task.stage_started))
        if phase != stage_before:
            observe(observer, data, folder/f'observation-{observations:03d}-{task.stage}')
            stage_before = phase; observations += 1
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
    # A privileged buffer search may choose a new supported target from the
    # measured held pose. Preserve both the initial plan and the executed label.
    put(folder/'action.json', action_metadata())
    if getattr(task, 'buffer_search', None) is not None:put(folder/'buffer-search.json', task.buffer_search)
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


def probe_final_transfer(model, data, layout, targets, item, buffer_path,
                         folder, observer, report, action_index, maximum_attempts):
    """Verify a buffer's proposed final leg in independent physical copies.

    A geometric endpoint is insufficient evidence for accepting a buffer whose
    only purpose is regrasp. Preserve every failed continuation and return the
    complete passing motor trace for exact main-scene replay, if one exists.
    The supplied post-buffer state and motor targets are never changed here.
    """
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, initial, state_spec)
    excluded, source_pose_cache = [], {}
    for attempt in range(maximum_attempts):
        trial = mujoco.MjData(model)
        mujoco.mj_setState(model, trial, initial, state_spec)
        mujoco.mj_forward(model, trial)
        trial_targets = targets.copy()
        task = TableTeacher(model, trial, layout)
        task.start(object_id=item, target=destination(layout, item),
                   excluded_grasps=excluded, source_pose_cache=source_pose_cache)
        task.update(trial_targets)
        path = folder/f'continuation-{buffer_path.name}-{attempt:02d}'
        if not task.active:
            failure = {'action_index': action_index, 'item': item,
                       'role': 'conditional_final_setting', 'grasp_attempt': attempt,
                       'conditional_on_buffer_path': buffer_path.relative_to(folder).as_posix(),
                       'message': task.message, 'search': task.search_log}
            report['planning_failures'].append(failure)
            put(path.with_suffix('.json'), failure)
            break
        result = execute(model, trial, layout, task, trial_targets, path, observer)
        result.update(role='final_setting', path=path.relative_to(folder).as_posix(),
                      candidate=task.chosen.name, initial_wrist_roll=task.chosen_roll,
                      action_index=action_index,
                      conditional_on_buffer_path=buffer_path.relative_to(folder).as_posix())
        report['physical_lookahead'].append(result)
        if result['demonstration_eligible']:
            return result, path, trial_targets
        excluded.append((task.side, task.chosen.name, task.chosen_roll))
    return None


def run(folder, seed, maximum_actions, only_item=None, lookahead_attempts=6, reset='joint', include_item_buffers=False,
        verify_relay_continuation=False):
    protocol_bytes = None
    continuation_protocol_bytes = None
    if verify_relay_continuation:
        continuation_protocol_bytes = (ROOT/'docs/robotics/experiments/verified-relay-continuation-development-v1.json').read_bytes()
        declaration = json.loads(continuation_protocol_bytes)
        if (seed != declaration['seed'] or reset != declaration['reset_distribution']
                or maximum_actions != declaration['maximum_accepted_actions']
                or lookahead_attempts != declaration['physical_lookahead_attempts_per_candidate']
                or only_item is not None or include_item_buffers):
            raise ValueError('Arguments differ from the fixed physical relay continuation declaration.')
    if reset == 'joint-direct-contact':
        protocol_bytes = (ROOT/'docs/robotics/experiments/joint-direct-contact-development-v1.json').read_bytes()
        protocol = json.loads(protocol_bytes)
        if (seed != protocol['protocol_seed']
                or maximum_actions != protocol['execution']['maximum_accepted_actions']
                or lookahead_attempts != protocol['execution']['physical_lookahead_attempts_per_candidate']
                or only_item is not None or include_item_buffers or verify_relay_continuation):
            raise ValueError('Arguments differ from the fixed joint-direct-contact V1 declaration. Declare a separate experiment instead.')
    if folder.exists():raise FileExistsError('Preserve every development attempt.')
    preflight = require_space(folder, 8*1024**3); folder.mkdir(parents=True)
    started = time.perf_counter(); report = {'seed': seed, 'preflight': preflight,
        'scope': 'Exposed development of a shared exact-state physical teacher; not learned evaluation.',
        'whole_table_complete': False, 'actions': [], 'planning_failures': [], 'remaining': list(OBJECTS),
        'only_item_diagnostic': only_item, 'maximum_actions': maximum_actions,
        'include_item_buffers': include_item_buffers,
        'verify_relay_continuation': verify_relay_continuation,
        'reset_distribution': reset, 'coverage_evaluation': False,
        'lookahead_attempts_per_item': lookahead_attempts, 'physical_lookahead': [],
        'lookahead_contract': 'Privileged training teacher searches independent copied physics states. Only a fully passing action is replayed through motor commands in the continuously evolving main scene. Failed searches remain; this is not learned online recovery.'}
    observer = None
    try:
        if continuation_protocol_bytes is not None:
            put(folder/'execution-protocol.json', continuation_protocol_bytes)
        for name in ('simulation_lab/table_teacher.py', 'simulation_lab/table_observation.py',
                     'simulation_lab/side_plate_grasp_candidates.py',
                     'simulation_lab/glass_grasp_candidates.py',
                     'simulation_lab/horizontal_glass_grasp_candidates.py',
                     'simulation_lab/table_buffers.py',
                     'simulation_lab/contact_reach.py', 'simulation_lab/direct_contact_dinner.py',
                     'simulation_lab/random_dinner.py', 'simulation_lab/autonomy.py',
                     'simulation_lab/dinner_autonomy.py', 'scripts/develop_whole_table.py'):
            put(folder/'source'/name, (ROOT/name).read_bytes())
        xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml)
        if reset == 'joint-direct-contact':
            put(folder/'protocol.json', protocol_bytes)
            def before_candidate(index):
                require_space(folder/'reset-candidates'/f'{index:03d}', 32*1024**2)
            def record_candidate(index, candidate_recipe, result, trace):
                path = folder/'reset-candidates'/f'{index:03d}'
                put(path/'recipe.json', candidate_recipe); put(path/'result.json', result)
                arrays(path/'states.npz', **trace)
                print({key: result[key] for key in ('candidate_index', 'candidate_seed', 'selected', 'reason', 'settling_replay_exact')}, flush=True)
            data, recipe = draw_candidate(model, seed, record=record_candidate,
                                          before_candidate=before_candidate)
        elif reset == 'joint':data, recipe = draw(model, seed)
        else:
            data = mujoco.MjData(model); data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
            mujoco.mj_forward(model, data)
            recipe = {'seed': seed, 'generated': True, 'reset': 'Existing narrow task preset for positive-control diagnosis only.'}
        put(folder/'reset-recipe.json', recipe)
        report['generated'] = recipe['generated']
        if not recipe['generated']:return
        if reset != 'joint-direct-contact':
            for _ in range(600):mujoco.mj_step(model, data)
        initial_geometry = assess(model, data); put(folder/'initial-geometry.json', initial_geometry)
        report['valid_initial_geometry'] = initial_geometry['valid']
        arrays(folder/'initial-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
        if not initial_geometry['valid'] and reset != 'task':return
        observer = TableObserver(model); observe(observer, data, folder/'initial-observation')
        targets = np.asarray(HOME*2); remaining = list(OBJECTS)
        seen_arrangements = {arrangement_key(data)}
        orientation_relays = {name: 0 for name in OBJECTS}
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        while len(report['actions']) < maximum_actions:
            action_index = len(report['actions'])
            selected = None
            # Source IK is independent of the proposed destination. Reuse only
            # within this unchanged main scene; discard after every real action.
            source_pose_cache = {}
            options = candidates(model, data, layout, [only_item] if only_item else remaining)
            if only_item and not include_item_buffers:
                options = [(only_item, destination(layout, only_item), 'isolated_diagnostic', None)]
            for attempt_index, (name, target, role, quaternion) in enumerate(options):
                excluded = []
                for grasp_attempt in range(lookahead_attempts):
                    initial = np.empty(mujoco.mj_stateSize(model, state_spec))
                    mujoco.mj_getState(model, data, initial, state_spec)
                    trial = mujoco.MjData(model); mujoco.mj_setState(model, trial, initial, state_spec)
                    mujoco.mj_forward(model, trial)
                    trial_targets = targets.copy()
                    task = TableTeacher(model, trial, layout)
                    task.start(object_id=name, target=target, target_quaternion=quaternion,
                               excluded_grasps=excluded, source_pose_cache=source_pose_cache,
                               allow_measured_buffer=role == 'relay_or_clearance')
                    task.update(trial_targets)
                    if not task.active:
                        failure = {'action_index': action_index, 'attempt_index': attempt_index,
                            'grasp_attempt': grasp_attempt, 'item': name, 'role': role,
                            'target_quaternion': None if quaternion is None else quaternion.tolist(),
                            'message': task.message, 'search': task.search_log}
                        report['planning_failures'].append(failure)
                        put(folder/f'planning-{action_index:02d}-{attempt_index:03d}-{grasp_attempt:02d}.json', failure)
                        print({k: v for k, v in failure.items() if k != 'search'}, flush=True)
                        break
                    attempt_path = folder/f'lookahead-{action_index:02d}-{attempt_index:03d}-{grasp_attempt:02d}-{name}'
                    result = execute(model, trial, layout, task, trial_targets, attempt_path, observer)
                    result.update(role=role, path=attempt_path.relative_to(folder).as_posix(),
                                  candidate=task.chosen.name, initial_wrist_roll=task.chosen_roll,
                                  action_index=action_index)
                    report['physical_lookahead'].append(result)
                    if result['demonstration_eligible']:
                        following = None
                        if role == 'relay_or_clearance':
                            if arrangement_key(trial) in seen_arrangements:
                                result['workflow_selection'] = 'Passing demonstration retained; repeated arrangement refused.'
                                break
                            continuation = TableTeacher(model, trial, layout)
                            continuation.start(object_id=name, target=destination(layout, name))
                            continuation.update(trial_targets.copy())
                            result['final_transfer_plan_exists_after_regrasp'] = continuation.active
                            before_blocked = blocked_relations(model, data, layout, remaining)
                            after_blocked = blocked_relations(model, trial, layout, remaining)
                            cleared = sorted(before_blocked-after_blocked)
                            introduced = sorted(after_blocked-before_blocked)
                            clearance_progress = bool(cleared) and not introduced
                            orientation_change = orientation_progress(data, trial, name, orientation_relays[name], introduced)
                            result['orientation_progress'] = orientation_change
                            result['cleared_blocking_relationships'] = cleared
                            result['introduced_blocking_relationships'] = introduced
                            put(attempt_path/'continuation-plan.json', {'planned': continuation.active,
                                'search': continuation.search_log, 'message': continuation.message,
                                'clearance_progress': clearance_progress,
                                'orientation_progress': orientation_change,
                                'cleared_blocking_relationships': cleared,
                                'introduced_blocking_relationships': introduced,
                                'meaning': 'Geometric next-action plan only, not yet physical transfer success.'})
                            if not continuation.active and not clearance_progress and not orientation_change['eligible']:
                                result['workflow_selection'] = 'Valid buffer demonstration retained, but final regrasp/transfer is unsolved.'
                                break
                            if (verify_relay_continuation and continuation.active
                                    and not clearance_progress and not orientation_change['eligible']):
                                if action_index+2 > maximum_actions:
                                    result['workflow_selection'] = 'Passing buffer retained; insufficient accepted-action budget to verify and commit both legs.'
                                    break
                                following = probe_final_transfer(
                                    model, trial, layout, trial_targets, name, attempt_path,
                                    folder, observer, report, action_index+1, lookahead_attempts)
                                result['final_transfer_physically_verified'] = following is not None
                                if following is None:
                                    result['workflow_selection'] = 'Passing buffer retained; geometric final transfer failed the bounded physical continuation search.'
                                    break
                            if clearance_progress:
                                result['workflow_selection'] = 'Physically verified clearance removes a destination obstruction without introducing another.'
                            elif following is not None:
                                result['workflow_selection'] = 'Both buffer and final-transfer motor traces pass; replay both in the continuous main scene.'
                                result['verified_final_transfer_path'] = following[1].relative_to(folder).as_posix()
                            elif verify_relay_continuation and orientation_change['eligible']:
                                # Orientation progress is why physical continuation
                                # could be skipped, even when geometric IK exists.
                                # Charge that reason against its two-relay budget.
                                result['workflow_selection'] = 'Physically verified buffer improves task orientation by over 30 degrees within the two-relay budget.'
                                result['orientation_relay_selected'] = True
                            elif continuation.active:
                                result['workflow_selection'] = 'Physically verified buffer has a geometric final-transfer plan; execution is still required.'
                            else:
                                result['workflow_selection'] = 'Physically verified buffer improves task orientation by over 30 degrees within the two-relay budget.'
                                result['orientation_relay_selected'] = True
                        selected = [(result, attempt_path, trial_targets)]
                        if following is not None:selected.append(following)
                        break
                    excluded.append((task.side, task.chosen.name, task.chosen_roll))
                if selected is not None:break
            if selected is None:
                report['stop_reason'] = 'No current candidate action solved; remaining items are unsolved, not classified impossible.'
                break
            for result, attempt_path, trial_targets in selected:
                with np.load(attempt_path/'states.npz') as archive:
                    values = {key: archive[key] for key in ('qpos', 'qvel', 'ctrl')}
                # Main scene continuity: no resetting or assigning qpos/qvel here.
                assert np.array_equal(data.qpos, values['qpos'][0])
                assert np.array_equal(data.qvel, values['qvel'][0])
                for i, ctrl in enumerate(values['ctrl']):
                    data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
                    assert np.array_equal(data.qpos, values['qpos'][i+1]), 'Main-scene position replay diverged.'
                    assert np.array_equal(data.qvel, values['qvel'][i+1]), 'Main-scene velocity replay diverged.'
                result['continuous_main_scene_replay_exact'] = True
                seen_arrangements.add(arrangement_key(data))
                if result.get('orientation_relay_selected'):orientation_relays[result['item']] += 1
                targets[:] = trial_targets
                report['actions'].append(result)
                put(folder/f"accepted-action-{len(report['actions'])-1:02d}.json", result)
                if result['role'] == 'final_setting':remaining.remove(result['item'])
            if only_item and (not include_item_buffers or result['role'] == 'final_setting'):break
            if not remaining:break
        if remaining and len(report['actions']) >= maximum_actions:
            report['stop_reason'] = 'Accepted-action budget exhausted; remaining items are unsolved, not classified impossible.'
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
        report['whole_table_complete'] = reset != 'task' and not remaining and only_item is None and all(r['passed'] for r in final_checks.values())
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
    parser.add_argument('--include-item-buffers', action='store_true',
                        help='Allow temporary placements in an explicitly single-item joint-scene diagnostic.')
    parser.add_argument('--verify-relay-continuation', action='store_true',
                        help='Physically verify both legs before accepting a buffer justified only by a final-transfer plan.')
    parser.add_argument('--lookahead-attempts', type=int, default=6)
    parser.add_argument('--reset', choices=('joint', 'joint-direct-contact', 'task'), default='joint')
    args = parser.parse_args()
    if args.include_item_buffers and not args.only_item:parser.error('--include-item-buffers requires --only-item.')
    run(args.output, args.seed, args.maximum_actions, args.only_item, args.lookahead_attempts, args.reset, args.include_item_buffers,
        args.verify_relay_continuation)
