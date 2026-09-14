"""Bounded V26 sideways-glass -> upright buffer -> final dependency.

The geometry stage uses reset-only recorded poses. Physical continuations replay
the accepted bottle and one passing sideways-glass action, with no reset between
legs. All common teacher gates remain unchanged; no learned-control claim.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space


def put(path, value):
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + '\n').encode()
    require_space(path, len(payload) + 1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def arrays(path, **values):
    require_space(path, sum(np.asarray(v).nbytes for v in values.values()) + 1024**2)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(out):
    log = out.with_suffix('.log')
    if out.exists() or log.exists():
        raise FileExistsError('Preserve both output and log')
    require_space(out, 2 * 1024**3)
    require_space(log, 32 * 1024**2)
    out.mkdir(parents=True)
    return log


def freeze(out, previous=None):
    names = ('table_teacher', 'table_buffers', 'autonomy', 'dinner_autonomy', 'dinner', 'scene',
             'random_dinner', 'side_plate_grasp_candidates', 'glass_grasp_candidates',
             'horizontal_glass_grasp_candidates', 'table_observation', 'direct_contact_dinner', 'contact_reach')
    manifest = {}
    paths = [f'simulation_lab/{name}.py' for name in names] + ['scripts/develop_whole_table.py',
                                                           'scripts/develop_glass_depth_two.py']
    for name in paths:
        source = previous / 'source' / name if previous is not None and name != 'scripts/develop_glass_depth_two.py' else ROOT / name
        payload = source.read_bytes()
        put(out / 'source' / name, payload)
        manifest[name] = hashlib.sha256(payload).hexdigest()
    # Load shared dependencies once and bind the frozen buffer/teacher modules.
    # All relative dependencies must still match their just-saved snapshots.
    for name in names:
        if name not in ('table_teacher', 'table_buffers'):
            __import__('simulation_lab.' + name)
            if hashlib.sha256((ROOT / f'simulation_lab/{name}.py').read_bytes()).hexdigest() != manifest[f'simulation_lab/{name}.py']:
                raise RuntimeError('Runtime dependency changed; preserve this attempt and resnapshot')
    load_module('simulation_lab.table_buffers', out / 'source/simulation_lab/table_buffers.py')
    teacher = load_module('simulation_lab._glass_depth_two_teacher', out / 'source/simulation_lab/table_teacher.py')
    whole = load_module('_glass_depth_two_whole', out / 'source/scripts/develop_whole_table.py')
    put(out / 'source-manifest.json', manifest)
    return teacher, whole


def geometry(args):
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out)
        from simulation_lab.scene import build_scene, HOME
        from simulation_lab.table_buffers import _horizontal_continuation
        source = ROOT / '.run/whole-table-development-v26-direct-contact'
        report = json.loads((source / 'report.json').read_text())
        xml, layout = build_scene(seed=report['seed'], scenario='dinner', dinner_preset='task')
        put(out / 'scene.xml', xml.encode())
        put(out / 'layout.json', layout)
        model = mujoco.MjModel.from_xml_string(xml)
        buffers = [r for r in report['physical_lookahead'] if r['item'] == 'glass' and r['status'] == 'succeeded']
        protocol = dict(scope=__doc__, source=source.relative_to(ROOT).as_posix(), seed=report['seed'],
                        passing_first_buffers=[r['path'] for r in buffers],
                        search='Existing common upright buffer candidates from each actual post-release state; require hypothetical horizontal final-regrasp geometry before source planning.',
                        maximum_new_physical_alternatives=2, thresholds_unchanged=True,
                        new_physical_steps=0, final_setting=[.075, .10, .76])
        put(out / 'protocol.json', protocol)
        rows, proofs = [], []
        for buffer_index, first in enumerate(buffers):
            folder = source / first['path']
            with np.load(folder / 'states.npz') as archive:
                saved = {k: archive[k].copy() for k in ('qpos', 'qvel', 'ctrl')}
            data = mujoco.MjData(model)
            data.qpos[:] = saved['qpos'][-1]
            data.qvel[:] = saved['qvel'][-1]
            data.ctrl[:] = saved['ctrl'][-1]
            mujoco.mj_forward(model, data)
            state_hash = hashlib.sha256((folder / 'states.npz').read_bytes()).hexdigest()
            arrays(out / f'buffer-{buffer_index:02d}-actual-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
            options = [r for r in whole.candidates(model, data, layout, ['glass'])
                       if r[2] == 'relay_or_clearance' and r[3] is not None and abs(r[3][0]) > .99999]
            print({'buffer': first['path'], 'pose': data.joint('glass_free').qpos.tolist(),
                   'upright_targets': [r[1].tolist() for r in options]}, flush=True)
            cache = {}
            for option_index, (_, target, role, quaternion) in enumerate(options):
                row = dict(buffer=first['path'], buffer_index=buffer_index, source_states_sha256=state_hash,
                           option_index=option_index, target=target.tolist(), quaternion=quaternion.tolist())
                rows.append(row)
                task = teacher.TableTeacher(model, data, layout)
                task.start(object_id='glass', target=target, target_quaternion=quaternion, source_pose_cache=cache)
                task._select_item('right', next(o for o in layout['objects'] if o['id'] == 'glass'))
                try:
                    row['final_regrasp_geometry'] = _horizontal_continuation(task, target, np.array([.075, .10, .76]), teacher.GraspCandidate)
                except teacher.PlanningError as exc:
                    row['error'] = str(exc)
                    put(out / f'check-{buffer_index:02d}-{option_index:02d}.json', row)
                    print({'buffer': buffer_index, 'option': option_index, 'future_regrasp': False}, flush=True)
                    continue
                targets = np.asarray(HOME * 2)
                task.update(targets)
                row.update(source_planned=task.active, source_search=task.search_log, message=task.message)
                if task.active:
                    row.update(arm=task.side, grasp=task.chosen.name, roll=task.chosen_roll)
                    proofs.append(row)
                    put(out / f'proof-{len(proofs):03d}.json', row)
                put(out / f'check-{buffer_index:02d}-{option_index:02d}.json', row)
                print({k: v for k, v in row.items() if k not in ('source_search', 'final_regrasp_geometry')}, flush=True)
                if len(proofs) >= 2:
                    break
            if len(proofs) >= 2:
                break
        put(out / 'report.json', dict(protocol=protocol, source_report_sha256=hashlib.sha256((source / 'report.json').read_bytes()).hexdigest(),
                                      rows=rows, proofs=proofs, physics_steps=0,
                                      conclusion='Checked geometric continuation exists; physical execution remains required.' if proofs else 'No checked continuation found in the declared bounded common candidate search; not proof of impossibility.'))


def physical(args):
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        proof = json.loads(args.proof.read_text())
        teacher, whole = freeze(out, args.proof.parent)
        xml = (args.proof.parent / 'scene.xml').read_bytes()
        layout = json.loads((args.proof.parent / 'layout.json').read_text())
        put(out / 'scene.xml', xml)
        put(out / 'layout.json', layout)
        put(out / 'proof.json', proof)
        model = mujoco.MjModel.from_xml_string(xml.decode())
        data = mujoco.MjData(model)
        source = ROOT / '.run/whole-table-development-v26-direct-contact'
        accepted = json.loads((source / 'accepted-action-00.json').read_text())
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        controls, positions, velocities, stages = [], [], [], []
        prefix = []
        initial = None
        for index, folder in enumerate((source / accepted['path'], source / proof['buffer'])):
            with np.load(folder / 'states.npz') as archive:
                values = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
            if index == 0:
                initial = values['initial_integration']
                mujoco.mj_setState(model, data, initial, state_spec)
                mujoco.mj_forward(model, data)
                positions.append(data.qpos.copy()); velocities.append(data.qvel.copy()); stages.append('prefix_bottle')
            assert np.array_equal(data.qpos, values['qpos'][0]) and np.array_equal(data.qvel, values['qvel'][0])
            for i, ctrl in enumerate(values['ctrl']):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                assert np.array_equal(data.qpos, values['qpos'][i + 1]) and np.array_equal(data.qvel, values['qvel'][i + 1])
                controls.append(ctrl.copy()); positions.append(data.qpos.copy()); velocities.append(data.qvel.copy())
                stages.append(('prefix_bottle_' if index == 0 else 'prefix_glass_') + str(values['stage'][i + 1]))
            arrays(out / f'prefix-{index:02d}-states.npz', **values)
            prefix.append(dict(path=folder.relative_to(ROOT).as_posix(), states_sha256=hashlib.sha256((folder / 'states.npz').read_bytes()).hexdigest(), motor_frames=len(values['ctrl']), exact=True))
        put(out / 'prefix.json', prefix)
        from simulation_lab.scene import HOME
        from simulation_lab.table_observation import TableObserver
        targets = np.asarray(HOME * 2)
        observer = TableObserver(model)
        results = []
        try:
            for leg, target, quaternion in ((2, proof['target'], proof['quaternion']), (3, [.075, .10, .76], [1., 0., 0., 0.])):
                task = teacher.TableTeacher(model, data, layout)
                task.start(object_id='glass', target=target, target_quaternion=quaternion)
                task.update(targets)
                if not task.active:
                    result = dict(leg=leg, status=task.status, message=task.message, search=task.search_log, physics_steps=0)
                    put(out / f'leg{leg}-planning-rejection.json', result)
                    results.append(result)
                    break
                if leg == 2:
                    assert (task.side, task.chosen.name, task.chosen_roll) == (proof['arm'], proof['grasp'], proof['roll'])
                leg_path = out / f'leg{leg}'
                require_space(leg_path, 384 * 1024**2)
                result = whole.execute(model, data, layout, task, targets, leg_path, observer)
                result['leg'] = leg
                results.append(result)
                with np.load(leg_path / 'states.npz') as archive:
                    controls.extend(archive['ctrl'].copy())
                    positions.extend(archive['qpos'][1:].copy())
                    velocities.extend(archive['qvel'][1:].copy())
                    stages.extend([f'leg{leg}_' + str(s) for s in archive['stage'][1:]])
                if result['status'] != 'succeeded' or not result['replay_exact']:
                    break
        finally:
            observer.close()
        arrays(out / 'continuous-states.npz', initial_integration=initial, ctrl=np.asarray(controls),
               qpos=np.asarray(positions), qvel=np.asarray(velocities), stage=np.asarray(stages))
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, initial, state_spec)
        mujoco.mj_forward(model, replay)
        max_error = 0.
        for i, ctrl in enumerate(controls):
            replay.ctrl[:] = ctrl
            mujoco.mj_step(model, replay)
            max_error = max(max_error, float(np.max(np.abs(replay.qpos - positions[i + 1]))),
                            float(np.max(np.abs(replay.qvel - velocities[i + 1]))))
        result = dict(scope='Privileged exposed three-leg glass dependency after accepted bottle; not whole-table or learned success.',
                      prefix=prefix, new_legs=results, continuous_frames=len(positions),
                      independent_continuous_replay_exact=max_error == 0., maximum_replay_error=max_error,
                      completed_three_leg_glass=len(results) == 2 and all(r['status'] == 'succeeded' for r in results) and max_error == 0.)
        put(out / 'report.json', result)
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    physical(args) if args.proof else geometry(args)
