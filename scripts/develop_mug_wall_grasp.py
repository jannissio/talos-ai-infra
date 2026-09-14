"""Bounded legacy-wall contact reuse on preserved actual V24/V19 mug states."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import prepare, freeze, put, arrays, load_module
from simulation_lab.storage import require_space


def setup(out, previous=None):
    teacher, whole = freeze(out, previous)
    for name in ('simulation_lab/mug_wall_grasp_candidates.py', 'scripts/develop_mug_wall_grasp.py'):
        source = previous / 'source' / name if previous is not None else ROOT / name
        put(out / 'source' / name, source.read_bytes())
    module = load_module('_frozen_mug_wall_candidates', out / 'source/simulation_lab/mug_wall_grasp_candidates.py')
    return teacher, whole, module


def contexts():
    return [('V24_post_accepted00_03_sideways', ROOT / '.run/whole-table-development-v24-seed4004', 2026114004, 4),
            ('V19_post_accepted00_upright', ROOT / '.run/whole-table-development-v19-seed4002', 2026114002, 1)]


def prefix_metadata(source, count):
    rows = []
    for index in range(count):
        accepted = json.loads((source / f'accepted-action-{index:02d}.json').read_text())
        assert accepted['status'] == 'succeeded' and accepted['continuous_main_scene_replay_exact']
        path = source / accepted['path']
        rows.append(dict(path=path.relative_to(ROOT).as_posix(), item=accepted['item'],
                         states_sha256=hashlib.sha256((path / 'states.npz').read_bytes()).hexdigest()))
    return rows


def geometry(args):
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole, module = setup(out)
        from simulation_lab.scene import build_scene, HOME
        camera_path = ROOT / '.run/raw-rgbd-mug-full-pose-v1/estimate-before-truth.json'
        camera = json.loads(camera_path.read_text())
        camera_pose = camera['estimate']['body_origin_m'] + camera['estimate']['body_quaternion_wxyz']
        put(out / 'camera-estimate-before-truth.json', camera_path.read_bytes())
        rows = []
        for context, source, seed, count in contexts():
            prefix = prefix_metadata(source, count)
            xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
            folder = out / context
            put(folder / 'scene.xml', xml.encode())
            put(folder / 'layout.json', layout)
            put(folder / 'prefix.json', prefix)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            with np.load(ROOT / prefix[-1]['path'] / 'states.npz') as archive:
                data.qpos[:] = archive['qpos'][-1]
                data.qvel[:] = archive['qvel'][-1]
                data.ctrl[:] = archive['ctrl'][-1]
            mujoco.mj_forward(model, data)
            arrays(folder / 'actual-source-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
            targets = np.asarray(HOME * 2)
            modes = [('privileged_teacher_pose', None)]
            if context.startswith('V24'):
                modes.append(('camera_derived_grasp_seed_actual_geometry_checks', camera_pose))
            row = dict(context=context, seed=seed, prefix=prefix, actual_mug_pose=data.joint('mug_free').qpos.tolist(), checks=[])
            for mode, pose in modes:
                teacher.grasp_candidates = lambda d, n, pose=pose: module.mug_wall_grasp_candidates(d, n, teacher.GraspCandidate, pose=pose)
                original = data.qpos.copy()
                task = teacher.TableTeacher(model, data, layout)
                task.start(object_id='mug', target=teacher.destination(layout, 'mug'))
                task.update(targets.copy())
                assert np.array_equal(original, data.qpos)
                check = dict(mode=mode, planned=task.active, message=task.message, search=task.search_log,
                             target=teacher.destination(layout, 'mug').tolist(), physics_steps=0)
                if task.active:
                    check.update(arm=task.side, candidate=task.chosen.name, roll=task.chosen_roll,
                                 grasp=task.grasp.tolist(), local_tool_point=task.chosen.local_tool_point.tolist())
                    proof = dict(context=context, seed=seed, prefix=prefix, check=check,
                                 source_geometry_folder=folder.relative_to(out).as_posix())
                    put(out / f'proof-{context}-{mode}.json', proof)
                row['checks'].append(check)
                put(folder / f'{mode}.json', check)
                print(dict(context=context, mode=mode, planned=task.active,
                           message=task.message, candidate=check.get('candidate')), flush=True)
            rows.append(row)
            if row['checks'][0]['planned']:
                break
        put(out / 'report.json', dict(scope=__doc__, new_physical_attempts=0, camera_only_control=False,
                                      comparison='Camera estimate changes grasp points/orientation only; real source and carry geometry remain privileged checks. The V24 estimate is not applied to the unrelated V19 mug pose.', rows=rows))


def physical(args):
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        proof = json.loads(args.proof.read_text())
        teacher, whole, module = setup(out, args.proof.parent)
        source_geometry = args.proof.parent / proof['source_geometry_folder']
        xml = (source_geometry / 'scene.xml').read_bytes()
        layout = json.loads((source_geometry / 'layout.json').read_text())
        put(out / 'scene.xml', xml)
        put(out / 'layout.json', layout)
        put(out / 'proof.json', proof)
        model = mujoco.MjModel.from_xml_string(xml.decode())
        data = mujoco.MjData(model)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        controls, qpos, qvel, stages = [], [], [], []
        initial = None
        for number, row in enumerate(proof['prefix']):
            source = ROOT / row['path'] / 'states.npz'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['states_sha256']
            with np.load(source) as archive:
                values = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
            if number == 0:
                initial = values['initial_integration']
                mujoco.mj_setState(model, data, initial, spec)
                mujoco.mj_forward(model, data)
                qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy()); stages.append('initial')
            assert np.array_equal(data.qpos, values['qpos'][0]) and np.array_equal(data.qvel, values['qvel'][0])
            for index, ctrl in enumerate(values['ctrl']):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                assert np.array_equal(data.qpos, values['qpos'][index + 1]) and np.array_equal(data.qvel, values['qvel'][index + 1])
                controls.append(ctrl.copy()); qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy())
                stages.append(f'prefix{number}_' + str(values['stage'][index + 1]))
            arrays(out / f'prefix-{number:02d}.npz', **values)
        from simulation_lab.scene import HOME
        from simulation_lab.table_observation import TableObserver
        teacher.grasp_candidates = lambda d, n: module.mug_wall_grasp_candidates(d, n, teacher.GraspCandidate)
        targets = np.asarray(HOME * 2)
        task = teacher.TableTeacher(model, data, layout)
        task.start(object_id='mug', target=teacher.destination(layout, 'mug'))
        task.update(targets)
        if not task.active:
            result = dict(status=task.status, message=task.message, search=task.search_log, new_physics_steps=0)
        else:
            expected = proof['check']
            assert (task.side, task.chosen.name, task.chosen_roll) == (expected['arm'], expected['candidate'], expected['roll'])
            observer = TableObserver(model)
            try:
                require_space(out / 'mug-final', 384 * 1024**2)
                result = whole.execute(model, data, layout, task, targets, out / 'mug-final', observer)
            finally:
                observer.close()
            with np.load(out / 'mug-final/states.npz') as archive:
                controls.extend(archive['ctrl'].copy()); qpos.extend(archive['qpos'][1:].copy()); qvel.extend(archive['qvel'][1:].copy())
                stages.extend(['mug_' + str(s) for s in archive['stage'][1:]])
        arrays(out / 'continuous-states.npz', initial_integration=initial, ctrl=np.asarray(controls),
               qpos=np.asarray(qpos), qvel=np.asarray(qvel), stage=np.asarray(stages))
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, initial, spec)
        mujoco.mj_forward(model, replay)
        max_error = 0.
        for index, ctrl in enumerate(controls):
            replay.ctrl[:] = ctrl
            mujoco.mj_step(model, replay)
            max_error = max(max_error, float(np.max(np.abs(replay.qpos - qpos[index + 1]))), float(np.max(np.abs(replay.qvel - qvel[index + 1]))))
        put(out / 'report.json', dict(context=proof['context'], result=result, prefix_motor_replay_exact=True,
                                     independent_continuous_replay_exact=max_error == 0., replay_max_error=max_error,
                                     frames=len(qpos), camera_only_control=False))
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    physical(args) if args.proof else geometry(args)
