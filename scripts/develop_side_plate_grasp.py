"""Immutable, exposed side-plate physical diagnostic; never a coverage claim."""
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
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def arrays(path, **values):
    require_space(path, sum(np.asarray(v).nbytes for v in values.values())+1024**2)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def run(args):
    folder = args.output.resolve()
    log = folder.with_suffix('.log')
    if folder.exists() or log.exists():
        raise FileExistsError('Both output and log must be unused; preserve all attempts.')
    preflight = require_space(folder, 256*1024**2)
    folder.mkdir(parents=True)
    with log.open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        started = time.perf_counter()
        manifest = {}
        for name in ('simulation_lab/table_teacher.py', 'simulation_lab/autonomy.py',
                     'simulation_lab/dinner_autonomy.py', 'simulation_lab/random_dinner.py',
                     'simulation_lab/dinner.py', 'simulation_lab/scene.py',
                     'simulation_lab/side_plate_grasp_candidates.py',
                     'scripts/develop_side_plate_grasp.py'):
            source = Path(args.teacher_source) if name == 'simulation_lab/table_teacher.py' and args.teacher_source else ROOT/name
            payload = source.read_bytes()
            put(folder/'source'/name, payload)
            manifest[name] = hashlib.sha256(payload).hexdigest()
        put(folder/'source-manifest.json', manifest)
        module_name = 'simulation_lab._side_plate_snapshot'
        spec = importlib.util.spec_from_file_location(module_name, folder/'source/simulation_lab/table_teacher.py')
        teacher = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = teacher
        spec.loader.exec_module(teacher)
        family_spec = importlib.util.spec_from_file_location('side_plate_frozen_candidates', folder/'source/simulation_lab/side_plate_grasp_candidates.py')
        family = importlib.util.module_from_spec(family_spec)
        family_spec.loader.exec_module(family)
        from simulation_lab.scene import build_scene, HOME
        from simulation_lab.random_dinner import draw, assess
        xml, layout = build_scene(seed=args.seed, scenario='dinner', dinner_preset='task')
        put(folder/'scene.xml', xml.encode())
        put(folder/'layout.json', layout)
        model = mujoco.MjModel.from_xml_string(xml)
        assert np.max(np.abs(model.actuator_forcerange)) <= 2.94+1e-8
        data, recipe = draw(model, args.seed)
        put(folder/'reset-recipe.json', recipe)
        for _ in range(600):
            mujoco.mj_step(model, data)
        put(folder/'initial-geometry.json', assess(model, data))
        def candidates(data, name):
            if args.family:
                return family.reverse_rim_candidates(data, name, teacher.GraspCandidate, teacher.OPEN)
            body = data.body(name)
            rotation = body.xmat.reshape(3, 3)
            radial = np.array([np.cos(args.angle), np.sin(args.angle), 0.])
            point = body.xpos+rotation@(radial*args.radius+[0., 0., args.height])
            world_radial = rotation@radial
            long_axis = world_radial*np.sin(args.pitch)+np.array([0., 0., np.cos(args.pitch)])
            closing_axis = args.closing_sign*(world_radial*np.cos(args.pitch)-np.array([0., 0., np.sin(args.pitch)]))
            return [teacher.GraspCandidate('side_diagnostic', point,
                np.array([args.tool_x, 0., args.tool_z]), 2, long_axis,
                closing_axis, teacher.OPEN, None, args.torque)]
        teacher.grasp_candidates = candidates
        task = teacher.TableTeacher(model, data, layout)
        targets = np.asarray(HOME*2)
        task.start(object_id='side_plate', side=args.arm, target=args.target)
        task.update(targets)
        put(folder/'parameters.json', {**vars(args), 'output': str(folder), 'preflight': preflight})
        put(folder/'search.json', task.search_log)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, initial, state_spec)
        controls, qpos, qvel, stages, sensors, relative = [], [data.qpos.copy()], [data.qvel.copy()], [task.stage], [], []
        last_stage = None
        while task.active and len(controls) < 24000:
            if task.stage != last_stage:
                print({'stage': task.stage, 'time': float(data.time)}, flush=True)
                last_stage = task.stage
            task.update(targets)
            if task.side:
                tool = data.body(task.side+'_gripper')
                item = data.body('side_plate')
                rotation = tool.xmat.reshape(3, 3)
                relative.append(np.r_[rotation.T@(item.xpos-tool.xpos), (rotation.T@item.xmat.reshape(3, 3)).ravel(), rotation[:, 2]])
            sensors.append([float(data.time), *task.metrics.get('finger_forces_n', [0.,0.]),
                task.metrics.get('external_contact_n', 0.), task.metrics.get('tilt_deg', 0.),
                task.metrics.get('lift_cm', 0.), float(data.qpos[task.offset+5])])
            if not task.active:
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            controls.append(data.ctrl.copy())
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            mujoco.mj_step(model, data)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            stages.append(task.stage)
        if task.active:
            task._finish('failed', 'Episode tick budget exhausted.', True)
        values = dict(initial_integration=initial, ctrl=np.asarray(controls), qpos=np.asarray(qpos),
            qvel=np.asarray(qvel), stage=np.asarray(stages), sensors=np.asarray(sensors), relative=np.asarray(relative))
        arrays(folder/'states.npz', **values)
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, initial, state_spec)
        mujoco.mj_forward(model, replay)
        error = 0.
        for i, ctrl in enumerate(controls):
            replay.ctrl[:] = ctrl
            mujoco.mj_step(model, replay)
            error = max(error, float(np.max(np.abs(replay.qpos-qpos[i+1]))), float(np.max(np.abs(replay.qvel-qvel[i+1]))))
        result = dict(status=task.status, message=task.message, metrics=task.metrics, history=task.history,
            frames=len(qpos), replay_exact=error == 0., replay_max_state_error=error,
            demonstration_eligible=task.status == 'succeeded' and error == 0.,
            coverage_evaluation=False, wall_s=time.perf_counter()-started)
        put(folder/'result.json', result)
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-source')
    parser.add_argument('--seed', type=int, default=2026114001)
    parser.add_argument('--angle', type=float, default=-np.pi)
    parser.add_argument('--radius', type=float, default=.048)
    parser.add_argument('--height', type=float, default=.0105)
    parser.add_argument('--tool-x', type=float, default=-.001)
    parser.add_argument('--tool-z', type=float, default=-.100)
    parser.add_argument('--torque', type=float, default=.8)
    parser.add_argument('--pitch', type=float, default=0.)
    parser.add_argument('--closing-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--target', type=float, nargs=3)
    parser.add_argument('--family', action='store_true')
    parser.add_argument('--arm', choices=('left', 'right', 'auto'), default='right')
    args = parser.parse_args()
    if not 0 < args.torque <= 2.94:
        parser.error('Torque must stay within rated real motors.')
    run(args)
