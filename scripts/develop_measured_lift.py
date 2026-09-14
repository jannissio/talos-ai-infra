"""Diagnose a failed lift after exact motor replay of an exposed saved grasp.

Only the optional continuation advances new physical motion. Geometry checks
use scratch data and never classify an unsolved route as physically impossible.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from scripts.develop_whole_table import arrays, execute, put
from simulation_lab.autonomy import CLOSED, PlanningError
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_observation import TableObserver
from simulation_lab.table_teacher import TableTeacher, grasp_candidates


def run(args):
    folder = args.output
    log = folder.with_suffix('.log')
    if folder.exists() or log.exists():
        raise FileExistsError('Preserve the previous output and console log.')
    preflight = require_space(folder, 512*1024**2)
    folder.mkdir(parents=True)
    with log.open('x', encoding='utf8') as stream, contextlib.redirect_stdout(stream):
        started = time.perf_counter()
        manifest = {}
        for name in ('scripts/develop_measured_lift.py', 'scripts/develop_whole_table.py',
                     'simulation_lab/table_teacher.py', 'simulation_lab/autonomy.py',
                     'simulation_lab/dinner_autonomy.py', 'simulation_lab/table_observation.py',
                     'simulation_lab/scene.py', 'simulation_lab/dinner.py',
                     'simulation_lab/glass_grasp_candidates.py',
                     'simulation_lab/side_plate_grasp_candidates.py'):
            payload = (ROOT/name).read_bytes()
            put(folder/'source'/name, payload)
            manifest[name] = hashlib.sha256(payload).hexdigest()
        put(folder/'source-manifest.json', manifest)
        action = json.loads((args.initial_from/'action.json').read_text())
        source_result = json.loads((args.initial_from/'result.json').read_text())
        if not source_result['replay_exact']:
            raise ValueError('An exact source replay is required.')
        xml, layout = build_scene(seed=args.seed, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        with np.load(args.initial_from/'states.npz') as archive:
            saved = {name: archive[name] for name in ('initial_integration', 'ctrl', 'qpos', 'qvel')}
        mujoco.mj_setState(model, data, saved['initial_integration'], state_spec)
        mujoco.mj_forward(model, data)
        initial_item_origin = data.body(action['item']).xpos.copy()
        candidate = next(c for c in grasp_candidates(data, action['item'])
                         if c.name == action['grasp_candidate'])
        for index, ctrl in enumerate(saved['ctrl']):
            data.ctrl[:] = ctrl
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            mujoco.mj_step(model, data)
            assert np.array_equal(data.qpos, saved['qpos'][index+1]), 'Prefix position replay diverged.'
            assert np.array_equal(data.qvel, saved['qvel'][index+1]), 'Prefix velocity replay diverged.'
        task = TableTeacher(model, data, layout)
        task.start(object_id=action['item'], side=action['arm'],
                   target=action['place_body_origin_m'], target_quaternion=action['place_body_quaternion_wxyz'])
        task._select_item(action['arm'], next(o for o in layout['objects'] if o['id'] == action['item']))
        task._configure(candidate)
        task.origin = initial_item_origin
        task.chosen = candidate
        task.chosen_roll = action['initial_wrist_roll']
        task.grasp = np.asarray(action['pick_point_m'])
        task.attempts = 1
        forces, lift, up, both = task._observe()
        if not both:
            raise ValueError('The exact prefix does not end with two-finger support.')
        original_state = data.qpos.copy()
        start = task.ik.point(data).copy()
        q = data.qpos[task.offset:task.offset+5].copy()
        grip = float(data.qpos[task.offset+5])
        carry = task._carry_reference()
        toward = data.body(task.side+'_shoulder').xpos-start
        toward[2] = 0.
        toward /= np.linalg.norm(toward)
        lateral = np.cross([0., 0., 1.], toward)
        shifts = [np.array([0., 0., z]) for z in (.04, .065, .03)]
        shifts += [toward*d+[0., 0., z] for d, z in ((.025, .04), (.05, .04), (.025, .065))]
        shifts += [toward*.025+lateral*d+[0., 0., .04] for d in (-.025, .025)]
        rows, passing = [], []
        for mode in ('original', 'measured_axis'):
            task._configure(candidate)
            if mode == 'measured_axis':
                rotation = data.body(task.side+'_gripper').xmat.reshape(3, 3)
                local_axis = np.eye(3)[:, task.ik.axis_index] if task.ik.axis_local is None else task.ik.axis_local
                task.ik.axis_target = rotation@local_axis
                task.ik.x_target = None
                task.ik.soft_placement = task.ik.placement_solve = True
            for shift in shifts:
                row = {'mode': mode, 'shift_m': shift.tolist(), 'passed_geometry': False}
                try:
                    points = task._cartesian(start, start+shift, q, grip, check=False)
                    task._check_path(points[:2], grip, True, carry=carry, support=task.base_geom)
                    task._check_path(points[2:], grip, True, carry=carry)
                    row.update(passed_geometry=True, path_samples=len(points), final_q=points[-1].tolist())
                    passing.append((len(rows), mode, points))
                except PlanningError as exc:
                    row['error'] = str(exc)
                rows.append(row)
                print(row, flush=True)
        assert np.array_equal(data.qpos, original_state), 'Geometry changed live state.'
        report = {'scope': 'Exposed motor-prefix lift diagnosis; not a complete table or learned result.',
                  'preflight': preflight, 'seed': args.seed,
                  'prefix': args.initial_from.as_posix(),
                  'prefix_states_sha256': hashlib.sha256((args.initial_from/'states.npz').read_bytes()).hexdigest(),
                  'prefix_motor_steps': len(saved['ctrl']), 'prefix_replay_exact': True,
                  'source_action': action, 'initial_forces': forces,
                  'geometry_checks': rows, 'physical_continuation': None,
                  'whole_table_complete': False}
        put(folder/'geometry.json', report)
        if args.execute_first and passing:
            index, mode, points = passing[0]
            # The stored motor route starts at the measured, physically attained
            # grasp. Its later transfer/lowering is still checked by the teacher.
            task._configure(candidate)
            task.metrics['lift_geometry_mode'] = mode
            task.metrics['lift_geometry_row'] = index
            task._move('lift', points, CLOSED, 3.)
            observer = TableObserver(model)
            try:
                targets = data.qpos[:12].copy()
                report['physical_continuation'] = execute(model, data, layout, task, targets,
                                                          folder/'continuation', observer)
            finally:
                observer.close()
        report['wall_s'] = time.perf_counter()-started
        put(folder/'report.json', report)
        print({k: v for k, v in report.items() if k not in ('geometry_checks', 'source_action')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-from', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execute-first', action='store_true')
    run(parser.parse_args())
