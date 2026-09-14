"""Finite shorter/strict-joint approach checks for six recorded V19 mug contacts.

No target contact waiver: approach checks call the common collision checker with
allow_tube=False, including every mug/jaw contact at the standard150um planning
penetration gate. Geometry uses scratch data; physics, if justified, uses only
the exact original accepted motor prefix and ordinary controls.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import freeze, prepare, put, arrays
from simulation_lab.storage import require_space

DEADLINE = datetime(2026, 9, 14, 15, 25, tzinfo=timezone.utc)


def before_deadline():
    if datetime.now(timezone.utc) >= DEADLINE:
        raise RuntimeError('Declared15:25UTC deadline reached; no further tuning')


def configure(task, teacher, row):
    task.start(side=row['side'], object_id='mug')
    task._select_item(row['side'], next(o for o in task.layout['objects'] if o['id'] == 'mug'))
    up = task.data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
    candidate = teacher.GraspCandidate('recorded_deep_wall_contact', np.asarray(row['point']),
        np.array([-.006, 0., -.092]), 2, up, np.asarray(row['radial']), .4, up.copy(), .35)
    task._configure(candidate)
    return candidate


def mug_contacts(task, q):
    scratch = mujoco.MjData(task.model)
    scratch.qpos[:] = task.data.qpos
    scratch.qpos[task.offset:task.offset + 5] = q
    scratch.qpos[task.offset + 5] = .4
    mujoco.mj_forward(task.model, scratch)
    rows = []
    for contact in scratch.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if ({a, b} & task.arm_geoms) and (task.model.geom_bodyid[a] == task.body or task.model.geom_bodyid[b] == task.body):
            rows.append(dict(geom1=task.model.geom(a).name, geom2=task.model.geom(b).name,
                             distance_m=float(contact.dist), position_m=contact.pos.tolist()))
    return rows


def geometry(args):
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out)
        put(out / 'source/scripts/develop_mug_wall_approach.py', Path(__file__).read_bytes())
        source = ROOT / '.run/mug-wall-coupled-geometry-v1/report.json'
        originals = [r for r in json.loads(source.read_text())['rows'] if 'source_q' in r]
        assert len(originals) == 6
        protocol = dict(scope=__doc__, declared_before_checks=True, maximum_new_physical_attempts=2,
                        deadline_utc='2026-09-14 15:25:00 UTC',
                        endpoints_source=source.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        endpoint_count=6, finite_alternatives=['bodyZ20mm', 'bodyZ10mm', 'joint_space_open_hand'],
                        open_grip=.4, grip_torque_nm=.35, radius_m=.023, height_m=.041,
                        local_tool_point_m=[-.006, 0., -.092],
                        planning_contact_gate_m=.00015, target_contact_waiver=False,
                        path_step_rad=.035, cartesian_step_m=.002,
                        physics_guards='Original teacher:1mm manipulated external penetration,.8mm unexpected robot contact,4mm unrelated displacement; full unsupported lift/hold/release/park.',
                        no_camera_control=True)
        put(out / 'protocol.json', protocol)
        geometry_source = ROOT / '.run/mug-wall-geometry-v2/V19_post_accepted00_upright'
        for name in ('scene.xml', 'layout.json', 'prefix.json', 'actual-source-state.npz'):
            put(out / name, (geometry_source / name).read_bytes())
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text())
        layout = json.loads((out / 'layout.json').read_text())
        data = mujoco.MjData(model)
        with np.load(out / 'actual-source-state.npz') as archive:
            data.qpos[:] = archive['qpos']; data.qvel[:] = archive['qvel']; data.ctrl[:] = archive['ctrl']
        mujoco.mj_forward(model, data)
        rows, proofs = [], []
        for number, original in enumerate(originals):
            before_deadline()
            task = teacher.TableTeacher(model, data, layout)
            candidate = configure(task, teacher, original)
            q = np.asarray(original['source_q'])
            contacts = mug_contacts(task, q)
            endpoint_error = None
            try:
                task._check_path([q], .4, allow_tube=False)
            except teacher.PlanningError as exc:
                endpoint_error = str(exc)
            for alternative, height in (('bodyZ20mm', .02), ('bodyZ10mm', .01), ('joint_space_open_hand', None)):
                before_deadline()
                row = dict(endpoint=number, original=original, alternative=alternative,
                           source_contacts=contacts, endpoint_collision_error=endpoint_error, passed=False)
                rows.append(row)
                task = teacher.TableTeacher(model, data, layout)
                candidate = configure(task, teacher, original)
                try:
                    # Endpoint is part of every route and may not be waived.
                    task._check_path([q], .4, allow_tube=False)
                    task._precheck_transfer(q)
                    task._configure(candidate)
                    if height is not None:
                        row['stage'] = 'hover_ik'
                        hover = candidate.point + height * candidate.axis_target
                        above = task.ik.solve(hover, q)
                        row['stage'] = 'strict_descent'
                        descent = task._cartesian(hover, candidate.point, above, .4, check=False)
                        task._check_path(descent, .4, allow_tube=False)
                        row['stage'] = 'strict_approach'
                        approach = task._route(data.qpos[task.offset:task.offset + 5], above, .4, allow_target=False)
                    else:
                        row['stage'] = 'strict_joint_route'
                        approach = task._route(data.qpos[task.offset:task.offset + 5], q, .4, allow_target=False)
                        descent = np.asarray([q, q])
                        hover = candidate.point.copy()
                    task._check_path(approach, .4, allow_tube=False)
                    task._check_path(descent, .4, allow_tube=False)
                    row.update(passed=True, hover=hover.tolist(), approach_samples=len(approach), descent_samples=len(descent))
                    proof_id = len(proofs) + 1
                    row['route_file'] = f'proof-{proof_id:03d}-route.npz'
                    arrays(out / row['route_file'], approach=approach, descent=descent)
                    proofs.append(row)
                    put(out / f'proof-{proof_id:03d}.json', row)
                except teacher.PlanningError as exc:
                    row['error'] = str(exc)
                put(out / f'check-{number:02d}-{alternative}.json', row)
                print(dict(endpoint=number, alternative=alternative, passed=row['passed'],
                           stage=row.get('stage', 'source_endpoint_collision'), error=row.get('error')), flush=True)
        put(out / 'report.json', dict(protocol=protocol, rows=rows, proofs=proofs, new_physics_steps=0,
                                      result='No strict checked approach' if not proofs else 'Checked approaches; physical verification still required'))


def physical(args):
    before_deadline()
    out = args.output.resolve()
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out, args.proof.parent)
        put(out / 'source/scripts/develop_mug_wall_approach.py', Path(__file__).read_bytes())
        proof = json.loads(args.proof.read_text())
        assert proof['passed']
        for name in ('scene.xml', 'layout.json', 'prefix.json', 'protocol.json'):
            put(out / name, (args.proof.parent / name).read_bytes())
        put(out / 'proof.json', proof)
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text())
        layout = json.loads((out / 'layout.json').read_text())
        data = mujoco.MjData(model)
        prefix = json.loads((out / 'prefix.json').read_text())
        assert len(prefix) == 1
        source = ROOT / prefix[0]['path'] / 'states.npz'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == prefix[0]['states_sha256']
        with np.load(source) as archive:
            saved = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(model, data, saved['initial_integration'], spec)
        mujoco.mj_forward(model, data)
        for index, ctrl in enumerate(saved['ctrl']):
            data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
            assert np.array_equal(data.qpos, saved['qpos'][index + 1]) and np.array_equal(data.qvel, saved['qvel'][index + 1])
        arrays(out / 'prefix-states.npz', **saved)
        with np.load(args.proof.parent / proof['route_file']) as archive:
            approach, descent = archive['approach'].copy(), archive['descent'].copy()
        original = proof['original']
        class ApproachTask(teacher.TableTeacher):
            def _plan(self, targets):
                self._select_item(original['side'], next(o for o in self.layout['objects'] if o['id'] == 'mug'))
                up = self.data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
                candidate = teacher.GraspCandidate('recorded_deep_wall_contact', np.asarray(original['point']),
                    np.array([-.006, 0., -.092]), 2, up, np.asarray(original['radial']), .4, up.copy(), .35)
                self._configure(candidate)
                self._check_path(approach, .4, allow_tube=False)
                self._check_path(descent, .4, allow_tube=False)
                self._precheck_transfer(np.asarray(original['source_q']))
                self._configure(candidate)
                self.grasp = candidate.point.copy(); self.chosen = candidate; self.chosen_roll = original['roll']
                self.hover = np.asarray(proof['hover']); self.approach_descent = descent
                self.search_log = [dict(scope='Preserved strict checked route, including target contact gate', proof=proof)]
                self.attempts = 1
                self._move('approach', approach, .4, 3.)
        from simulation_lab.scene import HOME
        from simulation_lab.table_observation import TableObserver
        targets = np.asarray(HOME * 2)
        task = ApproachTask(model, data, layout)
        task.start(side=original['side'], object_id='mug')
        task.update(targets)
        before_deadline()
        if not task.active:
            result = dict(status=task.status, message=task.message, search=task.search_log, new_physics_steps=0)
            controls, positions, velocities, stages = saved['ctrl'], saved['qpos'], saved['qvel'], saved['stage']
        else:
            observer = TableObserver(model)
            try:
                require_space(out / 'mug-final', 384 * 1024**2)
                result = whole.execute(model, data, layout, task, targets, out / 'mug-final', observer)
            finally:
                observer.close()
            with np.load(out / 'mug-final/states.npz') as archive:
                controls = np.concatenate([saved['ctrl'], archive['ctrl']])
                positions = np.concatenate([saved['qpos'], archive['qpos'][1:]])
                velocities = np.concatenate([saved['qvel'], archive['qvel'][1:]])
                stages = np.concatenate([saved['stage'], archive['stage'][1:]])
        arrays(out / 'continuous-states.npz', initial_integration=saved['initial_integration'],
               ctrl=controls, qpos=positions, qvel=velocities, stage=stages)
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, saved['initial_integration'], spec); mujoco.mj_forward(model, replay)
        error = 0.
        for index, ctrl in enumerate(controls):
            replay.ctrl[:] = ctrl; mujoco.mj_step(model, replay)
            error = max(error, float(np.max(np.abs(replay.qpos - positions[index + 1]))),
                        float(np.max(np.abs(replay.qvel - velocities[index + 1]))))
        put(out / 'report.json', dict(result=result, independent_continuous_replay_exact=error == 0.,
                                     replay_max_error=error, frames=len(positions), camera_control=False))
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    physical(args) if args.proof else geometry(args)
