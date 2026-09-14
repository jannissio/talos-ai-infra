"""Measured-normal clearance refinement of six fixed exposed mug contacts.

Finite offsets0.5/1/1.5/2mm; every open-hand endpoint/path checks the target at
the unchanged150um planning gate. No contact waiver, geometry edit or force
change. Physical continuation, if planned, replays the exact V19 motor prefix.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import prepare, freeze, put, arrays
from scripts.develop_mug_wall_approach import before_deadline, mug_contacts
from simulation_lab.storage import require_space


def measured_direction(model, data, original):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[6:11] = original['source_q']; scratch.qpos[11] = .4
    mujoco.mj_forward(model, scratch)
    body = model.body('mug').id
    contacts, weighted = [], np.zeros(3)
    for contact in scratch.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        an, bn = model.geom(a).name, model.geom(b).name
        if contact.dist >= -.00015:
            continue
        if 'right_fixed_jaw' in an and model.geom_bodyid[b] == body:
            away = -contact.frame[:3].copy()
        elif 'right_fixed_jaw' in bn and model.geom_bodyid[a] == body:
            away = contact.frame[:3].copy()
        else:
            continue
        weight = -float(contact.dist) - .00015
        weighted += away * weight
        contacts.append(dict(geom1=an, geom2=bn, depth_m=-float(contact.dist), away_world=away.tolist(),
                             weight_m=weight, position_m=contact.pos.tolist()))
    assert len(contacts) > 0 and np.linalg.norm(weighted) > 1e-10
    return weighted / np.linalg.norm(weighted), contacts


def geometry(args):
    out = args.output.resolve(); log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out)
        put(out / 'source/scripts/develop_mug_wall_clearance.py', Path(__file__).read_bytes())
        put(out / 'source/scripts/develop_mug_wall_approach.py', (ROOT / 'scripts/develop_mug_wall_approach.py').read_bytes())
        prior = ROOT / '.run/mug-wall-approach-geometry-v1'
        for name in ('scene.xml', 'layout.json', 'prefix.json', 'actual-source-state.npz'):
            put(out / name, (prior / name).read_bytes())
        originals = [r['original'] for r in json.loads((prior / 'report.json').read_text())['rows'][::3]]
        protocol = dict(scope=__doc__, declared_before_checks=True, deadline_utc='2026-09-14 15:25:00 UTC',
                        exposed_endpoint_count=6, measured_normal_offset_steps_m=[.0005, .001, .0015, .002],
                        maximum_actual_contact_point_change_m=.002,
                        normal_method='Depth-excess-weighted average of measured fixed-finger/mug outward separation normals; each weight=max(penetration-.15mm,0).',
                        coupled_solve='Shift the centerline target by measured world displacement, retain the original23mm radial contact relation and41mm band; solve free azimuth from the original source q. Reject actual point displacement exceeding2mm.',
                        original_local_tool_point_m=[-.006, 0., -.092], adjusted_local_tool_point_m=[-.006, 0., -.092],
                        radius_m=.023, band_m=.041, torque_nm=.35,
                        approach_alternatives=['bodyZ20mm', 'bodyZ10mm', 'joint_space_open_hand'],
                        target_inclusive_planning_gate_m=.00015, target_contact_waiver=False,
                        maximum_physical_attempts=2, camera_control=False,
                        source_report_sha256=hashlib.sha256((prior / 'report.json').read_bytes()).hexdigest())
        put(out / 'protocol.json', protocol)
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text())
        layout = json.loads((out / 'layout.json').read_text())
        data = mujoco.MjData(model)
        with np.load(out / 'actual-source-state.npz') as archive:
            data.qpos[:] = archive['qpos']; data.qvel[:] = archive['qvel']; data.ctrl[:] = archive['ctrl']
        mujoco.mj_forward(model, data)
        up = data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
        centerline = data.body('mug').xpos + .041 * up
        rows, proofs = [], []
        for endpoint, original in enumerate(originals):
            direction, contacts = measured_direction(model, data, original)
            for distance in (.0005, .001, .0015, .002):
                before_deadline()
                row = dict(endpoint=endpoint, distance_m=distance, original=original,
                           measured_contacts=contacts, separation_direction_world=direction.tolist(),
                           requested_displacement_world=(distance * direction).tolist(), passed_endpoint=False,
                           original_local_tool_point=[-.006, 0., -.092], adjusted_local_tool_point=[-.006, 0., -.092], approaches=[])
                task = teacher.TableTeacher(model, data, layout)
                task.start(side='right', object_id='mug'); task._select_item('right', next(o for o in layout['objects'] if o['id'] == 'mug'))
                try:
                    row['stage'] = 'coupled_source_ik'
                    task.ik.grasp_point = np.array([-.029, 0., -.092]); task.ik.axis_index = 2
                    task.ik.axis_target = up.copy(); task.ik.x_target = None
                    q = task.ik.solve(centerline + distance * direction, np.asarray(original['source_q']))
                    task.ik.data.qpos[6:11] = q; mujoco.mj_kinematics(model, task.ik.data)
                    rotation = task.ik.data.body('right_gripper').xmat.reshape(3, 3)
                    radial = rotation[:, 0] - up * float(rotation[:, 0] @ up); radial /= np.linalg.norm(radial)
                    point = centerline + distance * direction + .023 * radial
                    row.update(adjusted_point=point.tolist(), adjusted_radial=radial.tolist(), source_q=q.tolist(),
                               actual_point_displacement_m=float(np.linalg.norm(point - np.asarray(original['point']))))
                    row['stage'] = 'bounded_point_change'
                    if row['actual_point_displacement_m'] > .002 + 1e-10:
                        raise teacher.PlanningError('Actual contact point change exceeds declared2mm total budget')
                    candidate = teacher.GraspCandidate('measured_clearance_mug_wall', point, np.array([-.006, 0., -.092]),
                        2, up.copy(), radial, .4, up.copy(), .35)
                    task._configure(candidate)
                    q = task.ik.solve(point, q); row['source_q'] = q.tolist()
                    row['stage'] = 'target_inclusive_endpoint'
                    row['adjusted_contacts'] = mug_contacts(task, q)
                    task._check_path([q], .4, allow_tube=False)
                    row['passed_endpoint'] = True
                    row['stage'] = 'unchanged_goal'
                    task._precheck_transfer(q)
                    row['passed_goal'] = True
                    for label, height in (('bodyZ20mm', .02), ('bodyZ10mm', .01), ('joint_space_open_hand', None)):
                        before_deadline()
                        task._configure(candidate)
                        check = dict(alternative=label, passed=False); row['approaches'].append(check)
                        try:
                            if height is None:
                                approach = task._route(data.qpos[6:11], q, .4, allow_target=False)
                                descent = np.asarray([q, q]); hover = point.copy()
                            else:
                                hover = point + height * up
                                above = task.ik.solve(hover, q)
                                descent = task._cartesian(hover, point, above, .4, check=False)
                                task._check_path(descent, .4, allow_tube=False)
                                approach = task._route(data.qpos[6:11], above, .4, allow_target=False)
                            task._check_path(approach, .4, allow_tube=False); task._check_path(descent, .4, allow_tube=False)
                            check.update(passed=True, approach_samples=len(approach), descent_samples=len(descent))
                            number = len(proofs) + 1
                            proof = dict(endpoint=endpoint, offset_m=distance, original=original,
                                         adjusted_point=point.tolist(), adjusted_radial=radial.tolist(), source_q=q.tolist(),
                                         actual_point_displacement_m=row['actual_point_displacement_m'],
                                         local_tool_point=[-.006, 0., -.092], torque_nm=.35, alternative=label,
                                         hover=hover.tolist(), route_file=f'proof-{number:03d}-route.npz')
                            arrays(out / proof['route_file'], approach=approach, descent=descent)
                            put(out / f'proof-{number:03d}.json', proof); proofs.append(proof)
                        except teacher.PlanningError as exc:
                            check['error'] = str(exc)
                except teacher.PlanningError as exc:
                    row['error'] = str(exc)
                rows.append(row)
                put(out / f'check-{endpoint:02d}-{int(distance * 10000):02d}.json', row)
                print(dict(endpoint=endpoint, offset_mm=distance * 1000, stage=row['stage'],
                           endpoint_pass=row['passed_endpoint'], error=row.get('error'), approaches=row['approaches']), flush=True)
        put(out / 'report.json', dict(protocol=protocol, rows=rows, proofs=proofs, new_physics_steps=0))


def physical(args):
    before_deadline()
    out = args.output.resolve(); log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out, args.proof.parent)
        put(out / 'source/scripts/develop_mug_wall_clearance.py', Path(__file__).read_bytes())
        proof = json.loads(args.proof.read_text())
        for name in ('scene.xml', 'layout.json', 'prefix.json', 'protocol.json'):
            put(out / name, (args.proof.parent / name).read_bytes())
        put(out / 'proof.json', proof)
        short_lift = None
        if args.short_lift_proof:
            short_lift = json.loads(args.short_lift_proof.read_text())
            assert short_lift['passed'] and args.free_yaw_lift
            put(out / 'short-lift-proof.json', short_lift)
            put(out / 'short-lift-protocol.json', (args.short_lift_proof.parent / 'protocol.json').read_bytes())
        if args.free_yaw_lift:
            put(out / 'route-extension.json', dict(declared_before_new_physics=True,
                source_failed_attempt='.run/mug-wall-clearance-physics-v1',
                extension='During lift, preserve measured object-bodyZ through a tool-local axis constraint and free global closing yaw. Use the separately declared30mm measured lift proof if supplied;0.25mm/3degree IK and full carried-mug/handle collision gates remain.',
                same_contact_as_first_attempt=True, same_torque_nm=.35, maximum_remaining_physical_attempts=1,
                physical_surface_label='Adjusted wall surface target remains the pick label; the virtual centerline used for IK is never a pick label.'))
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text())
        layout = json.loads((out / 'layout.json').read_text()); data = mujoco.MjData(model)
        prefix = json.loads((out / 'prefix.json').read_text()); assert len(prefix) == 1
        source = ROOT / prefix[0]['path'] / 'states.npz'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == prefix[0]['states_sha256']
        with np.load(source) as archive:
            saved = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(model, data, saved['initial_integration'], spec); mujoco.mj_forward(model, data)
        for index, ctrl in enumerate(saved['ctrl']):
            data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
            assert np.array_equal(data.qpos, saved['qpos'][index + 1]) and np.array_equal(data.qvel, saved['qvel'][index + 1])
        arrays(out / 'prefix-states.npz', **saved)
        with np.load(args.proof.parent / proof['route_file']) as archive:
            approach, descent = archive['approach'].copy(), archive['descent'].copy()
        class RefinedWallTask(teacher.TableTeacher):
            def _collision(self, state, allow_tube=False, penetration=.00015):
                # Actual open-hand motion also includes the mug. The runtime
                # uses its unchanged0.8mm guard; planning uses unchanged0.15mm.
                if state is self.data and self.stage in ('approach', 'descend'):
                    allow_tube = False
                return super()._collision(state, allow_tube, penetration)

            def _lift(self):
                if args.free_yaw_lift:
                    reference = self._carry_reference()
                    self.ik.axis_local = reference[1][:, 2].copy()
                    self.ik.axis_target = self.data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
                    self.ik.x_target = None; self.ik.axis_in_plane = False
                    self.ik.soft_placement = True; self.ik.placement_solve = True
                    self.metrics['measured_body_normal_free_yaw_lift'] = True
                    self.lift_axis_local = self.ik.axis_local.copy()
                    self.lift_axis_world = self.ik.axis_target.copy()
                    if short_lift is not None:
                        start = self.ik.point(self.data).copy()
                        grip = float(self.data.qpos[self.offset + 5])
                        points = self._cartesian(start, start + np.asarray(short_lift['shift_m']),
                            self.data.qpos[self.offset:self.offset + 5], grip, check=False)
                        self._check_path(points[:2], grip, True, carry=reference, support=self.base_geom)
                        self._check_path(points[2:], grip, True, carry=reference)
                        self.metrics['measured_short_lift_shift_m'] = short_lift['shift_m']
                        self._move('lift', points, teacher.CLOSED, 3.)
                        return
                return super()._lift()

            def _plan(self, targets):
                self._select_item('right', next(o for o in self.layout['objects'] if o['id'] == 'mug'))
                up = self.data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
                candidate = teacher.GraspCandidate('measured_clearance_mug_wall', np.asarray(proof['adjusted_point']),
                    np.array([-.006, 0., -.092]), 2, up, np.asarray(proof['adjusted_radial']), .4, up.copy(), .35)
                self._configure(candidate)
                self._check_path(approach, .4, allow_tube=False); self._check_path(descent, .4, allow_tube=False)
                self._precheck_transfer(np.asarray(proof['source_q'])); self._configure(candidate)
                self.grasp = candidate.point.copy(); self.chosen = candidate; self.chosen_roll = proof['original']['roll']
                self.hover = np.asarray(proof['hover']); self.approach_descent = descent
                self.search_log = [dict(scope='Measured <=2mm normal clearance with strict target-inclusive source route', proof=proof)]
                self.attempts = 1; self._move('approach', approach, .4, 3.)
        from simulation_lab.scene import HOME
        from simulation_lab.table_observation import TableObserver
        if args.free_yaw_lift:
            # Independent read-only check against the exact measured held state
            # of attempt1 before spending the final physical continuation.
            held = mujoco.MjData(model)
            with np.load(ROOT / '.run/mug-wall-clearance-physics-v1/mug-final/states.npz') as old:
                held.qpos[:] = old['qpos'][-1]; held.qvel[:] = old['qvel'][-1]; held.ctrl[:] = old['ctrl'][-1]
            mujoco.mj_forward(model, held)
            probe = RefinedWallTask(model, held, layout); probe.start(side='right', object_id='mug')
            probe._select_item('right', next(o for o in layout['objects'] if o['id'] == 'mug'))
            candidate = teacher.GraspCandidate('measured_clearance_mug_wall', np.asarray(proof['adjusted_point']),
                np.array([-.006, 0., -.092]), 2, held.body('mug').xmat.reshape(3, 3)[:, 2].copy(),
                np.asarray(proof['adjusted_radial']), .4, None, .35)
            probe._configure(candidate)
            try:
                probe._lift()
                put(out / 'measured-held-lift-check.json', dict(passed=True,
                    axis_local=probe.lift_axis_local.tolist(), axis_world=probe.lift_axis_world.tolist(),
                    meaning='Scratch geometry from measured attempt1 held state; physical lift still required.'))
                arrays(out / 'measured-held-lift-route.npz', points=probe.motion.points)
            except teacher.PlanningError as exc:
                put(out / 'measured-held-lift-check.json', dict(passed=False, error=str(exc)))
                put(out / 'report.json', dict(result=dict(status='planning_rejected', message=str(exc)),
                    prefix_replay_exact=True, new_manipulation_steps=0, frames=len(saved['qpos']),
                    actual_replayed_trace='prefix-states.npz', camera_control=False))
                return
        targets = np.asarray(HOME * 2); task = RefinedWallTask(model, data, layout)
        task.start(side='right', object_id='mug'); task.update(targets); before_deadline()
        if not task.active:
            result = dict(status=task.status, message=task.message, new_physics_steps=0)
            controls, positions, velocities, stages = saved['ctrl'], saved['qpos'], saved['qvel'], saved['stage']
        else:
            observer = TableObserver(model)
            try:
                require_space(out / 'mug-final', 384 * 1024**2)
                result = whole.execute(model, data, layout, task, targets, out / 'mug-final', observer)
            finally:
                observer.close()
            with np.load(out / 'mug-final/states.npz') as archive:
                controls = np.concatenate([saved['ctrl'], archive['ctrl']]); positions = np.concatenate([saved['qpos'], archive['qpos'][1:]])
                velocities = np.concatenate([saved['qvel'], archive['qvel'][1:]]); stages = np.concatenate([saved['stage'], archive['stage'][1:]])
        arrays(out / 'continuous-states.npz', initial_integration=saved['initial_integration'], ctrl=controls,
               qpos=positions, qvel=velocities, stage=stages)
        replay = mujoco.MjData(model); mujoco.mj_setState(model, replay, saved['initial_integration'], spec); mujoco.mj_forward(model, replay)
        error = 0.
        for index, ctrl in enumerate(controls):
            replay.ctrl[:] = ctrl; mujoco.mj_step(model, replay)
            error = max(error, float(np.max(np.abs(replay.qpos - positions[index + 1]))), float(np.max(np.abs(replay.qvel - velocities[index + 1]))))
        put(out / 'report.json', dict(result=result, prefix_replay_exact=True, independent_continuous_replay_exact=error == 0.,
                                     replay_max_error=error, frames=len(positions), camera_control=False))
        if args.free_yaw_lift:
            with np.load(ROOT / '.run/mug-wall-clearance-physics-v1/mug-final/states.npz') as first, np.load(out / 'mug-final/states.npz') as second:
                count = len(first['ctrl'])
                same = bool(np.array_equal(first['ctrl'], second['ctrl'][:count]) and
                            np.array_equal(first['qpos'], second['qpos'][:count + 1]) and
                            np.array_equal(first['qvel'], second['qvel'][:count + 1]))
            put(out / 'first-attempt-prefix-comparison.json', dict(frames=count + 1, exact=same))
            assert same, 'Prior actual approach/close motor prefix diverged'
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--proof', type=Path)
    parser.add_argument('--free-yaw-lift', action='store_true')
    parser.add_argument('--short-lift-proof', type=Path)
    args = parser.parse_args(); physical(args) if args.proof else geometry(args)
