"""Nine measured-state30mm mug lift checks, unchanged25mm physical gate."""
import contextlib
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import prepare, freeze, put, arrays
from scripts.develop_mug_wall_approach import before_deadline


def main():
    out = ROOT / '.run/mug-short-lift-geometry-v1'; log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out)
        put(out / 'source/scripts/check_mug_short_lift.py', Path(__file__).read_bytes())
        source = ROOT / '.run/mug-wall-clearance-physics-v1'
        protocol = dict(scope=__doc__, declared_before_checks=True,
                        source='Exact recorded closed grip after attempt1',
                        source_states_sha256=hashlib.sha256((source / 'mug-final/states.npz').read_bytes()).hexdigest(),
                        shifts_m=[[x, y, .03] for x, y in itertools.product((-.02, 0., .02), repeat=2)],
                        normal='Measured bodyZ held through actual tool-local carried transform', yaw='free',
                        position_gate_m=.00025, normal_gate_deg=3,
                        carried_collision_gate_m=.00015,
                        support='Table allowed only for first two samples, as in existing lift. Every later carried-object external contact is checked; expected finger contacts remain the physical grasp.',
                        physical_lift_gate_m=.025, physical_hold_s=1.5, torque_nm=.35,
                        maximum_remaining_physical_attempts=1, deadline_utc='2026-09-14 15:25:00 UTC',
                        larger_or_adaptive_search=False, camera_control=False)
        put(out / 'protocol.json', protocol)
        for name in ('scene.xml', 'layout.json'):
            put(out / name, (source / name).read_bytes())
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text())
        layout = json.loads((out / 'layout.json').read_text())
        data = mujoco.MjData(model)
        with np.load(source / 'mug-final/states.npz') as archive:
            data.qpos[:] = archive['qpos'][-1]; data.qvel[:] = archive['qvel'][-1]; data.ctrl[:] = archive['ctrl'][-1]
        mujoco.mj_forward(model, data)
        arrays(out / 'actual-held-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
        contact = json.loads((source / 'proof.json').read_text())
        rows, proofs = [], []
        for index, shift in enumerate(protocol['shifts_m']):
            before_deadline()
            task = teacher.TableTeacher(model, data, layout); task.start(side='right', object_id='mug')
            task._select_item('right', next(o for o in layout['objects'] if o['id'] == 'mug'))
            candidate = teacher.GraspCandidate('measured_clearance_mug_wall', np.asarray(contact['adjusted_point']),
                np.array([-.006, 0., -.092]), 2, data.body('mug').xmat.reshape(3, 3)[:, 2].copy(),
                np.asarray(contact['adjusted_radial']), .4, None, .35)
            task._configure(candidate)
            reference = task._carry_reference()
            task.ik.axis_local = reference[1][:, 2].copy(); task.ik.axis_target = data.body('mug').xmat.reshape(3, 3)[:, 2].copy()
            task.ik.x_target = None; task.ik.placement_solve = True; task.ik.soft_placement = True
            start = task.ik.point(data).copy(); grip = float(data.qpos[11])
            row = dict(index=index, shift_m=shift, passed=False, stage='lift_ik',
                       actual_grip_q=grip, axis_local=task.ik.axis_local.tolist(), axis_world=task.ik.axis_target.tolist())
            try:
                points = task._cartesian(start, start + np.asarray(shift), data.qpos[6:11], grip, check=False)
                row['stage'] = 'carried_collision_and_support_release'
                task._check_path(points[:2], grip, True, carry=reference, support=task.base_geom)
                task._check_path(points[2:], grip, True, carry=reference)
                row.update(passed=True, samples=len(points), route_file=f'proof-{index:02d}-route.npz')
                arrays(out / row['route_file'], points=points)
                put(out / f'proof-{index:02d}.json', row); proofs.append(row)
            except teacher.PlanningError as exc:
                row['error'] = str(exc)
            rows.append(row); put(out / f'check-{index:02d}.json', row)
            print(row, flush=True)
        put(out / 'report.json', dict(protocol=protocol, rows=rows, proofs=proofs, new_physics_steps=0))


if __name__ == '__main__':
    main()
