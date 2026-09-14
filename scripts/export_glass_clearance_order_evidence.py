"""Compact outcome for the bounded physical glass-first V19 intervention."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import put, load_module
from simulation_lab.storage import require_space


def main():
    source = ROOT / '.run/glass-clearance-order-physics-v1'
    geometry = ROOT / '.run/glass-clearance-order-geometry-v1'
    out = ROOT / '.run/glass-clearance-order-outcome-v1'
    log = out.with_suffix('.log')
    evidence = ROOT / 'docs/robotics/evidence/glass-clearance-order-development-v1.json'
    for path in (out, log, evidence):
        if path.exists():
            raise FileExistsError(path.name)
        require_space(path, 64 * 1024**2)
    out.mkdir(parents=True)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        put(out / 'source.py', Path(__file__).read_bytes())
        result = json.loads((source / 'report.json').read_text())
        geo = json.loads((geometry / 'report.json').read_text())
        final = json.loads((ROOT / '.run/glass-clearance-final-geometry-v1/report.json').read_text())
        branches = []
        for number in (1, 2):
            folder = source / f'branch-{number:02d}'
            row = json.loads((folder / 'report.json').read_text())
            branches.append(dict(attempt=number, outcome=row['result'], source_grasp=row['source_grasp'],
                                 wrist_roll=row['wrist_roll'], continuous_frames=row['continuous_frames'],
                                 independent_continuous_replay_exact=row['independent_continuous_replay_exact'],
                                 replay_max_error=row['replay_max_error'],
                                 full_motor_trace_sha256=hashlib.sha256((folder / 'continuous-states.npz').read_bytes()).hexdigest()))
        with np.load(source / 'branch-01/glass-clearance/states.npz') as one, np.load(source / 'branch-02/glass-clearance/states.npz') as two:
            same_shape = one['ctrl'].shape == two['ctrl'].shape
            differences = dict(same_length=same_shape,
                max_control_difference=float(np.max(np.abs(one['ctrl'] - two['ctrl']))) if same_shape else None,
                max_qpos_difference=float(np.max(np.abs(one['qpos'] - two['qpos']))) if same_shape else None,
                max_qvel_difference=float(np.max(np.abs(one['qvel'] - two['qvel']))) if same_shape else None,
                interpretation='Different initial wrist guesses converged to nearly identical actual control/state paths. Both failures remain in the budget; they are not independent grasp-strategy coverage.')
        # Exact first-branch motor replay identifies which physical contacts
        # stopped approach, rather than inferring support from saved poses.
        model = mujoco.MjModel.from_xml_string((source / 'scene.xml').read_text())
        data = mujoco.MjData(model)
        with np.load(source / 'branch-01/continuous-states.npz') as archive:
            trace = {k: archive[k].copy() for k in ('initial_integration', 'ctrl', 'qpos', 'qvel')}
        mujoco.mj_setState(model, data, trace['initial_integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        for index, ctrl in enumerate(trace['ctrl']):
            data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
            assert np.array_equal(data.qpos, trace['qpos'][index + 1]) and np.array_equal(data.qvel, trace['qvel'][index + 1])
        teacher = load_module('simulation_lab._glass_clearance_outcome', source / 'source/simulation_lab/table_teacher.py')
        layout = json.loads((source / 'layout.json').read_text())
        task = teacher.TableTeacher(model, data, layout); task.start(side='right', object_id='glass')
        task._select_item('right', next(o for o in layout['objects'] if o['id'] == 'glass'))
        contacts = []
        for index, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if model.geom_bodyid[a] != task.body and model.geom_bodyid[b] != task.body:
                continue
            other = b if model.geom_bodyid[a] == task.body else a
            force = np.zeros(6); mujoco.mj_contactForce(model, data, index, force)
            if force[0] <= .01:
                continue
            contacts.append(dict(glass_geom=model.geom(a if model.geom_bodyid[a] == task.body else b).name,
                other_geom=model.geom(other).name, other_body=model.body(int(model.geom_bodyid[other])).name,
                category='fixed' if other in task.fixed_geoms else 'moving' if other in task.moving_geoms else 'external',
                normal_force_n=float(force[0]), distance_m=float(contact.dist), position_m=contact.pos.tolist()))
        put(out / 'contact-diagnosis.json', dict(exact_replay=True, final_contacts=contacts, redundancy=differences))
        hashes = {}
        for folder in (source, geometry):
            for path in (folder / 'source').rglob('*.py'):
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / 'scene.xml', source / 'report.json', source / 'protocol.json',
                     source / 'physical-selection-protocol.json', geometry / 'report.json',
                     ROOT / '.run/glass-clearance-final-geometry-v1/report.json', Path(__file__)):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = dict(scope='Explicit exposed V19 physical ordering intervention: attempt to clear the measured mug swept-route glass obstacle before a new mug final leg.',
            status='Glass clearance failed within the declared two-attempt budget. No mug continuation was attempted; whether actual clearance enables final mug placement remains untested.',
            protocol=result['protocol'],
            original_prior_results_unchanged=True,
            final_glass_plan=dict(planned=final['planned'], message=final['message'], source_search_records=len(final['search']),
                target=final['target'], note='Checked separately without requiring clearance of the previous fixed mug sweep; remains unsolved.'),
            temporary_clearance_geometry=dict(common_options_examined=len(geo['rows']),
                skipped_by_swept_aabb_filter=sum(not r['clears_recorded_sweep'] for r in geo['rows']),
                checked_plans=len(geo['proofs']), target=geo['proofs'][0]['target'], quaternion=geo['proofs'][0]['quaternion'],
                actual_previous_mug_sweep_margin_m=.006,
                scope='Conservative bounding-box clearance of the recorded swept path, not proof of all future routes.'),
            new_glass_physical_attempts=2, new_mug_physical_attempts=0, branches=branches,
            actual_first_failure_contacts=contacts, alternate_wrist_redundancy=differences,
            observation='Both attempts stop at approach timeout. Only the fixed finger loads the inverted glass; no moving-finger grasp, unsupported lift, hold or placement occurs.',
            independent_scene_branches='Each alternative starts with the original initial integration state once, replays the entire accepted V19 mug-buffer motor prefix, and then runs only normal motor/physics controls. Failed states are retained, not reset inside a branch. No branch reaches a glass pose eligible for mug continuation.',
            model_changed=False, force_or_contact_gates_changed=False, object_teleportation=False,
            camera_control=False, learned_ordering=False, full_table_success=False,
            preserved_outputs=[source.relative_to(ROOT).as_posix(), geometry.relative_to(ROOT).as_posix(),
                               '.run/glass-clearance-final-geometry-v1', out.relative_to(ROOT).as_posix()],
            source_hashes=hashes)
        assert str(ROOT) not in json.dumps(report)
        put(evidence, report)
        print(json.dumps(dict(glass_attempts=2, mug_attempts=0, glass_cleared=False,
                              redundancy=differences, first_final_contacts=contacts)))


if __name__ == '__main__':
    main()
