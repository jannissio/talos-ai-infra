"""Exact motor replay and compact final mug lift/hold/transfer-failure evidence."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import put, arrays, load_module
from simulation_lab.storage import require_space


def main():
    out = ROOT / '.run/mug-short-lift-outcome-v1'
    log = out.with_suffix('.log')
    evidence_path = ROOT / 'docs/robotics/evidence/mug-wall-clearance-development-v2.json'
    for path in (out, log, evidence_path):
        if path.exists():
            raise FileExistsError(path.name)
        require_space(path, 128 * 1024**2)
    out.mkdir(parents=True)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        source = ROOT / '.run/mug-wall-clearance-physics-v3'
        geometry = ROOT / '.run/mug-short-lift-geometry-v1'
        put(out / 'source.py', Path(__file__).read_bytes())
        result = json.loads((source / 'report.json').read_text())
        geo = json.loads((geometry / 'report.json').read_text())
        model = mujoco.MjModel.from_xml_string((source / 'scene.xml').read_text())
        data = mujoco.MjData(model)
        with np.load(source / 'continuous-states.npz') as archive:
            trace = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
        with np.load(source / 'prefix-states.npz') as archive:
            prefix_controls = len(archive['ctrl'])
            origin_z = float(archive['qpos'][-1, model.joint('mug_free').qposadr[0] + 2])
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(model, data, trace['initial_integration'], spec); mujoco.mj_forward(model, data)
        teacher = load_module('simulation_lab._mug_outcome_snapshot', source / 'source/simulation_lab/table_teacher.py')
        layout = json.loads((source / 'layout.json').read_text())
        probe = teacher.TableTeacher(model, data, layout); probe.start(side='right', object_id='mug')
        probe._select_item('right', next(o for o in layout['objects'] if o['id'] == 'mug'))
        rows, external = [], []
        error = 0.
        for index, ctrl in enumerate(trace['ctrl']):
            data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
            error = max(error, float(np.max(np.abs(data.qpos - trace['qpos'][index + 1]))),
                        float(np.max(np.abs(data.qvel - trace['qvel'][index + 1]))))
            if index < prefix_controls:
                continue
            forces = dict(fixed=0., moving=0., external=0.)
            contacts = []
            for cid, contact in enumerate(data.contact):
                a, b = int(contact.geom1), int(contact.geom2)
                if model.geom_bodyid[a] != probe.body and model.geom_bodyid[b] != probe.body:
                    continue
                other = b if model.geom_bodyid[a] == probe.body else a
                wrench = np.zeros(6); mujoco.mj_contactForce(model, data, cid, wrench)
                value = max(0., float(wrench[0]))
                key = 'fixed' if other in probe.fixed_geoms else 'moving' if other in probe.moving_geoms else 'external'
                forces[key] += value
                if key == 'external' and value > .01:
                    contacts.append(dict(other_geom=model.geom(other).name,
                        other_body=model.body(int(model.geom_bodyid[other])).name,
                        normal_force_n=value, distance_m=float(contact.dist), position_world_m=contact.pos.tolist(),
                        contact_frame_wrench=wrench.tolist()))
            stage = str(trace['stage'][index + 1])
            lift = float(data.body('mug').xpos[2] - origin_z)
            tilt = float(np.degrees(np.arccos(np.clip(data.body('mug').xmat[8], -1., 1.))))
            rows.append(dict(time_s=float(data.time), stage=stage, lift_m=lift, tilt_deg=tilt, **forces))
            if contacts and stage in ('lift', 'hold', 'align') and lift > .025:
                external.append(dict(time_s=float(data.time), stage=stage, lift_m=lift, tilt_deg=tilt, contacts=contacts))
        assert error == 0.
        arrays(out / 'actual-new-leg-observations.npz', time_s=[r['time_s'] for r in rows],
               stage=[r['stage'] for r in rows], lift_m=[r['lift_m'] for r in rows], tilt_deg=[r['tilt_deg'] for r in rows],
               fixed_n=[r['fixed'] for r in rows], moving_n=[r['moving'] for r in rows], external_n=[r['external'] for r in rows])
        holds = [r for r in rows if r['stage'] == 'hold']
        hold_summary = dict(measured_controller_hold_s=result['result']['metrics']['hold_verified_s'],
                            minimum_lift_m=min(r['lift_m'] for r in holds), maximum_tilt_deg=max(r['tilt_deg'] for r in holds),
                            minimum_fixed_force_n=min(r['fixed'] for r in holds), minimum_moving_force_n=min(r['moving'] for r in holds),
                            maximum_external_force_n=max(r['external'] for r in holds))
        replay_report = dict(exact=True, maximum_state_error=error, total_frames=len(trace['qpos']),
                             new_leg_frames=len(rows) + 1, prefix_motor_frames=prefix_controls,
                             unsupported_hold=hold_summary, external_contact_failure=external)
        put(out / 'report.json', replay_report)
        hashes = {}
        for path in (source / 'continuous-states.npz', source / 'scene.xml', source / 'report.json',
                     source / 'source/scripts/develop_mug_wall_clearance.py',
                     source / 'source/simulation_lab/table_teacher.py', source / 'source/simulation_lab/autonomy.py',
                     geometry / 'protocol.json', geometry / 'report.json', out / 'report.json', Path(__file__)):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence = dict(scope='Final bounded exposed V19 mug contact-clearance/lift dependency. No common edits or camera-control claim.',
                        status='Unsupported lift and1.805s hold pass; final transfer fails on external contact. No completed mug placement.',
                        prior_report='docs/robotics/evidence/mug-wall-clearance-development-v1.json',
                        finite_short_lift_checks=geo,
                        second_physical_attempt=result,
                        first_physical_approach_close_prefix=json.loads((source / 'first-attempt-prefix-comparison.json').read_text()),
                        exact_replay_diagnosis=replay_report,
                        hold_wording_correction=dict(required_unsupported_hold_s=1.8,
                            actual_measured_hold_s=result['result']['metrics']['hold_verified_s'],
                            nominal_motion_duration_s=1.5, unchanged_common_done_motion_extra_s=.3,
                            source='Frozen table_teacher.py schedules1.5s nominal hold; inherited autonomy._done_motion additionally requires0.3s and bounded joint error. Completion occurs at1.805s.',
                            correction='The earlier compact protocol field physical_hold_s=1.5 described nominal motion duration incorrectly as a physical gate. No1.5s acceptance occurred and no controller hold gate was changed.'),
                        total_new_physical_attempts=2, additional_physical_attempts_permitted=0,
                        no_model_or_force_changes=True, contact_and_normal_gates_unchanged=True,
                        integration_recommendation='Do not claim a shared successful mug primitive. The measured-normal <=2mm clearance and30mm free-yaw/body-normal lift are useful partial dependencies; carry/transfer remains unresolved. Keep dynamic coupled-contact solving and actual wall contact labels separate from virtual IK centerline targets.',
                        preserved_outputs=['.run/mug-short-lift-geometry-v1', '.run/mug-wall-clearance-physics-v3', out.relative_to(ROOT).as_posix()],
                        source_hashes=hashes, whole_table_complete=False)
        put(evidence_path, evidence)
        print(json.dumps(dict(hold=hold_summary, external_contact_failure=external, replay_exact=True)))


if __name__ == '__main__':
    main()
