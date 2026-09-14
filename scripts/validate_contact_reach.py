"""Read-only validation against preserved V24 actual states; no physics."""
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
from simulation_lab.contact_reach import initial_direct_contact_separation
from simulation_lab.storage import require_space


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value if isinstance(value, bytes) else json.dumps(value, indent=2).encode())


def main():
    output = ROOT / '.run/contact-reach-module-validation-v1'
    log = output.with_suffix('.log')
    evidence = ROOT / 'docs/robotics/evidence/sideplate-pregrasp-development-v1.json'
    for path in (output, log, evidence):
        if path.exists():
            raise FileExistsError(f'Preserve existing {path.name}')
    preflight = require_space(output, 64 * 1024**2)
    require_space(log, 1024**2)
    require_space(evidence, 4 * 1024**2)
    output.mkdir(parents=True)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        source = ROOT / '.run/sideplate-pregrasp-geometry-v1'
        xml = (source / 'scene.xml').read_bytes()
        model = mujoco.MjModel.from_xml_string(xml.decode())
        data = mujoco.MjData(model)
        prefix = json.loads((source / 'prefix.json').read_text())
        manifest = {}
        for path in (Path(__file__), ROOT / 'simulation_lab/contact_reach.py',
                     source / 'scene.xml', source / 'prefix.json',
                     source / 'actual-post-action03-state.npz'):
            payload = path.read_bytes()
            manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
            write(output / 'inputs' / path.relative_to(ROOT), payload)
        rows = []
        for action in prefix:
            path = ROOT / action['path']
            payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert payload_hash == action['sha256']
            archive = np.load(path)
            data.qpos[:] = archive['qpos'][0]
            data.qvel[:] = archive['qvel'][0]
            data.ctrl[:] = archive['ctrl'][0]
            mujoco.mj_forward(model, data)
            item = 'bottle' if action['action'] < 2 else 'glass'
            arm = 'left' if action['action'] == 0 else 'right'
            results = {side: initial_direct_contact_separation(model, data, item, side)
                       for side in ('left', 'right')}
            assert not results[arm]['excluded'], (action['action'], arm)
            rows.append(dict(action=action['action'], item=item, successful_arm=arm,
                             actual_source_pose=data.joint(item + '_free').qpos.tolist(),
                             source_sha256=payload_hash, results=results))
        with np.load(source / 'actual-post-action03-state.npz') as archive:
            data.qpos[:] = archive['qpos']
            data.qvel[:] = archive['qvel']
            data.ctrl[:] = archive['ctrl']
        mujoco.mj_forward(model, data)
        before_data = {name: getattr(data, name).copy() for name in
                       ('qpos', 'qvel', 'ctrl', 'xpos', 'xmat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat')}
        before_model = {name: getattr(model, name).copy() for name in
                        ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'jnt_range', 'geom_pos',
                         'geom_quat', 'geom_size', 'geom_contype', 'geom_conaffinity', 'mesh_vert')}
        actual = {side: initial_direct_contact_separation(model, data, 'side_plate', side)
                  for side in ('left', 'right')}
        for side, expected in (('left', .02890128243722434), ('right', .06488772187666225)):
            assert actual[side]['excluded']
            assert abs(actual[side]['gap_m'] - expected) < 1e-10
        pose = data.joint('side_plate_free').qpos.copy()
        pose_checks = []
        # Reset-only comparison validates the algebraic arbitrary-pose API for
        # full yaw, upright/inverted and two sideways orientations.
        for tilt in (0., np.pi / 2, -np.pi / 2, np.pi):
            for yaw in np.linspace(-np.pi, np.pi, 9):
                quat = np.array([np.cos(yaw / 2) * np.cos(tilt / 2),
                                 np.cos(yaw / 2) * np.sin(tilt / 2),
                                 np.sin(yaw / 2) * np.sin(tilt / 2),
                                 np.sin(yaw / 2) * np.cos(tilt / 2)])
                query = np.r_[pose[:3], quat]
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = data.qpos
                scratch.joint('side_plate_free').qpos[:] = query
                mujoco.mj_forward(model, scratch)
                for side in ('left', 'right'):
                    result = initial_direct_contact_separation(model, data, 'side_plate', side, pose=query)
                    reference = initial_direct_contact_separation(model, scratch, 'side_plate', side)
                    assert abs(result['gap_m'] - reference['gap_m']) < 1e-10
                pose_checks.append(dict(tilt=float(tilt), yaw=float(yaw)))
        for name, values in before_data.items():
            np.testing.assert_array_equal(getattr(data, name), values)
        for name, values in before_model.items():
            np.testing.assert_array_equal(getattr(model, name), values)
        report = dict(scope='Initial direct-contact geometry only; no new physics or pregrasp trial.',
                      preflight=preflight, physical_steps=0, live_state_or_model_mutation=False,
                      source_hashes=manifest, prefix=prefix, actual_post_action03_sideplate=actual,
                      successful_source_checks=rows, algebraic_pose_comparisons=pose_checks,
                      all_assertions_passed=True)
        write(output / 'report.json', report)
        geometry = json.loads((source / 'report.json').read_text())
        evidence_report = dict(
            scope='Exposed joint seed2026114004 side-plate initial direct-contact diagnosis after accepted V24 actions00-03.',
            outcome='No direct-contact start: both complete arm collision envelopes are strictly separated from the actual inverted side plate.',
            physical_pregrasp_attempts=0, physical_steps=0, preserved_geometry_searches=geometry['attempts'],
            checked_geometric_routes=geometry['proofs'],
            solver_failures_are_not_exclusion_evidence=True,
            actual_pose=pose.tolist(), source_prefix=prefix, source_hashes=manifest,
            independent_prefix_replay_performed=False,
            prefix_provenance='Read preserved actual final state after four root-verified continuous-main-scene exact motor replays. No new replay or manipulation was launched after the no-start certificate.',
            analytic_certificate=actual,
            earlier_bound='A root-centered ball allowed the 64.775mm first segment to point anywhere. Its apparent moving-jaw overlap was unresolved, not a contact proof. The actual fixed-axis circle removes that looseness.',
            derivation=[
                'Fixed shoulder-pan axis; shoulder-lift anchor sweeps full circle of radius35.470965mm with fixed axial component54.2mm.',
                'For each downstream collision geometry, use remaining serial anchor lengths plus farthest terminal collision radius as a ball. All hinge angles and complete mesh hulls are included.',
                'Support from the circle center in unit direction n is rho*sqrt(1-(n dot axis)^2)+R. Root-level arm geometry uses a separate root-anchor ball.',
                'Take maximum across all collision-enabled or explicitly paired arm geometry; subtract from exact minimum support of every current object primitive. Subtract contact-margin padding and require10um numerical reserve.'],
            limitations=['Only initial direct robot-arm contact at the queried item pose is certified.',
                         'Indirect object-assisted rearrangement is outside this certificate; no global task impossibility claim.',
                         'No sampler eligibility, scene starts, old results or denominators changed.',
                         'An unresolved bound does not establish a feasible grasp or transfer.',
                         'The pose API covers arbitrary rigid orientation without narrowing any yaw or stable-orientation family.'],
            validation=dict(successful_v24_bottle_glass_sources_not_excluded=4,
                            algebraic_pose_comparisons=len(pose_checks) * 2,
                            live_state_and_model_arrays_unchanged=True),
            preserved_outputs=['.run/sideplate-pregrasp-geometry-v1',
                               '.run/sideplate-pregrasp-separation-v1',
                               '.run/sideplate-pregrasp-separation-v2',
                               '.run/contact-reach-module-validation-v1'],
            incomplete_output='.run/sideplate-pregrasp-separation-v1 preserves a JSON numpy-bool serialization failure; no physics was run.')
        encoded = json.dumps(evidence_report, indent=2).encode()
        assert str(ROOT).encode() not in encoded
        require_space(evidence, len(encoded) + 1024**2)
        write(evidence, encoded)
        print(json.dumps(dict(passed=True, gaps_m={side: row['gap_m'] for side, row in actual.items()},
                              successful_source_gaps=[{s: r['gap_m'] for s, r in x['results'].items()} for x in rows],
                              actual_successful_source_poses=[x['actual_source_pose'] for x in rows],
                              orientation_comparisons=len(pose_checks) * 2)))


if __name__ == '__main__':
    main()
