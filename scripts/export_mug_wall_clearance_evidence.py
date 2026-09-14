"""Immutable contact-clearance refinement, physical pinch and lift rejection."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from simulation_lab.storage import require_space


def write(path, value):
    if path.exists():
        raise FileExistsError(path.name)
    payload = (json.dumps(value, indent=2) + '\n').encode()
    require_space(path, len(payload) + 1024**2)
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    source = ROOT / '.run/mug-wall-clearance-geometry-v1'
    first = ROOT / '.run/mug-wall-clearance-physics-v1'
    second = ROOT / '.run/mug-wall-clearance-physics-v2'
    output = ROOT / 'docs/robotics/evidence/mug-wall-clearance-development-v1.json'
    log = ROOT / '.run/mug-wall-clearance-evidence-v1.log'
    if output.exists() or log.exists():
        raise FileExistsError('Preserve output and log')
    require_space(output, 32 * 1024**2); require_space(log, 4 * 1024**2)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        geometry = json.loads((source / 'report.json').read_text())
        first_result = json.loads((first / 'report.json').read_text())
        rejection = json.loads((second / 'measured-held-lift-check.json').read_text())
        with np.load(second / 'prefix-states.npz') as prefix:
            prefix_frames = len(prefix['qpos'])
        # The original V2 harness raised after preserving its planning rejection.
        # Record that terminal result without overwriting any source/output.
        if not (second / 'report.json').exists():
            write(second / 'report.json', dict(result=dict(status='planning_rejected', message=rejection['error']),
                prefix_replay_exact=True, new_manipulation_steps=0, frames=prefix_frames,
                actual_replayed_trace='prefix-states.npz', camera_control=False,
                terminal_report_created_by_evidence_exporter=True,
                harness_exit_note='The preserved original harness raised PlanningError after writing the measured-held-lift rejection. No new manipulation followed.'))
        second_result = json.loads((second / 'report.json').read_text())
        hashes = {}
        for folder in (source, first, second):
            for path in folder.rglob('*.py'):
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ('report.json', 'scene.xml', 'proof.json', 'protocol.json', 'prefix-states.npz', 'continuous-states.npz', 'route-extension.json'):
                path = folder / name
                if path.exists():
                    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        first_proof = json.loads((first / 'proof.json').read_text())
        evidence = dict(scope='Exposed V19 upright-mug measured contact-normal refinement; privileged controller dependency, not camera control or learned success.',
                        status='One physical approach/pinch passes contact guards, then stops before unsupported lift. One measured free-yaw lift geometry extension is also unsolved.',
                        protocol=geometry['protocol'],
                        geometry=dict(offset_candidates=len(geometry['rows']),
                                      passing_target_inclusive_endpoints=sum(r['passed_endpoint'] for r in geometry['rows']),
                                      checked_approach_routes=len(geometry['proofs']),
                                      correction_rows=geometry['rows'],
                                      interpretation='Measured inward separation of the fixed finger clears the mug wall at1–1.5mm requested offsets;0.5mm remains colliding, and all nominal2mm cases exceed the actual2mm point-change budget after the coupled azimuth solve.'),
                        physical_attempt_count=1, physical_attempt=first_result,
                        selected_geometry=first_proof,
                        first_attempt_actual_control_frames=first_result['result']['frames'] - 1,
                        force_observation=first_result['result']['metrics']['finger_forces_n'],
                        physical_outcome='Both fingers contact the mug at4.37N/4.30N with0.35Nm torque. Maximum external penetration0.0836mm, unrelated displacement0.000374mm, zero unexpected contacts. Common lift IK fails before unsupported hold; there is no passing pick/place.',
                        measured_route_extension=dict(declaration=json.loads((second / 'route-extension.json').read_text()),
                                                      result=second_result, check=rejection,
                                                      physical_attempt_launched=False),
                        labels=dict(virtual_coupled_ik_target='Object centerline+.041bodyZ+measured clearance displacement; this is an IK construction only, not a surface pick label.',
                                    nominal_gripper_approach_point=first_proof['adjusted_point'],
                                    label_limit='The corrected tool-point target is an approach anchor, not a measured surface-contact coordinate. A future successful integration must emit the intended physical mug-wall contact and retain actual contact observations separately. These failed episodes are not eligible demonstrations.'),
                        continuous_replay='Attempt1 saves the full original accepted V19 motor prefix plus3188 new controls; independent replay is exact. V2 replays the original prefix exactly and launches no new manipulation after the measured-state planning rejection.',
                        no_common_edits=True, no_force_or_geometry_changes=True,
                        normal_and_contact_gates_unchanged=True, camera_control=False, whole_table_complete=False,
                        recommendation='Retain experimental geometry only. Do not promote this wall family as a successful shared mug primitive; unsupported lift and final placement remain unverified.',
                        preserved_outputs=[p.relative_to(ROOT).as_posix() for p in (source, first, second)], source_hashes=hashes)
        payload = json.dumps(evidence)
        assert str(ROOT) not in payload
        write(output, evidence)
        print(json.dumps(dict(offset_candidates=len(geometry['rows']), checked_routes=len(geometry['proofs']),
                              physical_attempts=1, physical_status=first_result['result']['status'],
                              second_route_check='planning_rejected', replay_exact=first_result['independent_continuous_replay_exact'])))


if __name__ == '__main__':
    main()
