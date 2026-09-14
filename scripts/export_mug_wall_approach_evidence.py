"""Preserve strict target-inclusive rejection of the six wall-pinch endpoints."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def main():
    source = ROOT / '.run/mug-wall-approach-geometry-v1'
    output = ROOT / 'docs/robotics/evidence/mug-wall-approach-development-v1.json'
    log = ROOT / '.run/mug-wall-approach-evidence-v1.log'
    snapshot = source / 'evidence-exporter.py'
    for path in (output, log, snapshot):
        if path.exists():
            raise FileExistsError(path.name)
        require_space(path, 8 * 1024**2)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        report = json.loads((source / 'report.json').read_text())
        contacts = []
        for row in report['rows'][::3]:
            depth = -min(c['distance_m'] for c in row['source_contacts'])
            contacts.append(dict(endpoint=row['endpoint'], original=row['original'],
                                 maximum_fixed_jaw_mug_penetration_mm=depth * 1000,
                                 collision=row['endpoint_collision_error'],
                                 contacts=row['source_contacts']))
        hashes = json.loads((source / 'source-manifest.json').read_text())
        for path in (source / 'protocol.json', source / 'report.json', source / 'scene.xml',
                     source / 'prefix.json', source / 'actual-source-state.npz',
                     source / 'source/scripts/develop_mug_wall_approach.py', Path(__file__)):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence = dict(scope='Exposed privileged V19 upright-mug approach dependency; fixed original six contact endpoints.',
                        status='All18 alternatives rejected at their shared target-inclusive source endpoint; no physical attempt.',
                        declaration=report['protocol'],
                        original_accepted_prefix=json.loads((source / 'prefix.json').read_text()),
                        alternatives=[dict(endpoint=r['endpoint'], alternative=r['alternative'],
                                           rejected_at='source_endpoint_collision', error=r['error']) for r in report['rows']],
                        endpoints=contacts, shortest_route_gate='Every approach contains its final source configuration. It cannot be accepted when that configuration violates the target-inclusive collision gate.',
                        new_finding='At open grip.4, the fixed-jaw spheres/boxes penetrate the actual mug wall by0.920–0.939mm across all six recorded endpoints. The unchanged common150um planning gate rejects every endpoint when allow_tube=False.',
                        correction_to_prior_interpretation='The previous six passing source collision checks used allow_tube=True, which deliberately skips jaw–target contacts. Those were not target-inclusive clear source endpoints. They remain preserved as the narrower original result.',
                        shorter_hover_ik_evaluated=False, joint_route_search_evaluated=False,
                        reason_not_evaluated='All finite alternatives first fail their mandatory common endpoint check; route expansion cannot repair the same fixed colliding endpoint.',
                        source_endpoint_count=6, finite_alternative_count=18,
                        passing_approaches=0, new_physical_attempts=0, new_physics_steps=0,
                        new_prefix_replay_performed=False,
                        camera_control=False, complete_table_success=False,
                        unchanged_contact_geometry=dict(radius_m=.023, height_m=.041,
                                                        local_tool_point_m=[-.006, 0., -.092], torque_nm=.35),
                        limits='No target collision waiver, contact tolerance increase, model edit, pose adjustment or new grasp sweep. A force-controlled contact approach that moves the mug is outside this fixed-endpoint collision-free route check; no global manipulation-impossibility claim.',
                        source_hashes=hashes, preserved_output=source.relative_to(ROOT).as_posix())
        payload = (json.dumps(evidence, indent=2) + '\n').encode()
        assert str(ROOT).encode() not in payload
        with output.open('xb') as stream_out:
            stream_out.write(payload)
        with snapshot.open('xb') as stream_out:
            stream_out.write(Path(__file__).read_bytes())
        print(json.dumps(dict(endpoints=6, alternatives=18, passed=0, physical_attempts=0,
                              penetration_mm=[r['maximum_fixed_jaw_mug_penetration_mm'] for r in contacts])))


if __name__ == '__main__':
    main()
