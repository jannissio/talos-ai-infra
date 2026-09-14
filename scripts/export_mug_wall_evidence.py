"""Sanitized deeper-wall and coupled-azimuth mug geometry evidence."""
import collections
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def main():
    output = ROOT / 'docs/robotics/evidence/mug-wall-grasp-development-v1.json'
    log = ROOT / '.run/mug-wall-evidence-v1.log'
    for path in (output, log):
        if path.exists():
            raise FileExistsError(path.name)
        require_space(path, 16 * 1024**2)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        discrete = []
        manifests = {}
        for version in (1, 2):
            folder = ROOT / f'.run/mug-wall-geometry-v{version}'
            report = json.loads((folder / 'report.json').read_text())
            checks = []
            for row in report['rows']:
                for check in row['checks']:
                    checks.append(dict(context=row['context'], mode=check['mode'], planned=check['planned'],
                                       records=len(check['search']),
                                       failure_categories=dict(collections.Counter(r.get('error', '').split(':')[0] for r in check['search'])),
                                       target=check['target'], actual_mug_pose=row['actual_mug_pose'], prefix=row['prefix']))
            discrete.append(dict(version=version, azimuth_phase='Object-relative legacy direction' if version == 1 else 'Exact legacy world direction projected onto actual body cross-section', checks=checks))
            for path in folder.rglob('*.py'):
                manifests[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifests[(folder / 'report.json').relative_to(ROOT).as_posix()] = hashlib.sha256((folder / 'report.json').read_bytes()).hexdigest()
        coupled_folder = ROOT / '.run/mug-wall-coupled-geometry-v1'
        coupled = json.loads((coupled_folder / 'report.json').read_text())
        for path in [coupled_folder / 'report.json', coupled_folder / 'coupled-source.py', Path(__file__)]:
            manifests[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        reached_source = [r for r in coupled['rows'] if 'source_q' in r]
        evidence = dict(scope='Exposed legacy mug-wall contact reuse on actual V24 sideways and V19 upright buffer states; privileged geometry dependency only.',
                        status='No passing source approach; zero new physical attempts.',
                        family=dict(api='mug_wall_grasp_candidates(data, name, candidate_type, pose=None, open_grip=.4)',
                                    radius_m=.023, local_height_m=.041, local_tool_point_m=[-.006, 0., -.092],
                                    tool_z='Actual or estimated mug body Z', tool_x='Radial around the mug wall',
                                    approach='Outward along mug body Z toward the lip', azimuths=12, torque_limit_nm=.35),
                        discrete_checks=discrete, discrete_search_records=sum(c['records'] for v in discrete for c in v['checks']),
                        coupled=dict(derivation='Let radial closing equal tool X. Then tool_origin+R*[-.006,0,-.092]=mug_origin+.041*bodyZ+.023*toolX. Solve the algebraically equivalent centerline target using temporary IK tool offset[-.029,0,-.092] and toolZ=bodyZ, with free azimuth. Restore the physical[-.006,0,-.092] contact and check its actual radius/orientation/collisions/goal.',
                                     attempts=len(coupled['rows']), successful_source_and_goal_endpoint_checks=len(reached_source),
                                     passing_approach_routes=len(coupled['proofs']), checked_contacts=reached_source,
                                     failure_stages=dict(collections.Counter(r['stage'] for r in coupled['rows']))),
                        diagnosis='V24 sideways source IK remains unsolved using both actual and camera-derived grasp poses. The coupled azimuth solve finds six V19 right-arm source contacts passing source collision and unchanged mug-goal endpoint checks, but their40mm lipward hover/approach IK remains unsolved.',
                        camera_comparison=dict(source='.run/raw-rgbd-mug-full-pose-v1/estimate-before-truth.json',
                                               source_sha256=hashlib.sha256((ROOT / '.run/raw-rgbd-mug-full-pose-v1/estimate-before-truth.json').read_bytes()).hexdigest(),
                                               actual_unchanged_scene='V24 postaccepted00-03',
                                               privileged_pose_and_camera_seed_both_source_ik_unsolved=True,
                                               estimate_applied_to_other_scene=False,
                                               policy_scope='Only candidate point/orientation comes from the camera estimate. Actual obstacle, carried-object and route geometry remain privileged; no camera-only controller or physical trial is claimed.'),
                        physics_steps=0, new_physical_attempts=0, new_prefix_replay_performed=False,
                        prefix_note='Every original accepted action motor trace is referenced by immutable hash with root continuous replay verification. No new prefix replay was launched because source approach geometry failed.',
                        source_hashes=manifests,
                        limitations=['All source/route failures remain unsolved, not analytic impossibility or invalid-start labels.',
                                     'No common controller, model, force/torque limits, task destination or guard changes.',
                                     'The V19 six-contact result is endpoint geometry only; it does not establish a safe approach or physical grip.',
                                     'The physical branch of the isolated harness was not exercised.'],
                        preserved_outputs=['.run/mug-wall-geometry-v1', '.run/mug-wall-geometry-v2', '.run/mug-wall-coupled-geometry-v1'],
                        whole_table_complete=False)
        payload = (json.dumps(evidence, indent=2) + '\n').encode()
        assert str(ROOT).encode() not in payload
        with output.open('xb') as target:
            target.write(payload)
        print(json.dumps(dict(discrete_records=evidence['discrete_search_records'], coupled_attempts=len(coupled['rows']),
                              source_and_goal_endpoints=len(reached_source), passing_approach_routes=len(coupled['proofs']), physics_steps=0)))


if __name__ == '__main__':
    main()
