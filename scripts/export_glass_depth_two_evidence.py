"""Sanitized immutable report of the bounded V26 three-leg glass search."""
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
    source = ROOT / '.run/glass-depth-two-geometry-v1'
    original = ROOT / '.run/whole-table-development-v26-direct-contact'
    output = ROOT / 'docs/robotics/evidence/glass-depth-two-development-v1.json'
    log = ROOT / '.run/glass-depth-two-evidence-v1.log'
    snapshot = source / 'evidence-exporter.py'
    for path in (output, log, snapshot):
        if path.exists():
            raise FileExistsError(f'Preserve {path.name}')
        require_space(path, 16 * 1024**2)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        report = json.loads((source / 'report.json').read_text())
        prior = json.loads((original / 'report.json').read_text())
        accepted = json.loads((original / 'accepted-action-00.json').read_text())
        prefixes = []
        for result in [accepted] + [r for r in prior['physical_lookahead'] if r['item'] == 'glass' and r['status'] == 'succeeded']:
            path = original / result['path']
            prefixes.append(dict(item=result['item'], path=path.relative_to(ROOT).as_posix(),
                                 states_sha256=hashlib.sha256((path / 'states.npz').read_bytes()).hexdigest(),
                                 action_sha256=hashlib.sha256((path / 'action.json').read_bytes()).hexdigest(),
                                 frames=result['frames'], original_independent_replay_exact=result['replay_exact'],
                                 originally_accepted_in_main_scene=bool(result.get('continuous_main_scene_replay_exact', False)),
                                 maximum_external_penetration_mm=result['metrics']['maximum_external_penetration_mm'],
                                 hold_verified_s=result['metrics']['hold_verified_s']))
        rows = []
        for row in report['rows']:
            result = {k: v for k, v in row.items() if k != 'source_search'}
            search = row.get('source_search', [])
            result['source_search_records'] = len(search)
            result['source_search_failure_categories'] = dict(collections.Counter(r.get('error', '').split(':')[0] for r in search))
            result['source_contact_candidates_without_upright_transfer'] = [r for r in search if r.get('error', '').startswith('Grasp has no solved destination approach:')]
            rows.append(result)
        hashes = json.loads((source / 'source-manifest.json').read_text())
        for path in (source / 'report.json', source / 'scene.xml', source / 'protocol.json', Path(__file__)):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence = dict(scope='Exposed V26 seed2026122001 sideways-glass three-leg regrasp dependency; read-only geometry after actual recorded release/park states.',
                        status='Unsolved in the declared bounded search; no impossibility certificate.',
                        experiment='Three passing first buffers, four existing common upright buffer candidates from each actual post-release state. Check horizontal final-regrasp geometry first, then unchanged common source/transfer planner.',
                        candidate_count=len(rows), hypothetical_final_regrasp_checks_passed=sum('final_regrasp_geometry' in r for r in rows),
                        checked_sideways_to_upright_plans=len(report['proofs']), source_arm_grasp_roll_records=sum(r['source_search_records'] for r in rows),
                        new_physical_alternatives=0, new_physics_steps=0, independently_replayed_new_continuous_prefix=False,
                        prefix_note='No new motor prefix replay was justified after geometry found no continuation. Original immutable controls/states and their existing exact-replay verification are retained by hash; the passing glass lookaheads were not accepted into the original main scene.',
                        source_prefixes=prefixes, source_hashes=hashes, rows=rows,
                        diagnosis='Four hypothetical upright buffer states admit horizontal final-regrasp geometry. The preceding sideways-source grasp/transfer does not solve:195 source IK failures,6 source collision failures and3 upright destination failures per checked plan. Only outside_body_band_0.045_-1 reaches source contact checks; its upright destination fails IK or right_camera_box2/table clearance.',
                        implication='A depth-two acceptance rule alone has no checked continuation to accept for these buffers. A different valid grasp/reorientation route remains an unsolved dependency.',
                        limitations=['The existing bounded buffer list is not the entire continuous table workspace.',
                                     'Planning failure is not physical impossibility or invalid-start evidence.',
                                     'No changed geometry, physics, thresholds, task destinations, original outcomes or scene denominators.',
                                     'No new upright or sideways glass physical placement is claimed.',
                                     'The optional physical harness branch was not exercised because no geometric proof passed.'],
                        preserved_outputs=['.run/glass-depth-two-geometry-v1'],
                        complete_table_success=False, learned_control=False)
        payload = (json.dumps(evidence, indent=2) + '\n').encode()
        assert str(ROOT).encode() not in payload
        with snapshot.open('xb') as target:
            target.write(Path(__file__).read_bytes())
        with output.open('xb') as target:
            target.write(payload)
        print(json.dumps({k: evidence[k] for k in ('status', 'candidate_count', 'hypothetical_final_regrasp_checks_passed', 'checked_sideways_to_upright_plans', 'source_arm_grasp_roll_records', 'new_physical_alternatives')}))


if __name__ == '__main__':
    main()
