"""Summarize physical demonstration coverage without treating repeats as variety."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier summaries.')
    require_space(args.output, 16*1024**2)
    inventory = json.loads(args.inventory.read_text())
    rows = {}
    bindings = []
    for sample in inventory['samples']:
        action = sample['privileged_teacher_label']['action']
        item = action['item']
        row = rows.setdefault(item, {'sample_count': 0, 'origins': Counter(), 'roles': Counter(),
                                    'source_families': Counter(), 'final_families': Counter(),
                                    'dependency_groups': set(), 'source_position_bins': set()})
        row['sample_count'] += 1
        row['origins'][sample['workflow_origin']] += 1
        row['roles'][action['role']] += 1
        row['source_families'][action['source_item_orientation']['orientation_family']] += 1
        row['final_families'][action['actual_final_item_orientation']['orientation_family']] += 1
        row['dependency_groups'].add(sample['dependency_group'])
        first = sample['privileged_teacher_label']['item_pose_keyframes'][0]
        xyz = first['item_body_origin_m']
        row['source_position_bins'].add(tuple(round(v/.005) for v in xyz))
        bindings.append({'sample_id': sample['sample_id'], 'run': sample['run'], 'episode': sample['episode'],
                         'item': item, 'workflow_origin': sample['workflow_origin'],
                         'dependency_group': sample['dependency_group'],
                         'source_files_sha256': sample['provenance']['source_files_sha256']})
    for row in rows.values():
        row['independent_scene_group_count'] = len(row.pop('dependency_groups'))
        row['source_position_bin_count_5mm'] = len(row.pop('source_position_bins'))
    report = {'schema': 'talos_spatial_demonstration_coverage_v1',
              'scope': 'Completed physical primitive inventory. Repeated exposed scenes are not generalization or complete workflows.',
              'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
              'runs_inspected': inventory['runs_inspected'],
              'sample_count': inventory['sample_count'],
              'workflow_origins': inventory['sample_count_by_workflow_origin'],
              'independent_scene_group_count': inventory['independent_scene_group_count'],
              'training_readiness': inventory['training_readiness'],
              'training_readiness_reason': inventory['training_readiness_reason'],
              'items': rows, 'samples': bindings, 'exclusions': inventory['exclusions'],
              'position_bin_interpretation': 'A 5 mm XYZ bin is a descriptive coarse source-position count, not independent coverage, an orientation bin or a dataset selection rule.',
              'raw_inputs': 'Original calibrated RGB-D, motor/state arrays, full keyframes and inventory remain in the preserved local run folders.'}
    payload = (json.dumps(report, indent=2)+'\n').encode()
    require_space(args.output, len(payload)+1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps({'sample_count': report['sample_count'], 'items': rows,
                      'independent_scene_group_count': report['independent_scene_group_count'],
                      'training_readiness': report['training_readiness']}))


if __name__ == '__main__':
    main()
