"""Export compact completed shared-teacher outcomes without raw private paths.

Original motor/state arrays, observations, recipes and sources remain in place.
This summarizes development attempts; it does not establish generalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder):
    report_path = folder/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    recipe = json.loads((folder/'reset-recipe.json').read_text(encoding='utf-8'))
    physical = report['physical_lookahead']
    return {
        'run': folder.name, 'seed': report['seed'],
        'scope': report['scope'], 'coverage_evaluation': False,
        'reset_distribution': report.get('reset_distribution', 'unspecified'),
        'verify_relay_continuation': report.get('verify_relay_continuation', False),
        'execution_protocol_sha256': digest(folder/'execution-protocol.json') if (folder/'execution-protocol.json').exists() else None,
        'candidate_scene_protocol': recipe.get('candidate_protocol'),
        'candidate_scene_summaries': recipe.get('candidate_summaries'),
        'candidate_protocol_sha256': digest(folder/'protocol.json') if (folder/'protocol.json').exists() else None,
        'candidate_scene_evidence': [
            {'candidate': path.parent.name, 'result_sha256': digest(path),
             'recipe_sha256': digest(path.parent/'recipe.json'),
             'states_sha256': digest(path.parent/'states.npz')}
            for path in sorted((folder/'reset-candidates').glob('*/result.json'))],
        'only_item_diagnostic': report['only_item_diagnostic'],
        'whole_table_complete': report['whole_table_complete'],
        'remaining': report['remaining'], 'stop_reason': report.get('stop_reason'),
        'valid_initial_geometry': report.get('valid_initial_geometry'),
        'physical_attempts': len(physical),
        'physical_frames': sum(row['frames'] for row in physical),
        'observations': sum(row['observations'] for row in physical),
        'passing_primitives': sum(row['status'] == 'succeeded' for row in physical),
        'planning_failures': len(report['planning_failures']),
        'all_motor_replays_exact': all(row['replay_exact'] and row['replay_max_state_error'] == 0 for row in physical),
        'accepted_actions': report['actions'],
        'physical_outcomes': physical,
        'planning_outcomes': [{key: value for key, value in row.items() if key != 'search'}
                              for row in report['planning_failures']],
        'final_task_checks': report.get('final_task_checks'),
        'report_sha256': digest(report_path),
        'reset_recipe_sha256': digest(folder/'reset-recipe.json'),
        'source_sha256': {path.relative_to(folder/'source').as_posix(): digest(path)
                          for path in sorted((folder/'source').rglob('*.py'))},
        'raw_preservation': 'Original arrays, every search, cameras and sources remain in the local run folder.',
    }


def run(output, folders):
    if output.exists():raise FileExistsError('Preserve the existing evidence export.')
    require_space(output, 16*1024**2)
    rows = [summarize(folder) for folder in folders]
    value = {'schema': 'talos_shared_table_followup_v1',
        'meaning': 'Completed exposed development; single-item and repeated-scene diagnostics are not full-table coverage.',
        'runs': rows, 'run_count': len(rows),
        'physical_attempts': sum(row['physical_attempts'] for row in rows),
        'physical_frames': sum(row['physical_frames'] for row in rows),
        'passing_primitives': sum(row['passing_primitives'] for row in rows),
        'accepted_actions': sum(len(row['accepted_actions']) for row in rows),
        'complete_tables': sum(row['whole_table_complete'] for row in rows)}
    payload = (json.dumps(value, indent=2)+'\n').encode()
    require_space(output, len(payload)+1024**2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:stream.write(payload)
    print(json.dumps({key: value[key] for key in
        ('run_count', 'physical_attempts', 'physical_frames', 'passing_primitives', 'accepted_actions', 'complete_tables')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    args = parser.parse_args(); run(args.output, args.run)
