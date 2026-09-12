"""Compare matched familiar-start checkpoints; separate motion learning from task success."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    def load(folder, name):
        return json.loads((folder / name).read_text())
    before, after = [load(folder, 'offline-fit5.json') for folder in (baseline, candidate)]
    if before.get('blank_images') or after.get('blank_images'):
        raise ValueError('The continuation gate requires normal-image scores.')
    if before.get('selected_episode') != after.get('selected_episode'):
        raise ValueError('Scoring episode selections differ.')
    if before['dataset_lineage'] != after['dataset_lineage']:
        raise ValueError('A matched-data comparison requires identical dataset lineage.')
    if before['sampled_observations'] != after['sampled_observations']:
        raise ValueError('Scoring samples differ.')
    trials = [load(folder, 'summary.json') for folder in (baseline, candidate)]
    if any(x['trials'] != 5 or len(x['results']) != 5 for x in trials):
        raise ValueError('All five physical trials must complete before deciding.')
    old_rows, new_rows = [sorted(x['results'], key=lambda row: row['episode']) for x in trials]
    if [x['episode'] for x in old_rows] != [x['episode'] for x in new_rows]:
        raise ValueError('Physical start sets differ.')
    metric = 'first_five_left_arm_mae_rad'
    gain = 1 - after[metric] / before[metric]
    supported_before = sum(x['best_supported_hold_s'] for x in old_rows)
    supported_after = sum(x['best_supported_hold_s'] for x in new_rows)
    # Budget gate, not a scientific convergence test: substantial inference-path
    # improvement or new physical success supports one further bounded run.
    extend = (trials[1]['successes'] > trials[0]['successes'] or
              (gain >= .25 and trials[1]['successes'] >= trials[0]['successes']))
    return {
        'baseline': str(baseline), 'candidate': str(candidate),
        'same_dataset_and_five_starts': True,
        'first_five_mae_before_rad': before[metric],
        'first_five_mae_after_rad': after[metric],
        'relative_prediction_improvement': gain,
        'current_position_reference_mae_rad': after['first_five_current_joint_reference_mae_rad'],
        'successes_before': trials[0]['successes'], 'successes_after': trials[1]['successes'],
        'supported_hold_sum_before_s': supported_before, 'supported_hold_sum_after_s': supported_after,
        'extend_training_gate_passed': extend,
        'gate_definition': 'New task success OR at least 25% lower first-five prediction MAE without fewer task successes.',
        'qualification': 'A budget decision on familiar data, not proof of convergence, generalization or task learning.'
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = compare(Path(a.baseline), Path(a.candidate))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
