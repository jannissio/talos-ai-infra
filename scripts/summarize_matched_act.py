"""Collect the completed matched ACT experiment without substituting replay for learning."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    checkpoint = ROOT / '.run/act-matched-1000/step-001000'
    evaluation = ROOT / '.run/act-matched-evaluation-1000'
    read = lambda path: json.loads(path.read_text())
    metrics = [json.loads(line) for line in (checkpoint.parent / 'metrics.jsonl').read_text().splitlines()]
    trials = read(evaluation / 'summary.json')
    audit = read(ROOT / 'docs/robotics/act-matched-reconstruction.json')
    assert metrics[-1]['step'] == metrics[-1]['successful_updates'] == 1000
    assert trials['trials'] == 5 and len(trials['results']) == 5
    assert len(audit) == 5 and all(row['physical_replays_passed'] == 2 for row in audit)
    assert all(row['max_state_error'] == row['max_applied_control_error'] == row['max_pixel_error'] == 0 for row in audit)
    scores = {}
    for name, filename in [('normal', 'offline-fit5.json'), ('blank_images', 'offline-fit5-blank.json')]:
        scores[name] = read(evaluation / filename)
        assert scores[name]['sampled_observations'] == 128
    with (checkpoint / 'model.safetensors').open('rb') as weights:
        checkpoint_hash = hashlib.file_digest(weights, 'sha256').hexdigest()
    report = {
        'experiment': 'Matched physical controller and NVIDIA camera capture; familiar starts only',
        'checkpoint': str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_hash,
        'training': {'first_logged_update': metrics[0], 'last_logged_update': metrics[-1]},
        'recording_audit': audit,
        'physical_evaluation': trials,
        'offline_scores': scores,
        'reserved_physical_poses_evaluated': 0,
        'caveat': 'Recording corrections and increased training exposure changed together; this does not isolate their effects.'
    }
    output = ROOT / 'docs/robotics/act-matched-results.json'
    output.write_text(json.dumps(report, indent=2))
    print(output)
    print(f"Learned physical successes: {trials['successes']}/{trials['trials']}")


if __name__ == '__main__':
    main()
