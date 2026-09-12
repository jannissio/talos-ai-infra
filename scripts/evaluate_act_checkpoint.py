"""One reproducible checkpoint review: physical trials, prediction scores and budget gate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(a):
    output = Path(a.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh evaluation folder to avoid mixing checkpoint results.')
    output.mkdir(parents=True)
    commands = [
        ('physical', ['scripts/evaluate_act_batch.py', '--checkpoint', a.checkpoint,
                      '--episode-root', a.episode_root, '--output', str(output), '--record-video']),
        ('offline', ['scripts/score_act.py', '--checkpoint', a.checkpoint,
                     '--dataset', a.dataset, '--samples', '128', '--output', str(output / 'offline-fit5.json')]),
        ('blank', ['scripts/score_act.py', '--checkpoint', a.checkpoint,
                   '--dataset', a.dataset, '--samples', '128', '--blank-images',
                   '--output', str(output / 'offline-fit5-blank.json')]),
        ('comparison', ['scripts/compare_act_runs.py', '--baseline', a.baseline,
                        '--candidate', str(output), '--output', str(output / 'comparison.json')]),
    ]
    for name, command in commands:
        print('Starting', name, flush=True)
        with (output / (name + '-driver.log')).open('w') as log:
            subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, timeout=1800, check=True)
        print('Completed', name, flush=True)
    print(json.dumps(json.loads((output / 'comparison.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline', default='.run/act-matched-evaluation-1000')
    p.add_argument('--dataset', default='.run/act-matched-nvidia-fit5')
    p.add_argument('--episode-root', default='.run/bottle-matched-nvidia')
    main(p.parse_args())
