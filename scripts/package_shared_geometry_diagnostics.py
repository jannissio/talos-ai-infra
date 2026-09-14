"""Retain completed, non-motion shared-table planning diagnoses compactly."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation_lab.storage import require_space


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != 'preflight'}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and value.startswith('.run/'):
        return value.removeprefix('.run/')
    if isinstance(value, str) and (re.search(r'[A-Za-z]:[/\\]', value)
                                   or '/Users/' in value or '/home/' in value):
        raise ValueError('Review a private absolute path before exporting.')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence.')
    preflight = require_space(args.output, 16*1024**2)
    rows = []
    for folder in args.run:
        report = json.loads((folder/'report.json').read_text())
        # These two frozen diagnoses have no new physical action. Refuse a
        # future successful motion result unless its scope is separately audited.
        if folder.name == 'shared-mug-continuation-v1':
            assert report['prefix_motor_steps'] == 38760
            assert all(row['status'] == 'failed' and 'search' in row for row in report['attempts'])
        elif folder.name == 'shared-glass-buffer-geometry-v1':
            assert report['prefix_motor_steps'] == 3497 and report['live_state_unchanged']
        else:
            raise ValueError('This exporter is scoped to the two named completed geometry diagnoses.')
        rows.append({'run': folder.name, 'result': clean(report), 'new_physical_action_steps': 0,
                     'report_sha256': digest(folder/'report.json'),
                     'source_sha256': {path.relative_to(folder/'source').as_posix(): digest(path)
                                       for path in sorted((folder/'source').rglob('*.py'))},
                     'array_sha256': {path.relative_to(folder).as_posix(): digest(path)
                                      for path in sorted(folder.glob('*.npz'))}})
    value = {'schema': 'talos_shared_geometry_diagnostics_v1', 'runs': rows,
             'meaning': 'Exact replayed prefixes followed by scratch-only geometry. No new physical successes, learned result or full table.',
             'preservation': 'All original reports, source snapshots and arrays remain in their local run folders.'}
    payload = (json.dumps(value, indent=2)+'\n').encode()
    require_space(args.output, len(payload)+1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps({'runs': len(rows), 'new_physical_action_steps': 0, 'bytes': len(payload),
                      'sha256': digest(args.output), 'preflight': preflight}))


if __name__ == '__main__':
    main()
