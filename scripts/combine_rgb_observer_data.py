"""Reference existing checked training shards without duplicating their images."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, 1024**2); args.output.mkdir(parents=True)
    result = {'schema': 'talos.rgb-observer-combined.v1', 'split': 'train', 'states': 0, 'completed_states': 0,
              'shards': [], 'sources': [p.as_posix() for p in args.inputs], 'storage': 'Relative references; source bytes preserved.'}
    for folder in args.inputs:
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['split'] != 'train' or manifest['completed_states'] != manifest['states']:
            raise ValueError('Only complete declared training sources may be combined.')
        for shard in manifest['shards']:
            result['shards'].append({**shard, 'file': os.path.relpath(folder/shard['file'], args.output).replace('\\', '/')})
        result['states'] += manifest['states']; result['completed_states'] += manifest['completed_states']
        result['views'] = manifest['views']; result['calibrations'] = manifest['calibrations']
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'states': result['states'], 'shards': len(result['shards'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
