"""Preserve completed shared-pipeline development with compact portable evidence.

Includes every attempted motor trace and search result. Raw RGB-D arrays remain
in the original local archive; their exact array hashes, calibrated recipes and
heightmap previews are included so a later renderer audit can check reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from simulation_lab.storage import require_space


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def put(path, value):
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:stream.write(payload)


def copy(source, target):
    require_space(target, source.stat().st_size+1024); target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:shutil.copyfileobj(src, dst)
    assert sha(source) == sha(target)


def package(output):
    if output.exists():raise FileExistsError('Preserve the existing evidence export.')
    folders = sorted(p for p in (ROOT/'.run').glob('whole-table-*')
                     if p.is_dir() and 'randomizer' not in p.name and (p/'report.json').exists())
    pending = sorted(p.name for p in (ROOT/'.run').glob('whole-table-*')
                     if p.is_dir() and 'randomizer' not in p.name and not (p/'report.json').exists())
    estimate = sum(p.stat().st_size for folder in folders for p in folder.rglob('*')
                   if p.is_file() and p.name != 'observation.npz')+32*1024**2
    preflight = require_space(output, estimate); output.mkdir(parents=True)
    rows = []; totals = {'physical_attempts': 0, 'trace_frames': 0, 'successful_primitives': 0,
                        'accepted_continuous_actions': 0, 'complete_randomized_tables': 0,
                        'rgbd_observations_hashed': 0, 'exact_replay_attempts': 0}
    for folder in folders:
        report = json.loads((folder/'report.json').read_text())
        raw_manifest = []
        for source in sorted(p for p in folder.rglob('*') if p.is_file()):
            name = source.relative_to(folder)
            raw_manifest.append({'path': name.as_posix(), 'bytes': source.stat().st_size, 'sha256': sha(source)})
            target = output/'runs'/folder.name/name
            if source.name == 'observation.npz':
                with np.load(source) as data:
                    hashes = {key: {'shape': list(data[key].shape), 'dtype': str(data[key].dtype),
                        'array_sha256': hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest()}
                        for key in data.files}
                put(target.with_name('observation-array-hashes.json'), hashes)
                totals['rgbd_observations_hashed'] += 1
            else:copy(source, target)
        results = [json.loads(p.read_text()) for p in folder.glob('*/result.json')]
        row = {'run': folder.name, 'seed': report['seed'], 'reset_distribution': report.get('reset_distribution', 'joint'),
            'only_item_diagnostic': report.get('only_item_diagnostic'),
            'valid_initial_geometry': report.get('valid_initial_geometry'),
            'physical_attempts': len(results), 'trace_frames': sum(r['frames'] for r in results),
            'successful_primitives': sum(r['demonstration_eligible'] for r in results),
            'exact_replay_attempts': sum(r['replay_exact'] for r in results),
            'accepted_continuous_actions': sum(r.get('continuous_main_scene_replay_exact', False) for r in report['actions']),
            'whole_table_complete': report['whole_table_complete'], 'remaining': report['remaining'],
            'error_type': report.get('error_type')}
        for key in ('physical_attempts', 'trace_frames', 'successful_primitives', 'accepted_continuous_actions', 'exact_replay_attempts'):
            totals[key] += row[key]
        totals['complete_randomized_tables'] += bool(row['whole_table_complete'])
        rows.append(row)
        put(output/'runs'/folder.name/'original-raw-manifest.json', raw_manifest)
    # These dependencies were unchanged throughout the recorded development.
    dependencies = ['simulation_lab/scene.py', 'simulation_lab/dinner.py', 'simulation_lab/storage.py',
                    'simulation_lab/rgb_servo_cameras.py']
    for name in dependencies:copy(ROOT/name, output/'unchanged-source'/name)
    assets = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT/'simulation_lab/assets/so101').rglob('*') if p.is_file()}
    put(output/'robot-assets-sha256.json', assets)
    for source in (ROOT/'.run/spatial-runtime-v1').iterdir():
        if source.is_file():copy(source, output/'spatial-runtime'/source.name)
    copy(ROOT/'.run/clip-rn50-v1/download.json', output/'clip-backbone-download.json')
    summary = {'schema': 'talos.shared-table-development.v1', 'preflight': preflight,
        'scope': 'Completed exploratory development attempts, not a frozen generalization evaluation. Repeated seeds are exposed. The shared visual policy has not been trained.',
        'totals': totals, 'runs': rows, 'unfinished_local_jobs_not_in_this_snapshot': pending,
        'camera_archive': 'Original raw RGB-D archives remain unchanged locally. This compact export retains canonical array hashes, calibration/scene/state recipes and every preview; raw images are not duplicated.',
        'claims': 'GPU runtime and camera geometry are verified. A complete randomized learned table-setting capability is unfinished.'}
    put(output/'summary.json', summary)
    lines = ['# Shared table pipeline: development evidence', '', summary['scope'], '',
        f"Retains {len(rows)} completed runs, {totals['physical_attempts']} physical attempts and {totals['trace_frames']:,} state frames. There are {totals['successful_primitives']} successful primitives and {totals['complete_randomized_tables']} completed randomized tables. These runs include explicitly labeled narrow positive controls; they are not broad success rates.", '',
        'The adapted CLIPort pick-location and 36-rotation placement heads pass one numerical optimizer step each on a real 256×320×6 calibrated observation: 5.49 GiB peak Torch reservation. No robotics policy checkpoint is trained or exported. `spatial-runtime/report.json` binds the exact tested source.', '',
        f"Every attempted physical motor trace, failed search and run source snapshot is retained. {totals['exact_replay_attempts']}/{totals['physical_attempts']} saved motor-command replays independently reproduce their recorded physics exactly; passing actions are additionally replayed into the continuously evolving main scene. The teacher uses privileged geometry and physics lookahead to generate demonstrations. It is not the learned controller and does not establish online recovery.", '',
        summary['camera_archive'], '',
        'See `summary.json` for every run, the invalid starts, unfinished local jobs and counts. Original outputs are never overwritten. GitHub/Hugging Face stay private; final Submit remains the user’s action.', '']
    put(output/'README.md', '\n'.join(lines).encode())
    manifest = {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in output.rglob('*') if p.is_file()}
    put(output/'manifest.json', manifest)
    for name, item in manifest.items():assert sha(output/name) == item['sha256'], name
    print({'completed_runs': len(rows), **totals, 'pending': pending,
           'bound_files': len(manifest), 'bytes': sum(item['bytes'] for item in manifest.values())}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    package(parser.parse_args().output)
