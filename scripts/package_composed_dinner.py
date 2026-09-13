"""Audit and preserve all composed-command trials without changing the models."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    destination = ROOT / 'docs/robotics/evidence/composed-dinner-v1'
    if destination.exists():
        raise FileExistsError(destination)
    freeze_path = ROOT / 'docs/robotics/experiments/composed-dinner-final-v1.json'
    freeze = json.loads(freeze_path.read_text())
    protocol = json.loads((ROOT / 'docs/robotics/experiments/composed-dinner-relay-v1.json').read_text())
    source_keys = ('source_sha256', 'model_files_sha256')
    for category in source_keys:
        for name, digest in freeze[category].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Frozen input changed: ' + name)
    result = json.loads((ROOT / '.run/composed-dinner-v1-final/summary.json').read_text())
    if result['attempted'] != result['planned'] or result['planned'] != 10:
        raise ValueError('The ten-scene batch is not complete.')
    require_space(destination, 100 * 1024**2)
    files, outcomes, total_frames = [], [], 0
    destination.mkdir(parents=True)

    def copy(source, target, scene=False):
        require_space(target, source.stat().st_size + 1024**2)
        payload = source.read_bytes()
        transformation = 'byte-identical copy'
        if scene:
            tree = ET.fromstring(payload)
            compiler = tree.find('compiler')
            meshes = (source.parent / compiler.get('meshdir')).resolve()
            if not meshes.is_relative_to(ROOT / 'simulation_lab/assets'):
                raise ValueError('Unexpected mesh directory.')
            compiler.set('meshdir', os.path.relpath(meshes, target.parent).replace('\\', '/'))
            payload = ET.tostring(tree, encoding='utf-8')
            transformation = 'Only compiler meshdir remapped to the same repository assets.'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if scene:
            model = mujoco.MjModel.from_xml_path(str(target))
            if model.neq:
                raise ValueError('Unexpected equality constraint.')
        files.append({'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
                      'file': target.relative_to(ROOT).as_posix(), 'sha256': sha(target),
                      'bytes': len(payload), 'transformation': transformation})

    copy(freeze_path, destination / 'frozen-selection.json')
    for name in freeze['source_sha256']:
        copy(ROOT / name, destination / 'evaluated-source' / name)
    copy(ROOT / 'scripts/evaluate_composed_dinner_suite.py', destination / 'evaluated-source/scripts/evaluate_composed_dinner_suite.py')
    for split in ('development', 'evaluation'):
        raw = ROOT / '.run' / ('composed-dinner-v1-development' if split == 'development' else 'composed-dinner-v1-final')
        for seed in protocol[split + '_seeds']:
            source, target = raw / str(seed), destination / split / str(seed)
            path = source / 'report.json'
            report = json.loads(path.read_text())
            for key, value in report['frozen_inputs'].items():
                if freeze[key] != value:
                    raise ValueError('A trial used different frozen inputs.')
            for name, digest in freeze['source_sha256'].items():
                if sha(source / 'evaluated-source' / name) != digest:
                    raise ValueError('Saved source differs from the frozen input.')
            if report['seed'] != seed or report['split'] != split:
                raise ValueError('Wrong trial identity.')
            if report['physics_state_writes_during_control'] or report['hidden_forces']:
                raise ValueError('Unexpected physical assistance.')
            if report.get('visual_plan', {}).get('steps') != protocol['expected_steps']:
                raise ValueError('The expected RGB-grounded workflow was not selected.')
            passed = report['status'] == 'succeeded'
            task = report.get('task', {})
            if passed and task.get('completed_steps') != protocol['expected_steps']:
                raise ValueError('Success lacks all seven completed learned steps.')
            for step in task.get('results', []):
                if step['teacher_updates'] or step['policy_details']['inference_demonstration_actions']:
                    raise ValueError('A trial used a teacher or retrieved demonstration actions.')
                if step['status'] == 'succeeded':
                    metrics = step['metrics']
                    if not metrics['both_arms_parked'] or metrics['unexpected_collisions']:
                        raise ValueError('A completed skill violates physical checks.')
                    if step['skill'] != 'drawer' and metrics['placement_error_mm'] > 8:
                        raise ValueError('A placement exceeds the unchanged threshold.')
            with np.load(source / 'states.npz', allow_pickle=False) as states:
                size = len(states['time'])
                if size == 0 or any(len(states[k]) != size for k in states.files):
                    raise ValueError('Missing synchronized states.')
                if np.any(np.diff(states['time']) <= 0):
                    raise ValueError('Nonmonotonic recording time.')
                if not all(np.isfinite(states[k]).all() for k in ('qpos', 'qvel', 'targets', 'time')):
                    raise ValueError('Nonfinite recording values.')
                total_frames += size
            for name in ('report.json', 'states.npz', 'scene.xml', 'frozen-inputs.json'):
                copy(source / name, target / name, scene=name == 'scene.xml')
            outcomes.append({'seed': seed, 'split': split, 'passed': passed,
                             'completed_steps': task.get('completed_steps'),
                             'message': report.get('message', task.get('message')),
                             'simulation_seconds': report['simulation_seconds'], 'wall_seconds': report['wall_seconds'],
                             'report': split + '/' + str(seed) + '/report.json', 'report_sha256': sha(path)})
    counts = {split: sum(r['passed'] for r in outcomes if r['split'] == split) for split in ('development', 'evaluation')}
    if counts['evaluation'] != result['passed']:
        raise ValueError('Final summary count disagrees with reports.')
    for row in result['rows']:
        if sha(ROOT / '.run/composed-dinner-v1-final' / row['report']) != row['report_sha256']:
            raise ValueError('Final batch report changed.')
    copy(ROOT / '.run/composed-dinner-v1-final/summary.json', destination / 'original-final-summary.json')
    audit = {'schema': 'talos.composed-dinner-audit.v1', 'passed_by_split': counts,
             'trials_by_split': {'development': 4, 'evaluation': 10}, 'outcomes': outcomes,
             'synchronized_state_frames': total_frames, 'source_files_verified': len(freeze['source_sha256']),
             'model_files_verified': len(freeze['model_files_sha256']), 'model_changes': 0,
             'final_gate_passed': counts['evaluation'] >= 8,
             'scope': 'Finite left-reach preset, unchanged models, actual language/RGB plan and seven physical neural steps. No general reachable-workspace or unrestricted language claim.'}
    for name, value in [('audit.json', audit), ('package-manifest.json', {'files': files, 'originals_modified': False})]:
        require_space(destination / name, 1024**2)
        (destination / name).write_bytes((json.dumps(value, indent=2) + '\n').encode())
    raw_bytes = sum(p.stat().st_size for d in (ROOT / '.run').glob('composed-dinner-v1*') if d.is_dir() for p in d.rglob('*') if p.is_file())
    package_bytes = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    if raw_bytes + package_bytes > protocol['budget']['maximum_raw_and_packaged_mib'] * 1024**2:
        raise ValueError('Combined compact evidence exceeds its declared budget.')
    print(json.dumps({'passes': counts, 'files': len(files), 'recorded_frames': total_frames,
                      'raw_bytes': raw_bytes, 'packaged_bytes': package_bytes}))


if __name__ == '__main__':
    run()
