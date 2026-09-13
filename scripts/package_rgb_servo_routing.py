"""Preserve both routing revisions, every final outcome and compact raw evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space
from scripts.summarize_rgb_servo_routing import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root):
    target = root / 'docs/robotics/evidence/rgb-servo-v2'
    if target.exists():
        raise FileExistsError(target)
    raw = root / '.run'
    selection = root / 'docs/robotics/experiments/rgb-servo-final-evaluation-v2.json'
    audit = summarize(raw / 'rgb-servo-v2-final-r1', selection)
    freeze = json.loads(selection.read_text())
    for name, digest in freeze['source_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Runtime changed after freeze: ' + name)
    require_space(target, 128 * 1024**2)
    target.mkdir(parents=True)
    files = []

    def copy(source, destination, scene=False):
        require_space(destination, source.stat().st_size + 1024**2)
        payload = source.read_bytes()
        transformation = 'byte-identical copy'
        if scene:
            tree = ET.fromstring(payload)
            compiler = tree.find('compiler')
            meshes = (source.parent / compiler.get('meshdir')).resolve()
            if not meshes.is_relative_to(root / 'simulation_lab/assets') or not meshes.is_dir():
                raise ValueError('Scene assets are outside the repository.')
            compiler.set('meshdir', os.path.relpath(meshes, destination.parent).replace('\\', '/'))
            payload = ET.tostring(tree, encoding='utf-8')
            transformation = 'Only compiler meshdir remapped to the same repository assets.'
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError('Conflicting content-addressed source.')
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        files.append({'source': source.relative_to(root).as_posix(), 'source_sha256': sha(source),
                      'file': destination.relative_to(root).as_posix(), 'sha256': sha(destination),
                      'bytes': len(payload), 'transformation': transformation})

    def write(destination, value):
        require_space(destination, 1024**2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((json.dumps(value, indent=2) + '\n').encode())

    def trial(source, destination):
        report = json.loads((source / 'report.json').read_text())
        for name, digest in report['source_sha256'].items():
            original = source / 'evaluated-source' / name
            if sha(original) != digest:
                raise ValueError('Historical source mismatch: ' + str(original))
            copy(original, target / 'source-blobs' / (digest + '.py'))
        for name in ('report.json', 'states.npz', 'image-counterfactuals.json', 'observations.json', 'scene.xml'):
            original = source / name
            if original.exists():
                copy(original, destination / name, scene=name == 'scene.xml')
        return report

    for name, digest in freeze['source_sha256'].items():
        copy(root / name, target / 'source-blobs' / (digest + '.py'))
    copy(selection, target / 'frozen-selection.json')
    batches = [('rgb-servo-v2-final-r1', 'final'), ('rgb-servo-v2-development-r1', 'development-r1'),
               ('rgb-servo-v2-development-r2', 'development-r2')]
    total = 0
    for original_name, name in batches:
        source, destination = raw / original_name, target / name
        result = json.loads((source / 'summary.json').read_text())
        if result['attempted'] != result['planned']:
            raise ValueError('Incomplete batch: ' + original_name)
        for item in ('summary.json', 'frozen-inputs.json'):
            copy(source / item, destination / item)
        for row in result['rows']:
            report_path = source / row['report']
            if sha(report_path) != row['report_sha256']:
                raise ValueError('Historical report hash mismatch.')
            trial(report_path.parent, destination / report_path.parent.name)
            total += 1
    training = []
    for source in sorted(raw.glob('rgb-servo-v2-routing-*-r*')):
        report = trial(source, target / 'training' / source.name)
        training.append({'seed': report['seed'], 'revision': source.name.rsplit('-', 1)[-1],
                         'status': report['status'], 'message': report.get('message'),
                         'report': 'training/' + source.name + '/report.json'})
    write(target / 'training-summary.json', {'all_exposed_attempts': training})
    for name in ('rgb-servo-v2-motor-probe', 'rgb-servo-v2-offset-diagnostic'):
        copy(raw / name / 'report.json', target / 'diagnostics' / name / 'report.json')
    copy(root / 'scripts/probe_rgb_servo_motor.py', target / 'diagnostics/probe_rgb_servo_motor.py')
    copy(raw / 'final-goal/diagnose_v2_offset_routes.py', target / 'diagnostics/diagnose_v2_offset_routes.py')
    write(target / 'audit.json', audit)
    write(target / 'package-manifest.json', {
        'schema': 'talos.rgb-servo-routing-package.v2', 'files': files,
        'batch_trials': total, 'exposed_training_trials': len(training), 'originals_modified': False,
        'weights': 'Unchanged models/bottle_servo_v1; no additional training or new weight copy.',
        'omissions': 'Camera PNGs remain in the raw archive. Every report, authoritative state trace, compact RGB observation, image-action counterfactual and unique evaluated source version is included.',
    })
    packaged = summarize(target / 'final', target / 'frozen-selection.json')
    if packaged != audit:
        raise ValueError('Packaged audit differs from original.')
    print(json.dumps({'packaged_files': len(files), 'bytes': sum(row['bytes'] for row in files),
                      'batch_trials': total, 'exposed_trials': len(training), 'audit': audit}))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    package(Path(__file__).resolve().parents[1])
