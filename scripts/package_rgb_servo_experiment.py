"""Package the completed RGB experiment without promoting it over the baseline.

Copies every final outcome. Compact training states preserve the exact rows
consumed by the image renderer; raw recordings and all candidates stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from simulation_lab.storage import require_space
from scripts.summarize_rgb_servo_evaluation import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root):
    model_dir = root/'models/bottle_servo_v1'
    training_dir = root/'training/bottle_servo_v1'
    evidence_dir = root/'docs/robotics/evidence/rgb-servo-v1'
    destinations = (model_dir, training_dir, evidence_dir)
    for folder in destinations:
        if folder.exists():
            raise FileExistsError(folder)
    require_space(root, 96*1024**2)
    batch = root/'.run/rgb-servo-final-v1'
    freeze_path = root/'docs/robotics/experiments/rgb-servo-final-evaluation-v1.json'
    freeze = json.loads(freeze_path.read_text())
    audit = summarize(batch, freeze_path)
    for name, sha in freeze['source_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Runtime changed after final evaluation: '+name)
    for folder in destinations:
        folder.mkdir(parents=True)
    files = []

    def record(source, target, transformation='byte-identical copy'):
        files.append({'source': source.relative_to(root).as_posix(), 'source_sha256': digest(source),
                      'file': target.relative_to(root).as_posix(), 'sha256': digest(target),
                      'bytes': target.stat().st_size, 'transformation': transformation})

    def copy(source, target, xml=False):
        require_space(target, source.stat().st_size+1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        if xml:
            tree = ET.fromstring(source.read_text())
            compiler = tree.find('compiler')
            meshdir = (source.parent/compiler.get('meshdir')).resolve()
            if not meshdir.is_relative_to(root/'simulation_lab/assets') or not meshdir.is_dir():
                raise ValueError('Scene mesh directory is outside packaged assets.')
            compiler.set('meshdir', os.path.relpath(meshdir, target.parent).replace('\\', '/'))
            target.write_bytes(ET.tostring(tree, encoding='utf-8'))
            record(source, target, 'Only compiler meshdir remapped to the same repository assets.')
        else:
            target.write_bytes(source.read_bytes())
            record(source, target)

    def write(target, obj):
        require_space(target, 1024**2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((json.dumps(obj, indent=2)+'\n').encode())

    for source, target in (
        ('.run/rgb-servo-observer-fit-v4/step-004000/model.safetensors', 'observer.safetensors'),
        ('.run/rgb-servo-motor-fit-v1/model.safetensors', 'motor.safetensors')):
        copy(root/source, model_dir/target)
    for name in ('observer.xml', 'observer.bin', 'motor.xml', 'motor.bin', 'parity.json'):
        copy(root/'.run/rgb-servo-openvino-v4'/name, model_dir/'openvino'/name)
    for name in freeze['source_sha256']:
        copy(root/name, evidence_dir/'evaluated-source'/name)
    copy(freeze_path, evidence_dir/'frozen-selection.json')
    copy(batch/'summary.json', evidence_dir/'summary.json')
    copy(batch/'frozen-inputs.json', evidence_dir/'frozen-inputs.json')
    write(evidence_dir/'audit.json', audit)
    final_summary = json.loads((batch/'summary.json').read_text())
    for row in final_summary['rows']:
        source = batch/Path(row['report']).parent
        target = evidence_dir/source.name
        for name in ('report.json', 'states.npz', 'image-counterfactuals.json'):
            copy(source/name, target/name)
        copy(source/'scene.xml', target/'scene.xml', xml=True)
    # Preserve selection failures and perception outcomes, not only the final fit.
    histories = (
        'rgb-servo-development-v4/summary.json',
        'rgb-servo-development-rigid-v1/summary.json',
        'rgb-servo-development-openvino-v1/summary.json',
        'rgb-servo-observer-frozen-evaluation-v1/report.json',
        'rgb-servo-observer-frozen-evaluation-v3/report.json',
        'rgb-servo-observer-frozen-evaluation-v4/report.json',
        'rgb-servo-rigid-frozen-evaluation-v1/report.json',
        'rgb-servo-rigid-frozen-openvino-v1/report.json')
    for name in histories:
        source = root/'.run'/name
        if not source.exists():
            source = source.with_name('results.json')
        copy(source, evidence_dir/'history'/source.parent.name/source.name)
    for version in range(1,6):
        copy(root/f'.run/rgb-servo-observer-fit-v{version}/training.json', training_dir/f'fits/observer-v{version}.json')
    copy(root/'.run/rgb-servo-motor-fit-v1/training.json', training_dir/'fits/motor-v1.json')
    for source in (root/'.run/rgb-servo-motor-data-v1').iterdir():
        if source.is_file():
            copy(source, training_dir/'motor'/source.name)
    pose_sources = []
    for old_name, name in (('full-dinner-42-transition-v6', 'dinner-42'), ('learned-relay-42-grip-v3', 'relay-42')):
        source = root/'.run/final-goal'/old_name/'states.npz'
        target = training_dir/'robot-poses'/name/'states.npz'
        require_space(target, 16*1024**2); target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as data:
            # The static collector consumes only these 12 joint positions.
            np.savez_compressed(target, qpos=data['qpos'][:, :12])
        record(source, target, 'Exact qpos[:, :12] array, unchanged dtype and row ordering.')
        with np.load(target, allow_pickle=False) as reduced:
            pose_sources.append({'folder': target.parent.relative_to(root).as_posix(), 'rows': len(reduced['qpos'])})
    motion_source = root/'.run/rgb-servo-motion-v1'
    copy(motion_source/'summary.json', training_dir/'motion/summary.json')
    motion_summary = json.loads((motion_source/'summary.json').read_text())
    for row in motion_summary['rows']:
        source = motion_source/f'seed-{row["seed"]}'
        target = training_dir/'motion'/source.name
        copy(source/'report.json', target/'report.json')
        if not row['training_eligible']:
            continue
        copy(source/'scene.xml', target/'scene.xml', xml=True)
        original = source/'replay-states.npz'
        reduced = target/'replay-states.npz'
        require_space(reduced, 4*1024**2)
        with np.load(original, allow_pickle=False) as data:
            indices = np.linspace(0, len(data['time'])-1, 128).astype(int)
            np.savez_compressed(reduced, **{k: data[k][indices] for k in ('qpos', 'qvel', 'time')})
        record(original, reduced, 'Exact 128 linspace-selected qpos/qvel/time rows used by render_rgb_servo_motion_labels.py.')
    write(training_dir/'motion/target-audit.json', {
        'original_reports_preserved': True,
        'correction': 'Original destination_m fields are requested goals. The original collector did not insert a bottle target; teacher and monitor used the fallback goal. Do not interpret these 15 passing motions as coverage of the requested destinations.',
        'training_use': 'Only physically replayed RGB poses supervise observer v3/v4. The motor network uses the separate kinematic dataset.',
        'reproduction': 'Use the included compact replay states. Rerunning the corrected collector creates different demonstrations.'})
    copy(root/'.run/rgb-servo-recovery-motion-v1/summary.json', evidence_dir/'history/recovery-motion/summary.json')
    for folder in ('rgb-servo-observer-train-v2', 'rgb-servo-observer-train-v4', 'rgb-servo-observer-dev-v1',
                   'rgb-servo-motion-labels-v1', 'rgb-servo-observer-combined-v4', 'rgb-servo-rigid-evaluation-data-v1'):
        copy(root/'.run'/folder/'manifest.json', training_dir/'original-dataset-manifests'/folder/'manifest.json')
    write(model_dir/'experiment.json', {
        'schema': 'talos.rgb-servo-package.v1', 'status': 'experimental; not promoted',
        'observer': 'observer.safetensors', 'observer_sha256': freeze['observer_sha256'],
        'motor': 'motor.safetensors', 'motor_sha256': freeze['motor_sha256'],
        'openvino': 'openvino', 'openvino_artifact_sha256': freeze['openvino_artifact_sha256'],
        'minimum_views': 2, 'rigid_geometry': True,
        'evidence': '../../docs/robotics/evidence/rgb-servo-v1/audit.json',
        'baseline': '../dinner_suite/suite.json', 'license': 'MIT',
        'scope': 'Known upright bottle; RGB correction during approach/descent, proprioceptive carry. Finite coverage test failed; no arbitrary-placement claim.'})
    write(training_dir/'provenance.json', {
        'schema': 'talos.rgb-servo-training-provenance.v1', 'robot_pose_sources': pose_sources,
        'selected_chain': ['observer-v2: 6000 steps from scratch', 'observer-v3: 4000 steps warm-started from v2',
                           'observer-v4: 4000 steps warm-started from v3 at learning rate 0.0003'],
        'observer_training_states_v4': 10112, 'selected_motion_examples': 15, 'selected_motion_frames': 1920,
        'motor_training_labels': 4056, 'motor_attempted_training_points': 6144,
        'total_experiment_adam_steps_including_rejected_fits_and_motor': 30000,
        'pixel_reproduction': 'Compact source arrays are exact. Re-rendered pixels may vary across OpenGL drivers; original shard hashes are retained for comparison, without claiming cross-driver bit identity.',
        'pretrained_external_weights': False, 'private_data': False})
    write(evidence_dir/'package-manifest.json', {'schema': 'talos.rgb-servo-package-manifest.v1', 'files': files,
          'all_final_trials': 48, 'originals_modified': False,
          'omitted_raw_data': 'Original RGB shards, per-frame observation JSON and camera PNGs remain local. All 48 physical reports, compact authoritative states, action counterfactuals, source snapshots and model artifacts are included.'})
    verified = summarize(evidence_dir, evidence_dir/'frozen-selection.json')
    if verified['results'] != audit['results']:
        raise ValueError('Packaged evaluation changed.')
    print(json.dumps({'files': len(files), 'copied_or_derived_bytes': sum(r['bytes'] for r in files),
                      'all_48_reports_verified': True, 'status': 'experimental; baseline unchanged'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    package(Path(__file__).resolve().parents[1])
