"""Preserve and independently audit a stopped observer-camera adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from simulation_lab.rgb_servo_cameras import KEYPOINTS, VIEWS, project
from simulation_lab.scene import build_scene
from simulation_lab.storage import GIB, require_space

SPLITS = ('train', 'development', 'evaluation')
LABELS = ('mask', 'keypoints_px', 'visible', 'world_points', 'present')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def summary(rows, gate, nested=False):
    observations = [row['observation'] if nested else row for row in rows]
    present = sum(row['scoring_only_present'] for row in rows)
    accepted = sum(row['scoring_only_present'] and obs['status'] == 'observed'
                   for row, obs in zip(rows, observations))
    false = sum(not row['scoring_only_present'] and obs['status'] == 'observed'
                for row, obs in zip(rows, observations))
    errors = [row['scoring_only_error_mm'] for row in rows if row.get('scoring_only_error_mm') is not None]
    p95, maximum = (float(np.quantile(errors, .95)), max(errors)) if errors else (None, None)
    passed = bool(present and len(rows) > present and errors
                  and accepted / present >= gate['minimum_present_acceptance_fraction']
                  and false <= gate['maximum_false_accepted_absent']
                  and p95 <= gate['maximum_accepted_p95_error_mm']
                  and maximum <= gate['maximum_accepted_error_mm'])
    return dict(states=len(rows), present=present, accepted_present=accepted, absent=len(rows)-present,
                false_accepted_absent=false, accepted_error_p95_mm=p95,
                accepted_error_max_mm=maximum, gate_passed=passed)


def audit(package):
    folder = package / 'validation'
    training, protocol = read(package/'training.json'), read(package/'protocol.json')
    selected, freeze = read(package/'selection.json'), read(folder/'fresh-perception-freeze.json')
    result, cpu = read(folder/'perception-verification.json'), read(folder/'cpu-development-audit.json')
    if (not training['training_completed'] or training['completed_steps'] != protocol['training']['steps']
            or selected != training['selection'] or result['protocol_sha256'] != sha(package/'protocol.json')
            or result['freeze_sha256'] != sha(folder/'fresh-perception-freeze.json')
            or freeze['cpu_development_audit_sha256'] != sha(folder/'cpu-development-audit.json')
            or freeze['export_parity_sha256'] != sha(folder/'openvino/parity.json')):
        raise ValueError('Training selection, CPU validation or fresh-test freeze changed.')
    for row in read(package/'training-manifest.json')['files']:
        if sha(package/row['file']) != row['sha256']:
            raise ValueError('The original training package changed: '+row['file'])
    inputs = read(folder/'reproduction-inputs.json')
    for name, digest in inputs['repository_asset_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A scene asset changed: '+name)
    for name, digest in inputs['source_sha256'].items():
        if sha(folder/'source'/name) != digest:
            raise ValueError('A reproduction source changed: '+name)
    for name, digest in training['source_sha256'].items():
        if sha(folder/'source'/name) != digest:
            raise ValueError('A training source differs from the original fit: '+name)
    for name, digest in freeze['openvino_artifact_sha256'].items():
        if sha(folder/'openvino'/name) != digest:
            raise ValueError('An exported artifact differs from the pre-test freeze: '+name)
    parity = read(folder/'openvino/parity.json')
    if (parity['observer_sha256'] != selected['checkpoint_sha256']
            or parity['motor_sha256'] != protocol['motor_sha256']
            or not all(row['parity_passed'] for row in parity['networks'].values())
            or sha(ROOT/protocol['warm_start']) != protocol['warm_start_sha256']
            or sha(ROOT/protocol['motor']/'motor.safetensors') != protocol['motor_sha256']):
        raise ValueError('Warm start, fixed motor or export parity mismatch.')
    gate = protocol['perception_gate']
    datasets, model = {}, mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
    data = mujoco.MjData(model)
    states_verified = 0
    for split in SPLITS:
        paired = []
        for config in protocol['camera_configurations']:
            name = split+'-'+config
            current = folder/'data'/name
            manifest, states, labels = read(current/'manifest.json'), arrays(current/'sampled-states.npz'), arrays(current/'labels.npz')
            count = protocol[split+'_states']
            if (manifest['states'] != count or manifest['completed_states'] != count
                    or manifest['protocol_sha256'] != sha(package/'protocol.json')
                    or manifest['camera_protocol_sha256'] != sha(folder/'camera-protocol.json')
                    or manifest['sampled_states_sha256'] != sha(current/'sampled-states.npz')
                    or manifest['generator_sha256'] != sha(folder/'source/scripts/collect_rgb_servo_camera_observer.py')
                    or any(len(value) != count for value in [*states.values(), *labels.values()])):
                raise ValueError('Incomplete or mismatched compact data: '+name)
            original = training['dataset_manifests'].get(name)
            if original and (manifest['original_manifest_sha256'] != original['sha256']
                             or manifest['shards'] != original['shards']):
                raise ValueError('A normalized training manifest lost its original provenance.')
            for pose_name, digest in manifest['arm_pose_source_sha256'].items():
                if sha(ROOT/pose_name) != digest:
                    raise ValueError('Arm-pose input changed.')
            truth = []
            for pose in states['qpos']:
                data.qpos[:] = pose
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                truth.append(KEYPOINTS @ data.body('bottle').xmat.reshape(3, 3).T + data.body('bottle').xpos)
            truth = np.asarray(truth, dtype=np.float32)
            if (not all(np.isfinite(states[key]).all() for key in ('qpos', 'light_diffuse', 'headlight_ambient'))
                    or not np.array_equal(truth, labels['world_points'])
                    or not np.array_equal(states['present'], labels['present'])):
                raise ValueError('Independent MuJoCo geometry differs from saved float32 labels.')
            hashes = read(current/'rgb-sha256.json')
            if len(hashes['states']) != count or any(len(row) != 3 for row in hashes['states']):
                raise ValueError('Incomplete per-image reproduction hashes.')
            paired.append(states)
            datasets[name] = labels
            states_verified += count
        if paired[0].keys() != paired[1].keys() or any(not np.array_equal(paired[0][key], paired[1][key]) for key in paired[0]):
            raise ValueError('Camera-paired scenes differ: '+split)
    discrepancy, observations_verified = 0., 0

    def score(rows, labels, nested=False):
        nonlocal discrepancy, observations_verified
        if len(rows) != len(labels['present']) or [r['index'] for r in rows] != list(range(len(rows))):
            raise ValueError('Missing, duplicate or out-of-order observations.')
        for index, row in enumerate(rows):
            present = bool(labels['present'][index])
            obs = row['observation'] if nested else row
            if row['scoring_only_present'] != present:
                raise ValueError('Recorded presence differs from compact labels.')
            if present and obs['status'] == 'observed':
                error = float(np.linalg.norm(np.asarray(obs['keypoints_m'])-labels['world_points'][index], axis=1).max()*1000)
                discrepancy = max(discrepancy, abs(error-row['scoring_only_error_mm']))
            elif row.get('scoring_only_error_mm') is not None:
                raise ValueError('An absent or refused observation has a position score.')
        observations_verified += len(rows)
        return summary(rows, gate, nested)

    for measured in [training['baseline_development'], *[c['development'] for c in training['candidates']]]:
        for config in protocol['camera_configurations']:
            entry = measured[config]
            if score(entry['rows'], datasets['development-'+config], True) != {k:v for k,v in entry.items() if k != 'rows'}:
                raise ValueError('A GPU development summary differs from its observations.')
    eligible = [c for c in training['candidates'] if c['development'][protocol['deployment_configuration']]['gate_passed']]
    choice = min(eligible, key=lambda c: (-c['development'][protocol['deployment_configuration']]['accepted_present'],
                                        c['development'][protocol['deployment_configuration']]['accepted_error_p95_mm']))
    if choice['step'] != selected['step'] or choice['sha256'] != selected['checkpoint_sha256']:
        raise ValueError('Selection violates the declared development ordering.')
    for candidate in training['candidates']:
        if sha(package/'candidates'/Path(candidate['checkpoint']).name) != candidate['sha256']:
            raise ValueError('A retained candidate changed.')
    derived = {}
    for split in ('development', 'evaluation'):
        derived[split] = {}
        for config in protocol['camera_configurations']:
            report = read(folder/(split+'-cpu-'+config)/'results.json')
            if report['checkpoint_sha256'] != selected['checkpoint_sha256'] or report['minimum_views'] != 2 or not report['rigid_geometry']:
                raise ValueError('CPU evaluation changed the selected model or decoder.')
            measured = score(report['rows'], datasets[split+'-'+config])
            if split == 'development':
                gpu = choice['development'][config]['rows']
                measured['changed_acceptance_decisions_from_cuda'] = sum(a['status'] != b['observation']['status'] for a,b in zip(report['rows'], gpu))
                measured['maximum_score_delta_from_cuda_mm'] = max((abs(a['scoring_only_error_mm']-b['scoring_only_error_mm'])
                    for a,b in zip(report['rows'], gpu) if a.get('scoring_only_error_mm') is not None and b['scoring_only_error_mm'] is not None), default=0.)
                if measured != cpu[config] or measured != result['development'][config]:
                    raise ValueError('CPU development audit differs from the frozen pre-test check.')
            elif measured != result['fresh_perception'][config]:
                raise ValueError('Fresh perception summary differs from observations.')
            derived[split][config] = measured
    if discrepancy > 1e-8:
        raise ValueError('Independent observation scoring disagrees with reported errors.')
    selected_gate = derived['evaluation'][protocol['deployment_configuration']]['gate_passed']
    if selected_gate or result['selected_fresh_gate_passed'] or result['physical_trials'] != 0:
        raise ValueError('This package is specifically for a stopped fresh-perception experiment.')
    collection_seconds = sum(read(folder/'data'/(split+'-'+config)/'manifest.json')['wall_seconds']
                             for split in SPLITS for config in protocol['camera_configurations'])
    if (collection_seconds > 60*protocol['budget']['maximum_collection_minutes']
            or training['wall_seconds'] > 60*protocol['budget']['maximum_training_minutes']
            or training['peak_torch_reserved_gib'] > protocol['budget']['maximum_peak_vram_gib']):
        raise ValueError('A declared collection or training resource limit was exceeded.')
    return dict(schema=protocol['schema'], all_observations_verified=observations_verified,
                camera_state_records_verified=states_verified, unique_paired_states=states_verified//2,
                maximum_scoring_discrepancy_mm=discrepancy, selected_step=selected['step'],
                collection_seconds=collection_seconds, training_seconds=training['wall_seconds'],
                peak_torch_reserved_gib=training['peak_torch_reserved_gib'], fresh_perception=derived['evaluation'],
                status='Stopped at fresh perception maximum-error gate', physical_trials=0, promoted=False,
                physical_training_seeds_exposed=[], physical_development_seeds_exposed=[], physical_final_seeds_exposed=[],
                scope='Static perception on exposed arm-pose families, OpenVINO CPU on AMD; no physical or Intel claim.')


def package(source, output):
    folder = output/'validation'
    if folder.exists():
        raise FileExistsError('Preserve the existing validation package.')
    protocol = read(output/'protocol.json')
    before = {r['file']:r['sha256'] for r in read(output/'training-manifest.json')['files']}
    used = sum(p.stat().st_size for base in (source, output) for p in base.rglob('*') if p.is_file())
    if used+64*1024**2 > protocol['budget']['maximum_new_data_gib']*GIB:
        raise ValueError('Packaging would exceed the cumulative data budget.')
    require_space(folder, 64*1024**2)
    folder.mkdir(parents=True)

    def write(name, payload):
        path = folder/name
        require_space(path, len(payload)+1024**2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)

    def json_write(name, value):
        write(name, (json.dumps(value, indent=2)+'\n').encode())

    for name in ('fresh-perception-freeze.json', 'cpu-development-audit.json', 'perception-verification.json'):
        write(name, (source/name).read_bytes())
    write('camera-protocol.json', (ROOT/protocol['camera_protocol']).read_bytes())
    for path in (source/'openvino-export').iterdir():
        if path.is_file():
            write('openvino/'+path.name, path.read_bytes())
    source_names = sorted(str(p.relative_to(ROOT).as_posix()) for p in (ROOT/'simulation_lab').rglob('*.py') if '__pycache__' not in p.parts)
    source_names += ['scripts/collect_rgb_servo_camera_observer.py', 'scripts/train_rgb_servo_camera_observer.py',
                     'scripts/train_rgb_servo_observer.py', 'scripts/evaluate_rgb_servo_observer.py',
                     'scripts/export_rgb_servo_openvino.py', 'scripts/package_rgb_servo_camera_training.py']
    source_hashes = {}
    for name in source_names:
        write('source/'+name, (ROOT/name).read_bytes())
        source_hashes[name] = sha(ROOT/name)
    write('orchestration/verify_observer_camera_v1.py', (ROOT/'.run/final-goal/verify_observer_camera_v1.py').read_bytes())
    xml, _ = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    document = ET.fromstring(xml)
    compiler = document.find('compiler')
    meshes = Path(compiler.get('meshdir')).resolve()
    assets = {}
    for element in document.iter():
        if 'file' in element.attrib:
            asset_base = meshes if element.tag == 'mesh' else ROOT
            path = (asset_base/element.attrib['file']).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('A simulation asset is outside the repository.')
            assets[path.relative_to(ROOT).as_posix()] = sha(path)
            element.set('file', Path(os.path.relpath(path, folder)).as_posix())
    compiler.set('meshdir', '.')
    write('scene.xml', ET.tostring(document, encoding='utf-8'))
    json_write('reproduction-inputs.json', dict(source_sha256=source_hashes, repository_asset_sha256=assets,
               scene_transformation='Same seed-42 task scene; XML asset paths made repository-relative. No sampled-state transformation.'))
    for split in SPLITS:
        for config in protocol['camera_configurations']:
            name = split+'-'+config
            raw = source/name
            manifest = read(raw/'manifest.json')
            # The original training runner passed an absolute protocol path. Keep
            # its hash, normalize only that field, and never publish the private path.
            manifest['original_manifest_sha256'] = sha(raw/'manifest.json')
            manifest['protocol'] = 'docs/robotics/experiments/rgb-servo-observer-camera-training-v1.json'
            manifest['packaging_transformation'] = 'Protocol path normalized; original manifest SHA retained; all other original fields preserved.'
            json_write('data/'+name+'/manifest.json', manifest)
            write('data/'+name+'/sampled-states.npz', (raw/'sampled-states.npz').read_bytes())
            labels, image_hashes = {key:[] for key in LABELS}, []
            for shard in manifest['shards']:
                path = raw/shard['file']
                if sha(path) != shard['sha256']:
                    raise ValueError('A raw training or evaluation shard changed.')
                values = arrays(path)
                for key in LABELS:
                    labels[key].append(values[key])
                image_hashes.extend([[hashlib.sha256(image.tobytes()).hexdigest() for image in state] for state in values['rgb']])
            target = folder/'data'/name/'labels.npz'
            require_space(target, 32*1024**2)
            with target.open('xb') as stream:
                np.savez_compressed(stream, **{key:np.concatenate(parts) for key,parts in labels.items()})
            json_write('data/'+name+'/rgb-sha256.json', dict(shape=[240,320,3], dtype='uint8', order='C', views=list(VIEWS), states=image_hashes,
                       scope='Exact original per-view pixels. Regenerated pixels may differ with renderer/driver; raw images remain local.'))
            if split != 'train':
                evaluation = source/(split+'-cpu-'+config)
                for path in evaluation.iterdir():
                    if path.is_file():
                        write(split+'-cpu-'+config+'/'+path.name, path.read_bytes())
                if split == 'evaluation':
                    report = read(evaluation/'results.json')
                    worst = max((row for row in report['rows'] if row.get('scoring_only_error_mm') is not None), key=lambda row:row['scoring_only_error_mm'])
                    offset = worst['index']
                    for shard in manifest['shards']:
                        if offset < shard['states']:
                            values = arrays(raw/shard['file'])
                            break
                        offset -= shard['states']
                    canvas = Image.new('RGB', (960,280), '#0d1c2b')
                    draw = ImageDraw.Draw(canvas)
                    for j, view in enumerate(VIEWS):
                        canvas.paste(Image.fromarray(values['rgb'][offset,j]), (j*320,0))
                        for u,v in worst['views'][view]['keypoints_px']:
                            draw.ellipse((j*320+u-3,v-3,j*320+u+3,v+3), outline='lime')
                        for u,v in project(values['world_points'][offset], manifest['calibrations'][view]['projection']):
                            draw.rectangle((j*320+u-2,v-2,j*320+u+2,v+2), outline='red')
                    draw.text((10,246), f"{config}, exposed frame {worst['index']}: {worst['scoring_only_error_mm']:.3f} mm; green prediction, red reference", fill='white')
                    require_space(folder/('worst-'+config+'.png'), 1024**2)
                    canvas.save(folder/('worst-'+config+'.png'))
    # Validate the source snapshot against every collection's actual Git parent.
    # The raw manifests preserve the collection revision even after normalization.
    revisions = {read(folder/'data'/(s+'-'+c)/'manifest.json')['parent_git_revision'] for s in SPLITS for c in protocol['camera_configurations']}
    for revision in revisions:
        for name in ('simulation_lab/scene.py', *assets):
            original = subprocess.check_output(['git', 'show', revision+':'+name])
            if hashlib.sha256(original).hexdigest() != sha(ROOT/name):
                raise ValueError('Scene source or assets changed since collection.')
    result = audit(output)
    json_write('audit.json', result)
    entries = [{'file':p.relative_to(folder).as_posix(), 'sha256':sha(p), 'bytes':p.stat().st_size}
               for p in sorted(folder.rglob('*')) if p.is_file()]
    json_write('manifest.json', dict(files=entries, source=source.relative_to(ROOT).as_posix(),
               training_package_unchanged=True, originals_modified=False,
               compact_labels='Exact concatenated original arrays excluding RGB; pixel hashes and rendering recipe retained.'))
    if any(sha(output/name) != digest for name,digest in before.items()):
        raise ValueError('The pre-existing training package changed.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'.run/rgb-servo-observer-camera-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/robotics/evidence/rgb-servo-observer-camera-v1')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.verify_only:
        for row in read(args.output/'validation/manifest.json')['files']:
            path = args.output/'validation'/row['file']
            if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
                raise ValueError('Validation package file changed: '+row['file'])
        result = audit(args.output)
        if result != read(args.output/'validation/audit.json'):
            raise ValueError('Saved independent audit changed.')
    else:
        result = package(args.source, args.output)
    print(json.dumps(result))
