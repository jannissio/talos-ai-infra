"""Preserve the complete isolated PC check of the downloadable Intel kit."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import os
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run():
    archive = ROOT/'submission/verification-v1/talos-intel-verification.zip'
    extracted = ROOT/'.run/intel-verification-kit-v1/extracted'
    raw = extracted/'.run/pc-bundle-cpu'
    output = ROOT/'submission/verification-v1/pc-portability'
    if output.exists():
        raise FileExistsError('Preserve existing kit evidence.')
    verification, physical = read(raw/'verification.json'), read(raw/'physical.json')
    if not verification['physical_passed'] or verification['strict_hardware_and_physics_passed']:
        raise ValueError('Expected physical success and a correctly false strict hardware result on this AMD/NVIDIA PC.')
    skills = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon']
    if [row['skill'] for row in physical['results']] != skills or any(row['status'] != 'succeeded' for row in physical['results']):
        raise ValueError('Every selected dinner skill must succeed in the preserved trial.')
    if any(physical[field] != 0 for field in ('physics_state_writes', 'hidden_forces', 'equality_constraints', 'teacher_updates')):
        raise ValueError('The kit trial contains privileged assistance.')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC failed.')
        manifest = json.loads(bundle.read('kit-manifest.json'))
        if set(bundle.namelist()) != set(manifest['files']) | {'kit-manifest.json'}:
            raise ValueError('Archive contents differ from the manifest.')
        for name, digest in manifest['files'].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest or sha(extracted/name) != digest:
                raise ValueError('An archived or extracted input changed: '+name)
        archive_members = len(bundle.namelist())
    suite = read(extracted/'models/dinner_suite/suite.json')
    for skill in skills:
        source = extracted/'models/dinner_suite'/suite[skill]/'primitive.safetensors'
        if sha(source) != physical['checkpoint_sha256'][skill] or sha(source) != verification['source_checkpoint_sha256'][skill]:
            raise ValueError('The physically evaluated model differs from the kit.')
        if not verification['benchmarks'][skill]['parity_passed']:
            raise ValueError('Every selected CPU export must pass parity.')
    mapping = {raw/name: name for name in ('verification.json', 'hardware.json', 'physical.json')}
    mapping.update({raw/skill/'benchmark.json': f'benchmarks/{skill}.json' for skill in skills})
    mapping[raw/'physical-recording/states.npz'] = 'states.npz'
    mapping[raw/'physical-recording/scene.xml'] = 'source-scene.xml'
    mapping[extracted/'kit-manifest.json'] = 'kit-manifest.json'
    preflight = require_space(output, sum(path.stat().st_size for path in mapping)+4*1024**2)
    output.mkdir(parents=True)
    for source, name in mapping.items():
        require_space(output/name, source.stat().st_size+1024)
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    scene = ET.parse(output/'source-scene.xml')
    source_meshdir = Path(scene.getroot().find('compiler').get('meshdir'))
    resolved = (raw/'physical-recording'/source_meshdir).resolve()
    relative_assets = resolved.relative_to(extracted)
    scene.getroot().find('compiler').set('meshdir', os.path.relpath(ROOT/relative_assets, output).replace('\\', '/'))
    require_space(output/'scene.xml', 1024**2)
    with (output/'scene.xml').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(ET.tostring(scene.getroot(), encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    data = mujoco.MjData(model)
    with np.load(output/'states.npz', allow_pickle=False) as state:
        if state['qpos'].shape[1] != model.nq or not np.isfinite(state['qpos']).all():
            raise ValueError('The portable scene does not match the complete trace.')
        for index in (0, len(state['time'])-1):
            data.qpos[:] = state['qpos'][index]
            data.qvel[:] = state['qvel'][index]
            mujoco.mj_forward(model, data)
        frames = len(state['time'])
        simulation_seconds = float(state['time'][-1])
    original = ET.parse(output/'source-scene.xml')
    original.getroot().find('compiler').set('meshdir', 'ASSET_ROOT')
    comparison = ET.parse(output/'scene.xml')
    comparison.getroot().find('compiler').set('meshdir', 'ASSET_ROOT')
    if ET.tostring(original.getroot()) != ET.tostring(comparison.getroot()):
        raise ValueError('Only the asset path may change in the portable scene.')
    audit = {'schema': 'talos.intel-verification-kit-pc-portability.v1',
        'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'source_git_revision': manifest['source_git_revision'], 'archive_members': archive_members,
        'all_archive_and_extracted_hashes_match': True, 'isolated_python_mode': True,
        'invocation': 'python -I scripts/verify_intel_submission.py --suite models/dinner_suite/suite.json --output .run/pc-bundle-cpu --device CPU --diagnostic',
        'physical_passed': verification['physical_passed'], 'all_six_cpu_exports_pass_parity': True,
        'strict_hardware_and_physics_passed': verification['strict_hardware_and_physics_passed'],
        'cpu': verification['hardware']['cpu'], 'renderer': verification['hardware']['opengl']['renderer'],
        'wall_seconds': physical['wall_seconds'], 'simulation_seconds': simulation_seconds, 'trace_frames': frames,
        'source_sha256': sha(Path(__file__)), 'storage_preflight': preflight,
        'scope': 'Exposed seed-42 PC portability regression of the downloadable kit. Actual Intel execution, hardware eligibility, human voice and final submission remain open.'}
    require_space(output/'audit.json', 1024**2)
    with (output/'audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(audit, indent=2)+'\n')
    files = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob('*') if path.is_file()}
    with (output/'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps({'files': files, 'bytes': sum((output/name).stat().st_size for name in files)}, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    run()
